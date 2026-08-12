from typing import List

from chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode

from Layer.Layer import BaseInferenceLayer, LayerEmission, MoeEpContext
from Layer.DenseBlockMath import DenseBlockMath
from Layer.MoeBlock import MoeBlock
from Utils.config import MoeConfig
from Utils.naming import comp_name, coll_name
from Utils.nodes import all_gather, allreduce, compute, reduce_scatter
from Utils.routing import RoutingPlan


class MoeInferenceLayer(BaseInferenceLayer):
    """A single Mixture-of-Experts inference block:
        attention -> [reduce-scatter] -> gating -> dispatch -> experts -> combine -> [all-gather]
    The attention half computes exactly what the dense block computes. What it also changes is the TENSOR-PARALLEL collective around it: here the
    expert half is sequence-parallel (each tensor rank owns 1/tp of the tokens and dispatches them) so the first all-reduce is split into the 
    reduce-scatter that feeds it and the all-gather that closes it
    With dp = 1 and tp > 1 vLLM activates no a2a kernel at all: tokens stay replicated on the tensor ranks and the block degenerates to the
    dense-like shape, two all-reduces around masked local-expert ffn compute (_emit_masked)."""

    def __init__(self, model_cfg, moe: MoeConfig, plan: RoutingPlan, layer_idx: int, tp_size: int = 1, dp_size: int = 1):
        # attention follows the tensor sharding; the MoE half has its own cost model, shared
        # with the training layer, which keeps the experts whole
        self.math_attn = DenseBlockMath.from_model_cfg(model_cfg, tp_size)
        self.moe_block = MoeBlock(model_cfg, plan, layer_idx)
        self.model_cfg = model_cfg
        self.moe = moe
        self.plan = plan
        self.layer_idx = layer_idx
        self.tp_size = tp_size
        self.dp_size = dp_size

    @property
    def attn_weight_elems(self) -> int:
        return self.math_attn.attn_weight_elems

    @property
    def ffn_weight_elems(self) -> int:
        """Every expert of this layer: what the model as a whole holds, not one device's share."""
        return self.moe_block.expert_math.ffn_weight_elems * self.moe.num_experts

    def prefill(self, name: str, pg_name: str | None, prompt_lens: List[int],cached_lens: List[int], ep_ctx: MoeEpContext | None = None, key: tuple | None = None, step: int = 0) -> LayerEmission:
        new_tokens, cached_tokens, score_entries = DenseBlockMath.prefill_counts(prompt_lens, cached_lens)
        attn_flops, attn_bytes = self.math_attn.attn_costs(
            query_tokens=new_tokens, kv_read_tokens=cached_tokens,
            kv_write_tokens=new_tokens, score_entries=score_entries)
        return self._emit(name, pg_name, attn_flops, attn_bytes, new_tokens, ep_ctx, key, step)

    def decode(self, name: str, pg_name: str | None, kv_lens: List[int],
               ep_ctx: MoeEpContext | None = None, key: tuple | None = None,
               step: int = 0) -> LayerEmission:
        batch_size, total_kv_tokens = len(kv_lens), sum(kv_lens)
        attn_flops, attn_bytes = self.math_attn.attn_costs(query_tokens=batch_size, kv_read_tokens=total_kv_tokens, kv_write_tokens=batch_size, score_entries=total_kv_tokens)
        return self._emit(name, pg_name, attn_flops, attn_bytes, batch_size, ep_ctx, key, step)

    def _emit(self, name, pg_name, attn_flops, attn_bytes, tokens, ep_ctx, key, step) -> LayerEmission:
        if self.tp_size > 1 and self.dp_size == 1:
            return self._emit_masked(name, pg_name, attn_flops, attn_bytes, tokens, ep_ctx, key)
        # sequence parallelism: this device owns 1/tp of its slice's tokens, full hidden each
        local_tokens = tokens // self.tp_size
        c = self.moe_block.costs(key=key, ep_ctx=ep_ctx, local_tokens=local_tokens)
        wire = c.traffic
        nodes: List[ChakraNode] = []

        # 1. attention, identical to the dense block
        attn = compute(attn_flops, attn_bytes, name=comp_name(name, "attn"))
        nodes.append(attn)
        attn_end = attn
        if self.tp_size > 1:
            attn_end = reduce_scatter(self.math_attn.allreduce_bytes(tokens), pg_name=pg_name,parents=[attn], name=coll_name(name, "rs"))
            nodes.append(attn_end)
        kv_ready = attn_end          # the cache exists once attention has run: unchanged by MoE

        # A COMP node is emitted only when it has work to do
        # 2. router: assigns each locally owned token to its top_k experts
        routed = [attn_end]
        if c.gate_bytes:
            gate = compute(c.gate_flops, c.gate_bytes, parents=[attn_end], name=comp_name(name, "gate"))
            nodes.append(gate)
            routed = [gate]

        # 3. dispatch: tokens travel to the devices hosting their experts
        arrived = self.moe_block.exchange(nodes, wire, ep_ctx, op="disp", parents=routed, it=step)

        # 4. expert FFN over everything routed here, local copies included. Skipped when the router sent this rank nothing
        after_experts = arrived + routed
        if c.expert_bytes:
            ffw = compute(c.expert_flops, c.expert_bytes, parents=arrived + routed, name=comp_name(name, "ffw"))
            nodes.append(ffw)
            after_experts = [ffw]

        # 5. combine: the transposed matrix returns the results to the token owners
        returned = self.moe_block.exchange(nodes, wire.T, ep_ctx, op="comb", parents=after_experts, it=step)

        # 6. all-gather closing the sequence-parallel window opened by the reduce-scatter
        if self.tp_size > 1:
            # the shard, not the whole tensor: all_gather's coll_size is per rank
            tail = all_gather(self.math_attn.allreduce_bytes(tokens) // self.tp_size, pg_name=pg_name, parents=returned + after_experts, name=coll_name(name, "ag"))
            nodes.append(tail)
        else:
            # tp = 1: no sequence-parallel window, so nothing closes the block and the combine recvs ARE the tail
            tail = returned or after_experts
        return LayerEmission(nodes=nodes, tail=tail, kv_ready=kv_ready)

    def _emit_masked(self, name, pg_name, attn_flops, attn_bytes, tokens, ep_ctx, key) -> LayerEmission:
        """dp = 1 with tp > 1: no token ever moves. Dense-like frame with the FFN
        replaced by router + masked local experts + weighted accumulation of the partials."""
        c = self.moe_block.masked_costs(key=key, ep_ctx=ep_ctx, tokens=tokens)
        ar_bytes = self.math_attn.allreduce_bytes(tokens)
        nodes: List[ChakraNode] = []

        attn = compute(attn_flops, attn_bytes, name=comp_name(name, "attn"))
        nodes.append(attn)
        attn_end = allreduce(ar_bytes, pg_name=pg_name, parents=[attn], name=coll_name(name, "attn"))
        nodes.append(attn_end)
        kv_ready = attn_end

        gate = compute(c.gate_flops, c.gate_bytes, parents=[attn_end], name=comp_name(name, "gate"))
        nodes.append(gate)

        # skipped when none of this rank's local experts was hit
        after_experts = gate
        if c.expert_bytes:
            ffw = compute(c.expert_flops, c.expert_bytes, parents=[gate], name=comp_name(name, "ffw"))
            nodes.append(ffw)
            after_experts = ffw

        # every rank holds a PARTIAL of the full output (its experts' contributions only)
        tail = allreduce(ar_bytes, pg_name=pg_name, parents=[after_experts], name=coll_name(name, "ffw"))
        nodes.append(tail)
        return LayerEmission(nodes=nodes, tail=tail, kv_ready=kv_ready)