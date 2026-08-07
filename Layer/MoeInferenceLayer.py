from typing import List

from chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode

from Layer.Layer import BaseInferenceLayer, LayerEmission, MoeEpContext
from Layer.DenseBlockMath import DenseBlockMath
from Layer.MoeBlock import MoeBlock
from Utils.config import MoeConfig
from Utils.naming import comp_name, coll_name
from Utils.nodes import allreduce, compute
from Utils.routing import RoutingPlan


class MoeInferenceLayer(BaseInferenceLayer):
    """A single Mixture-of-Experts inference block:
        attention -> [all-reduce] -> gating -> dispatch -> experts -> combine -> weighted sum
    The attention half is identical to the dense block: MoE changes the FFN, not the cache """

    def __init__(self, model_cfg, moe: MoeConfig, plan: RoutingPlan, layer_idx: int, tp_size: int = 1):
        # attention follows the tensor sharding; the MoE half has its own cost model, shared
        # with the training layer, which keeps the experts whole
        self.math_attn = DenseBlockMath.from_model_cfg(model_cfg, tp_size)
        self.moe_block = MoeBlock(model_cfg, plan, layer_idx)
        self.model_cfg = model_cfg
        self.moe = moe
        self.plan = plan
        self.layer_idx = layer_idx
        self.tp_size = tp_size

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
        # sequence parallelism: this device owns 1/tp of its slice's tokens, full hidden each
        local_tokens = tokens // self.tp_size
        c = self.moe_block.costs(key=key, ep_ctx=ep_ctx, local_tokens=local_tokens)
        nodes: List[ChakraNode] = []

        # 1. attention, identical to the dense block
        attn = compute(attn_flops, attn_bytes, name=comp_name(name, "attn"))
        nodes.append(attn)
        attn_end = attn
        if self.tp_size > 1:
            attn_end = allreduce(self.math_attn.allreduce_bytes(tokens), pg_name=pg_name, parents=[attn], name=coll_name(name, "attn"))
            nodes.append(attn_end)
        kv_ready = attn_end          # the cache exists once attention has run: unchanged by MoE

        # 2. router: assigns each locally owned token to its top_k experts
        gate = compute(c.gate_flops, c.gate_bytes, parents=[attn_end], name=comp_name(name, "gate"))
        nodes.append(gate)

        # 3. dispatch: tokens travel to the devices hosting their experts
        arrived = self.moe_block.exchange(nodes, c.traffic, ep_ctx, op="disp", parents=[gate], it=step, name_prefix=name)

        # 4. expert FFN over everything routed here, local copies included
        ffw = compute(c.expert_flops, c.expert_bytes, parents=arrived + [gate], name=comp_name(name, "ffw"))
        nodes.append(ffw)

        # 5. combine: the transposed matrix returns the results to the token owners
        returned = self.moe_block.exchange(nodes, c.traffic.T, ep_ctx, op="comb", parents=[ffw], it=step, name_prefix=name)

        # 6. weighted sum of each token's top_k expert outputs. Depending on every incoming
        #    combine edge, it is also the single tail the orchestrator chains the next layer on.
        tail = compute(c.combine_flops, c.combine_bytes, parents=returned + [ffw], name=comp_name(name, "comb"))
        nodes.append(tail)
        return LayerEmission(nodes=nodes, tail=tail, kv_ready=kv_ready)