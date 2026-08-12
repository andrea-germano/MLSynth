from typing import NamedTuple

from chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode

from Layer.DenseBlockMath import DenseBlockMath
from Layer.Layer import MoeEpContext
from Utils.config import ModelConfig
from Utils.naming import a2a_name
from Utils.nodes import alltoall_v
from Utils.routing import RoutingPlan


class MoeBlockCosts(NamedTuple):
    """Cost of the MoE half of a block for one device."""
    gate_flops: int
    gate_bytes: int
    expert_flops: int
    expert_bytes: int
    combine_flops: int # weighted sum of each token's top_k expert outputs
    combine_bytes: int
    traffic: object # [ep, ep] routed token copies; rows are sources, columns destinations


class MoeBlock:
    """The MoE half of a transformer block shared by the training and the inference layer the same way DenseBlockMath is shared for the dense half"""

    def __init__(self, model_cfg: ModelConfig, plan: RoutingPlan, layer_idx: int):
        self.expert_math = DenseBlockMath.from_model_cfg(model_cfg, tp_size=1)
        self.moe = model_cfg.moe
        self.plan = plan
        self.layer_idx = layer_idx
        self.hidden_size = self.expert_math.hidden_size
        self.bytes_per_val = model_cfg.bytes_per_val
        self.scale = model_cfg.scale

    def edge_bytes(self, copies: int) -> int:
        """Bytes of one dispatch/combine edge: `copies` whole tokens, full hidden dimension."""
        return int(self.scale * copies * self.hidden_size * self.bytes_per_val)

    def costs(self, *, key: tuple, ep_ctx: MoeEpContext, local_tokens: int) -> MoeBlockCosts:
        """`local_tokens` is what this device owns and routes."""
        num_experts = self.moe.num_experts
        hidden, b, scale = self.hidden_size, self.bytes_per_val, self.scale

        origin_tokens = ep_ctx.origin_tokens or [local_tokens] * ep_ctx.size
        # the tokens this device owns are its entry of origin_tokens, kept consistent with its
        # row of the traffic matrix (the caller's local_tokens is only the uniform fallback)
        local_tokens = origin_tokens[ep_ctx.ep_rank]

        # router: [tokens, hidden] x [hidden, E]. A small matrix, replicated across the tensor
        # ranks rather than sharded, so its cost is not divided further.
        gate_flops = int(scale * 2 * local_tokens * hidden * num_experts)
        gate_bytes = int(scale * local_tokens * hidden * b)

        traffic = self.plan.traffic_matrix(key, self.layer_idx, origin_tokens)
        expert_flops, expert_bytes = self._expert_costs(key, ep_ctx, origin_tokens, traffic)

        # the weighted sum reads what actually comes back: this rank's wire row, diagonal
        # included. One multiply-add per returned vector per element, plus the write of the local tokens' outputs.
        returned = int(traffic[ep_ctx.ep_rank].sum())
        combine_flops = int(scale * 2 * returned * hidden)
        combine_bytes = int(scale * (returned + local_tokens) * hidden * b)

        return MoeBlockCosts(gate_flops, gate_bytes, expert_flops, expert_bytes, combine_flops, combine_bytes, traffic)

    def _expert_costs(self, key, ep_ctx, origin_tokens, traffic) -> tuple[int, int]:
        """FFN cost of everything routed to this rank. The weights read are those of the
        experts actually HIT (grouped GEMM semantics), not the whole local block"""
        tokens_routed_here = int(traffic[:, ep_ctx.ep_rank].sum())   # column sum, diagonal included
        if not tokens_routed_here:
            return 0, 0
        hit = self.plan.experts_hit(key, self.layer_idx, origin_tokens, ep_ctx.ep_rank)
        return self.expert_math.ffn_costs(tokens_routed_here, weight_copies=hit)

    def masked_costs(self, *, key: tuple, ep_ctx: MoeEpContext, tokens: int) -> MoeBlockCosts:
        """The no-a2a regime (dp = 1 with tp > 1): vLLM activates no all-to-all kernel there, so after the attention all-reduce every tensor rank holds ALL tokens replicated."""
        top_k, num_experts = self.moe.top_k, self.moe.num_experts
        hidden, b, scale = self.hidden_size, self.bytes_per_val, self.scale
        origin_tokens = ep_ctx.origin_tokens or [tokens // ep_ctx.size] * ep_ctx.size

        # router replicated over the FULL token set, not this rank's 1/tp slice
        gate_flops = int(scale * 2 * tokens * hidden * num_experts)
        gate_bytes = int(scale * tokens * hidden * b)

        traffic = self.plan.traffic_matrix(key, self.layer_idx, origin_tokens)
        expert_flops, expert_bytes = self._expert_costs(key, ep_ctx, origin_tokens, traffic)

        # weighted accumulation of the local experts' outputs into this rank's partial: one multiply-add per routed copy per element; reads each copy and writes the whole partial, which is full-size
        copies_here = int(traffic[:, ep_ctx.ep_rank].sum())
        combine_flops = int(scale * 2 * copies_here * hidden)
        combine_bytes = int(scale * (copies_here + tokens) * hidden * b)

        return MoeBlockCosts(gate_flops, gate_bytes, expert_flops, expert_bytes,combine_flops, combine_bytes, traffic)

    def exchange(self, nodes: list, matrix, ep_ctx: MoeEpContext, *, op: str, parents, it) -> list[ChakraNode]:
        """Emit one routed all-to-all over `matrix` as explicit p2p SEND/RECV pairs"""
        if ep_ctx.size <= 1: # every expert is local: nothing to exchange
            return []

        def edge_name(src_ep, dst_ep):
            return a2a_name(pl=ep_ctx.pool, op=op, stage=ep_ctx.stage, se=src_ep, de=dst_ep, L=self.layer_idx, it=it,cl=ep_ctx.cluster if ep_ctx.clusters > 1 else None)

        emitted, recvs = alltoall_v(matrix, ep_ctx.peers, ep_ctx.ep_rank, parents=parents, size_for=self.edge_bytes, name_for=edge_name)
        nodes.extend(emitted)
        return recvs