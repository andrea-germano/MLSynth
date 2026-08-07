from typing import NamedTuple

from chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode

from Layer.DenseBlockMath import DenseBlockMath
from Layer.Layer import MoeEpContext
from Utils.config import ModelConfig
from Utils.naming import a2a_name
from Utils.nodes import alltoall, alltoall_v
from Utils.routing import RoutingPlan


class MoeBlockCosts(NamedTuple):
    """Cost of the MoE half of a block for one device."""
    gate_flops: int
    gate_bytes: int
    expert_flops: int
    expert_bytes: int
    combine_flops: int     # weighted sum of each token's top_k expert outputs
    combine_bytes: int
    traffic: object        # [ep, ep] routed token copies; rows are sources, columns destinations


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
        top_k, num_experts = self.moe.top_k, self.moe.num_experts
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
        tokens_routed_here = int(traffic[:, ep_ctx.ep_rank].sum())   # column sum, diagonal included
        expert_flops, expert_bytes = self.expert_math.ffn_costs(tokens_routed_here, weight_copies=num_experts // ep_ctx.size)

        # after the combine, a token sums its top_k expert outputs weighted by its gate scores: one multiply-add per element per expert
        combine_flops = int(scale * 2 * top_k * local_tokens * hidden)
        combine_bytes = int(scale * (top_k + 1) * local_tokens * hidden * b)

        return MoeBlockCosts(gate_flops, gate_bytes, expert_flops, expert_bytes, combine_flops, combine_bytes, traffic)

    def exchange(self, nodes: list, matrix, ep_ctx: MoeEpContext, *, op: str, parents, it, name_prefix: str) -> list[ChakraNode]:
        """Emit one all-to-all over `matrix`, appending to `nodes`"""
        if ep_ctx.size <= 1: # every expert is local: nothing to exchange
            return []

        if self.plan.routing.dispatch == "collective":
            # Training-only baseline (inference forbids it in config). comm_size is this device's
            # TOTAL routed volume, own share included, read off the matrix itself
            own_row = int(matrix[ep_ctx.ep_rank].sum())
            node = alltoall(self.edge_bytes(own_row), pg_name=ep_ctx.pg_name, parents=parents, name=f"{name_prefix}_ep_alltoall_{op}")
            nodes.append(node)
            return [node]

        def edge_name(src_ep, dst_ep):
            return a2a_name(pl=ep_ctx.pool, op=op, stage=ep_ctx.stage, se=src_ep, de=dst_ep, L=self.layer_idx, it=it,cl=ep_ctx.cluster if ep_ctx.clusters > 1 else None)

        emitted, recvs = alltoall_v(matrix, ep_ctx.peers, ep_ctx.ep_rank, parents=parents, size_for=self.edge_bytes, name_for=edge_name)
        nodes.extend(emitted)
        return recvs