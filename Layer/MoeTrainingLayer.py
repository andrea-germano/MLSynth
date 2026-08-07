# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import NamedTuple
from mlsynth.Layer.Layer import BaseTrainingLayer, MoeEpContext
from Layer.DenseBlockMath import DenseBlockMath
from Utils.config import ModelConfig
from Utils.naming import a2a_name
from Utils.nodes import allreduce, alltoall, alltoall_v, compute
from Utils.routing import PHASE_FWD, RoutingPlan
from chakra.schema.protobuf.et_def_pb2 import (
    Node as ChakraNode,
)


class BlockCosts(NamedTuple):
    """Everything the forward and the backward pass need, computed once per emission."""
    attn_flops: int
    attn_bytes: int
    tp_comm_size: int      # bytes of the attention all-reduce (tp > 1 only)
    gate_flops: int
    gate_bytes: int
    combine_flops: int     # weighted sum of the top_k expert outputs (the "unpermute")
    combine_bytes: int
    ffn_flops: int
    ffn_bytes: int
    traffic: object        # [ep, ep] routed token copies; rows are sources, columns destinations


class MoeTrainingLayer(BaseTrainingLayer):
    """A single Mixture-of-Experts training block:
        forward   attention -> [all-reduce] -> gating -> dispatch -> experts -> combine -> sum
        backward  sum -> grad-combine -> experts -> grad-dispatch -> gating -> attention -> [AR]
    """

    def __init__(self, model_cfg: ModelConfig, sequence_len: int, tp_size: int, plan: RoutingPlan, layer_idx: int):
        self.math = DenseBlockMath.from_model_cfg(model_cfg, tp_size)
        self.model_cfg = model_cfg
        self.moe = model_cfg.moe
        self.routing = plan.routing
        self.sequence_len = sequence_len
        self.tp_size = tp_size
        self.plan = plan
        self.layer_idx = layer_idx

    def _costs(self, num_batches, ep_ctx: MoeEpContext, microbatch: int) -> BlockCosts:
        tokens = int(num_batches * self.sequence_len)
        score_entries = num_batches * self.sequence_len * self.sequence_len
        attn_flops, attn_bytes = self.math.attn_costs(query_tokens=tokens, kv_read_tokens=0, kv_write_tokens=tokens, score_entries=score_entries)

        # sequence parallelism (which Megatron requires with TP+EP): this device owns 1/tp of the tokens, each with the full hidden dimension
        local_tokens = tokens // self.tp_size

        # the router is a small [hidden, E] matrix, replicated across TP ranks rather than sharded
        hidden, bytes_per_val = self.math.hidden_size, self.model_cfg.bytes_per_val
        gate_flops = int(self.math.scale * 2 * local_tokens * hidden * self.moe.num_experts)
        gate_bytes = int(self.math.scale * local_tokens * hidden * bytes_per_val)

        # after the combine, a token owns top_k expert outputs and sums them weighted by its gate scores
        # One multiply-add per element per expert
        top_k = self.moe.top_k
        combine_flops = int(self.math.scale * 2 * top_k * local_tokens * hidden)
        combine_bytes = int(self.math.scale * (top_k + 1) * local_tokens * hidden * bytes_per_val)

        # one realisation per EP cluster: clusters hold different tokens, so they route differently
        traffic = self.plan.traffic_matrix(
            (PHASE_FWD, microbatch, self.layer_idx, ep_ctx.cluster),
            self.layer_idx, [local_tokens] * ep_ctx.size)
        tokens_routed_here = int(traffic[:, ep_ctx.ep_rank].sum())   # column sum, diagonal included
        ffn_flops, ffn_bytes = self.math.ffn_costs(tokens_routed_here, weight_copies=self.moe.num_experts // ep_ctx.size)

        return BlockCosts(attn_flops, attn_bytes, self.math.allreduce_bytes(tokens), gate_flops, gate_bytes, combine_flops, combine_bytes, ffn_flops, ffn_bytes, traffic)

    def _edge_bytes(self, copies: int) -> int:
        """Bytes of one dispatch/combine edge carrying `copies` routed token copies, each a full
        hidden vector (experts are whole, so the exchange moves complete tokens)."""
        return int(self.math.scale * copies * self.math.hidden_size * self.model_cfg.bytes_per_val)

    def _exchange(self, nodes, matrix, ep_ctx: MoeEpContext, *, op, parents, pass_tag, name_prefix):
        """Emit one all-to-all over `matrix`, appending to `nodes`. Returns the nodes that gate
        whatever consumes the exchange: the RECVs on the p2p path, the collective otherwise."""
        if ep_ctx.size <= 1:                 # every expert is local: nothing to exchange
            return []

        if self.routing.dispatch == "collective":
            own_row = int(matrix[ep_ctx.ep_rank].sum())
            node = alltoall(self._edge_bytes(own_row), pg_name=ep_ctx.pg_name, parents=parents, name=f"{name_prefix}_ep_alltoall_{op}")
            nodes.append(node)
            return [node]

        def edge_name(src_ep, dst_ep):
            return a2a_name(pl="t", op=op, stage=ep_ctx.stage, se=src_ep, de=dst_ep, L=self.layer_idx, it=pass_tag, cl=ep_ctx.cluster)

        emitted, recvs = alltoall_v(matrix, ep_ctx.peers, ep_ctx.ep_rank, parents=parents, size_for=self._edge_bytes, name_for=edge_name)
        nodes.extend(emitted)
        return recvs

    def fwd(self, name="node_fwd", pg_name=None, num_batches=1, ep_ctx: MoeEpContext | None = None, microbatch: int = 0) -> list[ChakraNode]:
        c = self._costs(num_batches, ep_ctx, microbatch)
        pass_tag = f"f{microbatch}"
        nodes: list[ChakraNode] = []

        # 1. attention, identical to the dense block
        attention_compute = compute(c.attn_flops, c.attn_bytes, name=f"{name}_attention_compute")
        nodes.append(attention_compute)
        head = attention_compute
        if self.tp_size > 1:
            head = allreduce(c.tp_comm_size, pg_name=pg_name, parents=[attention_compute], name=f"{name}_attention_allreduce")
            nodes.append(head)

        # 2. router: assigns each locally owned token to its top_k experts
        gating_compute = compute(c.gate_flops, c.gate_bytes, parents=[head], name=f"{name}_gating_compute")
        nodes.append(gating_compute)

        # 3. dispatch: tokens travel to the devices hosting their experts
        arrived = self._exchange(nodes, c.traffic, ep_ctx, op="disp", parents=[gating_compute], pass_tag=pass_tag, name_prefix=name)

        # 4. expert FFN over everything routed here, local copies included
        ffwd_compute = compute(c.ffn_flops, c.ffn_bytes, parents=arrived + [gating_compute], name=f"{name}_ffwd_compute")
        nodes.append(ffwd_compute)

        # 5. combine: the transposed matrix returns the results to the token owners
        returned = self._exchange(nodes, c.traffic.T, ep_ctx, op="comb", parents=[ffwd_compute], pass_tag=pass_tag, name_prefix=name)

        # 6. weighted sum of the top_k expert outputs of each token.
        nodes.append(compute(c.combine_flops, c.combine_bytes, parents=returned + [ffwd_compute], name=f"{name}_combine_compute"))
        return nodes

    def bckwd(self, name="node_bckwd", pg_name=None, num_batches=1, ep_ctx: MoeEpContext | None = None, microbatch: int = 0) -> list[ChakraNode]:
        c = self._costs(num_batches, ep_ctx, microbatch)   # same key => same matrix as the forward
        pass_tag = f"b{microbatch}"
        nodes: list[ChakraNode] = []

        # 1. gradient of the combine weighted sum
        entry = compute(2 * c.combine_flops, c.combine_bytes, name=f"{name}_combine_compute")
        nodes.append(entry)

        # 2. gradient of the combine: it travels the dispatch edges, hence the same matrix
        arrived = self._exchange(nodes, c.traffic, ep_ctx, op="gcomb", parents=[entry], pass_tag=pass_tag, name_prefix=name)

        # 3. expert FFN backward
        ffwd_compute = compute(2 * c.ffn_flops, c.ffn_bytes, parents=arrived + [entry], name=f"{name}_ffwd_compute")
        nodes.append(ffwd_compute)

        # 4. gradient of the dispatch: back along the combine edges
        returned = self._exchange(nodes, c.traffic.T, ep_ctx, op="gdisp", parents=[ffwd_compute], pass_tag=pass_tag, name_prefix=name)

        # 5. router backward, which also waits for every incoming grad-dispatch edge
        gating_compute = compute(2 * c.gate_flops, c.gate_bytes, parents=returned + [ffwd_compute], name=f"{name}_gating_compute")
        nodes.append(gating_compute)

        # 6. attention backward, identical to the dense block
        attention_compute = compute(2 * c.attn_flops, c.attn_bytes, parents=[gating_compute], name=f"{name}_attention_compute")
        nodes.append(attention_compute)
        if self.tp_size > 1:
            nodes.append(allreduce(c.tp_comm_size, pg_name=pg_name, parents=[attention_compute], name=f"{name}_attention_allreduce"))
        return nodes