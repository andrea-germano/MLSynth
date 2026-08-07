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

from Layer.Layer import BaseTrainingLayer, MoeEpContext
from Layer.DenseBlockMath import DenseBlockMath
from Layer.MoeBlock import MoeBlock
from Utils.config import ModelConfig
from Utils.nodes import allreduce, compute
from Utils.routing import PHASE_FWD, RoutingPlan
from chakra.schema.protobuf.et_def_pb2 import (Node as ChakraNode)


class MoeTrainingLayer(BaseTrainingLayer):
    """A single Mixture-of-Experts training block"""

    def __init__(self, model_cfg: ModelConfig, sequence_len: int, tp_size: int, plan: RoutingPlan, layer_idx: int):
        self.math = DenseBlockMath.from_model_cfg(model_cfg, tp_size)
        self.moe_block = MoeBlock(model_cfg, plan, layer_idx)
        self.model_cfg = model_cfg
        self.moe = model_cfg.moe
        self.sequence_len = sequence_len
        self.tp_size = tp_size
        self.layer_idx = layer_idx

    def _attn_costs(self, num_batches):
        """(flops, bytes, all-reduce bytes) of the dense half, which is not MoE-specific."""
        tokens = int(num_batches * self.sequence_len)
        score_entries = num_batches * self.sequence_len * self.sequence_len
        flops, mem = self.math.attn_costs(query_tokens=tokens, kv_read_tokens=0, kv_write_tokens=tokens, score_entries=score_entries)
        return flops, mem, self.math.allreduce_bytes(tokens)

    def _moe_costs(self, num_batches, ep_ctx: MoeEpContext, microbatch: int):
        """Sequence parallelism (which Megatron requires with TP+EP) gives this device 1/tp of
        the tokens, each with the full hidden dimension"""
        local_tokens = int(num_batches * self.sequence_len) // self.tp_size
        key = (PHASE_FWD, microbatch, self.layer_idx, ep_ctx.cluster)
        return self.moe_block.costs(key=key, ep_ctx=ep_ctx, local_tokens=local_tokens)

    def fwd(self, name="node_fwd", pg_name=None, num_batches=1, ep_ctx: MoeEpContext | None = None, microbatch: int = 0) -> list[ChakraNode]:
        attn_flops, attn_bytes, tp_comm_size = self._attn_costs(num_batches)
        c = self._moe_costs(num_batches, ep_ctx, microbatch)
        pass_tag = f"f{microbatch}"
        nodes: list[ChakraNode] = []

        # 1. attention, identical to the dense block
        attention_compute = compute(attn_flops, attn_bytes, name=f"{name}_attention_compute")
        nodes.append(attention_compute)
        head = attention_compute
        if self.tp_size > 1:
            head = allreduce(tp_comm_size, pg_name=pg_name, parents=[attention_compute], name=f"{name}_attention_allreduce")
            nodes.append(head)

        # 2. router: assigns each locally owned token to its top_k experts
        gating_compute = compute(c.gate_flops, c.gate_bytes, parents=[head], name=f"{name}_gating_compute")
        nodes.append(gating_compute)

        # 3. dispatch: tokens travel to the devices hosting their experts
        arrived = self.moe_block.exchange(nodes, c.traffic, ep_ctx, op="disp", parents=[gating_compute], it=pass_tag, name_prefix=name)

        # 4. expert FFN over everything routed here, local copies included
        ffwd_compute = compute(c.expert_flops, c.expert_bytes, parents=arrived + [gating_compute], name=f"{name}_ffwd_compute")
        nodes.append(ffwd_compute)

        # 5. combine: the transposed matrix returns the results to the token owners
        returned = self.moe_block.exchange(nodes, c.traffic.T, ep_ctx, op="comb", parents=[ffwd_compute], it=pass_tag, name_prefix=name)

        # 6. weighted sum of the top_k expert outputs of each token
        nodes.append(compute(c.combine_flops, c.combine_bytes, parents=returned + [ffwd_compute], name=f"{name}_combine_compute"))
        return nodes

    def bckwd(self, name="node_bckwd", pg_name=None, num_batches=1, ep_ctx: MoeEpContext | None = None, microbatch: int = 0) -> list[ChakraNode]:
        attn_flops, attn_bytes, tp_comm_size = self._attn_costs(num_batches)
        c = self._moe_costs(num_batches, ep_ctx, microbatch)  # same key => same matrix as the forward
        pass_tag = f"b{microbatch}"
        nodes: list[ChakraNode] = []

        # 1. gradient of the combine weighted sum: scatters each token's gradient back to its top_k expert copies.
        entry = compute(2 * c.combine_flops, c.combine_bytes, name=f"{name}_combine_compute")
        nodes.append(entry)

        # 2. gradient of the combine: it travels the dispatch edges, hence the same matrix
        arrived = self.moe_block.exchange(nodes, c.traffic, ep_ctx, op="gcomb", parents=[entry], it=pass_tag, name_prefix=name)

        # 3. expert FFN backward
        ffwd_compute = compute(2 * c.expert_flops, c.expert_bytes, parents=arrived + [entry], name=f"{name}_ffwd_compute")
        nodes.append(ffwd_compute)

        # 4. gradient of the dispatch: back along the combine edges
        returned = self.moe_block.exchange(nodes, c.traffic.T, ep_ctx, op="gdisp", parents=[ffwd_compute], it=pass_tag, name_prefix=name)

        # 5. router backward, which also waits for every incoming grad-dispatch edge
        gating_compute = compute(2 * c.gate_flops, c.gate_bytes, parents=returned + [ffwd_compute], name=f"{name}_gating_compute")
        nodes.append(gating_compute)

        # 6. attention backward, identical to the dense block
        attention_compute = compute(2 * attn_flops, attn_bytes, parents=[gating_compute], name=f"{name}_attention_compute")
        nodes.append(attention_compute)
        if self.tp_size > 1:
            nodes.append(allreduce(tp_comm_size, pg_name=pg_name, parents=[attention_compute], name=f"{name}_attention_allreduce"))
        return nodes