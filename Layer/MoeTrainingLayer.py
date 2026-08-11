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

from Layer.Layer import BaseTrainingLayer
from Layer.DenseBlockMath import DenseBlockMath
from Utils.config import ModelConfig
from Utils.nodes import allreduce, alltoall, compute
from chakra.schema.protobuf.et_def_pb2 import (Node as ChakraNode)


class MoeTrainingLayer(BaseTrainingLayer):
    """A single Mixture-of-Experts training block:

        attention -> [TP all-reduce] -> gating -> [EP all-to-all] -> FFN -> [TP all-reduce]

    The MoE half is coarse, as in the original: no router (the all-to-all is sized by a fixed
    `capacity_factor * top_k`), dispatch only and no combine, and the FFN keeps the dense cost.
    Those are inherited choices, listed in Adel_bugs.md §6. The wiring is not inherited: the
    original left the gating and both all-reduces without `parents`, which lets ASTRA-sim run a
    collective before the compute that feeds it.

    The arithmetic comes from DenseBlockMath, so GQA, SwiGLU and the tp division hold."""

    def __init__(self, model_cfg: ModelConfig, sequence_len: int, tp_size: int, ep_size: int):
        self.math = DenseBlockMath.from_model_cfg(model_cfg, tp_size)
        self.model_cfg = model_cfg
        self.moe = model_cfg.moe
        self.sequence_len = sequence_len
        self.tp_size = tp_size
        self.ep_size = ep_size

    def _costs(self, num_batches):
        tokens = num_batches * self.sequence_len
        # every query attends to the full sequence (no /2 causal factor)
        score_entries = num_batches * self.sequence_len * self.sequence_len
        attn_flops, attn_bytes = self.math.attn_costs(
            query_tokens=tokens, kv_read_tokens=0,
            kv_write_tokens=tokens, score_entries=score_entries)
        ffn_flops, ffn_bytes = self.math.ffn_costs(tokens)
        return attn_flops, attn_bytes, ffn_flops, ffn_bytes, self.math.allreduce_bytes(tokens)

    def _gate_costs(self, num_batches):
        """One pass over the tokens, as in the original."""
        tokens = num_batches * self.sequence_len
        scale, b, hidden = self.math.scale, self.math.bytes_per_val, self.math.hidden_size
        return int(scale * 2 * tokens * hidden), int(scale * tokens * hidden * b)

    def _a2a_bytes(self, num_batches) -> int:
        """Dispatch volume of one rank: every token to `top_k` experts, buffers sized for
        `capacity_factor` times the average."""
        tokens = num_batches * self.sequence_len
        return int(self.math.scale * self.math.bytes_per_val * self.math.hidden_size
                   * tokens * self.moe.capacity_factor * self.moe.top_k)

    def fwd(self, name="node_fwd", pg_name=None, num_batches=1) -> list[ChakraNode]:
        attn_flops, attn_bytes, ffn_flops, ffn_bytes, tp_comm_size = self._costs(num_batches)
        gate_flops, gate_bytes = self._gate_costs(num_batches)
        nodes: list[ChakraNode] = []

        attention_compute = compute(attn_flops, attn_bytes, name=f"{name}_attention_compute")
        nodes.append(attention_compute)
        head = attention_compute

        if self.tp_size > 1:
            head = allreduce(tp_comm_size, pg_name=pg_name, parents=[attention_compute],
                             name=f"{name}_attention_allreduce")
            nodes.append(head)

        gating_compute = compute(gate_flops, gate_bytes, parents=[head],
                                 name=f"{name}_gating_compute")
        nodes.append(gating_compute)
        head = gating_compute

        if self.ep_size > 1:
            # dispatch only; the combine is not emitted, as in the original
            head = alltoall(self._a2a_bytes(num_batches), pg_name=pg_name,
                            parents=[gating_compute], name=f"{name}_ep_alltoall")
            nodes.append(head)

        ffwd_compute = compute(ffn_flops, ffn_bytes, parents=[head], name=f"{name}_ffwd_compute")
        nodes.append(ffwd_compute)

        if self.tp_size > 1:
            nodes.append(allreduce(tp_comm_size, pg_name=pg_name, parents=[ffwd_compute],
                                   name=f"{name}_mlp_allreduce"))
        return nodes

    def bckwd(self, name="node_bckwd", pg_name=None, num_batches=1) -> list[ChakraNode]:
        """The forward walked backwards, with 2x the FLOPs."""
        attn_flops, attn_bytes, ffn_flops, ffn_bytes, tp_comm_size = self._costs(num_batches)
        gate_flops, gate_bytes = self._gate_costs(num_batches)
        nodes: list[ChakraNode] = []

        ffwd_compute = compute(2 * ffn_flops, ffn_bytes, name=f"{name}_ffwd_compute")
        nodes.append(ffwd_compute)
        head = ffwd_compute

        if self.tp_size > 1:
            head = allreduce(tp_comm_size, pg_name=pg_name, parents=[ffwd_compute],
                             name=f"{name}_mlp_allreduce")
            nodes.append(head)

        if self.ep_size > 1:
            head = alltoall(self._a2a_bytes(num_batches), pg_name=pg_name, parents=[head],
                            name=f"{name}_ep_alltoall_back")
            nodes.append(head)

        gating_grad = compute(2 * gate_flops, gate_bytes, parents=[head],
                              name=f"{name}_gating_grad")
        nodes.append(gating_grad)

        attention_compute = compute(2 * attn_flops, attn_bytes, parents=[gating_grad],
                                    name=f"{name}_attention_compute")
        nodes.append(attention_compute)

        if self.tp_size > 1:
            nodes.append(allreduce(tp_comm_size, pg_name=pg_name, parents=[attention_compute],
                                   name=f"{name}_attention_allreduce"))
        return nodes
