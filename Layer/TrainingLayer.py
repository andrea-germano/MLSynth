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

from Layer.Interfaces import BaseTrainingLayer
from Layer.DenseBlockMath import DenseBlockMath
from Utils.config import ModelConfig
from Utils.nodes import allreduce, compute
from chakra.schema.protobuf.et_def_pb2 import (
    Node as ChakraNode,
)


class TrainingLayer(BaseTrainingLayer):
    """A single dense training block. The cost model lives in DenseBlockMath (shared with the
    inference layer); the backward pass is modeled as 2x the forward FLOPs, emitted in reverse
    order (FFN first, then attention). Attention scores are not halved for causal masking."""

    def __init__(self, model_cfg: ModelConfig, sequence_len: int, tp_size: int):
        self.math = DenseBlockMath.from_model_cfg(model_cfg, tp_size)
        self.sequence_len = sequence_len
        self.tp_size = tp_size

    def _costs(self, num_batches):
        tokens = num_batches * self.sequence_len
        # every query attends to the full sequence (no /2 causal factor)
        score_entries = num_batches * self.sequence_len * self.sequence_len
        attn_flops, attn_bytes = self.math.attn_costs(
            query_tokens=tokens, kv_read_tokens=0,
            kv_write_tokens=tokens, score_entries=score_entries)
        ffn_flops, ffn_bytes = self.math.ffn_costs(tokens)
        return attn_flops, attn_bytes, ffn_flops, ffn_bytes, self.math.allreduce_bytes(tokens)

    def fwd(self, name="node_fwd", pg_name=None, num_batches=1) -> list[ChakraNode]:
        attn_flops, attn_bytes, ffn_flops, ffn_bytes, tp_comm_size = self._costs(num_batches)

        attention_compute = compute(attn_flops, attn_bytes, name=f"{name}_attention_compute")

        # tensor parallel allreduce
        attention_allreduce = None
        if self.tp_size > 1:
            attention_allreduce = allreduce(tp_comm_size, pg_name=pg_name, parents=[attention_compute], name=f"{name}_attention_allreduce")

        ffwd_parent = [attention_allreduce] if attention_allreduce else [attention_compute]
        ffwd_compute = compute(ffn_flops, ffn_bytes, parents=ffwd_parent, name=f"{name}_ffwd_compute")

        # tensor parallel allreduce
        ffwd_allreduce = None
        if self.tp_size > 1:
            ffwd_allreduce = allreduce(tp_comm_size, pg_name=pg_name, parents=[ffwd_compute], name=f"{name}_mlp_allreduce")

        nodes: list[ChakraNode] = [attention_compute]
        if attention_allreduce is not None:
            nodes.append(attention_allreduce)
        nodes.append(ffwd_compute)
        if ffwd_allreduce is not None:
            nodes.append(ffwd_allreduce)
        return nodes


    def bckwd(self, name="node_bckwd", pg_name=None, num_batches=1) -> list[ChakraNode]:
        attn_flops, attn_bytes, ffn_flops, ffn_bytes, tp_comm_size = self._costs(num_batches)

        ffwd_compute = compute(2 * ffn_flops, ffn_bytes, name=f"{name}_ffwd_compute")

        # tensor parallel allreduce
        ffwd_allreduce = None
        if self.tp_size > 1:
            ffwd_allreduce = allreduce(tp_comm_size, pg_name=pg_name, parents=[ffwd_compute], name=f"{name}_mlp_allreduce")

        attention_parent = [ffwd_allreduce] if ffwd_allreduce else [ffwd_compute]
        attention_compute = compute(2 * attn_flops, tensor_size=attn_bytes, parents=attention_parent, name=f"{name}_attention_compute")

        # tensor parallel allreduce
        attention_allreduce = None
        if self.tp_size > 1:
            attention_allreduce = allreduce(tp_comm_size, pg_name=pg_name, parents=[attention_compute], name=f"{name}_attention_allreduce")

        nodes: list[ChakraNode] = [ffwd_compute]
        if ffwd_allreduce is not None:
            nodes.append(ffwd_allreduce)
        nodes.append(attention_compute)
        if attention_allreduce is not None:
            nodes.append(attention_allreduce)
        return nodes
