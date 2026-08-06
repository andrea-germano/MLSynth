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

from typing import List, Optional, Tuple

from Wrapper.Interfaces import BaseWrapper
from Layer.Interfaces import LayerEmission
from Utils.config import ParallelismConfig, WrapperCondition, WrapperConfig
from Utils.nodes import attr_val, compute
from chakra.schema.protobuf.et_def_pb2 import (
    Node as ChakraNode,
    NodeType as ChakraNodeType,
)

import numpy as np


class ComputeWrapper(BaseWrapper):
    """A decorator that inserts compute slowdown nodes into the compute graph.
    Works both in training (fwd/bckwd) and in inference (prefill/decode)."""

    def __init__(self, model, wrapper_cfg: WrapperConfig):
        super().__init__(model)
        self.wrapper_cfg = wrapper_cfg
        self.rng = np.random.default_rng(wrapper_cfg.seed)

    def with_parallelism(self, parallelism: ParallelismConfig) -> "ComputeWrapper":
        # each derived view gets a fresh RNG seeded with the same seed: deterministic,
        # but prefill/decode pools draw from independent identically-seeded streams
        return ComputeWrapper(self.model.with_parallelism(parallelism), self.wrapper_cfg)

    # ------------- training -------------

    def fwd(self, name, npu_id, layer, num_batches, pg_name=None, microbatch: int = 0, ep_ctx=None) -> list[ChakraNode]:
        ops = self.model.fwd(name, npu_id, layer, num_batches, pg_name, microbatch=microbatch, ep_ctx=ep_ctx)
        condition = self.should_slowdown(npu_id, layer)
        if condition and condition.applies_to("forward"):
            ops, _ = self._insert_slowdown(ops, self._slowdown_factor(condition))
        return ops

    def bckwd(self, name, npu_id, layer, num_batches, pg_name=None, microbatch: int = 0, ep_ctx=None) -> list[ChakraNode]:
        ops = self.model.bckwd(name, npu_id, layer, num_batches, pg_name, microbatch=microbatch, ep_ctx=ep_ctx)
        condition = self.should_slowdown(npu_id, layer)
        if condition and condition.applies_to("backward"):
            ops, _ = self._insert_slowdown(ops, self._slowdown_factor(condition))
        return ops

    # ------------- inference -------------

    def prefill(self, name, npu_id, layer, prompt_lens, cached_lens, pg_name=None) -> LayerEmission:
        emission = self.model.prefill(name, npu_id, layer, prompt_lens, cached_lens, pg_name)
        condition = self.should_slowdown(npu_id, layer)
        if condition and condition.applies_to("prefill"):
            emission = self._apply_to_emission(emission, self._slowdown_factor(condition))
        return emission

    def decode(self, name, npu_id, layer, kv_lens, pg_name=None) -> LayerEmission:
        emission = self.model.decode(name, npu_id, layer, kv_lens, pg_name)
        condition = self.should_slowdown(npu_id, layer)
        if condition and condition.applies_to("decode"):
            emission = self._apply_to_emission(emission, self._slowdown_factor(condition))
        return emission

    # ------------- slowdown machinery -------------

    def should_slowdown(self, npu_id: int, layer: int) -> Optional[WrapperCondition]:
        for condition in self.wrapper_cfg.conditions:
            if condition.matches(npu_id, layer):
                return condition
        return None

    def _slowdown_factor(self, condition: WrapperCondition) -> float:
        spec = condition.slowdown
        if spec.type == "constant":
            return spec.value
        return self.rng.normal(spec.mean, spec.std)

    def _insert_slowdown(self, ops: List[ChakraNode], slowdown: float) -> Tuple[List[ChakraNode], dict]:
        """Insert after every COMP_NODE a slowdown compute node scaled by `slowdown`.
        Returns the updated list and a map {original comp node id -> slowdown node}."""
        replaced: dict[int, ChakraNode] = {}
        if slowdown <= 0:
            return ops, replaced

        # Iterate in reverse order to avoid index issues when inserting elements
        for i in range(len(ops) - 1, -1, -1):
            op = ops[i]
            if op.type == ChakraNodeType.COMP_NODE:
                if attr_val(op, "num_ops") == 0 and attr_val(op, "tensor_size") == 0:
                    continue  # zero-cost barrier nodes (MoE tails): a slowdown of 0 is pure noise
                slow_node = compute(int(attr_val(op, "num_ops") * slowdown),
                                    int(attr_val(op, "tensor_size") * slowdown),
                                    parents=[op], name=f"{op.name}_slowdown")
                # Rewire EVERY dependent of op onto the slowdown node.
                for j in range(i + 1, len(ops)):
                    if op.id in ops[j].data_deps:
                        ops[j].data_deps.remove(op.id)
                        ops[j].data_deps.append(slow_node.id)
                ops.insert(i+1, slow_node)
                replaced[op.id] = slow_node
        return ops, replaced

    def _apply_to_emission(self, emission: LayerEmission, slowdown: float) -> LayerEmission:
        """Apply the slowdown to a LayerEmission, retargeting tail/kv_ready when the node they
        point to gained a trailing slowdown node (otherwise KV/PP sends would not wait for it)."""
        nodes, replaced = self._insert_slowdown(list(emission.nodes), slowdown)
        return LayerEmission(
            nodes=nodes,
            tail=replaced.get(emission.tail.id, emission.tail),
            kv_ready=replaced.get(emission.kv_ready.id, emission.kv_ready),
        )