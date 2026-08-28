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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
from chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode


@dataclass
class LayerEmission:
    """Contract between an inference layer and the orchestrator"""
    nodes: List[ChakraNode]
    tail: ChakraNode | List[ChakraNode]
    kv_ready: ChakraNode


@dataclass(frozen=True)
class MoeEpContext:
    """The expert-parallel group an MoE layer emits its all-to-all over. INFERENCE ONLY"""
    peers: List[int]    # npu_ids of the group; index i == EP rank i == row/col i of the matrix
    ep_rank: int        # this device's position in `peers`, i.e. its row/column in the matrix
    stage: int          # this device's pipeline stage (naming only)
    pool: str           # "p" or "d", the inference pool this device belongs to (naming only)
    origin_tokens: List[int] | None = None  # tokens each peer routes; None = every peer routes the caller's local count

    @property
    def size(self) -> int:
        return len(self.peers)


class BaseTrainingLayer(ABC):
    """Interface for a layer in a model that supports training.

    Implement the fundamental operations that occur at each
    level of the model, specifically forward and backward computations.
    """

    @abstractmethod
    def fwd(self, name: str, pg_name: str | None = None, num_batches: float = 1) -> List[ChakraNode]:
        """Execute forward computation for this layer and return nodes."""
        raise NotImplementedError

    @abstractmethod
    def bckwd(self, name: str, pg_name: str | None = None, num_batches: float = 1) -> List[ChakraNode]:
        """Execute backward computation for this layer and return nodes."""
        raise NotImplementedError


class BaseInferenceLayer(ABC):
    """Interface for a layer in a model that supports inference.

    Implement the fundamental operations that occur at different
    phases of inference, specifically prefill and decode computations.
    """

    @abstractmethod
    def prefill(self, name: str, pg_name: str | None, prompt_lens: List[int], cached_lens: List[int]) -> LayerEmission:
        """Return Chakra nodes for the prefill phase of this layer."""
        raise NotImplementedError

    @abstractmethod
    def decode(self, name: str, pg_name: str | None, kv_lens: List[int]) -> LayerEmission:
        """Return Chakra nodes for a single decode step of this layer."""
        raise NotImplementedError