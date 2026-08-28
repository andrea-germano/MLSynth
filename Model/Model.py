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

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode

from Layer.Layer import LayerEmission
from Utils.config import ModelConfig, ParallelismConfig


class BaseTrainingModel(ABC):
    """Interface for a training-mode model composed of layers.

    The Model interface provides functions that the orchestrator uses
    to fetch computation and communication operations at each layer."""

    @abstractmethod
    def fwd(self, name: str, npu_id: int, layer: int, num_batches: float, pg_name: str | None = None, microbatch: int = 0) -> List[ChakraNode]:
        """Return forward-pass operations for the given layer and microbatch count"""
        raise NotImplementedError

    @abstractmethod
    def bckwd(self, name: str, npu_id: int, layer: int, num_batches: float, pg_name: str | None = None, microbatch: int = 0) -> List[ChakraNode]:
        """Return backward-pass operations for the given layer and microbatch count."""
        raise NotImplementedError

    @property
    @abstractmethod
    def num_params(self) -> float:
        """Whole-model parameter count."""
        raise NotImplementedError


class BaseInferenceModel(ABC):
    """Interface for an inference-mode model composed of inference layers.
    Mirrors the role of BaseTrainingModel but exposes prefill/decode rather than fwd/bckwd."""

    @abstractmethod
    def prefill(self, name: str, npu_id: int, layer: int, prompt_lens: List[int], cached_lens: List[int], pg_name: str | None = None) -> LayerEmission:
        raise NotImplementedError

    @abstractmethod
    def decode(self, name: str, npu_id: int, layer: int, kv_lens: List[int], pg_name: str | None = None) -> LayerEmission:
        raise NotImplementedError

    @abstractmethod
    def with_parallelism(self, parallelism: ParallelismConfig) -> "BaseInferenceModel":
        """Return a view of this model with a different parallelism config but the SAME ModelConfig."""
        raise NotImplementedError

    @property
    @abstractmethod
    def model_cfg(self) -> ModelConfig:
        raise NotImplementedError

    @property
    @abstractmethod
    def parallelism(self) -> ParallelismConfig:
        raise NotImplementedError