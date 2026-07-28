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
from typing import List
from chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode

from Layer.Interfaces import LayerEmission


class BaseWrapper(ABC):
    """Base class for decorators around a model (a BaseTrainingModel or BaseInferenceModel
    implementation). A wrapper intercepts the emission methods of both modes (fwd/bckwd for
    training, prefill/decode for inference) to alter the emitted graph.

    Every other attribute is delegated to the wrapped model, so orchestrators can use a
    wrapped model transparently (num_params, getters, model_cfg, with_parallelism, ...).
    """

    def __init__(self, model):
        # assign before anything else: __getattr__ delegates to self.model
        self.model = model

    def __getattr__(self, item):
        if item == "model":  # not set yet: avoid infinite recursion
            raise AttributeError(item)
        return getattr(self.model, item)

    @abstractmethod
    def fwd(self, name: str, npu_id: int, layer: int, num_batches: float, pg_name: str | None = None) -> List[ChakraNode]:
        """Return the wrapped model's forward-pass operations, possibly altered."""
        raise NotImplementedError

    @abstractmethod
    def bckwd(self, name: str, npu_id: int, layer: int, num_batches: float, pg_name: str | None = None) -> List[ChakraNode]:
        """Return the wrapped model's backward-pass operations, possibly altered."""
        raise NotImplementedError

    @abstractmethod
    def prefill(self, name: str, npu_id: int, layer: int, prompt_lens: List[int], cached_lens: List[int], pg_name: str | None = None) -> LayerEmission:
        """Return the wrapped model's prefill emission, possibly altered."""
        raise NotImplementedError

    @abstractmethod
    def decode(self, name: str, npu_id: int, layer: int, kv_lens: List[int], pg_name: str | None = None) -> LayerEmission:
        """Return the wrapped model's decode emission, possibly altered."""
        raise NotImplementedError
