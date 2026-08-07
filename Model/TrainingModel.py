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

from Model.Model import BaseTrainingModel
from Layer.TrainingLayer import TrainingLayer
from Utils.config import TrainRunConfig
from chakra.schema.protobuf.et_def_pb2 import (
    Node as ChakraNode,
)


class TrainingModel(BaseTrainingModel):
    """Dense transformer model for training. The orchestrator fetches per-layer
    computation and communication operations through fwd/bckwd."""

    def __init__(self, run: TrainRunConfig):
        self._model_cfg = run.model
        self._training = run.training
        self._tp_size = run.parallelism.tp_size

        # Model is composed of Transformer layers, one instance per layer
        self.layers = [
            TrainingLayer(
                model_cfg=run.model,
                sequence_len=run.training.sequence_len,
                tp_size=run.parallelism.tp_size,
            )
            for _ in range(run.model.num_layers)
        ]

    def fwd(self, name, npu_id, layer, num_batches, pg_name=None, microbatch=0, ep_ctx=None) -> list[ChakraNode]:
        # microbatch/ep_ctx belong to the BaseTrainingModel signature; a dense model has no use for them
        return self._layer_for(layer).fwd(name=name, num_batches=num_batches, pg_name=pg_name)

    def bckwd(self, name, npu_id, layer, num_batches, pg_name=None, microbatch=0, ep_ctx=None) -> list[ChakraNode]:
        return self._layer_for(layer).bckwd(name=name, num_batches=num_batches, pg_name=pg_name)

    def _layer_for(self, idx: int) -> TrainingLayer:
        return self.layers[idx]

    def get_layers(self) -> list[TrainingLayer]:
        return self.layers

    @property
    def model_cfg(self):
        return self._model_cfg

    @property
    def num_params(self) -> float:
        d, L, V = self._model_cfg.hidden_size, self._model_cfg.num_layers, self._model_cfg.vocab_size
        layer_weights = sum(layer.math.attn_weight_elems + layer.math.ffn_weight_elems for layer in self.layers)
        embedding = 2 * V * d # embedding + lm head
        norms = (2 * L + 1) * d
        return float(layer_weights + embedding + norms)

    def get_num_params(self) -> float:
        return self.num_params

    def get_num_layers(self) -> int:
        return self._model_cfg.num_layers

    def get_name(self) -> str:
        return self._model_cfg.name

    def get_hidden_size(self) -> int:
        return self._model_cfg.hidden_size

    def get_sequence_len(self) -> int:
        return self._training.sequence_len

    def get_vocab_size(self) -> int:
        return self._model_cfg.vocab_size

    def get_batch_size(self) -> int:
        return self._training.batch_size

    def get_bytes_per_val(self) -> int:
        return self._model_cfg.bytes_per_val

    def get_tp_size(self) -> int:
        return self._tp_size

    def get_scale(self) -> float:
        return self._model_cfg.scale
