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

from Model.Interfaces import BaseTrainingModel
from Layer.MoeTrainingLayer import MoeTrainingLayer
from Utils.config import TrainRunConfig
from chakra.schema.protobuf.et_def_pb2 import (
    Node as ChakraNode,
)


class MoeTrainingModel(BaseTrainingModel):
    """Mixture-of-Experts transformer model for training.
    Unlike the dense models, the model is NOT homogeneous: each layer gets its own
    MoeTrainingLayer instance (experts/routing may differ per layer).

    ⚠ INCOMPLETE and not wired into any entry point: the layer is the ORIGINAL legacy
    implementation (see MoeTrainingLayer's warning), num_params uses the legacy formula.
    MoE-specific knobs stay explicit kwargs since the config schema does not model them."""

    def __init__(self, run: TrainRunConfig, ep_size: int, top_k: int = 1, capacity_factor: float = 1.25):
        self._model_cfg = run.model
        self._training = run.training
        self._tp_size = run.parallelism.tp_size
        self.ep_size = ep_size

        # Model is composed of MoE layers, one instance per layer
        self.layers = [
            MoeTrainingLayer(
                num_layers=run.model.num_layers,
                hidden_size=run.model.hidden_size,
                sequence_len=run.training.sequence_len,
                vocab_size=run.model.vocab_size,
                ep_size=ep_size,
                tp_size=run.parallelism.tp_size,
                top_k=top_k,
                capacity_factor=capacity_factor,
                bytes_per_val=run.model.bytes_per_val,
                scale=run.model.scale,
            )
            for _ in range(run.model.num_layers)
        ]

    def fwd(self, name, npu_id, layer, num_batches, pg_name=None) -> list[ChakraNode]:
        return self._layer_for(layer).fwd(name=name, num_batches=num_batches, pg_name=pg_name)

    def bckwd(self, name, npu_id, layer, num_batches, pg_name=None) -> list[ChakraNode]:
        return self._layer_for(layer).bckwd(name=name, num_batches=num_batches, pg_name=pg_name)

    def _layer_for(self, idx: int) -> MoeTrainingLayer:
        return self.layers[idx]

    def get_layers(self) -> list[MoeTrainingLayer]:
        return self.layers

    @property
    def model_cfg(self):
        return self._model_cfg

    @property
    def num_params(self) -> float:
        # ⚠ ORIGINAL legacy formula, kept on purpose (like the layer): dense 4d²-attention
        # accounting, no GQA/SwiGLU, no per-expert FFN replication. TODO align with DenseBlockMath.
        d, L, V = self._model_cfg.hidden_size, self._model_cfg.num_layers, self._model_cfg.vocab_size
        S = self._training.sequence_len
        return 12 * L * d * d * (1 + (13 / (12 * L * d)) + ((V + S) / (12 * L * d)))

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
