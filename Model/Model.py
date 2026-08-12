from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode

from Layer.Layer import LayerEmission
from Utils.config import ModelConfig, ParallelismConfig


class BaseTrainingModel(ABC):
    """Interface for a training-mode model composed of training layers.

    The orchestrator uses fwd/bckwd to fetch computation and communication operations at
    each layer. `npu_id` identifies the emitting NPU; it is unused by plain models but
    consumed by wrappers (e.g. per-NPU slowdowns)."""

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
        """Whole-model parameter count. Sizes the DP gradient all-reduce."""
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
        """Return a view of this model with a different parallelism config but the SAME (by identity) ModelConfig."""
        raise NotImplementedError

    @property
    @abstractmethod
    def model_cfg(self) -> ModelConfig:
        raise NotImplementedError

    @property
    @abstractmethod
    def parallelism(self) -> ParallelismConfig:
        raise NotImplementedError