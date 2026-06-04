from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

from Layer.InferenceLayer import LayerEmission
from parser import ModelConfig, ParallelismConfig

class InferenceModel(ABC):
    """Interface for an inference-mode model composed of inference layers.
    Mirrors the role of `Model` for training but exposes prefill/decode rather
    than fwd/bckwd """

    @abstractmethod
    def prefill(self, name: str, npu_id: int, layer: int, prompt_lens: List[int], pg_name: str | None = None) -> LayerEmission:
        raise NotImplementedError

    @abstractmethod
    def decode(self, name: str, npu_id: int, layer: int, kv_lens: List[int], pg_name: str | None = None) -> LayerEmission:
        raise NotImplementedError
    
    @abstractmethod
    def with_parallelism(self, parallelism: ParallelismConfig) -> "InferenceModel":
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
