from abc import ABC, abstractmethod
from typing import List
from chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode

from Layer.TransformerInferenceLayer import TransformerInferenceLayer

class InferenceModel(ABC):
    """Interface for an inference-mode model composed of inference layers.
    Mirrors the role of `Model` for training but exposes prefill/decode rather
    than fwd/bckwd """

    @abstractmethod
    def prefill(self,name: str,npu_id: int,layer: int,num_batches: int,prompt_len: int,pg_name: str | None = None) -> List[ChakraNode]:
        raise NotImplementedError

    @abstractmethod
    def decode(self,name: str,npu_id: int,layer: int,num_batches: int,kv_len: int,pg_name: str | None = None) -> List[ChakraNode]:
        raise NotImplementedError
