from abc import ABC, abstractmethod
from typing import List, NamedTuple
from chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode

class LayerEmission(NamedTuple):
    """Output of a layer's prefill or decode method. Contains the list of Chakra nodes emitted by the layer, and a node that can be used to stream the KV cache to the next layer (if bulk transfer is used it is not used)"""
    nodes: List[ChakraNode]
    kv_ready: ChakraNode

class InferenceLayer(ABC):
    """Interface for a layer in a model that supports inference.
    
    Inference has two distinct phases that must be modelled separately:
    * prefill: the phase where the model is processing the initial input and filling up its context window
    * decode: the phase where the model is generating new tokens based on the filled context window, attending to the exixsting KV cache
    """
    
    @abstractmethod
    def prefill(self, 
                name: str,
                pg_name: str | None,
                prompt_lens: List[int]
            ) -> LayerEmission:
        """Return Chakra nodes for the prefill phase of this layer."""
        raise NotImplementedError
    
    @abstractmethod
    def decode(self, 
               name: str, 
               pg_name: str | None, 
               kv_lens: List[int]
            ) -> LayerEmission:
        """Return Chakra nodes for a single decode step of this layer."""
        raise NotImplementedError