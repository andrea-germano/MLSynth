from abc import ABC, abstractmethod
from typing import List
from chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode

class InferenceLayer(ABC):
    """Interface for a layer in a model that supports inference.
    
    Inference has two distinct phases that must be modelled separately:
    * prefill: the phase where the model is processing the initial input and filling up its context window
    * decode: the phase where the model is generating new tokens based on the filled context window, attending to the exixsting KV cache

    pg_name and tp_size are used to emit tensor-parallel all reduce operations, mirroring the Megatron-LM scheme used in the training implementation
    """
    
    @abstractmethod
    def prefill(self, 
                name: str = "node_prefill", 
                pg_name: str | None = None, 
                num_batches: int = 1, 
                prompt_len: int = 1
            ) -> List[ChakraNode]:
        """Return Chakra nodes for the prefill phase of this layer."""
        raise NotImplementedError
    
    @abstractmethod
    def decode(self, 
               name: str = "node_decode", 
               pg_name: str | None = None, 
               num_batches: int = 1, 
               kv_len: int = 1
            ) -> List[ChakraNode]:
        """Return Chakra nodes for a single decode step of this layer."""
        #kv_len is the length of the KV cache including the token being produced in the current step
        raise NotImplementedError