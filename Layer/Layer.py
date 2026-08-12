from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
from chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode


@dataclass
class LayerEmission:
    """Contract between an inference layer and the orchestrator: `tail` is what the next node on
    this NPU must depend on, `kv_ready` is the node after which the KV cache of this layer exists
    `tail` may be a LIST when no single node closes the block: that happens in the MoE layer with tp = 1 and ep > 1 """
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

    The orchestrator fetches computation and communication operations for the forward and
    backward pass of each layer through fwd/bckwd.
    """

    @abstractmethod
    def fwd(self, name: str, pg_name: str | None = None, num_batches: float = 1) -> List[ChakraNode]:
        """Return Chakra nodes for the forward pass of this layer."""
        raise NotImplementedError

    @abstractmethod
    def bckwd(self, name: str, pg_name: str | None = None, num_batches: float = 1) -> List[ChakraNode]:
        """Return Chakra nodes for the backward pass of this layer."""
        raise NotImplementedError


class BaseInferenceLayer(ABC):
    """Interface for a layer in a model that supports inference.

    Inference has two distinct phases that must be modelled separately:
    * prefill: the phase where the model is processing the initial input and filling up its context window
    * decode: the phase where the model is generating new tokens based on the filled context window, attending to the existing KV cache

    pg_name and tp_size are used to emit tensor-parallel all reduce operations, mirroring the Megatron-LM scheme used in the training implementation
    """

    @abstractmethod
    def prefill(self, name: str, pg_name: str | None, prompt_lens: List[int], cached_lens: List[int]) -> LayerEmission:
        """Return Chakra nodes for the prefill phase of this layer."""
        raise NotImplementedError

    @abstractmethod
    def decode(self, name: str, pg_name: str | None, kv_lens: List[int]) -> LayerEmission:
        """Return Chakra nodes for a single decode step of this layer. `kv_lens[i]` is the length
        of request i's KV cache *including* the token produced in this step."""
        raise NotImplementedError