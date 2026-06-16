from typing import List
from chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode

from Layer.InferenceLayer import InferenceLayer, LayerEmission
from Utils.utils import compute, allreduce
from Utils.naming import comp_name, coll_name

# With tensor parallelism, each TP rank does 1/tp of the FLOPs and holds 1/tp of the weights, but 2 all reduce operations
# are required per layer (after output projection and FFW down projection) to reassemble the partial sums.
# Q,K,V,O matrices are each of size d*d/tp per TP rank and the MLP up and down projection matrices are 4d^2/tp each
# K and V are produced by column-parallel projections, so each TP rank stores 1/tp slice of the KV cache

#! It is supposed that a method like flash attention is used, since with a naive attention implementation the memory access for the attention scores would be much higher

#! Forse possiamo omettere la lettura e la scrittura degli input e output activations, perchè spesso sono ottimizzati e non sono presenti
class TransformerInferenceLayer(InferenceLayer):
    """ A single dense inference block, accounting for FLOPs and activations of a single transformer layer, for both prefill and decode phases
    Attention: MHA and GQA (via query_dim / key_value_dim)
    FFN: classic (2 matrix multiplications) or SwiGLU (3 matrix multiplications)"""

    def __init__(self, hidden_size: int, query_dim: int | None = None,
                key_value_dim: int | None = None,
                ffn_intermediate_size: int | None = None,
                ffn_type: str = "classic", 
                bytes_per_val: int = 2, tp_size: int = 1, scale: float = 1.0):
        self.hidden_size = hidden_size
        self.query_dim = query_dim if query_dim is not None else hidden_size
        self.key_value_dim = key_value_dim if key_value_dim is not None else hidden_size
        self.ffn_intermediate_size = ffn_intermediate_size if ffn_intermediate_size is not None else 4 * hidden_size
        self.ffn_type = ffn_type
        self.bytes_per_val = bytes_per_val
        self.tp_size = tp_size
        self.scale = scale

        self.attn_weight_elems = (
            2 * hidden_size * self.query_dim       # Q proj + O proj
            + 2 * hidden_size * self.key_value_dim # K proj + V proj
        )
        num_ffn_matrices = 3 if ffn_type == "swiglu" else 2
        self.ffn_weight_elems = num_ffn_matrices * hidden_size * self.ffn_intermediate_size
        self.ffn_flops_per_token = 2* num_ffn_matrices * hidden_size * self.ffn_intermediate_size

    def prefill(self, name: str, pg_name: str | None, prompt_lens: List[int]) -> LayerEmission:
        b = self.bytes_per_val

        total_prompt_tokens = sum(prompt_lens)
        score_entries = sum(length * length for length in prompt_lens)   # Σ l²: coppie query–key

        attn_flops = int(self.scale * (
            2 * total_prompt_tokens * self.hidden_size * self.query_dim       # Q projection
            + 4 * total_prompt_tokens * self.hidden_size * self.key_value_dim # K+V projections
            + 2 * total_prompt_tokens * self.query_dim * self.hidden_size     # O projections
            + 4 * score_entries * self.query_dim                         # QKᵀ + (scores · V)
        ) // self.tp_size)

        attn_bytes = int(self.scale * (
            self.attn_weight_elems * b // self.tp_size                 # Q,K,V,O weights (sharded)
            + total_prompt_tokens * self.hidden_size * b               # input activations
            + 2 * total_prompt_tokens * self.key_value_dim * b // self.tp_size  # KV written to cache
            + total_prompt_tokens * self.hidden_size * b               # output activations
        ))

        ffn_flops = int(self.scale * (total_prompt_tokens * self.ffn_flops_per_token) // self.tp_size)
        ffn_bytes = int(self.scale * (
            self.ffn_weight_elems * b // self.tp_size                  # FFN weights (sharded)
            + 2 * total_prompt_tokens * self.hidden_size * b           # input + output activations
        ))

        return self._emit(name, pg_name, attn_flops, attn_bytes, ffn_flops, ffn_bytes,
                          allreduce_tokens=total_prompt_tokens)

    def decode(self, name: str, pg_name: str | None, kv_lens: List[int]) -> LayerEmission:
        b = self.bytes_per_val

        batch_size = len(kv_lens)
        total_kv_tokens = sum(kv_lens)   

        attn_flops = int(self.scale * (
            2 * batch_size * self.hidden_size * self.query_dim       # Q projection (1 token/richiesta)
            + 4 * batch_size * self.hidden_size * self.key_value_dim # K+V projections
            + 2 * batch_size * self.query_dim * self.hidden_size     # O projections
            + 4 * total_kv_tokens * self.query_dim              # scores + context su tutta la KV
        ) // self.tp_size)

        attn_bytes = int(self.scale * (
            self.attn_weight_elems * b // self.tp_size                  # weights
            + batch_size * self.hidden_size * b                         # input activation
            + 2 * total_kv_tokens * self.key_value_dim * b // self.tp_size   # reading KV cache (memory bound)
            + 2 * batch_size * self.key_value_dim * b // self.tp_size        # new K,V written
            + batch_size * self.hidden_size * b                         # output activation
        ))

        ffn_flops = int(self.scale * (batch_size * self.ffn_flops_per_token) // self.tp_size)
        ffn_bytes = int(self.scale * (
            self.ffn_weight_elems * b // self.tp_size
            + batch_size * self.hidden_size * b
            + batch_size * self.hidden_size * b
        ))

        return self._emit(name, pg_name, attn_flops, attn_bytes, ffn_flops, ffn_bytes, allreduce_tokens=batch_size)

    
    def _emit(self, name: str, pg_name: str | None, attn_flops: int, attn_bytes: int, ffn_flops: int, ffn_bytes: int, allreduce_tokens: int) -> LayerEmission:
        allreduce_bytes = int(self.scale * allreduce_tokens * self.hidden_size * self.bytes_per_val)

        nodes: List[ChakraNode] = []

        attn = compute(attn_flops, attn_bytes, name=comp_name(name, "attn"))
        nodes.append(attn)
        attn_end = attn

        if self.tp_size > 1:
            attn_ar = allreduce(allreduce_bytes, pg_name=pg_name, parents=[attn], name=coll_name(name, "attn"))
            nodes.append(attn_ar)
            attn_end = attn_ar

        kv_ready = attn_end

        ffw = compute(ffn_flops, ffn_bytes, parents=[attn_end], name=comp_name(name, "ffw"))
        nodes.append(ffw)
        tail = ffw

        if self.tp_size > 1:
            ffn_ar = allreduce(allreduce_bytes, pg_name=pg_name, parents=[ffw], name=coll_name(name, "ffw"))
            nodes.append(ffn_ar)
            tail = ffn_ar

        return LayerEmission(nodes=nodes, tail=tail, kv_ready=kv_ready)

