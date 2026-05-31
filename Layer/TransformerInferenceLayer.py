from typing import List
from chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode

from Layer.InferenceLayer import InferenceLayer, LayerEmission
from Evaluator.Evaluator import Evaluator
from utils import compute, allreduce, compute_time

# With tensor parallelism, each TP rank does 1/tp of the FLOPs and holds 1/tp of the weights, but 2 all reduce operations
# are required per layer (after output projection and FFW down projection) to reassemble the partial sums.
# Q,K,V,O matrices are each of size d*d/tp per TP rank and the MLP up and down projection matrices are 4d^2/tp each
# K and V are produced by column-parallel projections, so each TP rank stores 1/tp slice of the KV cache

#! It is supposed that a method like flash attention is used, since with a naive attention implementation the memory access for the attention scores would be much higher

#! Forse possiamo omettere la lettura e la scrittura degli input e output activations, perchè spesso sono ottimizzati e non sono presenti
class TransformerInferenceLayer(InferenceLayer):
    """ A single dense inference block, accounting for FLOPs and activations of a single transformer layer, for both prefill and decode phases"""

    def __init__(self, hidden_size: int, bytes_per_val: int, tp_size: int, evaluator: Evaluator, scale: float = 1.0):
        self.hidden_size = hidden_size
        self.bytes_per_val = bytes_per_val
        self.tp_size = tp_size
        self.evaluator = evaluator
        self.scale = scale

    def prefill(self, name: str, pg_name: str | None, prompt_lens: List[int]) -> LayerEmission:
        d, b, tp= self.hidden_size, self.bytes_per_val, self.tp_size
        T = sum(prompt_lens) #Total number of tokens (replace B*S in the equation)
        Q = sum(l*l for l in prompt_lens) #Replace B*S*S in the equations of the attention scores


        # Input and output activations are not sharded since they are needed in full for the attention and MLP computations, so each TP rank holds a full copy of them
        attn_tensor = int(self.scale*(
            4*d*d*b // tp #weights read for q,k,v,o
            + T*d*b #input activations read
            + 2*T*d*b // tp #KV cache written at the end of the attention block
            + T*d*b #output activations written
        ))   
                                                                            
        attn_flops = int(self.scale * (8*T*d*d + 4*Q*d) // tp) # 8*B*S*d^2 = 6B*S*d^2 for q,k,v matmul + 2B*S*d^2 for output projection, 4*B*S*S*d accounts for attention scores and context vector matmuls

        ffwd_tensor = int(self.scale * (
            8*d*d*b // tp # 2 matrices for the MLP (each big 4d^2, since they are dx4d and 4dxd)
            + 2*T*d*b # accounts for the input and output activations, which are read and written during the MLP block
        ))

        ffwd_flops = int(self.scale * (16*T*d*d) // tp) # 16*B*S*d^2 for the two matrix multiplications in the MLP, condidering d_ff=4*d
        
        return self._emit(name, pg_name, attn_flops, attn_tensor, ffwd_flops, ffwd_tensor, ar_tokens=T)
        
    def decode(self, name: str, pg_name: str | None, kv_lens: List[int]) -> LayerEmission:
        d, b, tp = self.hidden_size, self.bytes_per_val, self.tp_size
        B = len(kv_lens) #Batch size
        K = sum(kv_lens) #Total number of tokens in the KV cache

        attn_tensor = int(self.scale * (
            4*d*d*b // tp #weights read for q,k,v,o
            + B*1*d*b #input activation read for the single token being processed in the current decode step
            + (2*B*K*d*b) // tp #! KV cache read (makes this memory bound), sharded across TP ranks since K and V are produced by column-parallel projections
            + (2*B*1*d*b) // tp  #new K,V written to the KV cache at the end of the attention block
            + B*1*d*b #output activation written 
        ))
        
        attn_flops = int(self.scale * (8*B*1*d*d + 4*B*1*K*d) // tp) # 8*B*1*d^2 = 6B*1*d^2 for q,k,v matmul (input only one token)+ 2B*1*d^2 for output projection, 4*B*1*S_kv*d accounts for attention scores and context vector matmuls with KV cache of length S_kv

        ffwd_tensor = int(self.scale * (
            8*d*d*b // tp #weights read for the 2 MLP matrices
            + B*1*d*b #input activation read for the single token being processed 
            + B*1*d*b #output activation written
        ))

        ffwd_flops = int(self.scale * (16*B*1*d*d) // tp) # 16*B*1*d^2 for the two matrix multiplications in the MLP

        return self._emit(name, pg_name, attn_flops, attn_tensor, ffwd_flops, ffwd_tensor, ar_tokens=B)
    
    def _emit(self, name:str, pg_name: str | None, attn_flops: int, attn_tensor: int, ffwd_flops: int, ffwd_tensor: int, ar_tokens: int) -> LayerEmission:
        d, b = self.hidden_size, self.bytes_per_val
        ar_size = int(self.scale * ar_tokens*d*b) #Size of the tensor to be all reduced

        attn_us= self.evaluator.evaluate(attn_flops, attn_tensor)
        ffwd_us = self.evaluator.evaluate(ffwd_flops, ffwd_tensor)

        attn_node = compute_time(attn_us, name=f"{name}_attn")
        nodes = [attn_node]
        last = attn_node
        if self.tp_size > 1:
            ar1_node = allreduce(ar_size, pg_name=pg_name, parents=[attn_node], name=f"{name}_attn_ar")
            nodes.append(ar1_node)
            last = ar1_node
        
        ffwd_node = compute_time(ffwd_us, parents=[last], name=f"{name}_ffwd")
        nodes.append(ffwd_node)
        last = ffwd_node

        if self.tp_size > 1:
            ar2_node = allreduce(ar_size, pg_name=pg_name, parents=[ffwd_node], name=f"{name}_ffwd_ar")
            nodes.append(ar2_node)
            last = ar2_node
        
        return LayerEmission(nodes=nodes, kv_ready=attn_node)

