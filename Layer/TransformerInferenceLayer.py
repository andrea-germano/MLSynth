from typing import List
from chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode

from Layer.InferenceLayer import InferenceLayer
from utils import compute

#! Forse possiamo omettere la lettura e la scrittura degli input e output activations, perchè spesso sono ottimizzati e non sono presenti
class TransformerInferenceLayer(InferenceLayer):
    """ A single dense inference block, accounting for FLOPs and activations of a single transformer layer, for both prefill and decode phases"""

    def __init__(self, hidden_size: int, bytes_per_val: int, scale: float):
        self.hidden_size = hidden_size
        self.bytes_per_val = bytes_per_val
        self.scale = scale

    def prefill(self, name = "node_prefill", pg_name = None, num_batches = 1, prompt_len = 1) -> List[ChakraNode]:
        d, b, B, S= self.hidden_size, self.bytes_per_val, num_batches, prompt_len
        attn_tensor = int(self.scale*(
            4*d*d*b #weights read for q,k,v,o
            + B*S*d*b #input activations read
            + 2*B*S*d*b #KV cache written at the end of the attention block
            + B*S*d*b #output activations written
        ))   
                                                                            
        attn_flops = int(self.scale * (8*B*S*d*d + 4*B*S*S*d)) # 8*B*S*d^2 = 6B*S*d^2 for q,k,v matmul + 2B*S*d^2 for output projection, 4*B*S*S*d accounts for attention scores and context vector matmuls
        attn_node = compute(attn_flops, attn_tensor, name=f"{name}_attention_compute")
        ffwd_tensor = int((8*d*d*b + 2*B*S*d*b) * self.scale)   # 2 matrices for the MLP (each big 4d^2, since they are dx4d and 4dxd)
                                                                # 2*B*S*d*b accounts for the input and output activations, which are read and written during the MLP block
                                                                #? This should not make big difference since with roofline model the compute should be dominant, right?
        ffwd_flops = int(self.scale * (16*B*S*d*d)) # 16*B*S*d^2 for the two matrix multiplications in the MLP, condidering d_ff=4*d
        ffwd_node = compute(ffwd_flops, ffwd_tensor, parents=[attn_node], name=f"{name}_ffwd_compute")
        return [attn_node, ffwd_node]

    def decode(self, name = "node_decode", pg_name = None, num_batches = 1, kv_len = 1) -> List[ChakraNode]:
        d, b, B, S_kv = self.hidden_size, self.bytes_per_val, num_batches, kv_len
        attn_tensor = int(self.scale * (
            4*d*d*b #weights read for q,k,v,o
            + B*1*d*b #input activation read for the single token being processed in the current decode step
            + 2*B*S_kv*d*b #! KV cache read (makes this memory bound)
            + 2*B*1*d*b #new K,V written to the KV cache at the end of the attention block
            + B*1*d*b #output activation written 
        ))
        
        attn_flops = int(self.scale * (8*B*1*d*d + 4*B*1*S_kv*d)) # 8*B*1*d^2 = 6B*1*d^2 for q,k,v matmul (input only one token)+ 2B*1*d^2 for output projection, 4*B*1*S_kv*d accounts for attention scores and context vector matmuls with KV cache of length S_kv
        attn_node = compute(attn_flops, attn_tensor, name=f"{name}_attention_compute")
        ffwd_tensor = int(self.scale * (
            8*d*d*b #weights read for the 2 MLP matrices
            + B*1*d*b #input activation read for the single token being processed 
            + B*1*d*b #output activation written
        ))
        ffwd_flops = int(self.scale * (16*B*1*d*d)) # 16*B*1*d^2 for the two matrix multiplications in the MLP
        ffwd_node = compute(ffwd_flops, ffwd_tensor, parents=[attn_node], name=f"{name}_ffwd_compute")
        return [attn_node, ffwd_node]

