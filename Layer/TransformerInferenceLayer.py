from typing import List
from chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode

from Layer.InferenceLayer import InferenceLayer
from utils import compute, allreduce

# With tensor parallelism, each TP rank does 1/tp of the FLOPs and holds 1/tp of the weights, but 2 all reduce operations
# are required per layer (after output projection and FFW down projection) to reassemble the partial sums.
# Q,K,V,O matrices are each of size d*d/tp per TP rank and the MLP up and down projection matrices are 4d^2/tp each
# K and V are produced by column-parallel projections, so each TP rank stores 1/tp slice of the KV cache

#! It is supposed that a method like flash attention is used, since with a naive attention implementation the memory access for the attention scores would be much higher

#! Forse possiamo omettere la lettura e la scrittura degli input e output activations, perchè spesso sono ottimizzati e non sono presenti
class TransformerInferenceLayer(InferenceLayer):
    """ A single dense inference block, accounting for FLOPs and activations of a single transformer layer, for both prefill and decode phases"""

    def __init__(self, hidden_size: int, bytes_per_val: int = 2, tp_size: int = 1, scale: float = 1.0):
        self.hidden_size = hidden_size
        self.bytes_per_val = bytes_per_val
        self.tp_size = tp_size
        self.scale = scale

    def prefill(self, name = "node_prefill", pg_name = None, num_batches = 1, prompt_len = 1) -> List[ChakraNode]:
        d, b, B, S= self.hidden_size, self.bytes_per_val, num_batches, prompt_len

        # Input and output activations are not sharded since they are needed in full for the attention and MLP computations, so each TP rank holds a full copy of them
        attn_tensor = int(self.scale*(
            4*d*d*b // self.tp_size #weights read for q,k,v,o
            + B*S*d*b #input activations read
            + 2*B*S*d*b // self.tp_size #KV cache written at the end of the attention block
            + B*S*d*b #output activations written
        ))   
                                                                            
        attn_flops = int(self.scale * (8*B*S*d*d + 4*B*S*S*d) // self.tp_size) # 8*B*S*d^2 = 6B*S*d^2 for q,k,v matmul + 2B*S*d^2 for output projection, 4*B*S*S*d accounts for attention scores and context vector matmuls
        attn_node = compute(attn_flops, attn_tensor, name=f"{name}_attention_compute")

        attn_allreduce = None
        if self.tp_size > 1:
            tp_comm_size = int(self.scale * B*S*d*b) #! DA CONTROLLARE Communication size for the all reduce after attention is the size of the output activation
            attn_allreduce = allreduce(tp_comm_size, pg_name=pg_name, parents=[attn_node], name=f"{name}_attention_allreduce")


        ffwd_tensor = int(self.scale * (
            8*d*d*b // self.tp_size # 2 matrices for the MLP (each big 4d^2, since they are dx4d and 4dxd)
            + 2*B*S*d*b # accounts for the input and output activations, which are read and written during the MLP block
        ))   #? This should not make big difference since with roofline model the compute should be dominant, right?

        ffwd_flops = int(self.scale * (16*B*S*d*d) // self.tp_size) # 16*B*S*d^2 for the two matrix multiplications in the MLP, condidering d_ff=4*d
        ffwd_parent=[attn_allreduce] if attn_allreduce else [attn_node]
        ffwd_node = compute(ffwd_flops, ffwd_tensor, parents=ffwd_parent, name=f"{name}_ffwd_compute")

        ffwd_allreduce = None
        if self.tp_size > 1:
            tp_comm_size = int(self.scale * B*S*d*b) #Communication size for the all reduce after MLP is the size of the output activation
            ffwd_allreduce = allreduce(tp_comm_size, pg_name=pg_name, parents=[ffwd_node], name=f"{name}_ffwd_allreduce")
        
        output_nodes : list[ChakraNode] = [attn_node]
        if attn_allreduce: 
            output_nodes.append(attn_allreduce)
        output_nodes.append(ffwd_node)
        if ffwd_allreduce:
            output_nodes.append(ffwd_allreduce)
        return output_nodes

    def decode(self, name = "node_decode", pg_name = None, num_batches = 1, kv_len = 1) -> List[ChakraNode]:
        d, b, B, S_kv = self.hidden_size, self.bytes_per_val, num_batches, kv_len

        attn_tensor = int(self.scale * (
            4*d*d*b // self.tp_size #weights read for q,k,v,o
            + B*1*d*b #input activation read for the single token being processed in the current decode step
            + 2*B*S_kv*d*b // self.tp_size #! KV cache read (makes this memory bound), sharded across TP ranks since K and V are produced by column-parallel projections
            + 2*B*1*d*b // self.tp_size  #new K,V written to the KV cache at the end of the attention block
            + B*1*d*b #output activation written 
        ))
        
        attn_flops = int(self.scale * (8*B*1*d*d + 4*B*1*S_kv*d) // self.tp_size) # 8*B*1*d^2 = 6B*1*d^2 for q,k,v matmul (input only one token)+ 2B*1*d^2 for output projection, 4*B*1*S_kv*d accounts for attention scores and context vector matmuls with KV cache of length S_kv
        attn_node = compute(attn_flops, attn_tensor, name=f"{name}_attention_compute")

        attn_allreduce = None
        if self.tp_size > 1:
            tp_comm_size = int(self.scale * B*1*d*b) #Communication size for the all reduce after attention is the size of the output activation (only one token)
            attn_allreduce = allreduce(tp_comm_size, pg_name=pg_name, parents=[attn_node], name=f"{name}_attention_allreduce")

        ffwd_tensor = int(self.scale * (
            8*d*d*b // self.tp_size #weights read for the 2 MLP matrices
            + B*1*d*b #input activation read for the single token being processed 
            + B*1*d*b #output activation written
        ))

        ffwd_flops = int(self.scale * (16*B*1*d*d) // self.tp_size) # 16*B*1*d^2 for the two matrix multiplications in the MLP
        ffwd_parent = [attn_allreduce] if attn_allreduce else [attn_node]
        ffwd_node = compute(ffwd_flops, ffwd_tensor, parents=ffwd_parent, name=f"{name}_ffwd_compute")

        ffwd_allreduce = None
        if self.tp_size > 1:
            tp_comm_size = int(self.scale * B*1*d*b) #Communication size for the all reduce after MLP is the size of the output activation (only one token)
            ffwd_allreduce = allreduce(tp_comm_size, pg_name=pg_name, parents=[ffwd_node], name=f"{name}_ffwd_allreduce")
        
        output_nodes : list[ChakraNode] = [attn_node]
        if attn_allreduce: 
            output_nodes.append(attn_allreduce)
        output_nodes.append(ffwd_node)
        if ffwd_allreduce:
            output_nodes.append(ffwd_allreduce)
        return output_nodes

