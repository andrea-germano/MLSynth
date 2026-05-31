from collections import defaultdict
from typing import List, Optional, Dict

from chakra.schema.protobuf.et_def_pb2 import GlobalMetadata, Node as ChakraNode

from Config.Config import ParallelismConfig
from Model.InferenceModel import InferenceModel
from Scheduler.Request import Request
from utils import add_dependencies, send, receive


class DisaggregatedInference:
    """Two pools of NPUs, one for prefill and one for decode, with possible different parallelism configurations"""
    def __init__(self, 
        prefill_model: InferenceModel, 
        decode_model: InferenceModel,
        prefill_parallelism: ParallelismConfig,
        decode_parallelism: ParallelismConfig,
        kv_transfer_strategy: str = "streaming"):
    
        if prefill_model.get_num_layers() != decode_model.get_num_layers():
            raise ValueError("Prefill and decode models must have the same number of layers")
        
        self.prefill_model = prefill_model
        self.decode_model = decode_model
        self.kv_transfer_strategy = kv_transfer_strategy
        self.tp_p, self.pp_p = prefill_parallelism.tp_size, prefill_parallelism.pp_size
        self.tp_d, self.pp_d = decode_parallelism.tp_size, decode_parallelism.pp_size

        self.prefill_npus_count = self.tp_p * self.pp_p
        self.decode_npus_count = self.tp_d * self.pp_d
        self.total_npus = self.prefill_npus_count + self.decode_npus_count

        self.nodes: Dict[str, List[ChakraNode]] = defaultdict(list)
        self.last_node: Dict[int, Optional[ChakraNode]] = {
            npu: None for npu in range(self.total_npus)
        }

        for npu in range(self.total_npus):
            self.nodes[npu].append(GlobalMetadata(version="0.0.4"))
    
    def step(self, iteration: int, prefill_requests: List[Request], decode_requests: List[Request]) -> None:
        # Generate and store nodes for prefill requests
        for req in prefill_requests:
            emission = self.prefill_model.prefill(
                name=req.name,
                npu_id=req.npu_id,
                layer=req.layer,
                prompt_lens=req.prompt_len,
                pg_name=req.pg_name
            )
            self.nodes[req.npu_id].extend(emission.nodes)
            self.last_node[req.npu_id] = emission.kv_ready
        
        # Generate and store nodes for decode requests
        for req in decode_requests:
            emission = self.decode_model.decode(
                name=req.name,
                npu_id=req.npu_id,
                layer=req.layer,
                kv_lens=req.kv_len,
                pg_name=req.pg_name
            )
            self.nodes[req.npu_id].extend(emission.nodes)
            self.last_node[req.npu_id] = emission.kv_ready

        # Handle KV transfer between prefill and decode NPUs if using streaming strategy
        if self.kv_transfer_strategy == "streaming":
            for req in decode_requests:
                if req.npu_id < self.prefill_npus_count:  # This request is assigned to a prefill NPU, but needs to send KV to a decode NPU
                    target_npu = req.npu_id + self.prefill_npus_count  # Assuming a simple mapping where the first decode NPU corresponds to the first prefill NPU, etc.
                    kv_node = self.last_node[req.npu_id]
                    if kv_node is not None:
                        send(kv_node, target_npu)
                        receive(target_npu)  # Wait for acknowledgment that the KV has been received
    
    def generate_comm_groups(self) -> Dict[str, List[int]]:
        comm_groups = Dict[str, List[int]] = defaultdict(list)

        if self.tp_p > 1:
            for stage in range(self.pp_p):
                for shard in range(self.tp_p):
                    npu_id = stage * self.tp_p + shard
                    comm_groups[f"prefill_pp_{stage}_tp_{shard}"].append(npu_id)
        
