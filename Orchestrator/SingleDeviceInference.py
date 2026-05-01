from collections import defaultdict
from chakra.schema.protobuf.et_def_pb2 import GlobalMetadata

from Orchestrator.Orchestrator import Orchestrator
from Model.TransformerInference import InferenceModel
from utils import add_dependencies

class SingleDeviceInference(Orchestrator):
    """Single-device inference orchestrator (no TP and no PP)
    generates a Chakra ET for one GPU executing:
    * one prefill over prompt_len tokens
    * num_generated_tokens sequential decode steps, each attending to a KV cache of increasing lenght
    
    All layers are chained so that the resulting ET respects causal data dependencies, prefill of layer i depends on prefill of layer i-1
    and decode step t starts only after decode step t-1 of the last layer is complete.
    """

    def __init__(self, model: InferenceModel, config):
        self.model = model
        self.config = config
        
        infer_cfg = config.get("inference", {})
        self.prompt_len = int(infer_cfg.get("prompt_len", 128))
        self.num_generated_tokens = int(infer_cfg.get("num_generated_tokens", 32))

        self.batch_size = self.model.get_batch_size()
        self.num_layers = self.model.get_num_layers()

        self.num_npus = 1 # for now only single device inference
    
    def generate_comm_groups(self):
        return {} #No collectives in single-device inference, so no comm groups
    
    def exec(self) -> dict:
        nodes = defaultdict(list)
        npu_id=0
        nodes[npu_id].append(GlobalMetadata(version="0.0.4"))

        B = self.batch_size
        prev_node = None

        for layer in range(self.num_layers):
            layer_nodes =self.model.prefill(
                name=f"COMP_NODE_PREFILL_L{layer}",
                npu_id=npu_id,
                layer=layer,
                num_batches=B,
                prompt_len=self.prompt_len,
            )
            
            #Hook the first node of this layer onto the tail of the previous (nothing for layer 0)
            if prev_node is not None:
                add_dependencies(layer_nodes[0], [prev_node])

            for n in layer_nodes:
                nodes[npu_id].append(n)
            prev_node = layer_nodes[-1]
        
        for t in range(self.num_generated_tokens):
            kv_len = self.prompt_len + t + 1 # +1 because we include the token being produced in the current step
            for layer in range(self.num_layers):
                layer_nodes = self.model.decode(
                    name=f"COMP_NODE_DECODE_T{t}_L{layer}",
                    npu_id=npu_id,
                    layer=layer,
                    num_batches=B,
                    kv_len=kv_len,
                )

                if prev_node is not None:
                    add_dependencies(layer_nodes[0], [prev_node])
                
                for n in layer_nodes:
                    nodes[npu_id].append(n)
                
                prev_node = layer_nodes[-1]
                
        return nodes