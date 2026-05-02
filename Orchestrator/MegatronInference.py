from collections import defaultdict
from chakra.schema.protobuf.et_def_pb2 import GlobalMetadata

from Orchestrator.Orchestrator import Orchestrator
from Model.TransformerInference import InferenceModel
from utils import add_dependencies, send, receive

#! For now some major assumpions were introduced:
#! *static batching (uniform requests size, no arrival times)
#! *fixed prompt lenght and fixed number of generated tokens
#! *only one inference instance --> no DP

# Each layer of the model is owned by exactly one PP stage, so layers_per_stage = num_layers / pp_size and PP stage i owns layers [i*layers_per_stage, (i+1)*layers_per_stage)

class MegatronInference(Orchestrator):
    """Inference orchestrator for a Megatron-LM style transformer model, supporting tensor and pipeline parallelism (no DP)"""
    def __init__(self, model: InferenceModel, config: dict):
        self.model = model
        self.config = config

        infer_cfg = config.get("inference", {})
        self.prompt_len = int(infer_cfg.get("prompt_len", 128))
        self.num_generated_tokens = int(infer_cfg.get("num_generated_tokens", 32))

        par=config.get("parallelism", {})
        self.tp_size = int(par.get("tp_size", 1))
        self.pp_size = int(par.get("pp_size", 1))
        self.num_npus = self.tp_size * self.pp_size

        self.batch_size = self.model.get_batch_size()
        self.num_layers = self.model.get_num_layers()
        self.bytes_per_val = self.model.get_bytes_per_val()
        self.hidden_size = self.model.get_hidden_size()
        self.scale = self.model.get_scale()

        # Layer partitioning across PP stages
        if self.num_layers % self.pp_size != 0:
            raise ValueError(f"Number of layers ({self.num_layers}) must be divisible by pipeline parallel size ({self.pp_size}).")
        self.layers_per_stage = self.num_layers // self.pp_size

        # The communication pattern for the prefill phase is the same as in training, the entire prompt activation is transferred
        self.pp_prefill_comm_size = int(self.scale*self.batch_size*self.prompt_len*self.hidden_size*self.bytes_per_val)
        # During decoding, only the last token's activation is transferred between pipeline stages
        self.pp_decode_comm_size = int(self.scale*self.batch_size* 1*self.hidden_size*self.bytes_per_val)

    def generate_comm_groups(self) -> dict:
        """Generate one TP group per PP stage. No PP groups since PP is p2p"""
        comm_groups = defaultdict(list)

        if self.tp_size <= 1:
            return {}
        
        for pp_stage in range(self.pp_size):
            group_name = f"tp_{pp_stage}"
            for tp_rank in range(self.tp_size):
                npu_id = pp_stage * self.tp_size + tp_rank
                comm_groups[group_name].append(npu_id)
        return dict(comm_groups)

    def exec(self) -> dict:
        nodes = defaultdict(list)

        for npu_id in range(self.num_npus):
            nodes[npu_id].append(GlobalMetadata(version="0.0.4"))
        
        # Keep track of the last emitted node for each gpu, both within a phase and accross phases (prefill -> decode -> decode ...)
        last_node={npu_id: None for npu_id in range(self.num_npus)}

        self._emit_pp_iteration(nodes, last_node, phase="prefill", phase_kwargs={"prompt_len": self.prompt_len}, pp_comm_size=self.pp_prefill_comm_size, iter_label="PREFILL")

        #decode iterations, one for each generated token. Each iteration is a full traversal of the PP stages (stage0 -> stage P-1) for each token
        for gen_token_id in range(self.num_generated_tokens):
            kv_len = self.prompt_len + gen_token_id + 1 #+1 because the token being produced in the current step is also added to the KV cache
            self._emit_pp_iteration(nodes, last_node, phase="decode", phase_kwargs={"kv_len": kv_len, "token_id": gen_token_id}, pp_comm_size=self.pp_decode_comm_size, iter_label=f"DECODE_t{gen_token_id}")
        
        return nodes
    
    def _emit_pp_iteration(self, nodes, last_node, phase, phase_kwargs, pp_comm_size, iter_label):
        """Emit one iteration of the PP loop (prefill or decode)"""
        B = self.batch_size
        
        #For each GPU:
        for pp_stage in range(self.pp_size):
            for tp_shard in range(self.tp_size):
                # Compute the global npu_id based on PP stage and TP shard, used the same convention as in training
                npu_id = pp_stage * self.tp_size + tp_shard
                tg_pg_name = f"tp_{pp_stage}" if self.tp_size > 1 else None

                #1) optional COMM_RECV from previous PP stage (except for the first stage)
                if pp_stage > 0:
                    scr_node = npu_id - self.tp_size
                    recv_node = receive(
                        sender=scr_node, receiver=npu_id, size=pp_comm_size, 
                        parents=[last_node[npu_id]] if last_node[npu_id] else None,
                        name=f"COMM_RECV_{iter_label}_pp{pp_stage}_tp{tp_shard}"
                    )
                    nodes[npu_id].append(recv_node)
                    last_node[npu_id] = recv_node

                # 2) compute nodes for each layer owned by this PP stage (TP parallelism is aalready accounted for within the layer implementation)
                for local_layer in range(self.layers_per_stage):
                    global_layer = pp_stage*self.layers_per_stage + local_layer
                    name = f"COMP_NODE_{iter_label}_L{global_layer}_pp{pp_stage}_tp{tp_shard}"

                    if phase == "prefill":
                        layer_nodes= self.model.prefill(name=name, npu_id=npu_id, layer=global_layer, num_batches=B, prompt_len=phase_kwargs["prompt_len"], pg_name=tg_pg_name)
                    elif phase == "decode":
                        layer_nodes = self.model.decode(name=name, npu_id=npu_id, layer=global_layer, kv_len=phase_kwargs["kv_len"], pg_name=tg_pg_name)
                    else:
                        raise ValueError(f"Unsupported phase: {phase}")

                    # We have to hook the head of this layer to the last emitted node for this GPU, which could be either the COMM_RECV, a layer or a node from a previous phase (in case this is not the first iteration of the loop)
                    if last_node[npu_id]:
                        add_dependencies(layer_nodes[0], [last_node[npu_id]])
                    
                    for node in layer_nodes:
                        nodes[npu_id].append(node)
                    last_node[npu_id] = layer_nodes[-1]

                # 3) optional COMM_SEND to the same TP shard in the next PP stage (except for the last PP stage)
                if pp_stage < self.pp_size - 1:
                    dst_node = npu_id + self.tp_size
                    send_node = send(
                        sender=npu_id, receiver=dst_node, size=pp_comm_size, 
                        parents=[last_node[npu_id]] if last_node[npu_id] else None,
                        name=f"COMM_SEND_{iter_label}_pp{pp_stage}_tp{tp_shard}"
                    )
                    nodes[npu_id].append(send_node)
                    last_node[npu_id] = send_node