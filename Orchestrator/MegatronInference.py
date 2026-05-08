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

        # In single-request PP inference stage 0 cannot embed token t+1 until stage P-1 has sampled it from token t's logit.
        # We model this with a small COMM_SEND/COMM_RECV pair per TP shard from stage P-1 to stage 0 between consecutive
        # iterations (see `_emit_sampling_broadcast`). 
        #! Set to False only when modelling a saturated pipeline with multiple requests in flight (continuous batching with micro-batched pipelining)
        self.serialize_decode_iterations = bool(infer_cfg.get("serialize_decode_iterations", True))

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

        self.emit_prefill(nodes, last_node)

        self._emit_autoregressive_feedback(nodes, last_node, label="POST_PREFILL")

        for token_idx in range(self.num_generated_tokens):
            kv_len = self.prompt_len + token_idx + 1 #the length of the KV cache grows by one at each decode step
            self.emit_decode(nodes, last_node, token_idx=token_idx, kv_len=kv_len)

            # serialization for the next token. Skip after the last token
            if token_idx < self.num_generated_tokens - 1:
                self._emit_autoregressive_feedback(nodes, last_node, label=f"POST_DECODE_t{token_idx}")
        
        return nodes
    

    def emit_prefill(self, nodes, last_node) -> None:
        """Emit a single PP traversal for the prefill phase. Each PP stage executes its owned layers over the full prompt_len tokens, with TP all-reduces already accounted for in the layer implementation."""
        B = self.batch_size

        for pp_stage in range(self.pp_size):
            for tp_shard in range(self.tp_size):
                npu_id = pp_stage * self.tp_size + tp_shard
                tg_pg_name = f"tp_{pp_stage}" if self.tp_size > 1 else None

                # optional COMM_RECV from previous PP stage (except for the first stage)
                if pp_stage > 0:
                    scr_node = npu_id - self.tp_size
                    recv_node = receive(
                        sender=scr_node, receiver=npu_id, size=self.pp_prefill_comm_size, 
                        parents=[last_node[npu_id]] if last_node[npu_id] else None,
                        name=f"COMM_RECV_PREFILL_pp{pp_stage}_tp{tp_shard}"
                    )
                    nodes[npu_id].append(recv_node)
                    last_node[npu_id] = recv_node

                # compute nodes for each layer owned by this PP stage (TP parallelism is aalready accounted for within the layer implementation)
                for local_layer in range(self.layers_per_stage):
                    global_layer = pp_stage*self.layers_per_stage + local_layer
                    name = f"COMP_NODE_PREFILL_L{global_layer}_pp{pp_stage}_tp{tp_shard}"
                    layer_nodes= self.model.prefill(name=name, npu_id=npu_id, layer=global_layer, num_batches=B, prompt_len=self.prompt_len, pg_name=tg_pg_name)

                    # We have to hook the head of this layer to the last emitted node for this GPU, which could be either the COMM_RECV, a layer or a node from a previous phase (in case this is not the first iteration of the loop)
                    if last_node[npu_id]:
                        add_dependencies(layer_nodes[0], [last_node[npu_id]])
                    
                    for node in layer_nodes:
                        nodes[npu_id].append(node)
                    last_node[npu_id] = layer_nodes[-1]

                # optional COMM_SEND to the same TP shard in the next PP stage (except for the last PP stage)
                if pp_stage < self.pp_size - 1:
                    dst_node = npu_id + self.tp_size
                    send_node = send(
                        sender=npu_id, receiver=dst_node, size=self.pp_prefill_comm_size, 
                        parents=[last_node[npu_id]] if last_node[npu_id] else None,
                        name=f"COMM_SEND_PREFILL_pp{pp_stage}_tp{tp_shard}"
                    )
                    nodes[npu_id].append(send_node)
                    last_node[npu_id] = send_node

    def emit_decode(self, nodes, last_node, token_idx, kv_len) -> None:
        """Emit a single PP traversal for one decode step. Each step processes exactly one new token; every layer reads a KV cache of length `kv_len`. """

        B = self.batch_size
        
        for pp_stage in range(self.pp_size):
            for tp_shard in range(self.tp_size):
                npu_id = pp_stage * self.tp_size + tp_shard
                tp_pg_name = f"tp_{pp_stage}" if self.tp_size > 1 else None

                # Receive single-token activation from previous stage
                if pp_stage > 0:
                    scr_npu = npu_id - self.tp_size
                    recv_node = receive(
                        sender=scr_npu, receiver=npu_id, size=self.pp_decode_comm_size, 
                        parents=[last_node[npu_id]] if last_node[npu_id] else None,
                        name=f"COMM_RECV_DECODE_t{token_idx}_pp{pp_stage}_tp{tp_shard}"
                    )
                    nodes[npu_id].append(recv_node)
                    last_node[npu_id] = recv_node

                #compute nodes for each layer owned by this PP stage
                for local_layer in range(self.layers_per_stage):
                    global_layer = pp_stage*self.layers_per_stage + local_layer
                    name = f"COMP_NODE_DECODE_t{token_idx}_L{global_layer}_pp{pp_stage}_tp{tp_shard}"
                    layer_nodes= self.model.decode(name=name, npu_id=npu_id, layer=global_layer, num_batches=B, kv_len=kv_len, pg_name=tp_pg_name)

                    if last_node[npu_id]:
                        add_dependencies(layer_nodes[0], [last_node[npu_id]])
                
                    for node in layer_nodes:
                        nodes[npu_id].append(node)
                    last_node[npu_id] = layer_nodes[-1]

                # send single-token activation to next stage
                if pp_stage < self.pp_size - 1:
                    dst_node = npu_id + self.tp_size
                    send_node = send(
                        sender=npu_id, receiver=dst_node, size=self.pp_decode_comm_size, 
                        parents=[last_node[npu_id]] if last_node[npu_id] else None,
                        name=f"COMM_SEND_DECODE_t{token_idx}_pp{pp_stage}_tp{tp_shard}"
                    )
                    nodes[npu_id].append(send_node)
                    last_node[npu_id] = send_node
    
    def _emit_autoregressive_feedback(self, nodes, last_node, label) -> None:
        """Cross-iteration auto-regressive serialization: the sampled token must travel from the tail PP stage back to the head PP stage before the next iteration can start.
        Realized as a small COMM_SEND/COMM_RECV pair per TP shard."""

        if not (self.serialize_decode_iterations and self.pp_size > 1):
            return

        EMIT_BYTES = 8 # We can get away with sending just the token ids, which are typically 4 or 8 bytes, instead of the full activation
        for tp_shard in range(self.tp_size):
            src_npu = (self.pp_size - 1) * self.tp_size + tp_shard
            dst_npu = tp_shard

            #send on stage P-1
            send_node = send(
                sender=src_npu, receiver=dst_npu, size=EMIT_BYTES, 
                parents=[last_node[src_npu]] if last_node[src_npu] else None,
                name=f"COMM_SEND_SAMPLE_{label}_tp{tp_shard}"
            )

            nodes[src_npu].append(send_node)
            last_node[src_npu] = send_node 

            #recv on stage 0
            recv_node = receive(
                sender=src_npu, receiver=dst_npu, size=EMIT_BYTES, 
                parents=[last_node[dst_npu]] if last_node[dst_npu] else None,
                name=f"COMM_RECV_SAMPLE_{label}_tp{tp_shard}"
            )
            nodes[dst_npu].append(recv_node)
            last_node[dst_npu] = recv_node            
