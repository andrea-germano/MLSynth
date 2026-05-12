from collections import defaultdict
from chakra.schema.protobuf.et_def_pb2 import GlobalMetadata

from Orchestrator.Orchestrator import Orchestrator
from Model.TransformerInference import InferenceModel
from utils import add_dependencies, send, receive

class DisaggregatedInference(Orchestrator):
    """Separate pool for prefill and decode to model a disaggregated inference system where prefill and decode can be executed on different hardware.
    Each pool can have its own TP and PP topology. The KV cache is transferred with a streaming mechanism: each layer can send its KV cache to the decode pool
    assynchronously as soon as it is produced in the prefil phase, overlapping comms with computation of next layer"""

    #Prefill pool: NPU ids [0, num_prefill_npus),
    #Decode pool: NPU ids [num_prefill_npus, num_prefill_npus + num_decode_npus)

    def __init__(self, model: InferenceModel, config: dict):
        self.model=model
        self.config = config

        infer_cfg = config.get("inference", {})
        self.prompt_len = int(infer_cfg.get("prompt_len", 128))
        self.num_generated_tokens = int(infer_cfg.get("num_generated_tokens", 32))
        self.serialize_decode_iterations = bool(infer_cfg.get("serialize_decode_iterations", False))
        
        disagg = config.get("disaggregation")
        if disagg is None:
            raise ValueError("Disaggregation config is required for DisaggregatedInference orchestrator")
        self.tp_p = int(disagg.get("prefill", {}).get("tp_size", 1))
        self.pp_p = int(disagg.get("prefill", {}).get("pp_size", 1))
        self.tp_d = int(disagg.get("decode", {}).get("tp_size", 1))
        self.pp_d = int(disagg.get("decode", {}).get("pp_size", 1))

        #! For now we assume tp_p == tp_d for semplicity, this needs to be relaxed to support different TP sizes in the prefill and decode pools (require KV resharding)
        if self.tp_p != self.tp_d:
            raise NotImplementedError("Different TP sizes in prefill and decode pools is not implemented yet, since it requires resharding the KV cache")

        self.num_prefill_npus = self.tp_p * self.pp_p
        self.num_decode_npus = self.tp_d * self.pp_d
        self.num_npus = self.num_prefill_npus + self.num_decode_npus

        self.batch_size = self.model.get_batch_size()
        self.num_layers = self.model.get_num_layers()
        self.bytes_per_val = self.model.get_bytes_per_val()
        self.hidden_size = self.model.get_hidden_size()
        self.scale = self.model.get_scale()

        if self.num_layers % self.pp_p != 0 or self.num_layers % self.pp_d != 0:
            raise ValueError(f"num_layers ({self.num_layers}) must be divisible by both pp_p ({self.pp_p}) and pp_d ({self.pp_d})")    
        
        self.layers_per_prefill_stage = self.num_layers // self.pp_p
        self.layers_per_decode_stage = self.num_layers // self.pp_d

        #activation size that needs to be exchanged between PP stages in the prefill and decode pools
        self.pp_prefill_act_size = int(self.scale * self.batch_size * self.prompt_len * self.hidden_size * self.bytes_per_val)
        self.pp_decode_act_size = int(self.scale * self.batch_size * 1 * self.hidden_size * self.bytes_per_val) #only the single token being generated at each step

        # KV cache per layer, per TP rank. Each TP rank holds 1/tp_size of the per_layer KV cache since K and V are produced by column-parallel projections.
        # Each prefill TP shard sends only its own slice to the corresponding decode TP shard.
        self.kv_transfer_size_layer = int( self.scale * 2 * self.batch_size * self.prompt_len * self.hidden_size * self.bytes_per_val // self.tp_p)
    
    def generate_comm_groups(self) -> dict:
        comm_groups = defaultdict(list)

        # TP groups for prefill pool
        if self.tp_p > 1:
            for pp_stage in range(self.pp_p):
                for tp_shard in range(self.tp_p):
                    npu_id = pp_stage * self.tp_p + tp_shard
                    comm_groups[f"prefill_tp_{pp_stage}"].append(npu_id)

        # TP groups for decode pool
        if self.tp_d > 1:
            for pp_stage in range(self.pp_d):
                for tp_shard in range(self.tp_d):
                    npu_id = self.num_prefill_npus + pp_stage * self.tp_d + tp_shard
                    comm_groups[f"decode_tp_{pp_stage}"].append(npu_id)

        return dict(comm_groups)
    
    def exec(self) -> dict:
        nodes = defaultdict(list)
        for npu_id in range(self.num_npus):
            nodes[npu_id].append(GlobalMetadata(version="0.0.4"))
        
        last_node = {npu_id: None for npu_id in range(self.num_npus)}

        self._emit_prefill(nodes, last_node)

        # Registriamo i nodi di receive del KV cache sulla gpu di decode
        self._emit_kv_receives(nodes, last_node)

        # Manda l'ultimo token della prefill pool al primo stadio della decode pool per permettere di iniziare il decode non appena il primo token è pronto
        self._emit_first_token_transfer(nodes, last_node)

        for token_idx in range(self.num_generated_tokens):
            kv_len = self.prompt_len + token_idx + 1 #the length of the KV cache grows by one at each decode step
            self._emit_decode(nodes, last_node, token_idx=token_idx, kv_len=kv_len)

            # serialization for the next token. Skip after the last token
            if token_idx < self.num_generated_tokens - 1:
                self._emit_autoregressive_feedback(nodes, last_node, label=f"POST_DECODE_t{token_idx}")

        return nodes

    def _emit_prefill(self, nodes, last_node) -> None:
        """Emit the prefill phase on the prefill pool. After the prefill phase, the KV cache is transferred from the last PP stage of the prefill pool to all PP stages of the decode pool (one transfer per layer, sharded accross TP ranks)"""
        B = self.batch_size

        for pp_stage in range(self.pp_p):
            pg_name = f"prefill_tp_{pp_stage}" if self.tp_p > 1 else None
            for tp_shard in range(self.tp_p):
                npu_id = pp_stage * self.tp_p + tp_shard                

                if pp_stage > 0:
                    scr_npu = npu_id - self.tp_p
                    recv_node = receive(
                        sender=scr_npu, receiver=npu_id, size=self.pp_prefill_act_size, 
                        parents=[last_node[npu_id]] if last_node[npu_id] else None,
                        name=f"COMM_RECV_PREFILL_pp{pp_stage}_tp{tp_shard}"
                    )
                    nodes[npu_id].append(recv_node)
                    last_node[npu_id] = recv_node
                
                for local_layer in range(self.layers_per_prefill_stage):
                    global_layer = pp_stage*self.layers_per_prefill_stage + local_layer
                    name = f"COMP_NODE_PREFILL_L{global_layer}_pp{pp_stage}_tp{tp_shard}"
                    layer_nodes= self.model.prefill(name=name, npu_id=npu_id, layer=global_layer, num_batches=B, prompt_len=self.prompt_len, pg_name=pg_name)
                    if last_node[npu_id]:
                        add_dependencies(layer_nodes[0], [last_node[npu_id]])
                    for n in layer_nodes:
                        nodes[npu_id].append(n)
                    
                    # computation of this layer is finished, KV cache ready to be sent to decode pool
                    compute_end_node = layer_nodes[-1]

                    dst_pp_decode = global_layer // self.layers_per_decode_stage
                    dst_npu = self.num_prefill_npus + dst_pp_decode * self.tp_d + tp_shard
                    send_kv_node = send(
                        sender=npu_id, receiver=dst_npu, size=self.kv_transfer_size_layer, 
                        parents=[compute_end_node],
                        name=f"COMM_SEND_KV_L{global_layer}_tp{tp_shard}"
                    )
                    nodes[npu_id].append(send_kv_node)
                    # Here is not send_kv_node because we want the next layer to be able to start computing as soon as the current layer has finished, without waiting for the KV transfer to complete
                    last_node[npu_id] = compute_end_node

                if pp_stage < self.pp_p - 1:
                    dst = npu_id + self.tp_p
                    send_node = send(
                        sender=npu_id, receiver=dst, size=self.pp_prefill_act_size, 
                        parents=[last_node[npu_id]] if last_node[npu_id] else None,
                        name=f"COMM_SEND_PREFILL_pp{pp_stage}_tp{tp_shard}"
                    )
                    nodes[npu_id].append(send_node)
                    last_node[npu_id] = send_node
    
    def _emit_kv_receives(self, nodes, last_node) -> None:
        for layer in range(self.num_layers):
            src_pp_prefill = layer // self.layers_per_prefill_stage
            #destination is the decode pp stage that will own this layer
            dst_pp_decode = layer // self.layers_per_decode_stage

            for tp_shard in range(self.tp_p):
                src_npu = src_pp_prefill * self.tp_p + tp_shard
                dst_npu = self.num_prefill_npus + dst_pp_decode * self.tp_d + tp_shard

                recv_node = receive(
                    sender=src_npu, receiver=dst_npu, size=self.kv_transfer_size_layer, 
                    parents=[last_node[dst_npu]] if last_node[dst_npu] else None,
                    name=f"COMM_RECV_KV_L{layer}_tp{tp_shard}"
                )
                nodes[dst_npu].append(recv_node)
                # Decode must wait for the KV cache to be received before starting to compute the layer
                last_node[dst_npu] = recv_node
    
    def _emit_first_token_transfer(self, nodes, last_node) -> None:
        """Transfer the first token to be generated from the last stage of the prefill pool to the first stage of the decode pool"""
        for tp_shard in range(self.tp_p):
            # last npu in the prefill pool for this TP shard
            src_npu = (self.pp_p - 1) * self.tp_p + tp_shard
            dst_npu = self.num_prefill_npus + 0 * self.tp_d + tp_shard
            send_node = send(
                sender=src_npu, receiver=dst_npu, size=self.pp_decode_act_size,
                parents=[last_node[src_npu]] if last_node[src_npu] else None,
                name=f"COMM_SEND_FIRST_TOKEN_tp{tp_shard}"
            )
            nodes[src_npu].append(send_node)
            last_node[src_npu] = send_node

            recv_node = receive(
                sender=src_npu, receiver=dst_npu, size=self.pp_decode_act_size,
                parents=[last_node[dst_npu]] if last_node[dst_npu] else None,
                name=f"COMM_RECV_FIRST_TOKEN_tp{tp_shard}"
            )
            nodes[dst_npu].append(recv_node)
            last_node[dst_npu] = recv_node

    def _emit_decode(self, nodes, last_node, token_idx, kv_len) -> None:
        """Emit a single PP traversal for one decode step. Each step processes exactly one new token; every layer reads a KV cache of length `kv_len`. """
        B = self.batch_size

        for pp_stage in range(self.pp_d):
            pg_name = f"decode_tp_{pp_stage}" if self.tp_d > 1 else None
            for tp_shard in range(self.tp_d):
                npu_id = self.num_prefill_npus + pp_stage * self.tp_d + tp_shard

                if pp_stage > 0:
                    scr_npu = npu_id - self.tp_d
                    recv_node = receive(
                        sender=scr_npu, receiver=npu_id, size=self.pp_decode_act_size, 
                        parents=[last_node[npu_id]] if last_node[npu_id] else None,
                        name=f"COMM_RECV_DECODE_t{token_idx}_pp{pp_stage}_tp{tp_shard}"
                    )
                    nodes[npu_id].append(recv_node)
                    last_node[npu_id] = recv_node
                
                for local_layer in range(self.layers_per_decode_stage):
                    global_layer = pp_stage*self.layers_per_decode_stage + local_layer
                    name = f"COMP_NODE_DECODE_t{token_idx}_L{global_layer}_pp{pp_stage}_tp{tp_shard}"
                    layer_nodes= self.model.decode(name=name, npu_id=npu_id, layer=global_layer, num_batches=B, kv_len=kv_len, pg_name=pg_name)
                    if last_node[npu_id]:
                        add_dependencies(layer_nodes[0], [last_node[npu_id]])
                    for n in layer_nodes:
                        nodes[npu_id].append(n)
                    last_node[npu_id] = layer_nodes[-1]
                
                if pp_stage < self.pp_d - 1:
                    dst = npu_id + self.tp_d
                    send_node = send(
                        sender=npu_id, receiver=dst, size=self.pp_decode_act_size, 
                        parents=[last_node[npu_id]] if last_node[npu_id] else None,
                        name=f"COMM_SEND_DECODE_t{token_idx}_pp{pp_stage}_tp{tp_shard}"
                    )
                    nodes[npu_id].append(send_node)
                    last_node[npu_id] = send_node
    
    def _emit_autoregressive_feedback(self, nodes, last_node, label) -> None:
        if not (self.serialize_decode_iterations and self.pp_d > 1):
            return
        
        EMIT_BYTES = 8
        for tp_shard in range(self.tp_d):
            src_npu = self.num_prefill_npus + (self.pp_d - 1) * self.tp_d + tp_shard 
            dst_npu = self.num_prefill_npus + tp_shard 

            send_node = send(
                sender=src_npu, receiver=dst_npu, size=EMIT_BYTES, 
                parents=[last_node[src_npu]] if last_node[src_npu] else None,
                name=f"COMM_SEND_{label}_tp{tp_shard}"
            )
            nodes[src_npu].append(send_node)
            last_node[src_npu] = send_node

            recv_node = receive(
                sender=src_npu, receiver=dst_npu, size=EMIT_BYTES, 
                parents=[last_node[dst_npu]] if last_node[dst_npu] else None,
                name=f"COMM_RECV_{label}_tp{tp_shard}"
            )
            nodes[dst_npu].append(recv_node)
            last_node[dst_npu] = recv_node