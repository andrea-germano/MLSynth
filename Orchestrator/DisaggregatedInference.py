from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from chakra.schema.protobuf.et_def_pb2 import GlobalMetadata

from config import RunConfig
from Orchestrator.Orchestrator import Orchestrator
from Model.InferenceModel import InferenceModel
from utils import add_dependencies, send, receive

# size of bytes of a pull request message
REQUEST_BYTES = 8
# size of bytes of the sampled token feedback (used for serialization of decode iterations and for the first token handoff from prefill to decode)
#! Maybe this should be calculated as the size of lm_head * bytes_per_val * scale, to more accurately reflect the size of the autoregressive feedback, but for now we keep it fixed and small as it is only used for synchronization and does not carry actual data in this model of the system.
SAMPLE_BYTES = 8

class DisaggregatedInference(Orchestrator):
    """Separate pool for prefill and decode to model a disaggregated inference system where prefill and decode can be executed on different hardware.
    Each pool can have its own TP and PP topology. The KV cache is transferred with a streaming mechanism: each layer can send its KV cache to the decode pool
    assynchronously as soon as it is produced in the prefil phase, overlapping comms with computation of next layer"""

    #Prefill pool: NPU ids [0, num_prefill_npus),
    #Decode pool: NPU ids [num_prefill_npus, num_prefill_npus + num_decode_npus)

    def __init__(self, model: InferenceModel, run: RunConfig):
        self.run = run
        self.model = model

        self.prefill_cfg = run.prefill
        self.decode_cfg = run.decode
        self.kv_cfg = run.inference.kv_transfer
        self.serialize_decode = run.inference.serialize_decode_iterations

        #different views of the same model with different parallelism configs for prefill and decode pools, same model_cfg shared by identity
        self.prefill_model = model.with_parallelism(run.prefill)
        self.decode_model = model.with_parallelism(run.decode)

        #Model metadata
        model_cfg = run.model
        self.num_layers = model_cfg.num_layers
        self.hidden_size = model_cfg.hidden_size
        self.bytes_per_val = model_cfg.bytes_per_val
        self.scale = model_cfg.scale
        self.vocab_size = model_cfg.vocab_size

        if self.num_layers % self.prefill_cfg.pp_size != 0:
            raise ValueError(f"num_layers ({self.num_layers}) must be divisible by prefill pp_size ({self.prefill_cfg.pp_size})")
        if self.num_layers % self.decode_cfg.pp_size != 0:
            raise ValueError(f"num_layers ({self.num_layers}) must be divisible by decode pp_size ({self.decode_cfg.pp_size})")
        
        self.layer_per_stage_prefill = self.num_layers // self.prefill_cfg.pp_size
        self.layer_per_stage_decode = self.num_layers // self.decode_cfg.pp_size

        self.prefill_npus = self.prefill_cfg.num_npus
        self.decode_npus = self.decode_cfg.num_npus
        self.total_npus = self.prefill_npus + self.decode_npus

        #static request blob of token
        self.requests = run.inference.requests
        self.prompt_lens = [req.prompt_len for req in self.requests]
        self.max_decode_steps = max(req.gen_len for req in self.requests)
        self.total_prompt_tokens = sum(self.prompt_lens)

        self.kv_bytes_per_layer = int(self.scale * 2 * self.hidden_size * self.bytes_per_val * self.total_prompt_tokens)

        # PP aactivations transfer size
        self.pp_prefill_bytes= int(self.scale * self.total_prompt_tokens * self.hidden_size * self.bytes_per_val)

        self._stream_recv: Dict[Tuple[int,int], List] = defaultdict(list) # (layer, dst) -> recvs
        self._bulk_recv: Dict[int, List] = defaultdict(list) # dst -> recvs

    def _get_prefill_npu_id(self, pp_stage:int, tp_rank:int) -> int:
        return (pp_stage * self.prefill_cfg.tp_size) + tp_rank
    
    def _get_decode_npu_id(self, pp_stage:int, tp_rank:int) -> int:
        return self.prefill_npus + (pp_stage * self.decode_cfg.tp_size) + tp_rank

    # SINGLE SOURCE OF TRUTH for the pg_name strings. They are used both when building comm_groups.json (generate_comm_groups) and when tagging the COMM_COLL nodes (_emit_prefill / _emit_decode).
    def _prefill_tp_pg(self, pp_stage: int) -> str:
        return f"prefill_tp_{pp_stage}"

    def _decode_tp_pg(self, pp_stage: int) -> str:
        return f"decode_tp_{pp_stage}"

    def generate_comm_groups(self) -> dict:
        groups: Dict[str, List[int]] = {}
        if self.prefill_cfg.tp_size > 1:
            for stage in range(self.prefill_cfg.pp_size):
                group_name = self._prefill_tp_pg(stage)
                groups[group_name] = [self._get_prefill_npu_id(stage, rank) for rank in range(self.prefill_cfg.tp_size)]
        if self.decode_cfg.tp_size > 1:
            for stage in range(self.decode_cfg.pp_size):
                group_name = self._decode_tp_pg(stage)
                groups[group_name] = [self._get_decode_npu_id(stage, rank) for rank in range(self.decode_cfg.tp_size)]
        return groups
    
    def exec(self) -> dict:
        nodes: Dict[int, list] = defaultdict(list) # npu_id -> list of nodes
        for npu in range(self.total_npus):
            nodes[npu].append(GlobalMetadata(version="0.0.4"))
        last_node_per_npu: Dict[int, Optional[object]] = {npu: None for npu in range(self.total_npus)}

        kv_ready_hooks = self._emit_prefill(nodes, last_node_per_npu)
        self._emit_kv_transfer(nodes, last_node_per_npu, kv_ready_hooks)
        self._first_token_recv = self._emit_first_token(nodes, last_node_per_npu)
        self._emit_decode(nodes, last_node_per_npu)

        return nodes
    
    def _kv_edges(self, layer: int) -> List[Tuple[int,int,int]]:
        """Return the list of edges needed to transfer the KV cache of a given layer from the prefill pool to the decode pool. Each edge is a tuple (src_npu, dst_npu, bytes)."""
        tp_prefill, tp_decode = self.prefill_cfg.tp_size, self.decode_cfg.tp_size
        prefill_stage = layer // self.layer_per_stage_prefill
        decode_stage = layer // self.layer_per_stage_decode
        prefill_npus = [self._get_prefill_npu_id(prefill_stage, rank) for rank in range(tp_prefill)]
        decode_npus = [self._get_decode_npu_id(decode_stage, rank) for rank in range(tp_decode)]
        full_size = self.kv_bytes_per_layer
        edges: List[Tuple[int,int,int]] = []

        if tp_prefill == tp_decode:
            #Mapping 1:1
            bytes_per_edge = full_size // tp_prefill
            edges = [(prefill_npus[rank], decode_npus[rank], bytes_per_edge) for rank in range(tp_prefill)]
        elif tp_decode % tp_prefill == 0:
            # decode pool has more TP shards, each prefill shard sends to multiple decode shards (1:k)
            k = tp_decode // tp_prefill
            bytes_per_edge = full_size // tp_decode
            for rank in range(tp_prefill):
                for j in range(k):
                    edges.append((prefill_npus[rank], decode_npus[rank*k + j], bytes_per_edge))
        elif tp_prefill % tp_decode == 0:
            # prefill pool has more TP shards, each decode shard receives from multiple prefill shards and merges them (k:1)
            k = tp_prefill // tp_decode
            bytes_per_edge = full_size // tp_prefill
            for rank in range(tp_prefill):
                edges.append((prefill_npus[rank], decode_npus[rank // k], bytes_per_edge))
        else:
            # This should be prevented by the config validation, but we check again for safety
            raise ValueError("TP sizes must divide one another (checked in Config)")        
        return edges
    
    # ------------------------------------------------------------------ #
    # Firts phase: prefill
    # ------------------------------------------------------------------ #
    def _emit_prefill(self, nodes: Dict, last_node_per_npu: Dict) -> Dict[Tuple[int,int], object]:
        """Emit the prefill pass, return the dict describing the kv_ready[(layer, src_npu)] nodes after which the KV cache of each layer is ready to be sent to the decode pool."""
        kv_ready_hooks: Dict[Tuple[int,int], object] = {}
        tp_size = self.prefill_cfg.tp_size

        for stage in range(self.prefill_cfg.pp_size):
            for rank in range(tp_size):
                npu = self._get_prefill_npu_id(stage, rank)
                process_group = self._prefill_tp_pg(stage) if tp_size > 1 else None

                #Receive the activations from the previous stage
                if stage > 0:
                    prev_stage_npu = npu - tp_size
                    recv_node = receive(
                        sender=prev_stage_npu, receiver=npu, size=self.pp_prefill_bytes,
                        parents=[last_node_per_npu[npu]] if last_node_per_npu[npu] else None,
                        name=f"COMM_RECV_PREFILL_pp{stage}_tp{rank}"
                    )
                    nodes[npu].append(recv_node)
                    last_node_per_npu[npu] = recv_node

                # Emit the prefill computation for the layers owned by this stage
                for local_layer_idx in range(self.layer_per_stage_prefill):
                    global_layer_idx = (stage*self.layer_per_stage_prefill) + local_layer_idx
                    emit_result = self.prefill_model.prefill(
                        name=f"COMP_PREFILL_L{global_layer_idx}_pp{stage}_tp{rank}", 
                        layer=global_layer_idx, prompt_lens=self.prompt_lens, pg_name=process_group
                    )
                    if last_node_per_npu[npu]:
                        add_dependencies(emit_result.nodes[0], [last_node_per_npu[npu]])
                    nodes[npu].extend(emit_result.nodes)
                    last_node_per_npu[npu] = emit_result.tail
                    kv_ready_hooks[(global_layer_idx, npu)] = emit_result.kv_ready

                # Send the activations to the next stage
                if stage < self.prefill_cfg.pp_size - 1:
                    next_stage_npu = npu + tp_size
                    send_node = send(
                        sender=npu, receiver=next_stage_npu, size=self.pp_prefill_bytes,
                        parents=[last_node_per_npu[npu]] if last_node_per_npu[npu] else None,
                        name=f"COMM_SEND_PREFILL_pp{stage}_tp{rank}"
                    )
                    nodes[npu].append(send_node)
                    last_node_per_npu[npu] = send_node 
        return kv_ready_hooks

    # ------------------------------------------------------------------ #
    # Second phase: KV transfer from prefill to decode
    # ------------------------------------------------------------------ #
    def _emit_kv_transfer(self, nodes: Dict, last_node_per_npu: Dict, kv_ready_hooks: Dict) -> None:
        is_pull_explicit = (self.kv_cfg.direction == "pull" and self.kv_cfg.explicit_request)
        if self.kv_cfg.mode == "streaming":
            self._emit_streaming_transfer(nodes, kv_ready_hooks, is_pull_explicit)
        else:
            self._emit_bulk_transfer(nodes, last_node_per_npu, is_pull_explicit)

    def _emit_streaming_transfer(self, nodes: Dict, kv_ready_hooks: Dict, is_pull_explicit: bool) -> None:
        """Emit the streaming transfer of the KV cache for each layer as soon as it is ready in the prefill pool, overlapping communication with the rest of prefill (Splitwise approach)"""
        for layer in range(self.num_layers):
            for (src_npu, dst_npu, size) in self._kv_edges(layer):
                tag = f"KV_L{layer}_{src_npu}to{dst_npu}"
                ready_hook = kv_ready_hooks[(layer, src_npu)]
                recv_node = self._create_kv_transfer_pair(nodes, src_npu, dst_npu, size, ready_hook, tag, is_pull_explicit)
                self._stream_recv[(layer, dst_npu)].append(recv_node)
    
    def _emit_bulk_transfer(self, nodes: Dict, last_node_per_npu: Dict, is_pull_explicit: bool) -> None:
        """One SEND/RECV per (src,dst) pair carrying the whole KV (all owned layers), hung off the prefill tail of the source (DistServe approach)"""
        aggregated_bytes: Dict[Tuple[int,int], int] = defaultdict(int) # (src,dst) -> total bytes
        for layer in range(self.num_layers):
            for (src_npu, dst_npu, size) in self._kv_edges(layer):
                aggregated_bytes[(src_npu, dst_npu)] += size
        for (src_npu, dst_npu), size in aggregated_bytes.items():
            tag = f"KV_BULK_{src_npu}to{dst_npu}"
            ready_hook = last_node_per_npu[src_npu]
            recv_node = self._create_kv_transfer_pair(nodes, src_npu, dst_npu, size, ready_hook, tag, is_pull_explicit)
            self._bulk_recv[dst_npu].append(recv_node)

    def _create_kv_transfer_pair(self, nodes: Dict, src_npu: int, dst_npu: int, size: int, ready_hook: object, tag: str, is_pull_explicit: bool) -> object:
        """Emit the send and receive nodes for a single KV transfer, with dependencies to ensure the send happens after `ready`. If `pull_explicit` is True, also emit a request SEND/RECV pair before the transfer."""
        if is_pull_explicit:
            #send request
            req_send = send(
                sender=dst_npu, receiver=src_npu, size=REQUEST_BYTES, 
                parents=None,name=f"COMM_SEND_KVREQ_{tag}"
            )
            nodes[dst_npu].append(req_send)
            req_recv = receive(
                sender=dst_npu, receiver=src_npu, size=REQUEST_BYTES, 
                parents=None, name=f"COMM_RECV_KVREQ_{tag}"
            )
            nodes[src_npu].append(req_recv)

            #send KV cache after receiving the request and after the data is ready
            kv_send = send(
                sender=src_npu, receiver=dst_npu, size=size, 
                parents=[ready_hook, req_recv], name=f"COMM_SEND_KV_{tag}"
            )
            nodes[src_npu].append(kv_send)
            kv_recv = receive(
                sender=src_npu, receiver=dst_npu, size=size, 
                parents=[req_send], name=f"COMM_RECV_KV_{tag}"
            )
            nodes[dst_npu].append(kv_recv)
        else:
            #Pull or implicit push: prefill just sends as soon as ready, decode just receives without waiting for an explicit request
            kv_send = send(
                sender=src_npu, receiver=dst_npu, size=size, 
                parents=[ready_hook], name=f"COMM_SEND_KV_{tag}"
            )
            nodes[src_npu].append(kv_send)
            kv_recv = receive(
                sender=src_npu, receiver=dst_npu, size=size, 
                parents=None, name=f"COMM_RECV_KV_{tag}"
            )
            nodes[dst_npu].append(kv_recv)

        return kv_recv

    def _get_kv_arrival_dependencies(self, layer: int, npu: int, local_idx: int) -> list:
        """KV nodes that the first decode step of `layer` on `npu` must wait for. 
        Streaming gates per layer; bulk gates the whole NPU on its first owned layer."""
        if self.kv_cfg.mode == "streaming":
            return list(self._stream_recv.get((layer, npu), []))
        if local_idx == 0:
            return list(self._bulk_recv.get(npu, []))
        return []
    
    # ------------------------------------------------------------------ #
    # Second phase: emit the first token computed by the prefill
    # ------------------------------------------------------------------ #
    def _emit_first_token(self, nodes: Dict, last_node_per_npu: Dict) -> Dict:
        """Causal handoff of the FIRST token: prefill last stage -> decode stage 0.
        Transport only (token id, SAMPLE_BYTES). Mirrors the autoregressive feedback"""
        last_stage = self.prefill_cfg.pp_size - 1
        tp_p = self.prefill_cfg.tp_size
        token_recv_per_npu = {}
        for dst_rank in range(self.decode_cfg.tp_size):
            src_npu = self._get_prefill_npu_id(last_stage, dst_rank % tp_p)
            dst_npu = self._get_decode_npu_id(0, dst_rank)
            send_node = send(sender=src_npu, receiver=dst_npu, size=SAMPLE_BYTES,
                            parents=[last_node_per_npu[src_npu]],
                            name=f"COMM_SEND_FIRSTTOK_tp{dst_rank}")
            nodes[src_npu].append(send_node)
            last_node_per_npu[src_npu] = send_node
            recv_node = receive(sender=src_npu, receiver=dst_npu, size=SAMPLE_BYTES,
                                parents=None, name=f"COMM_RECV_FIRSTTOK_tp{dst_rank}")
            nodes[dst_npu].append(recv_node)
            token_recv_per_npu[dst_npu] = recv_node
        return token_recv_per_npu
    
    # ------------------------------------------------------------------ #
    # Fourth phase: decode
    # ------------------------------------------------------------------ #
    def _emit_decode(self, nodes: Dict, last_node_per_npu: Dict) -> None:
        """Emit a single PP traversal for one decode step. Each step processes exactly one new token; every layer reads a KV cache of length `kv_len`. """
        tp_size = self.decode_cfg.tp_size
        for npu in range(self.prefill_npus, self.total_npus):
            # Streaming overlaps KV *transfer* with prefill, but the first decode step is still gated by the first-token handoff (see _emit_first_token).
            # Done just for safety, not useful since prefill does not touch decode NPUs traces
            last_node_per_npu[npu] = None 
        
        for step in range(self.max_decode_steps):
            active_requests = [i for i, req in enumerate(self.requests) if req.gen_len > step]
            current_kv_lens = [self.requests[i].prompt_len + step + 1 for i in active_requests] # each active request has a KV cache length equal to its prompt length + number of decode steps already processed + 1 for the new token
            active_batch_size = len(active_requests)
            pp_decode_bytes = int(self.scale * active_batch_size * self.hidden_size * self.bytes_per_val)

            for stage in range(self.decode_cfg.pp_size):
                for rank in range(tp_size):
                    npu = self._get_decode_npu_id(stage, rank)
                    process_group = self._decode_tp_pg(stage) if tp_size > 1 else None

                    #Receives the activations from the previous stage (or the KV cache for the first stage)
                    if stage > 0:
                        recv_node = receive(
                            sender=npu - tp_size, receiver=npu, size=pp_decode_bytes, 
                            parents=[last_node_per_npu[npu]] if last_node_per_npu[npu] else None,
                            name=f"COMM_RECV_DECODE_step{step}_pp{stage}_tp{rank}"
                        )
                        nodes[npu].append(recv_node)
                        last_node_per_npu[npu] = recv_node
                    
                    # Emit the decode computation for the layers owned by this stage
                    for local_layer_idx in range(self.layer_per_stage_decode):
                        global_layer_idx = stage*self.layer_per_stage_decode + local_layer_idx
                        emit_result = self.decode_model.decode(
                            name=f"COMP_DECODE_step{step}_L{global_layer_idx}_pp{stage}_tp{rank}", 
                            layer=global_layer_idx, kv_lens=current_kv_lens, pg_name=process_group
                        )

                        dependencies=[]
                        if last_node_per_npu[npu]:
                            dependencies.append(last_node_per_npu[npu])
                        if step == 0:
                            # Make sure the first decode step waits for the KV cache to be ready
                            dependencies += self._get_kv_arrival_dependencies(global_layer_idx, npu, local_layer_idx)
                            if local_layer_idx == 0:
                                # The first layer of the first decode stage must also wait for the first token from prefill
                                token_recv = self._first_token_recv.get(npu)
                                if token_recv:
                                    dependencies.append(token_recv)
                        if dependencies:
                            add_dependencies(emit_result.nodes[0], dependencies)
                        
                        nodes[npu].extend(emit_result.nodes)
                        last_node_per_npu[npu] = emit_result.tail
                    
                    if stage < self.decode_cfg.pp_size - 1:
                        next_stage_npu = npu + tp_size
                        send_node = send(
                            sender=npu, receiver=next_stage_npu, size=pp_decode_bytes, 
                            parents=[last_node_per_npu[npu]] if last_node_per_npu[npu] else None,
                            name=f"COMM_SEND_DECODE_step{step}_pp{stage}_tp{rank}"
                        )
                        nodes[npu].append(send_node)
                        last_node_per_npu[npu] = send_node
            
            # Autoregressive serizialization
            if step < self.max_decode_steps - 1:
                self._emit_autoregressive_feedback(nodes, last_node_per_npu, label=f"DECODE_step{step}")
    
    def _emit_autoregressive_feedback(self, nodes: Dict, last_node_per_npu: Dict, label: str) -> None:
        """The token sampled at the tail PP stage must reach the head stage before the next decode step starts"""
        if not (self.serialize_decode and self.decode_cfg.pp_size > 1):
            # With pp_d == 1 the per-NPU chain already serialises steps, so this is only needed for pp_d > 1.
            return
        
        tp_size = self.decode_cfg.tp_size
        for rank in range(tp_size):
            src_npu = self._get_decode_npu_id(self.decode_cfg.pp_size - 1, rank)
            dst_npu = self._get_decode_npu_id(0, rank) 

            send_node = send(
                sender=src_npu, receiver=dst_npu, size=SAMPLE_BYTES, 
                parents=[last_node_per_npu[src_npu]] if last_node_per_npu[src_npu] else None,
                name=f"COMM_SEND_SAMPLE_{label}_tp{rank}"
            )
            nodes[src_npu].append(send_node)
            last_node_per_npu[src_npu] = send_node

            recv_node = receive(
                sender=src_npu, receiver=dst_npu, size=SAMPLE_BYTES, 
                parents=[last_node_per_npu[dst_npu]] if last_node_per_npu[dst_npu] else None,
                name=f"COMM_RECV_SAMPLE_{label}_tp{rank}"
            )
            nodes[dst_npu].append(recv_node)
            last_node_per_npu[dst_npu] = recv_node