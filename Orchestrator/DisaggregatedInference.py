from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from chakra.schema.protobuf.et_def_pb2 import GlobalMetadata

from Utils.config import InferenceRunConfig
from Orchestrator.Orchestrator import Orchestrator
from Model.Model import BaseInferenceModel
from Utils.nodes import add_dependencies, send, receive
from Layer.Layer import MoeEpContext
from Utils.routing import PHASE_DECODE, PHASE_PREFILL
from Utils.naming import (comp_base, pp_name, kv_name, firsttok_name, decfb_name, comm_tag)

# size of bytes of the sampled token feedback (used for serialization of decode iterations and for the first token handoff from prefill to decode)
#! Maybe this should be calculated as the size of lm_head * bytes_per_val * scale, to more accurately reflect the size of the autoregressive feedback, but for now we keep it fixed and small as it is only used for synchronization and does not carry actual data in this model of the system.
SAMPLE_BYTES = 8

class DisaggregatedInference(Orchestrator):
    """Separate pool for prefill and decode to model a disaggregated inference system where prefill and decode can be executed on different hardware.
    Each pool can have its own TP and PP topology. The KV cache is transferred with a streaming mechanism: each layer can send its KV cache to the decode pool
    asynchronously as soon as it is produced in the prefill phase, overlapping comms with computation of next layer"""

    #Prefill pool: NPU ids [0, num_prefill_npus),
    #Decode pool: NPU ids [num_prefill_npus, num_prefill_npus + num_decode_npus)

    def __init__(self, model: BaseInferenceModel, run: InferenceRunConfig):
        self.run = run
        self.model = model

        self.prefill_cfg = run.prefill
        self.decode_cfg = run.decode
        self.kv_mode = run.inference.kv_transfer
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
        self.cached_lens = [req.cached_len for req in self.requests]
        self.computed_tokens = sum(prompt_len - cached_len for prompt_len, cached_len in zip(self.prompt_lens, self.cached_lens))
        self.max_decode_steps = max(req.gen_len for req in self.requests)

        self.kv_dim=run.model.key_value_dim # = hidden for MHA and kv_heads*head_dim for GQA
        self.moe = run.model.moe

        self._stream_recv: Dict[Tuple[int,int], List] = defaultdict(list) # (layer, dst) -> recvs
        self._bulk_recv: Dict[int, List] = defaultdict(list) # dst -> recvs

    # Device layout of a pool: pipeline stages are contiguous blocks of tp*dp devices
    def _get_prefill_npu_id(self, pp_stage: int, tp_rank: int, dp_rank: int = 0) -> int:
        cfg = self.prefill_cfg
        return (pp_stage * cfg.dp_size + dp_rank) * cfg.tp_size + tp_rank

    def _get_decode_npu_id(self, pp_stage: int, tp_rank: int, dp_rank: int = 0) -> int:
        cfg = self.decode_cfg
        return self.prefill_npus + (pp_stage * cfg.dp_size + dp_rank) * cfg.tp_size + tp_rank

    def _get_prefill_coord(self, npu_id: int) -> Tuple[int, int, int]:
        """(pp_stage, tp_rank, dp_rank) of an npu in the prefill pool."""
        cfg = self.prefill_cfg
        stage, within = divmod(npu_id, cfg.tp_size * cfg.dp_size)
        dp_rank, tp_rank = divmod(within, cfg.tp_size)
        return (stage, tp_rank, dp_rank)

    def _get_decode_coord(self, npu_id: int) -> Tuple[int, int, int]:
        """(pp_stage, tp_rank, dp_rank) of an npu in the decode pool."""
        cfg = self.decode_cfg
        stage, within = divmod(npu_id - self.prefill_npus, cfg.tp_size * cfg.dp_size)
        dp_rank, tp_rank = divmod(within, cfg.tp_size)
        return (stage, tp_rank, dp_rank)

    # Requests are assigned round-robin to the dp ranks of a pool. That is the static limit of what vLLM's router does
    def _owner(self, request_idx: int, dp_size: int) -> int:
        return request_idx % dp_size

    def _slice(self, dp_rank: int, dp_size: int) -> List[int]:
        """Indices of the requests owned by one dp rank."""
        return [i for i in range(len(self.requests)) if i % dp_size == dp_rank]

    @staticmethod
    def _dp_field(dp_rank: int, cfg) -> int | None:
        """dp coordinate for a node name, omitted when there is a single slice so that dense traces keep their original names."""
        return dp_rank if cfg.dp_size > 1 else None

    def _ep_context(self, pool: str, stage: int, dp_rank: int, tp_rank: int,
                    origin_tokens: List[int]) -> MoeEpContext:
        """The expert-parallel group of one device: every device of its pipeline stage, since in
        serving ep = tp * dp with whole experts (etp = 1, edp = 1)"""
        cfg = self.prefill_cfg if pool == "p" else self.decode_cfg
        npu_of = self._get_prefill_npu_id if pool == "p" else self._get_decode_npu_id
        peers = [npu_of(stage, tp, dp)
                 for dp in range(cfg.dp_size) for tp in range(cfg.tp_size)]
        return MoeEpContext(peers=peers, ep_rank=dp_rank * cfg.tp_size + tp_rank, cluster=0,
                            stage=stage, pool=pool, origin_tokens=origin_tokens)

    def _moe_kwargs(self, pool: str, stage: int, dp_rank: int, tp_rank: int,
                    origin_tokens, key: tuple, step: int = 0) -> dict:
        """Extra arguments the MoE layers need; empty for a dense model, so the dense call site
        is unchanged."""
        if not self.moe:
            return {}
        return {"ep_ctx": self._ep_context(pool, stage, dp_rank, tp_rank, origin_tokens),
                "key": key, "step": step}

    def _origin_tokens(self, pool: str, tokens_per_slice: List[int]) -> List[int]:
        """Tokens owned by each device of a stage. Sequence parallelism splits a slice's tokens
        across its tp ranks, so a device owns 1/tp of its slice."""
        cfg = self.prefill_cfg if pool == "p" else self.decode_cfg
        return [tokens_per_slice[dp] // cfg.tp_size
                for dp in range(cfg.dp_size) for _ in range(cfg.tp_size)]

    def _new_tokens(self, request_idxs: List[int]) -> int:
        """Prompt tokens actually computed for a set of requests (prefix cache already excluded)."""
        return sum(self.requests[i].prompt_len - self.requests[i].cached_len for i in request_idxs)

    def _kv_bytes(self, tokens: int) -> int:
        """Bytes of one layer's KV cache for `tokens` tokens, full hidden dimension."""
        return int(self.scale * 2 * self.kv_dim * self.bytes_per_val * tokens)

    def _activation_bytes(self, tokens: int) -> int:
        return int(self.scale * tokens * self.hidden_size * self.bytes_per_val)

    # SINGLE SOURCE OF TRUTH for the pg_names. They are used both when building comm_groups.json (generate_comm_groups) and when tagging the COMM_COLL nodes (_emit_prefill / _emit_decode).
    #They must be integers since astra-sim only supports integer pg names
    def _prefill_tp_pg(self, pp_stage: int, dp_rank: int = 0) -> str:
        return str(pp_stage * self.prefill_cfg.dp_size + dp_rank + 1)

    def _decode_tp_pg(self, pp_stage: int, dp_rank: int = 0) -> str:
        base = self.prefill_cfg.pp_size * self.prefill_cfg.dp_size
        return str(base + pp_stage * self.decode_cfg.dp_size + dp_rank + 1)

    def generate_comm_groups(self) -> dict:
        groups: Dict[str, List[int]] = {}
        if self.prefill_cfg.tp_size > 1:
            for stage in range(self.prefill_cfg.pp_size):
                for dp_rank in range(self.prefill_cfg.dp_size):
                    groups[self._prefill_tp_pg(stage, dp_rank)] = [
                        self._get_prefill_npu_id(stage, rank, dp_rank)
                        for rank in range(self.prefill_cfg.tp_size)]
        if self.decode_cfg.tp_size > 1:
            for stage in range(self.decode_cfg.pp_size):
                for dp_rank in range(self.decode_cfg.dp_size):
                    groups[self._decode_tp_pg(stage, dp_rank)] = [
                        self._get_decode_npu_id(stage, rank, dp_rank)
                        for rank in range(self.decode_cfg.tp_size)]
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
    
    def _kv_edges(self, layer: int) -> List[Tuple[int, int, int]]:
        """Edges carrying one layer's KV cache from the prefill pool to the decode pool, as
        (src_npu, dst_npu, bytes).

        Two independent reshardings compose here. Along dp the KV follows the REQUESTS: a prefill
        dp rank sends to a decode dp rank only the cache of the requests they both own, so the
        payload of a (src_dp, dst_dp) pair is the intersection of their slices. Along tp it
        follows the existing head sharding, unchanged."""
        cfg_p, cfg_d = self.prefill_cfg, self.decode_cfg
        prefill_stage = layer // self.layer_per_stage_prefill
        decode_stage = layer // self.layer_per_stage_decode

        edges: List[Tuple[int, int, int]] = []
        for src_dp in range(cfg_p.dp_size):
            owned = self._slice(src_dp, cfg_p.dp_size)
            for dst_dp in range(cfg_d.dp_size):
                shared = [i for i in owned if self._owner(i, cfg_d.dp_size) == dst_dp]
                if not shared:
                    continue
                prefill_npus = [self._get_prefill_npu_id(prefill_stage, rank, src_dp)
                                for rank in range(cfg_p.tp_size)]
                decode_npus = [self._get_decode_npu_id(decode_stage, rank, dst_dp)
                               for rank in range(cfg_d.tp_size)]
                edges.extend(self._tp_reshard(prefill_npus, decode_npus, self._kv_bytes(self._new_tokens(shared))))
        return edges

    @staticmethod
    def _tp_reshard(src_npus: List[int], dst_npus: List[int],
                    full_size: int) -> List[Tuple[int, int, int]]:
        """Split one (src_dp, dst_dp) payload across the tensor shards.

        Both pools shard the KV heads, so cut the payload into the FINEST common sharding --
        max(tp_src, tp_dst) pieces -- and give each piece its one owner on either side. That
        covers 1:1, the 1:k fan-out when the decode pool has more shards, and the k:1 merge when
        it has fewer, without a branch each. The tp degrees divide one another (config-checked),
        so every piece maps to exactly one shard per side."""
        tp_src, tp_dst = len(src_npus), len(dst_npus)
        if max(tp_src, tp_dst) % min(tp_src, tp_dst):
            raise ValueError("TP sizes must divide one another (checked in Config)")
        shards = max(tp_src, tp_dst)
        per_shard = full_size // shards
        return [(src_npus[s * tp_src // shards], dst_npus[s * tp_dst // shards], per_shard)
                for s in range(shards)]

    # ------------------------------------------------------------------ #
    # First phase: prefill
    # ------------------------------------------------------------------ #
    def _emit_prefill(self, nodes: Dict, last_node_per_npu: Dict) -> Dict[Tuple[int,int], object]:
        """Emit the prefill pass, return the dict describing the kv_ready[(layer, src_npu)] nodes after which the KV cache of each layer is ready to be sent to the decode pool."""
        kv_ready_hooks: Dict[Tuple[int,int], object] = {}
        cfg = self.prefill_cfg
        tp_size, dp_size = cfg.tp_size, cfg.dp_size
        stage_stride = tp_size * dp_size          # devices per pipeline stage

        tokens_per_slice = [self._new_tokens(self._slice(dp, dp_size)) for dp in range(dp_size)]
        origin_tokens = self._origin_tokens("p", tokens_per_slice) if self.moe else None

        for stage in range(cfg.pp_size):
          for dp_rank in range(dp_size):
            owned = self._slice(dp_rank, dp_size)
            prompt_lens = [self.requests[i].prompt_len for i in owned]
            cached_lens = [self.requests[i].cached_len for i in owned]
            pp_bytes = self._activation_bytes(self._new_tokens(owned))
            for rank in range(tp_size):
                npu = self._get_prefill_npu_id(stage, rank, dp_rank)
                process_group = self._prefill_tp_pg(stage, dp_rank) if tp_size > 1 else None

                #Receive the activations from the previous stage
                if stage > 0:
                    prev_stage_npu = npu - stage_stride
                    name=pp_name(pl="p", src_stage=stage-1, dst_stage=stage, sh=rank, it=0,
                                 dp=self._dp_field(dp_rank, cfg))
                    recv_node = receive(
                        sender=prev_stage_npu, receiver=npu, size=pp_bytes,
                        parents=[last_node_per_npu[npu]] if last_node_per_npu[npu] else None,
                        name=name, tag = comm_tag(name)
                    )
                    nodes[npu].append(recv_node)
                    last_node_per_npu[npu] = recv_node

                # Emit the prefill computation for the layers owned by this stage
                for local_layer_idx in range(self.layer_per_stage_prefill):
                    global_layer_idx = (stage*self.layer_per_stage_prefill) + local_layer_idx
                    emit_result = self.prefill_model.prefill(
                        name=comp_base(pl="p", ss=stage, sh=rank, L=global_layer_idx, it=0,
                                       dp=self._dp_field(dp_rank, cfg)),
                        npu_id=npu, layer=global_layer_idx, prompt_lens=prompt_lens, cached_lens=cached_lens,
                        pg_name=process_group, **self._moe_kwargs("p", stage, dp_rank, rank,
                                                                 origin_tokens,
                                                                 (PHASE_PREFILL, global_layer_idx))
                    )
                    if last_node_per_npu[npu]:
                        add_dependencies(emit_result.nodes[0], [last_node_per_npu[npu]])
                    nodes[npu].extend(emit_result.nodes)
                    last_node_per_npu[npu] = emit_result.tail
                    kv_ready_hooks[(global_layer_idx, npu)] = emit_result.kv_ready

                # Send the activations to the next stage
                if stage < cfg.pp_size - 1:
                    next_stage_npu = npu + stage_stride
                    name=pp_name(pl="p", src_stage=stage, dst_stage=stage+1, sh=rank, it=0,
                                 dp=self._dp_field(dp_rank, cfg))
                    send_node = send(
                        sender=npu, receiver=next_stage_npu, size=pp_bytes,
                        parents=[last_node_per_npu[npu]] if last_node_per_npu[npu] else None,
                        name=name, tag = comm_tag(name)
                    )
                    nodes[npu].append(send_node)
                    last_node_per_npu[npu] = send_node 
        return kv_ready_hooks

    # ------------------------------------------------------------------ #
    # Second phase: KV transfer from prefill to decode
    # ------------------------------------------------------------------ #
    def _emit_kv_transfer(self, nodes: Dict, last_node_per_npu: Dict, kv_ready_hooks: Dict) -> None:
        if self.kv_mode == "streaming":
            self._emit_streaming_transfer(nodes, kv_ready_hooks)
        else:
            self._emit_bulk_transfer(nodes, last_node_per_npu)

    def _emit_streaming_transfer(self, nodes: Dict, kv_ready_hooks: Dict) -> None:
        """Emit the streaming transfer of the KV cache for each layer as soon as it is ready in the prefill pool, overlapping communication with the rest of prefill (Splitwise approach)"""
        for layer in range(self.num_layers):
            for (src_npu, dst_npu, size) in self._kv_edges(layer):
                ps, sr, sdp = self._get_prefill_coord(src_npu)
                ds, dr, ddp = self._get_decode_coord(dst_npu)
                name_kv = kv_name(src_stage=ps, dst_stage=ds, ssh=sr, dsh=dr, it=0, L=layer,
                                  sdp=self._dp_field(sdp, self.prefill_cfg),
                                  ddp=self._dp_field(ddp, self.decode_cfg))
                ready_hook = kv_ready_hooks[(layer, src_npu)]
                recv_node = self._create_kv_transfer_pair(nodes, src_npu, dst_npu, size, ready_hook, name_kv)
                self._stream_recv[(layer, dst_npu)].append(recv_node)
    
    def _emit_bulk_transfer(self, nodes: Dict, last_node_per_npu: Dict) -> None:
        """One SEND/RECV per (src,dst) pair carrying the whole KV (all owned layers), hung off the prefill tail of the source (DistServe approach)"""
        aggregated_bytes: Dict[Tuple[int,int], int] = defaultdict(int) # (src,dst) -> total bytes
        for layer in range(self.num_layers):
            for (src_npu, dst_npu, size) in self._kv_edges(layer):
                aggregated_bytes[(src_npu, dst_npu)] += size
        for (src_npu, dst_npu), size in aggregated_bytes.items():
            ps, sr, sdp = self._get_prefill_coord(src_npu)
            ds, dr, ddp = self._get_decode_coord(dst_npu)
            name_kv = kv_name(src_stage=ps, dst_stage=ds, ssh=sr, dsh=dr, seg="all", it=0,
                              sdp=self._dp_field(sdp, self.prefill_cfg),
                              ddp=self._dp_field(ddp, self.decode_cfg))
            ready_hook = last_node_per_npu[src_npu]
            recv_node = self._create_kv_transfer_pair(nodes, src_npu, dst_npu, size, ready_hook, name_kv)
            self._bulk_recv[dst_npu].append(recv_node)

    def _create_kv_transfer_pair(self, nodes: Dict, src_npu: int, dst_npu: int, size: int, ready_hook: object, name_kv: str) -> object:
        """Emit the SEND/RECV pair for a single KV transfer. The SEND is gated on
        `ready_hook` so the cache is only pushed once prefill has produced it; the
        RECV is dependency-free on the decode side and is consumed by the first
        decode step that needs this layer's KV."""
        kv_send = send(
            sender=src_npu, receiver=dst_npu, size=size,
            parents=[ready_hook], name=name_kv, tag=comm_tag(name_kv)
        )
        nodes[src_npu].append(kv_send)
        kv_recv = receive(
            sender=src_npu, receiver=dst_npu, size=size,
            parents=None, name=name_kv, tag=comm_tag(name_kv)
        )
        nodes[dst_npu].append(kv_recv)
        return kv_recv

    def _get_kv_arrival_dependencies(self, layer: int, npu: int, local_idx: int) -> list:
        """KV nodes that the first decode step of `layer` on `npu` must wait for. 
        Streaming gates per layer; bulk gates the whole NPU on its first owned layer."""
        if self.kv_mode == "streaming":
            return list(self._stream_recv.get((layer, npu), []))
        if local_idx == 0:
            return list(self._bulk_recv.get(npu, []))
        return []
    
    # ------------------------------------------------------------------ #
    # Second phase: emit the first token computed by the prefill
    # ------------------------------------------------------------------ #
    def _emit_first_token(self, nodes: Dict, last_node_per_npu: Dict) -> Dict:
        """Causal handoff of the FIRST token: prefill last stage -> decode stage 0. Transport
        only (token id, SAMPLE_BYTES), mirroring the autoregressive feedback.

        Like the KV cache, the handoff follows the REQUESTS: a decode slice cannot start until
        every prefill device that produced one of its first tokens has handed it over, so there
        is one edge per (src_dp, dst_dp) pair whose slices intersect. Gating on a single device
        would let a slice start while some of its requests had not been prefilled yet."""
        last_stage = self.prefill_cfg.pp_size - 1
        cfg_p, cfg_d = self.prefill_cfg, self.decode_cfg
        token_recv_per_npu: Dict[int, List] = defaultdict(list)

        for dst_dp in range(cfg_d.dp_size):
            owned = self._slice(dst_dp, cfg_d.dp_size)
            producers = sorted({self._owner(i, cfg_p.dp_size) for i in owned})
            for src_dp in producers:
                for dst_rank in range(cfg_d.tp_size):
                    src_npu = self._get_prefill_npu_id(last_stage, dst_rank % cfg_p.tp_size, src_dp)
                    dst_npu = self._get_decode_npu_id(0, dst_rank, dst_dp)
                    name = firsttok_name(src_stage=last_stage, dst_stage=0, dsh=dst_rank, it=0,
                                         sdp=self._dp_field(src_dp, cfg_p),
                                         ddp=self._dp_field(dst_dp, cfg_d))
                    send_node = send(sender=src_npu, receiver=dst_npu, size=SAMPLE_BYTES,
                                     parents=[last_node_per_npu[src_npu]],
                                     name=name, tag=comm_tag(name))
                    nodes[src_npu].append(send_node)
                    recv_node = receive(sender=src_npu, receiver=dst_npu, size=SAMPLE_BYTES,
                                        parents=None, name=name, tag=comm_tag(name))
                    nodes[dst_npu].append(recv_node)
                    token_recv_per_npu[dst_npu].append(recv_node)
        return token_recv_per_npu

    # ------------------------------------------------------------------ #
    # Fourth phase: decode
    # ------------------------------------------------------------------ #
    def _emit_decode(self, nodes: Dict, last_node_per_npu: Dict) -> None:
        """Emit a single PP traversal for one decode step. Each step processes exactly one new token; every layer reads a KV cache of length `kv_len`. """
        cfg = self.decode_cfg
        tp_size, dp_size = cfg.tp_size, cfg.dp_size
        stage_stride = tp_size * dp_size
        for npu in range(self.prefill_npus, self.total_npus):
            # Streaming overlaps KV *transfer* with prefill, but the first decode step is still gated by the first-token handoff (see _emit_first_token).
            # Done just for safety, not useful since prefill does not touch decode NPUs traces
            last_node_per_npu[npu] = None 
        
        for step in range(self.max_decode_steps):
            for stage in range(cfg.pp_size):
              for dp_rank in range(dp_size):
                # a slice keeps only its own still-generating requests; a slice that runs dry
                # still walks the whole schedule, which is what vLLM does with dummy forward
                # passes so that the DP ranks stay in lockstep for the expert layers
                active_requests = [i for i in self._slice(dp_rank, dp_size)
                                   if self.requests[i].gen_len > step]
                # KV length = prompt + steps already done + the token produced now
                current_kv_lens = [self.requests[i].prompt_len + step + 1 for i in active_requests]
                pp_decode_bytes = self._activation_bytes(len(active_requests))
                origin_tokens = None
                if self.moe:
                    active_per_slice = [len([i for i in self._slice(dp, dp_size)
                                             if self.requests[i].gen_len > step])
                                        for dp in range(dp_size)]
                    origin_tokens = self._origin_tokens("d", active_per_slice)

                for rank in range(tp_size):
                    npu = self._get_decode_npu_id(stage, rank, dp_rank)
                    process_group = self._decode_tp_pg(stage, dp_rank) if tp_size > 1 else None

                    #Receives the activations from the previous stage (or the KV cache for the first stage)
                    if stage > 0:
                        name = pp_name(pl="d", src_stage=stage-1, dst_stage=stage, sh=rank, it=step,
                                       dp=self._dp_field(dp_rank, cfg))
                        recv_node = receive(
                            sender=npu - stage_stride, receiver=npu, size=pp_decode_bytes, 
                            parents=[last_node_per_npu[npu]] if last_node_per_npu[npu] else None,
                            name=name, tag=comm_tag(name)
                        )
                        nodes[npu].append(recv_node)
                        last_node_per_npu[npu] = recv_node
                    
                    # Emit the decode computation for the layers owned by this stage
                    for local_layer_idx in range(self.layer_per_stage_decode):
                        global_layer_idx = stage*self.layer_per_stage_decode + local_layer_idx
                        emit_result = self.decode_model.decode(
                            name=comp_base(pl="d", ss=stage, sh=rank, L=global_layer_idx, it=step,
                                           dp=self._dp_field(dp_rank, cfg)),
                            npu_id=npu, layer=global_layer_idx, kv_lens=current_kv_lens,
                            pg_name=process_group,
                            **self._moe_kwargs("d", stage, dp_rank, rank, origin_tokens,
                                               (PHASE_DECODE, step, global_layer_idx), step)
                        )

                        dependencies=[]
                        if last_node_per_npu[npu]:
                            dependencies.append(last_node_per_npu[npu])
                        if step == 0:
                            # Make sure the first decode step waits for the KV cache to be ready
                            dependencies += self._get_kv_arrival_dependencies(global_layer_idx, npu, local_layer_idx)
                            if local_layer_idx == 0:
                                # The first layer of the first decode stage must also wait for the first token from prefill
                                # every prefill device that produced one of this slice's
                                # first tokens must have handed it over
                                dependencies += self._first_token_recv.get(npu, [])
                        if dependencies:
                            add_dependencies(emit_result.nodes[0], dependencies)
                        
                        nodes[npu].extend(emit_result.nodes)
                        last_node_per_npu[npu] = emit_result.tail
                    
                    if stage < cfg.pp_size - 1:
                        next_stage_npu = npu + stage_stride
                        name = pp_name(pl="d", src_stage=stage, dst_stage=stage+1, sh=rank, it=step,
                                       dp=self._dp_field(dp_rank, cfg))
                        send_node = send(
                            sender=npu, receiver=next_stage_npu, size=pp_decode_bytes, 
                            parents=[last_node_per_npu[npu]] if last_node_per_npu[npu] else None,
                            name=name, tag=comm_tag(name)
                        )
                        nodes[npu].append(send_node)
                        last_node_per_npu[npu] = send_node
            
            # Autoregressive serialization
            if step < self.max_decode_steps - 1:
                self._emit_autoregressive_feedback(nodes, last_node_per_npu, step)
    
    def _emit_autoregressive_feedback(self, nodes: Dict, last_node_per_npu: Dict, step: int) -> None:
        """The token sampled at the tail PP stage must reach the head stage before the next decode step starts"""
        if not (self.serialize_decode and self.decode_cfg.pp_size > 1):
            # With pp_d == 1 the per-NPU chain already serialises steps, so this is only needed for pp_d > 1.
            return
        
        cfg = self.decode_cfg
        for dp_rank in range(cfg.dp_size):
          for rank in range(cfg.tp_size):
            src_npu = self._get_decode_npu_id(cfg.pp_size - 1, rank, dp_rank)
            dst_npu = self._get_decode_npu_id(0, rank, dp_rank)
            name = decfb_name(pl="d", src_stage=cfg.pp_size - 1, dst_stage=0, sh=rank, it=step,
                              dp=self._dp_field(dp_rank, cfg))

            send_node = send(
                sender=src_npu, receiver=dst_npu, size=SAMPLE_BYTES, 
                parents=[last_node_per_npu[src_npu]] if last_node_per_npu[src_npu] else None,
                name=name, tag=comm_tag(name)
            )
            nodes[src_npu].append(send_node)
            last_node_per_npu[src_npu] = send_node

            recv_node = receive(
                sender=src_npu, receiver=dst_npu, size=SAMPLE_BYTES, 
                parents=[last_node_per_npu[dst_npu]] if last_node_per_npu[dst_npu] else None,
                name=name, tag=comm_tag(name)
            )
            nodes[dst_npu].append(recv_node)
            last_node_per_npu[dst_npu] = recv_node