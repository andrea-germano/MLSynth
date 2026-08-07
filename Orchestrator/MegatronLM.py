# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from Orchestrator.Orchestrator import Orchestrator
from Utils.config import TrainRunConfig
from Utils.nodes import add_dependencies, allreduce, receive, send
from Layer.Layer import MoeEpContext
from chakra.schema.protobuf.et_def_pb2 import (GlobalMetadata)


class MegatronLM(Orchestrator):
    """3D-parallel (DP x PP x TP) training orchestrator. `model` is a TrainingModel or a Wrapper around one"""
    def __init__(self, model, run: TrainRunConfig):
        self.model = model
        self.dp_size = run.parallelism.dp_size
        self.pp_size = run.parallelism.pp_size
        self.tp_size = run.parallelism.tp_size
        self.num_npus = self.dp_size * self.pp_size * self.tp_size
        self.num_microbatches = run.training.num_microbatches
        self.moe = run.model.moe
        self.routing = run.moe_routing
        self.ep_size = run.parallelism.ep
        self.edp_size = run.parallelism.edp
        self.scale = run.model.scale

    def _npu_id(self, dp_group: int, pp_stage: int, tp_shard: int) -> int:
        """The device layout. Everything else that needs npu ids goes through here."""
        return dp_group * (self.pp_size * self.tp_size) + (pp_stage * self.tp_size) + tp_shard

    def _flat(self, dp_group: int, tp_shard: int) -> int:
        """Index of a device within its pipeline stage, tp FASTEST so that consecutive EP ranks
        are physically adjacent (Megatron keeps EP x TP inside one NVLink domain)."""
        return dp_group * self.tp_size + tp_shard

    def _ep_context(self, dp_group: int, pp_stage: int, tp_shard: int) -> MoeEpContext:
        """Locate a device in the MoE mesh of its stage: etp(=1) x ep x edp over the tp*dp
        devices, i.e. edp contiguous clusters of ep ranks. The cluster is the unit the
        all-to-all runs over; the position inside it is the row/column of the traffic matrix."""
        flat = self._flat(dp_group, tp_shard)
        cluster, ep_rank = divmod(flat, self.ep_size)
        return MoeEpContext(peers=self._cluster_peers(cluster, pp_stage),
                            ep_rank=ep_rank, cluster=cluster, clusters=self.edp_size,
                            stage=pp_stage, pg_name=self._ep_pg_name(cluster, pp_stage))

    def _cluster_peers(self, cluster: int, pp_stage: int) -> list[int]:
        """npu_ids of one EP cluster, ordered by their position in it."""
        return [self._npu_id(flat // self.tp_size, pp_stage, flat % self.tp_size)
                for flat in range(cluster * self.ep_size, (cluster + 1) * self.ep_size)]

    def _expdp_peers(self, offset: int, pp_stage: int) -> list[int]:
        """One expert-DP group: the ranks at the same position across the edp clusters. Carries
        the expert-gradient all-reduce, and exists only when the experts are replicated."""
        flats = (cluster * self.ep_size + offset for cluster in range(self.edp_size))
        return [self._npu_id(flat // self.tp_size, pp_stage, flat % self.tp_size)
                for flat in flats]

    @staticmethod
    def _ep_pg_name(cluster: int, pp_stage: int) -> str:
        return f"ep_{cluster}_{pp_stage}"

    @staticmethod
    def _expdp_pg_name(offset: int, pp_stage: int) -> str:
        return f"expdp_{offset}_{pp_stage}"

    def _dp_pg_name(self, pp_stage: int, tp_shard: int) -> str:
        # dp all-reduce group consists of all ranks that share the same pipeline stage and tensor parallel shard
        pp_name = "" if self.pp_size <= 1 else f"pp_{pp_stage}"
        tp_name = "" if self.tp_size <= 1 else f"_tp_{tp_shard}"
        return f"{pp_name}{tp_name}"

    def generate_comm_groups(self):
        comm_groups = defaultdict(list)

        # generate comm groups for data parallel groups
        for dp_group in range(self.dp_size):
            for pp_stage in range(self.pp_size):
                for tp_shard in range(self.tp_size):
                    comm_groups[self._dp_pg_name(pp_stage, tp_shard)].append(self._npu_id(dp_group, pp_stage, tp_shard))
        
        # generate comm groups for each tensor parallel group
        if self.tp_size > 1:
            for tp_group in range(self.num_npus // self.tp_size):
                tp_comm_group = []
                base = tp_group * self.tp_size
                for npu in range(self.tp_size):
                    npu_id = base + npu
                    tp_comm_group.append(npu_id)
                comm_groups[f"tp_{tp_group}"] = tp_comm_group

        # MoE groups. An EP cluster is `ep` consecutive devices of a stage and is only needed as
        # a process group by the native ALL_TO_ALL fast path (the p2p path addresses peers
        # directly). The expert-DP group exists only when the experts are replicated.
        if self.moe is not None:
            for pp_stage in range(self.pp_size):
                if self.routing.dispatch == "collective" and self.ep_size > 1:
                    for cluster in range(self.edp_size):
                        comm_groups[self._ep_pg_name(cluster, pp_stage)] = self._cluster_peers(
                            cluster, pp_stage)
                if self.edp_size > 1:
                    for offset in range(self.ep_size):
                        comm_groups[self._expdp_pg_name(offset, pp_stage)] = self._expdp_peers(offset, pp_stage)
        return comm_groups

    def exec(self) -> dict:
        B = self.model.get_batch_size()
        S = self.model.get_sequence_len()
        d = self.model.get_hidden_size()
        bytes_per_val = self.model.get_bytes_per_val()

        layers_per_pipeline_stage = self.model.get_num_layers() // self.pp_size
        pp_comm_size = int((B*S*d*bytes_per_val * self.scale) / self.num_microbatches)
        # MoE models exclude the expert weights here: those are reduced over the expert-DP group
        # instead, and only when they are actually replicated (edp > 1).
        dp_comm_size = int(self.scale * self.model.dp_sync_params * bytes_per_val / self.tp_size / self.pp_size)
        expert_comm_size = int(self.scale * self.model.expert_sync_params * bytes_per_val / self.pp_size)

        # print(f"Pipeline comm size: {pp_comm_size / 1024 / 1024:,.2f} MB")
        # print(f"DP comm size: {dp_comm_size / 1024 / 1024 / 1024:,.2f} GB")

        nodes = defaultdict(list)

        for dp_group in range(self.dp_size):
            #print(f"------------ DP GROUP {dp_group} ------------")
            for pp_stage in range(self.pp_size):
                for tp_shard in range(self.tp_size):
                    npu_id = self._npu_id(dp_group, pp_stage, tp_shard)
                    tp_group = npu_id // self.tp_size
                    ep_ctx = self._ep_context(dp_group, pp_stage, tp_shard) if self.moe else None
                    nodes[npu_id].append(GlobalMetadata(version="0.0.4"))
                    # print(f"NPU {npu_id} - dp group: {dp_group}, pp stage: {pp_stage}, tp shard: {tp_shard}")
                    # -------------
                    # Forward pass
                    # -------------
                    prev_rcv = None
                    prev_comp = None
                    for b in range(self.num_microbatches):
                        rcv_node = None
                        if pp_stage != 0 and self.pp_size > 1:
                            #print(f"RCV ({npu_id - tp_size} -> {npu_id})")
                            rcv_node = receive(npu_id - self.tp_size, npu_id, pp_comm_size, parents=[prev_rcv], name=f"COMM_RECV_NODE_FWD_b{b}_dp{dp_group}pp{pp_stage}tp{tp_shard}")
                            nodes[npu_id].append(rcv_node)
                            prev_rcv = rcv_node
                        
                        for layer in range(layers_per_pipeline_stage):
                            current_layer = pp_stage * layers_per_pipeline_stage + layer
                            cmp_nodes = self.model.fwd(name=f"COMP_NODE_FWD_b{b}", npu_id=npu_id, layer=current_layer, num_batches=B/self.num_microbatches, pg_name=f"tp_{tp_group}", microbatch=b, ep_ctx=ep_ctx)
                            if layer == 0:
                                add_dependencies(cmp_nodes[0], [rcv_node, prev_comp])
                            else:
                                add_dependencies(cmp_nodes[0], [prev_comp])
                            for node in cmp_nodes:
                                nodes[npu_id].append(node)
                            prev_comp = cmp_nodes[-1]
                        
                        if pp_stage != self.pp_size - 1 and self.pp_size > 1:
                            #print(f"SND ({npu_id} -> {npu_id + tp_size})")
                            snd_node = send(npu_id, npu_id + self.tp_size, pp_comm_size, parents=[prev_comp], name=f"COMM_SEND_NODE_FWD_b{b}_dp{dp_group}pp{pp_stage}tp{tp_shard}")
                            nodes[npu_id].append(snd_node)                
                    # -------------
                    # Backward pass
                    # -------------
                    #print("Backward pass")
                    for b in range(self.num_microbatches):
                        bck_rcv_node = None
                        if pp_stage != self.pp_size - 1 and self.pp_size > 1:
                            #print(f"RCV ({npu_id + tp_size} -> {npu_id})")
                            bck_rcv_node = receive(npu_id + self.tp_size, npu_id, pp_comm_size, parents=[prev_rcv], name=f"COMM_RECV_NODE_BCKWD_b{b}_dp{dp_group}pp{pp_stage}tp{tp_shard}")
                            nodes[npu_id].append(bck_rcv_node)
                            prev_rcv = bck_rcv_node
                        
                        for layer in range(layers_per_pipeline_stage):
                            current_layer = pp_stage * layers_per_pipeline_stage + layer
                            bck_cmp_nodes = self.model.bckwd(name=f"COMP_NODE_BCKWD_b{b}", npu_id=npu_id, layer=current_layer, num_batches=B/self.num_microbatches, pg_name=f"tp_{tp_group}", microbatch=b, ep_ctx=ep_ctx)
                            if layer == 0:
                                add_dependencies(bck_cmp_nodes[0], [bck_rcv_node, prev_comp])
                            else:
                                add_dependencies(bck_cmp_nodes[0], [prev_comp])
                            for node in bck_cmp_nodes:
                                nodes[npu_id].append(node)
                            prev_comp = bck_cmp_nodes[-1]
                        
                        
                        if pp_stage != 0 and self.pp_size > 1:
                            #print(f"SND ({npu_id} -> {npu_id - tp_size})")
                            bck_snd_node = send(npu_id, npu_id - self.tp_size, pp_comm_size, parents=[prev_comp], name=f"COMM_SEND_NODE_BCKWD_b{b}_dp{dp_group}pp{pp_stage}tp{tp_shard}")
                            nodes[npu_id].append(bck_snd_node)

                    if self.dp_size > 1:
                        dp_comm_node = allreduce(dp_comm_size, parents=[prev_comp], pg_name=self._dp_pg_name(pp_stage, tp_shard), name=f"COMM_COLL_NODE_DP_All-Reduce_dp{dp_group}pp{pp_stage}tp{tp_shard}")
                        nodes[npu_id].append(dp_comm_node)
                        prev_comp = dp_comm_node
                    if expert_comm_size:
                        # Expert weights are replicated edp times, so their gradients are reduced over the expert-DP group
                        node = allreduce(expert_comm_size, parents=[prev_comp],
                                         pg_name=self._expdp_pg_name(self._flat(dp_group, tp_shard) % self.ep_size, pp_stage),
                                         name=f"COMM_COLL_NODE_EXPDP_All-Reduce_dp{dp_group}pp{pp_stage}tp{tp_shard}")
                        nodes[npu_id].append(node)
        return nodes