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
        self.scale = run.model.scale
        self.stage_stride = self.dp_size * self.tp_size
        # ASTRA-sim parses pg names with std::stoi, so they must be integers >= 1 ("" and "0"
        # are the world group)
        self._dp_pg_base = 1
        self._tp_pg_base = self._dp_pg_base + self.pp_size * self.tp_size

    def _npu_id(self, dp_group: int, pp_stage: int, tp_shard: int) -> int:
        """Megatron's rank order `tp-dp-pp`: tp fastest, pp outermost, so a pipeline stage is a
        contiguous block of dp*tp ids."""
        return pp_stage * self.stage_stride + dp_group * self.tp_size + tp_shard

    def _tp_pg(self, tp_group: int) -> str | None:
        """None when tp = 1: no tensor group is registered then, and a name pointing at a group
        that is not in comm_groups.json makes ASTRA-sim abort (Workload::extract_comm_group).
        The dense layer never asks in that case, the MoE all-to-all does, and it falls back to
        the world group -- which is where the original emitted it too, having no group at all."""
        return str(self._tp_pg_base + tp_group) if self.tp_size > 1 else None

    def _dp_pg(self, pp_stage: int, tp_shard: int) -> str | None:
        """The ranks sharing (pipeline stage, tensor shard). None when that is every npu
        (pp = tp = 1), which stays on ASTRA-sim's implicit world group."""
        if self.pp_size == 1 and self.tp_size == 1:
            return None
        return str(self._dp_pg_base + pp_stage * self.tp_size + tp_shard)

    def generate_comm_groups(self):
        comm_groups = defaultdict(list)

        # data parallel groups
        if self.dp_size > 1 and self._dp_pg(0, 0) is not None:
            for dp_group in range(self.dp_size):
                for pp_stage in range(self.pp_size):
                    for tp_shard in range(self.tp_size):
                        comm_groups[self._dp_pg(pp_stage, tp_shard)].append(self._npu_id(dp_group, pp_stage, tp_shard))

        # tensor parallel groups
        if self.tp_size > 1:
            for tp_group in range(self.num_npus // self.tp_size):
                base = tp_group * self.tp_size
                comm_groups[self._tp_pg(tp_group)] = [base + npu for npu in range(self.tp_size)]

        # no expert-parallel group: the MoE all-to-all rides the tensor one (world group at tp=1)
        return comm_groups

    def exec(self) -> dict:
        B = self.model.get_batch_size()
        S = self.model.get_sequence_len()
        d = self.model.get_hidden_size()
        bytes_per_val = self.model.get_bytes_per_val()

        # batch_size is the GLOBAL batch: each DP replica works on its 1/dp share
        replica_batch = B / self.dp_size

        layers_per_pipeline_stage = self.model.get_num_layers() // self.pp_size
        pp_comm_size = int((replica_batch*S*d*bytes_per_val * self.scale) / self.num_microbatches)
        # every parameter goes through the one DP all-reduce, experts included: no expert-DP group
        dp_comm_size = int(self.scale * self.model.num_params * bytes_per_val / self.tp_size / self.pp_size)

        # print(f"Pipeline comm size: {pp_comm_size / 1024 / 1024:,.2f} MB")
        # print(f"DP comm size: {dp_comm_size / 1024 / 1024 / 1024:,.2f} GB")

        nodes = defaultdict(list)

        for dp_group in range(self.dp_size):
            #print(f"------------ DP GROUP {dp_group} ------------")
            for pp_stage in range(self.pp_size):
                for tp_shard in range(self.tp_size):
                    npu_id = self._npu_id(dp_group, pp_stage, tp_shard)
                    tp_group = npu_id // self.tp_size
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
                            rcv_node = receive(npu_id - self.stage_stride, npu_id, pp_comm_size, parents=[prev_rcv], name=f"COMM_RECV_NODE_FWD_b{b}_dp{dp_group}pp{pp_stage}tp{tp_shard}")
                            nodes[npu_id].append(rcv_node)
                            prev_rcv = rcv_node
                        
                        for layer in range(layers_per_pipeline_stage):
                            current_layer = pp_stage * layers_per_pipeline_stage + layer
                            cmp_nodes = self.model.fwd(name=f"COMP_NODE_FWD_b{b}", npu_id=npu_id, layer=current_layer, num_batches=replica_batch/self.num_microbatches, pg_name=self._tp_pg(tp_group), microbatch=b)
                            if layer == 0:
                                add_dependencies(cmp_nodes[0], [rcv_node, prev_comp])
                            else:
                                add_dependencies(cmp_nodes[0], [prev_comp])
                            for node in cmp_nodes:
                                nodes[npu_id].append(node)
                            prev_comp = cmp_nodes[-1]
                        
                        if pp_stage != self.pp_size - 1 and self.pp_size > 1:
                            #print(f"SND ({npu_id} -> {npu_id + tp_size})")
                            snd_node = send(npu_id, npu_id + self.stage_stride, pp_comm_size, parents=[prev_comp], name=f"COMM_SEND_NODE_FWD_b{b}_dp{dp_group}pp{pp_stage}tp{tp_shard}")
                            nodes[npu_id].append(snd_node)                
                    # -------------
                    # Backward pass
                    # -------------
                    #print("Backward pass")
                    for b in range(self.num_microbatches):
                        bck_rcv_node = None
                        if pp_stage != self.pp_size - 1 and self.pp_size > 1:
                            #print(f"RCV ({npu_id + tp_size} -> {npu_id})")
                            bck_rcv_node = receive(npu_id + self.stage_stride, npu_id, pp_comm_size, parents=[prev_rcv], name=f"COMM_RECV_NODE_BCKWD_b{b}_dp{dp_group}pp{pp_stage}tp{tp_shard}")
                            nodes[npu_id].append(bck_rcv_node)
                            prev_rcv = bck_rcv_node
                        
                        for layer in range(layers_per_pipeline_stage):
                            current_layer = pp_stage * layers_per_pipeline_stage + layer
                            bck_cmp_nodes = self.model.bckwd(name=f"COMP_NODE_BCKWD_b{b}", npu_id=npu_id, layer=current_layer, num_batches=replica_batch/self.num_microbatches, pg_name=self._tp_pg(tp_group), microbatch=b)
                            if layer == 0:
                                add_dependencies(bck_cmp_nodes[0], [bck_rcv_node, prev_comp])
                            else:
                                add_dependencies(bck_cmp_nodes[0], [prev_comp])
                            for node in bck_cmp_nodes:
                                nodes[npu_id].append(node)
                            prev_comp = bck_cmp_nodes[-1]
                        
                        
                        if pp_stage != 0 and self.pp_size > 1:
                            #print(f"SND ({npu_id} -> {npu_id - tp_size})")
                            bck_snd_node = send(npu_id, npu_id - self.stage_stride, pp_comm_size, parents=[prev_comp], name=f"COMM_SEND_NODE_BCKWD_b{b}_dp{dp_group}pp{pp_stage}tp{tp_shard}")
                            nodes[npu_id].append(bck_snd_node)

                    if self.dp_size > 1:
                        dp_comm_node = allreduce(dp_comm_size, parents=[prev_comp], pg_name=self._dp_pg(pp_stage, tp_shard), name=f"COMM_COLL_NODE_DP_All-Reduce_dp{dp_group}pp{pp_stage}tp{tp_shard}")
                        nodes[npu_id].append(dp_comm_node)
                        prev_comp = dp_comm_node
        return nodes