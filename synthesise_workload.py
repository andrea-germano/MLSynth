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

"""Single entry point for MLSynth: reads a YAML config and synthesises either a training
workload (top-level `training` block) or a disaggregated-inference workload (top-level
`inference` block)."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

from chakra.src.third_party.utils.protolib import encodeMessage as encode_message
from chakra.schema.protobuf.et_def_pb2 import COMM_SEND_NODE, Node as ChakraNode

from Utils.config import TrainRunConfig, load_config
from Utils.nodes import attr_val
from Model.TrainingModel import TrainingModel
from Model.InferenceModel import InferenceModel
from Model.MoeTrainingModel import MoeTrainingModel
from Model.MoeInferenceModel import MoeInferenceModel
from Wrapper.ComputeWrapper import ComputeWrapper
from Orchestrator.MegatronLM import MegatronLM
from Orchestrator.DisaggregatedInference import DisaggregatedInference


def write_comm_groups(comm_groups, path: str | Path) -> None:
    with open(os.path.join(path, "comm_groups.json"), "w") as f:
        json.dump(comm_groups, f, indent=2, sort_keys=True)

def write_nodes(nodes, name: str, path: str | Path) -> None:
    for npu_id in nodes.keys():
        with open(os.path.join(path, f"{name}.{npu_id}.et"), "wb") as et:
            for node in nodes[npu_id]:
                encode_message(et, node)

def assert_tag_uniqueness(nodes) -> None:
    """Assert that all communication tags are unique across all nodes. This is important to avoid collisions in astra-sim.
    Legacy training PP/DP edges carry tag 0 and are matched by emission order, so they are skipped."""
    seen = {}  # (src, dst, tag) -> name
    for npu_nodes in nodes.values():
        for n in npu_nodes:
            if not isinstance(n, ChakraNode):
                continue
            if n.type == COMM_SEND_NODE:
                src = attr_val(n, "comm_src")
                dst = attr_val(n, "comm_dst")
                tag = attr_val(n, "comm_tag")
                if tag == 0:
                    continue
                key = (src, dst, tag)
                if key in seen and seen[key] != n.name:
                    raise ValueError(
                        f"Collision tag on (src,dst)=({src},{dst}): "
                        f"'{seen[key]}' vs '{n.name}' (tag={tag}). "
                        f"This would break concurrent RECV matching in ASTRA-sim: disambiguate the names"
                    )
                seen[key] = n.name


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Synthesize a training or inference workload from a YAML config")
    parser.add_argument("-c", "--config", default="input.yaml",
                        help="Path to the YAML config file (its `training`/`inference` block selects the mode)")
    parser.add_argument("-o", "--out-dir", default="output",
                        help="Base output directory (default: output)")
    args = parser.parse_args(argv)

    run = load_config(args.config)
    training = isinstance(run, TrainRunConfig)

    moe = run.model.moe is not None
    if training:
        m, p, t = run.model, run.parallelism, run.training
        name = f"{m.name}_{p.dp_size}dp_{p.pp_size}pp_{p.tp_size}tp_{t.batch_size}B_{t.sequence_len}S_{m.vocab_size}V_{m.hidden_size}d_{m.bytes_per_val}b_{int(m.scale*100)}scale"
        if moe:
            name += f"_{p.ep}ep_{m.moe.num_experts}E"
        model = MoeTrainingModel(run) if moe else TrainingModel(run)
    else:
        name = run.model.name
        # per-run auto-name, kept for reference:
        # p, d = run.prefill, run.decode
        # name = f"{run.model.name}_p{p.tp_size}tp{p.pp_size}pp_d{d.tp_size}tp{d.pp_size}pp_{len(run.inference.requests)}req_{run.inference.kv_transfer}"
        # one shared model; the orchestrator derives prefill/decode views
        model = (MoeInferenceModel(run.model, run.moe_routing, run.prefill) if moe
                 else InferenceModel(run.model, run.prefill))

    if run.wrapper:
        model = ComputeWrapper(model, run.wrapper)
    orchestrator = MegatronLM(model, run) if training else DisaggregatedInference(model, run)

    out_dir = Path(args.out_dir) / name
    et_dir = out_dir / "et"
    et_dir.mkdir(parents=True, exist_ok=True)

    write_comm_groups(orchestrator.generate_comm_groups(), path=out_dir)

    nodes = orchestrator.exec()
    # inference and MoE traces carry per-name comm tags that must not collide; legacy tag-0
    # training edges (PP/DP) are matched by order and skipped inside the check
    assert_tag_uniqueness(nodes)
    write_nodes(nodes, name, path=et_dir)

    print(f"Wrote {'training' if training else 'inference'} ET for {len(nodes)} npus to {et_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
