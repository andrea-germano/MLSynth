from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from chakra.src.third_party.utils.protolib import encodeMessage as encode_message

from parser import RunConfig
from Model.TransformerInference import TransformerInference
from Orchestrator.DisaggregatedInference import DisaggregatedInference

def write_comm_groups(comm_groups, path="") -> None:
    with open(os.path.join(path, "comm_groups.json"), "w") as f:
        json.dump(comm_groups, f, indent=2, sort_keys=True)
    
def write_nodes(nodes, name: str, path : str ="") -> None:
    for npu_id in nodes.keys():
        with open(os.path.join(path, f"{name}.{npu_id}.et"), "wb") as et:
            for node in nodes[npu_id]:
                encode_message(et, node)

def main(argv = None) -> int:
    parser = argparse.ArgumentParser(description="Synthesize disaggregated inference workload")
    parser.add_argument("-c", "--config", default="input_inference.yaml",
                        help="Path to inference YAML config file")
    parser.add_argument("-o", "--out-dir", default="output_inference",
                        help="Base path to output directory (default: output_inference)")
    args = parser.parse_args(argv)

    run = RunConfig.from_yaml(args.config)

    out_dir = Path(args.out_dir) / run.model.name
    
    et_dir = out_dir / "et"
    et_dir.mkdir(parents=True, exist_ok=True)

    # one shared model; the orchestrator derives prefill/decode views
    model = TransformerInference(run.model, run.prefill)
    orch = DisaggregatedInference(model, run)

    comm_groups = orch.generate_comm_groups()
    write_comm_groups(comm_groups, path=str(out_dir))

    nodes = orch.exec()
    write_nodes(nodes, run.model.name, path=str(et_dir))

    kv = run.inference.kv_transfer
    print(f"Wrote inference ET to {et_dir}")
    print(f" prefill pool : tp={run.prefill.tp_size} pp={run.prefill.pp_size} ({run.prefill.num_npus} npus)")
    print(f" decode  pool : tp={run.decode.tp_size} pp={run.decode.pp_size} ({run.decode.num_npus} npus)")
    print(f" total npus : {len(nodes)}")
    print(f" requests : {len(run.inference.requests)} decode steps : {orch.max_decode_steps}")
    print(f" kv transfer : {kv.mode}/{kv.direction} {' (explicit req)' if kv.direction == 'pull' and kv.explicit_request else ''}")
    print(f" comm_groups : {list(comm_groups.keys()) or 'none'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())