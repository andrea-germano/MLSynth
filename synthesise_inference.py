import argparse
import json
import os
from pathlib import Path

import yaml
from chakra.src.third_party.utils.protolib import encodeMessage as encode_message

from Model.TransformerInference import TransformerInference
from Orchestrator.SingleDeviceInference import SingleDeviceInference

def write_comm_groups(comm_groups, path="") -> None:
    with open(os.path.join(path, "comm_groups.json"), "w") as f:
        json.dump(comm_groups, f, indent=2, sort_keys=True)
    
def write_nodes(nodes, name: str, path : str ="") -> None:
    for npu_id in nodes.keys():
        out = os.path.join(path, f"{name}.{npu_id}.et")
        with open(out, "wb") as et:
            for node in nodes[npu_id]:
                encode_message(et, node)

def validate_config(cfg) -> None:
    if "model" not in cfg:
        raise ValueError("Config must define a `model` section.")
    if "inference" not in cfg:
        raise ValueError("Config must define an `inference` section for the inference entrypoint.")
    
    m = cfg["model"]
    inf = cfg["inference"]

    required_model = ["name", "num_layers", "sequence_len", "vocab_size",
                      "hidden_size", "batch_size", "bytes_per_val"]
    for k in required_model:
        if k not in m:
            raise ValueError(f"Missing required model field: {k}")
    
    if "prompt_len" not in inf or "num_generated_tokens" not in inf:
        raise ValueError("`inference` section must define `prompt_len` and `num_generated_tokens`.")
    
def build_run_name(cfg: dict) -> str:
    m = cfg["model"]
    inf = cfg["inference"]
    return (
        f"{m['name']}_inference_"
        f"P{inf['prompt_len']}_G{inf['num_generated_tokens']}_"
        f"B{m['batch_size']}_S{m['sequence_len']}_V{m['vocab_size']}_"
        f"d{m['hidden_size']}_b{m['bytes_per_val']}_"
        f"scale{int(float(m.get('scale', 1.0))*100)}"
    )

def main(argv = None) -> int:
    parser = argparse.ArgumentParser(description="Synthesize inference workload from YAML config")
    parser.add_argument("-c", "--config", default="input_inference.yaml",
                        help="Path to inference YAML config file")
    args = parser.parse_args(argv)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg)

    name = build_run_name(cfg)
    out_dir = Path("output_inference") / name
    et_dir = out_dir / "et"
    et_dir.mkdir(parents=True, exist_ok=True)

    model = TransformerInference(cfg)
    orchestrator = SingleDeviceInference(model, cfg)

    comm_groups = orchestrator.generate_comm_groups()
    write_comm_groups(comm_groups, path=str(out_dir))

    nodes = orchestrator.exec()
    write_nodes(nodes, name, path=str(et_dir))

    print(f"Wrote inference ET to {et_dir} (npus={len(nodes)})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())