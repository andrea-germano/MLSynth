import argparse
import json
import os
from pathlib import Path

import yaml
from chakra.src.third_party.utils.protolib import encodeMessage as encode_message

from Model.TransformerInference import TransformerInference
from Orchestrator.SingleDeviceInference import SingleDeviceInference
from Orchestrator.MegatronInference import MegatronInference

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
    par = cfg.get("parallelism", {})

    required_model = ["name", "num_layers", "sequence_len", "vocab_size",
                      "hidden_size", "batch_size", "bytes_per_val"]
    
    for k in required_model:
        if k not in m:
            raise ValueError(f"Missing required model field: {k}")
    
    if "prompt_len" not in inf or "num_generated_tokens" not in inf:
        raise ValueError("`inference` section must define `prompt_len` and `num_generated_tokens`.")
    
    total_seq = int(inf["prompt_len"]) + int(inf["num_generated_tokens"])
    if total_seq > int(m["sequence_len"]):
        raise ValueError(f"Total sequence length (prompt_len + num_generated_tokens = {total_seq}) cannot exceed model's max sequence length ({m['sequence_len']}).")
    
    pp_size = int(par.get("pp_size", 1))
    if int(m["num_layers"]) % pp_size != 0:
        raise ValueError(f"Number of layers ({m['num_layers']}) must be divisible by pipeline parallel size ({pp_size}).")
    
    if int(par.get("dp_size", 1)) != 1:
        raise ValueError(f"DP is not supported in inference (it would only replicate the model)")
    
def build_run_name(cfg: dict) -> str:
    m = cfg["model"]
    inf = cfg["inference"]
    par = cfg.get("parallelism", {})
    tp = int(par.get("tp_size", 1))
    pp = int(par.get("pp_size", 1))
    return (
        f"{m['name']}_inference_"
        f"P{inf['prompt_len']}_G{inf['num_generated_tokens']}_"
        f"B{m['batch_size']}_d{m['hidden_size']}_L{m['num_layers']}_"
        f"tp{tp}_pp{pp}_b{m['bytes_per_val']}"
    )

def main(argv = None) -> int:
    parser = argparse.ArgumentParser(description="Synthesize inference workload from YAML config")
    parser.add_argument("-c", "--config", default="input_inference.yaml", help="Path to inference YAML config file")
    args = parser.parse_args(argv)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg)

    name = build_run_name(cfg)
    out_dir = Path("output_inference") / name
    et_dir = out_dir / "et"
    et_dir.mkdir(parents=True, exist_ok=True)

    model = TransformerInference(cfg)
    orchestrator = MegatronInference(model, cfg)
    #orchestrator = SingleDeviceInference(model, cfg)

    comm_groups = orchestrator.generate_comm_groups()
    write_comm_groups(comm_groups, path=str(out_dir))

    nodes = orchestrator.exec()
    write_nodes(nodes, name, path=str(et_dir))

    print(f"Wrote inference ET to {et_dir}")
    print(f"npus : {len(nodes)} (tp={cfg.get('parallelism',{}).get('tp_size',1)}, pp={cfg.get('parallelism',{}).get('pp_size',1)})")
    print(f" comm groups: {list(comm_groups.keys()) or 'none'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())