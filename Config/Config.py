from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class ModelConfig:
    name: str
    num_layers: int
    hidden_size: int
    sequence_length: int #? Is it useful anymore?
    vocab_size: int
    bytes_per_val: int
    scale: float

@dataclass
class HardwareConfig:
    peak_flops: float
    memory_bandwidth: float

@dataclass
class ParallelismConfig:
    tp_size: int
    pp_size: int

@dataclass
class ClusterConfig:
    model: ModelConfig
    hardware: HardwareConfig
    prefill: ParallelismConfig
    decode: ParallelismConfig

def load_config(path: str | Path) -> ClusterConfig:
    """Load the cluster configuration from a YAML file."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    
    _require(raw, ("model", "hardware", "prefill_parallelism", "decode_parallelism"), ctx="root")
    
    model = _build_model(raw["model"])
    hardware = _build_hardware(raw["hardware"])
    prefill = _build_parallelism(raw["prefill_parallelism"], label="prefill")
    decode = _build_parallelism(raw["decode_parallelism"], label="decode")

    _cross_validate(model, prefill, label="prefill")
    _cross_validate(model, decode, label="decode")

    return ClusterConfig(model=model,hardware=hardware,prefill=prefill,decode=decode)

def _build_model(raw: dict) -> ModelConfig:
    _require(raw, ("name", "num_layers", "hidden_size", "sequence_length", "vocab_size", "bytes_per_val"), ctx="model")
    cfg = ModelConfig(
        name=str(raw["name"]),
        num_layers=int(raw["num_layers"]),
        hidden_size=int(raw["hidden_size"]),
        sequence_length=int(raw["sequence_length"]),
        vocab_size=int(raw["vocab_size"]),
        bytes_per_val=int(raw["bytes_per_val"]),
        scale=float(raw.get("scale", 1.0))
    )
    if cfg.num_layers <= 0 or cfg.hidden_size <= 0 or cfg.sequence_length <= 0 or cfg.vocab_size <= 0 or cfg.bytes_per_val <= 0:
        raise ValueError(f"All model parameters must be positive integers")
    if cfg.bytes_per_val not in (1, 2, 4, 8):
        raise ValueError(f"bytes_per_val must be 1/2/4/8, got {cfg.bytes_per_val}")
    return cfg

def _build_hardware(raw: dict) -> HardwareConfig:
    _require(raw, ("peak_flops", "memory_bandwidth"), ctx="hardware")
    pf= float(raw["peak_flops"])
    mbw = float(raw["memory_bandwidth"])
    if pf <= 0 or mbw <= 0:
        raise ValueError(f"Hardware parameters must be positive numbers")
    return HardwareConfig(peak_flops=pf, memory_bandwidth=mbw)

def _build_parallelism(raw: dict, label: str) -> ParallelismConfig:
    _require(raw, ("tp_size", "pp_size"), ctx=f"{label}_parallelism")
    tp = int(raw["tp_size"])
    pp = int(raw["pp_size"])
    if tp < 1 or pp < 1:
        raise ValueError(f"Parallelism sizes must be positive integers")
    return ParallelismConfig(tp_size=tp, pp_size=pp)

def _cross_validate(m: ModelConfig, p: ParallelismConfig, label: str) -> None:
    if m.num_layers % p.pp_size != 0:
        raise ValueError(f"Number of layers {m.num_layers} is not divisible by {label} pipeline parallel size {p.pp_size}")
    if m.hidden_size % p.tp_size != 0:
        raise ValueError(f"Hidden size {m.hidden_size} is not divisible by {label} tensor parallel size {p.tp_size}")

def _require(d: dict, keys: tuple, ctx: str)-> None:
    """Helper function to check that required keys are present in a dictionary."""
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"Missing required keys in {ctx} config: {missing}")
