from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml

@dataclass(frozen=True)
class ModelConfig:
    """Architectural model description. A single instance is shared between prefill and decode in
      disaggregation since the underlying architecture is identical, only the parallelism strategy may differ"""
    name: str
    num_layers: int
    hidden_size: int
    sequence_len: int
    vocab_size: int
    bytes_per_val: int = 2
    scale: float = 1.0

@dataclass(frozen=True) 
class ParallelismConfig:
    """Tensor- and pipeline-parallel degrees for one pool. No DP in inference (it would only replicate the model)."""
    tp_size: int = 1
    pp_size: int = 1

    @property
    def num_npus(self) -> int:
        return self.pp_size * self.tp_size

@dataclass(frozen=True)
class Request:
    """A single inference request. `prompt_len` tokens are processed in prefill; `gen_len` tokens are produced autoregressively in decode."""
    prompt_len: int
    gen_len: int

@dataclass(frozen=True)
class KVTransferConfig:
    """Configuration for KV cache transfer between prefill and decode pools."""
    mode: str
    direction: str
    explicit_request: bool = False  # if True, pull-based transfer requires an explicit request from the decode side

    VALID_MODES={"bulk", "streaming"}
    VALID_DIRECTIONS={"push", "pull"}

@dataclass(frozen=True)
class InferenceConfig:
    requests: List[Request]
    kv_transfer: KVTransferConfig
    serialize_decode_iterations: bool = True

@dataclass(frozen=True)
class RunConfig:
    model: ModelConfig
    prefill: ParallelismConfig
    decode: ParallelismConfig
    inference: InferenceConfig

    @staticmethod
    def from_yaml(path: str| Path) -> RunConfig:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        _require(data, ("model","inference"), ctx="root")

        model = _build_model(data["model"])

        prefill, decode = _build_parallelism(data, model)
        inference = _build_inference(data["inference"], model)
        return RunConfig(model=model, prefill=prefill, decode=decode, inference=inference)
       
def _build_model(data: dict) -> ModelConfig:
    _require(data, ("name", "num_layers", "hidden_size", "sequence_len", "vocab_size", "bytes_per_val"), ctx="model")

    cfg= ModelConfig(
        name=str(data["name"]),
        num_layers=int(data["num_layers"]),
        hidden_size=int(data["hidden_size"]),
        sequence_len=int(data["sequence_len"]),
        vocab_size=int(data["vocab_size"]),
        bytes_per_val=int(data["bytes_per_val"]),
        scale=float(data.get("scale", 1.0)),
    )
    if min(cfg.num_layers, cfg.hidden_size, cfg.sequence_len, cfg.vocab_size, cfg.bytes_per_val) <= 0:
        raise ValueError("All model parameters must be positive")
    if cfg.bytes_per_val not in (1, 2, 4, 8):
        raise ValueError(f"bytes_per_val must be one of 1/2/4/8, got {cfg.bytes_per_val}")
    return cfg

def _build_parallelism(data: dict, model: ModelConfig) -> tuple[ParallelismConfig, ParallelismConfig]:
    def one(block: dict, label: str) -> ParallelismConfig:
        tp = int(block.get("tp_size", 1))
        pp = int(block.get("pp_size", 1))
        if int(block.get("dp_size", 1)) != 1:
            raise ValueError("DP is not supported in inference (it only replicates the model).")
        if tp < 1 or pp < 1:
            raise ValueError(f"{label}: parallelism sizes must be >= 1.")
        if model.num_layers % pp != 0:
            raise ValueError(f"{label}: num_layers ({model.num_layers}) not divisible by pp_size ({pp}).")
        if model.hidden_size % tp != 0:
            raise ValueError(f"{label}: hidden_size ({model.hidden_size}) not divisible by tp_size ({tp}).")
        return ParallelismConfig(tp_size=tp, pp_size=pp)

    if "prefill_parallelism" in data or "decode_parallelism" in data:
        _require(data, ("prefill_parallelism", "decode_parallelism"), ctx="root")
        prefill = one(data["prefill_parallelism"], "prefill")
        decode = one(data["decode_parallelism"], "decode")
    else:
        # single shared block
        block = data.get("parallelism", {})
        prefill = decode = one(block, "parallelism")
    
    #For kv resharding we need that the tp size must be divisible between prefill and decode (they can differ, but one must be a multiple of the other)
    tp_p, tp_d = prefill.tp_size, decode.tp_size
    if max(tp_p, tp_d) % min(tp_p, tp_d) != 0:
        raise ValueError(f"KV resharding requires that one TP size is a multiple of the other, got prefill.tp_size={tp_p} and decode.tp_size={tp_d}")
    return prefill, decode

def _build_inference(data: dict, model: ModelConfig) -> InferenceConfig:
    requests = _build_requests(data, model)
    kv_transfer = _build_kv_transfer(data.get("kv_transfer", {}))
    serialize = bool(data.get("serialize_decode_iterations", True))
    return InferenceConfig(requests=requests, kv_transfer=kv_transfer, serialize_decode_iterations=serialize)

def _build_requests(data: dict, model: ModelConfig) -> List[Request]:
    if "requests" in data:
        reqs = [Request(prompt_len=int(r["prompt_len"]), gen_len=int(r["gen_len"])) for r in data["requests"]]
    else:
        # homogeneous shorthand
        _require(data, ("num_requests", "prompt_len", "gen_len"), ctx="inference")
        n = int(data["num_requests"])
        p = int(data["prompt_len"])
        g = int(data["gen_len"])
        reqs = [Request(prompt_len=p, gen_len=g) for _ in range(n)]

    if not reqs:
        raise ValueError("inference: at least one request is required.")
    for i, r in enumerate(reqs):
        if r.prompt_len < 1 or r.gen_len < 1:
            raise ValueError(f"request[{i}]: prompt_len and gen_len must be >= 1.")
        if r.prompt_len + r.gen_len > model.sequence_len:
            raise ValueError(f"request[{i}]: prompt_len+gen_len ({r.prompt_len + r.gen_len}) exceeds model sequence_len ({model.sequence_len}).")
    return reqs

def _build_kv_transfer(data: dict) -> KVTransferConfig:
    mode = str(data.get("mode", "streaming")).lower()
    direction = str(data.get("direction", "push")).lower()
    explicit = bool(data.get("explicit_request", True))
    if mode not in KVTransferConfig.VALID_MODES:
        raise ValueError(f"kv_transfer.mode must be one of {KVTransferConfig.VALID_MODES}, got {mode!r}")
    if direction not in KVTransferConfig.VALID_DIRECTIONS:
        raise ValueError(f"kv_transfer.direction must be one of {KVTransferConfig.VALID_DIRECTIONS}, got {direction!r}")
    return KVTransferConfig(mode=mode, direction=direction, explicit_request=explicit)

def _require(data:dict, keys: tuple, ctx: str) -> None:
    missing = [k for k in keys if k not in data]
    if missing:
        raise ValueError(f"Missing required keys {missing} in {ctx} config")