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
    vocab_size: int
    bytes_per_val: int = 2
    scale: float = 1.0
    
    num_attention_heads: int = 0 #default is q_dim/kv_dim = hidden_size
    num_kv_heads: int = 0 #set to num_attention heads for MHA, otherwise it's GQA
    head_dim: int = 0  #default is hidden_size//num_attention_heads
    intermediate_size: int = 0 #default is 4*hidden_size
    ffn_type: str = "classic" # for now only supported classic or swiglu

    @property
    def effective_head_dim(self) -> int:
        if self.head_dim:
            return self.head_dim
        return self.hidden_size // self.num_attention_heads if self.num_attention_heads else 0

    @property
    def query_dim(self) -> int:
        return self.num_attention_heads * self.effective_head_dim if self.num_attention_heads else self.hidden_size

    @property
    def key_value_dim(self) -> int:
        num_kv_heads = self.num_kv_heads or self.num_attention_heads
        return num_kv_heads * self.effective_head_dim if self.num_attention_heads else self.hidden_size

    @property
    def ffn_intermediate_size(self) -> int:
        return self.intermediate_size or 4 * self.hidden_size

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
    cached_len: int = 0 # number of tokens already present in the KV cache (for resuming a previous request)

@dataclass(frozen=True)
class InferenceConfig:
    requests: List[Request]
    kv_transfer: str #bulk or streaming
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
        inference = _build_inference(data["inference"])
        return RunConfig(model=model, prefill=prefill, decode=decode, inference=inference)
       
def _build_model(data: dict) -> ModelConfig:
    _require(data, ("name", "num_layers", "hidden_size", "vocab_size", "bytes_per_val"), ctx="model")

    ffn_type = str(data.get("ffn_type", "classic")).lower()
    if ffn_type not in ("classic", "swiglu"):
        raise ValueError(f"ffn_type must be 'classic' or 'swiglu', got {ffn_type!r}")

    cfg= ModelConfig(
        name=str(data["name"]),
        num_layers=int(data["num_layers"]),
        hidden_size=int(data["hidden_size"]),
        vocab_size=int(data["vocab_size"]),
        bytes_per_val=int(data["bytes_per_val"]),
        scale=float(data.get("scale", 1.0)),
        num_attention_heads=int(data.get("num_attention_heads", 0)),
        num_kv_heads=int(data.get("num_kv_heads", 0)),
        head_dim=int(data.get("head_dim", 0)),
        intermediate_size=int(data.get("intermediate_size", 0)),
        ffn_type=ffn_type,
    )
    if min(cfg.num_layers, cfg.hidden_size, cfg.vocab_size, cfg.bytes_per_val) <= 0:
        raise ValueError("All model parameters must be positive")
    if cfg.bytes_per_val not in (1, 2, 4, 8):
        raise ValueError(f"bytes_per_val must be one of 1/2/4/8, got {cfg.bytes_per_val}")
    if cfg.num_attention_heads and cfg.num_kv_heads and cfg.num_attention_heads % cfg.num_kv_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_kv_heads (GQA).")
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
        if model.num_kv_heads and max(tp, model.num_kv_heads) % min(tp, model.num_kv_heads) != 0:
            raise ValueError(f"{label}: tp_size ({tp}) and num_kv_heads ({model.num_kv_heads}) must be multiples of each other (GQA).")
        if model.num_kv_heads and tp > model.num_kv_heads:
            raise ValueError(f"{label}: tp_size ({tp}) cannot exceed num_kv_heads in the current configuration ({model.num_kv_heads}).")   
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

def _build_inference(data: dict) -> InferenceConfig:
    requests = _build_requests(data)
    kv_transfer = _build_kv_transfer(data.get("kv_transfer", "streaming"))
    serialize = bool(data.get("serialize_decode_iterations", True))
    return InferenceConfig(requests=requests, kv_transfer=kv_transfer, serialize_decode_iterations=serialize)

def _build_requests(data: dict) -> List[Request]:
    has_explicit = "requests" in data
    has_shortand = "num_requests" in data
    if has_explicit and has_shortand:
        raise ValueError("inference: specify either 'requests' or the shorthand 'num_requests', not both.")
    if not has_explicit and not has_shortand:
        raise ValueError("inference: either 'requests' or the shorthand 'num_requests' must be specified.")
    if has_explicit:
        entries = data["requests"]
        if not entries:
            raise ValueError("inference: 'requests' must contaain at least one entry")
        return [_build_request(f"request[{i}]", entry) for i, entry in enumerate(entries)]
    
    _require(data, ("num_requests", "prompt_len", "gen_len"), ctx="inference")
    n = int(data["num_requests"])
    if n < 1:
        raise ValueError("inference: num_requests must be >= 1")
    template = _build_request("inference", data)
    return [template for _ in range(n)]

def _build_request(ctx: str, data: dict) -> Request:
    prompt_len = int(data["prompt_len"])
    gen_len = int(data["gen_len"])
    if prompt_len < 1 or gen_len < 1:
        raise ValueError(f"{ctx}: prompt_len and gen_len must be >= 1")
    cached_len = _resolve_cached_len(data, prompt_len, ctx)
    return Request(prompt_len=prompt_len, gen_len=gen_len, cached_len=cached_len)

def _resolve_cached_len(r:dict, prompt_len:int, ctx:str) -> int:
    has_len = "cached_len" in r
    has_frac = "cached_frac" in r
    if has_len and has_frac:
        raise ValueError(f"{ctx}: specify either cached_len or cached_frac, not both.")
    if has_frac:
        frac = float(r["cached_frac"])
        if not (0 <= frac < 1):
            raise ValueError(f"{ctx}: cached_frac must be in [0,1), got {frac}")
        cached_len = int(frac * prompt_len)
    else:
        cached_len = int(r.get("cached_len", 0))
    if not (0 <= cached_len < prompt_len):
        raise ValueError(f"{ctx}: cached_len must be in [0, prompt_len), got {cached_len} for prompt_len={prompt_len}")
    return cached_len

def _build_kv_transfer(value) -> str:
    VALID_KV_MODES = {"bulk", "streaming"}
    if not isinstance(value, str):
        raise ValueError(f"inference.kv_transfer must be a string, one of {sorted(VALID_KV_MODES)}")
    mode = value.lower()
    if mode not in VALID_KV_MODES:
        raise ValueError(f"inference.kv_transfer must be one of {sorted(VALID_KV_MODES)}, got {mode!r}")
    return mode

def _require(data:dict, keys: tuple, ctx: str) -> None:
    missing = [k for k in keys if k not in data]
    if missing:
        raise ValueError(f"Missing required keys {missing} in {ctx} config")