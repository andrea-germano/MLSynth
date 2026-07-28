from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

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
    requests: list[Request]
    kv_transfer: str #bulk or streaming
    serialize_decode_iterations: bool = True

@dataclass(frozen=True)
class TrainingConfig:
    """Workload description for one training run (per-iteration shapes)."""
    batch_size: int
    sequence_len: int
    num_microbatches: int

@dataclass(frozen=True)
class TrainParallelismConfig:
    """Data-, pipeline- and tensor-parallel degrees for training."""
    dp_size: int = 1
    pp_size: int = 1
    tp_size: int = 1

    @property
    def num_npus(self) -> int:
        return self.dp_size * self.pp_size * self.tp_size

@dataclass(frozen=True)
class SlowdownSpec:
    type: str  # "constant" or "random" (normal distribution)
    value: float = 0.0  # constant
    mean: float = 0.0   # random
    std: float = 0.0    # random

@dataclass(frozen=True)
class WrapperCondition:
    """Selects the (npu, layer, phase) combinations a slowdown applies to. Unset fields match everything.

    npu_id/npu_id_range refer to the GLOBAL NPU id, with the same semantics in both modes:
    - training (MegatronLM): npu_id = dp_group*(pp_size*tp_size) + pp_stage*tp_size + tp_shard
    - inference (DisaggregatedInference): the prefill pool occupies ids [0, prefill_npus),
      the decode pool ids [prefill_npus, prefill_npus + decode_npus)
    layer_id/layer_id_range refer to the global layer index in both modes."""
    slowdown: SlowdownSpec
    npu_id: int | None = None
    npu_id_range: tuple[int, int] | None = None
    layer_id: int | None = None
    layer_id_range: tuple[int, int] | None = None
    phase: str | None = None  # forward | backward | prefill | decode; None = all phases

    def matches(self, npu_id: int, layer: int) -> bool:
        if self.npu_id is not None and self.npu_id != npu_id:
            return False
        elif self.npu_id_range is not None and (npu_id < self.npu_id_range[0] or npu_id > self.npu_id_range[1]):
            return False
        if self.layer_id is not None and self.layer_id != layer:
            return False
        elif self.layer_id_range is not None and (layer < self.layer_id_range[0] or layer > self.layer_id_range[1]):
            return False
        return True

    def applies_to(self, phase: str) -> bool:
        return self.phase is None or self.phase == phase

@dataclass(frozen=True)
class WrapperConfig:
    seed: int
    conditions: tuple[WrapperCondition, ...]
    type: str = "compute"

@dataclass(frozen=True)
class RunConfig:
    model: ModelConfig
    prefill: ParallelismConfig
    decode: ParallelismConfig
    inference: InferenceConfig
    wrapper: WrapperConfig | None = None

    @staticmethod
    def from_yaml(path: str | Path) -> RunConfig:
        with open(path, "r") as f:
            return RunConfig.from_data(yaml.safe_load(f))

    @staticmethod
    def from_data(data: dict) -> RunConfig:
        _require(data, ("model", "inference"), ctx="root")

        model = _build_model(data["model"])

        prefill, decode = _build_parallelism(data, model)
        inference = _build_inference(data["inference"])
        wrapper = _build_wrapper(data.get("wrapper"))
        return RunConfig(model=model, prefill=prefill, decode=decode, inference=inference, wrapper=wrapper)

@dataclass(frozen=True)
class TrainRunConfig:
    model: ModelConfig
    parallelism: TrainParallelismConfig
    training: TrainingConfig
    wrapper: WrapperConfig | None = None

    @staticmethod
    def from_yaml(path: str | Path) -> TrainRunConfig:
        with open(path, "r") as f:
            return TrainRunConfig.from_data(yaml.safe_load(f))

    @staticmethod
    def from_data(data: dict) -> TrainRunConfig:
        _require(data, ("model", "training", "parallelism"), ctx="root")

        model = _build_model(data["model"])
        parallelism = _build_train_parallelism(data["parallelism"], model)
        training = _build_training(data["training"], parallelism)
        wrapper = _build_wrapper(data.get("wrapper"))
        return TrainRunConfig(model=model, parallelism=parallelism, training=training, wrapper=wrapper)
       
def _build_model(data: dict) -> ModelConfig:
    _require(data, ("name", "num_layers", "hidden_size", "vocab_size", "bytes_per_val"), ctx="model")

    ffn_type = str(data.get("ffn_type", "classic")).lower()
    if ffn_type not in ("classic", "swiglu"):
        raise ValueError(f"ffn_type must be 'classic' or 'swiglu', got {ffn_type!r}")

    cfg = ModelConfig(
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
    if (cfg.num_kv_heads or cfg.head_dim) and not cfg.num_attention_heads:
        raise ValueError("num_kv_heads/head_dim require num_attention_heads to be set explicitly")
    return cfg

def load_config(path: str | Path):
    """Load a YAML config and dispatch on its mode: a top-level `inference` block yields a
    RunConfig, a top-level `training` block yields a TrainRunConfig."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    has_inference = "inference" in data
    has_training = "training" in data
    if has_inference and has_training:
        raise ValueError("Config must contain either an 'inference' or a 'training' block, not both.")
    if has_inference:
        return RunConfig.from_data(data)
    if has_training:
        return TrainRunConfig.from_data(data)
    raise ValueError("Config must contain either an 'inference' or a 'training' block.")

def _validate_model_parallelism(model: ModelConfig, tp: int, pp: int, label: str) -> None:
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

def _build_parallelism(data: dict, model: ModelConfig) -> tuple[ParallelismConfig, ParallelismConfig]:
    def one(block: dict, label: str) -> ParallelismConfig:
        tp = int(block.get("tp_size", 1))
        pp = int(block.get("pp_size", 1))
        if int(block.get("dp_size", 1)) != 1:
            raise ValueError("DP is not supported in inference (it only replicates the model).")
        _validate_model_parallelism(model, tp, pp, label)
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

def _build_requests(data: dict) -> list[Request]:
    has_explicit = "requests" in data
    has_shorthand = "num_requests" in data
    if has_explicit and has_shorthand:
        raise ValueError("inference: specify either 'requests' or the shorthand 'num_requests', not both.")
    if not has_explicit and not has_shorthand:
        raise ValueError("inference: either 'requests' or the shorthand 'num_requests' must be specified.")
    if has_explicit:
        entries = data["requests"]
        if not entries:
            raise ValueError("inference: 'requests' must contain at least one entry")
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

def _resolve_cached_len(r: dict, prompt_len: int, ctx: str) -> int:
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

def _build_train_parallelism(block: dict, model: ModelConfig) -> TrainParallelismConfig:
    dp = int(block.get("dp_size", 1))
    pp = int(block.get("pp_size", 1))
    tp = int(block.get("tp_size", 1))
    if dp < 1:
        raise ValueError("parallelism: dp_size must be >= 1.")
    _validate_model_parallelism(model, tp, pp, "parallelism")
    return TrainParallelismConfig(dp_size=dp, pp_size=pp, tp_size=tp)

def _build_training(data: dict, parallelism: TrainParallelismConfig) -> TrainingConfig:
    _require(data, ("batch_size", "sequence_len", "num_microbatches"), ctx="training")
    cfg = TrainingConfig(
        batch_size=int(data["batch_size"]),
        sequence_len=int(data["sequence_len"]),
        num_microbatches=int(data["num_microbatches"]),
    )
    if min(cfg.batch_size, cfg.sequence_len, cfg.num_microbatches) < 1:
        raise ValueError("training: batch_size, sequence_len and num_microbatches must be >= 1")
    if cfg.batch_size < parallelism.dp_size:
        raise ValueError(f"num batches (batch_size={cfg.batch_size}) must be greater than num dp groups (dp_size={parallelism.dp_size})!")
    if cfg.batch_size // parallelism.dp_size < cfg.num_microbatches:
        raise ValueError(f"num batches (batch_size={cfg.batch_size}) must be greater than num microbatches (num_microbatches={cfg.num_microbatches})!")
    return cfg

_VALID_PHASES = ("forward", "backward", "prefill", "decode")

def _build_wrapper(data: dict | None) -> WrapperConfig | None:
    if data is None:
        return None
    _require(data, ("seed", "conditions"), ctx="wrapper")
    wrapper_type = str(data.get("type", "compute")).lower()
    if wrapper_type != "compute":
        raise ValueError(f"wrapper.type must be 'compute', got {wrapper_type!r}")
    entries = data["conditions"]
    if not entries:
        raise ValueError("wrapper: 'conditions' must contain at least one entry")
    conditions = tuple(_build_wrapper_condition(f"wrapper.conditions[{i}]", entry) for i, entry in enumerate(entries))
    return WrapperConfig(seed=int(data["seed"]), conditions=conditions, type=wrapper_type)

def _build_wrapper_condition(ctx: str, data: dict) -> WrapperCondition:
    if "npu_id" in data and "npu_id_range" in data:
        raise ValueError(f"{ctx}: specify either npu_id or npu_id_range, not both.")
    if "layer_id" in data and "layer_id_range" in data:
        raise ValueError(f"{ctx}: specify either layer_id or layer_id_range, not both.")
    phase = data.get("pass")
    if phase is not None:
        phase = str(phase).lower()
        if phase not in _VALID_PHASES:
            raise ValueError(f"{ctx}: pass must be one of {list(_VALID_PHASES)}, got {phase!r}")
    return WrapperCondition(
        slowdown=_build_slowdown(ctx, data.get("slowdown")),
        npu_id=int(data["npu_id"]) if "npu_id" in data else None,
        npu_id_range=_build_id_range(ctx, "npu_id_range", data.get("npu_id_range")),
        layer_id=int(data["layer_id"]) if "layer_id" in data else None,
        layer_id_range=_build_id_range(ctx, "layer_id_range", data.get("layer_id_range")),
        phase=phase,
    )

def _build_id_range(ctx: str, key: str, value) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{ctx}: {key} must be a [lo, hi] pair")
    lo, hi = int(value[0]), int(value[1])
    if lo > hi:
        raise ValueError(f"{ctx}: {key} must satisfy lo <= hi, got [{lo}, {hi}]")
    return (lo, hi)

def _build_slowdown(ctx: str, data) -> SlowdownSpec:
    if not isinstance(data, dict):
        raise ValueError(f"{ctx}: 'slowdown' block is required")
    slowdown_type = str(data.get("type", "")).lower()
    if slowdown_type == "constant":
        _require(data, ("value",), ctx=f"{ctx}.slowdown")
        return SlowdownSpec(type=slowdown_type, value=float(data["value"]))
    if slowdown_type == "random":
        _require(data, ("mean", "std"), ctx=f"{ctx}.slowdown")
        return SlowdownSpec(type=slowdown_type, mean=float(data["mean"]), std=float(data["std"]))
    raise ValueError(f"{ctx}: slowdown.type must be 'constant' or 'random', got {slowdown_type!r}")

def _build_kv_transfer(value) -> str:
    VALID_KV_MODES = {"bulk", "streaming"}
    if not isinstance(value, str):
        raise ValueError(f"inference.kv_transfer must be a string, one of {sorted(VALID_KV_MODES)}")
    mode = value.lower()
    if mode not in VALID_KV_MODES:
        raise ValueError(f"inference.kv_transfer must be one of {sorted(VALID_KV_MODES)}, got {mode!r}")
    return mode

def _require(data: dict, keys: tuple, ctx: str) -> None:
    missing = [k for k in keys if k not in data]
    if missing:
        raise ValueError(f"Missing required keys {missing} in {ctx} config")