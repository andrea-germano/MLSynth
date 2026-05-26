from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml

@dataclass(frozen=True)
class ModelConfig:
    """Architectural model description. A ssingle instance is shared between prefill and decode in
      disaggregation since the underlying architecture is identical, only the parallelism strategy may differ"""
    name: str
    num_layers: int
    hidden_size: int
    sequence_len: int
    vocab_size: int
    batch_size: int
    bytes_per_val: int = 2
    scale: float = 1.0

    def num_params(self) -> int:
        d, L = self.hidden_size, self.num_layers
        V, S = self.vocab_size, self.sequence_len
        return int(12 * L * d * d * (1 + ((13)/(12*L*d)) + ((V + S)/(12*L*d)))) # Potrebbe servire poi quando dobbiamo modellare l'occupazione della memoria della GPU per lo swapping

@dataclass(frozen=True) 
class ParallelismConfig:
    """ Parallelism strategy description. In disaggregation, prefill and decode can have different parallelism configs, but they share the same underlying model architecture (and thus the same ModelConfig)"""
    pp_size: int = 1
    tp_size: int = 1

    def total_npus(self) -> int:
        return self.pp_size * self.tp_size

@dataclass(frozen=True)
class InferenceConfig:
    """Inference-specific configuration, such as prompt length and number of generated tokens. This is separate from the ModelConfig since it does not affect the architecture of the model, but only how it is executed."""
    prompt_len: int
    num_generated_tokens: int
    serialize_decode_iteration: bool = True #Used for serializing the decode loop when there is no continuos batching

@dataclass(frozen=True)
class Config:
    """Top-level config object that is deserialized from YAML. Contains the model configuration, parallelism configuration and inference configuration."""
    model: ModelConfig
    inference: InferenceConfig
    parallelism: Optional[ParallelismConfig] = None
    # for disaggregation, we can have separate parallelism configs for prefill and decode
    prefill_parallelism: Optional[ParallelismConfig] = None
    decode_parallelism: Optional[ParallelismConfig] = None

    @property
    def is_disaggregated(self) -> bool:
        return self.prefill_parallelism is not None and self.decode_parallelism is not None
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
            if not isinstance(raw, dict):
                raise ValueError("Config file must contain a top-level dictionary.")

            if "model" not in raw:
                raise ValueError("Config must contain a `model` section.")
            
            if "inference" not in raw:
                raise ValueError("Config must contain an `inference` section.")
            
            model = ModelConfig(**raw["model"])
            inference = InferenceConfig(**raw["inference"])

            if "disaggregation" in raw:
                disagg = raw["disaggregation"]
                if "prefill" not in disagg or "decode" not in disagg:
                    raise ValueError("`disaggregation` section must contain both `prefill` and `decode` subsections.")
                cfg = cls(model=model, inference=inference, 
                    prefill_parallelism=ParallelismConfig(**disagg.get("prefill", {})),
                    decode_parallelism=ParallelismConfig(**disagg.get("decode", {}))
                )

            else:
                if "parallelism" not in raw:
                    raise ValueError("Config must contain a `parallelism` section if `disaggregation` is not specified.")
                cfg = cls(model=model, inference=inference, 
                    parallelism=ParallelismConfig(**raw.get("parallelism", {}))
                )
            cfg._validate()
            return cfg
    
    def _validate(self) -> None:
        """Validate the config for internal consistency. For example, check that the number of layers is divisible by the pipeline parallel size."""
        total = self.inference.prompt_len + self.inference.num_generated_tokens
        if total > self.model.sequence_len:
            raise ValueError(f"Total sequence length (prompt_len + num_generated_tokens = {total}) cannot exceed model's max sequence length ({self.model.sequence_len}).")
        L = self.model.num_layers
        if self.is_disaggregated:
            pp_p = self.prefill_parallelism.pp_size
            pp_d = self.decode_parallelism.pp_size
            if L % pp_p != 0:
                raise ValueError(f"Number of layers ({L}) must be divisible by prefill pipeline parallel size ({pp_p}).")
            if L % pp_d != 0:
                raise ValueError(f"Number of layers ({L}) must be divisible by decode pipeline parallel size ({pp_d}).")
        else:
            pp = self.parallelism.pp_size
            if L % pp != 0:
                raise ValueError(f"Number of layers ({L}) must be divisible by pipeline parallel size ({pp}).")
        