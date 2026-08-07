from typing import List

from mlsynth.Model.Model import BaseInferenceModel
from mlsynth.Layer.Layer import LayerEmission
from Layer.InferenceLayer import InferenceLayer
from Utils.config import ModelConfig, ParallelismConfig

class InferenceModel(BaseInferenceModel):
    """Dense transformer model for inference. A single instance is shared between prefill and
    decode pools; `with_parallelism` derives per-pool views over the same ModelConfig."""

    def __init__(self, model_cfg: ModelConfig, parallelism: ParallelismConfig = ParallelismConfig()):
        self._model_cfg = model_cfg
        self._parallelism = parallelism
        # Model is composed of inference layers, one instance per layer
        self.layers = [
            InferenceLayer(
                hidden_size=model_cfg.hidden_size,
                query_dim=model_cfg.query_dim,
                key_value_dim=model_cfg.key_value_dim,
                ffn_intermediate_size=model_cfg.ffn_intermediate_size,
                ffn_type=model_cfg.ffn_type,
                bytes_per_val=model_cfg.bytes_per_val,
                tp_size=parallelism.tp_size,
                scale=model_cfg.scale,
            )
            for _ in range(model_cfg.num_layers)
        ]

    def with_parallelism(self, parallelism: ParallelismConfig) -> "InferenceModel":
        """Return a view of this model with a different parallelism config but the SAME (by identity) ModelConfig."""
        return InferenceModel(self._model_cfg, parallelism)

    def prefill(self, name: str, npu_id: int, layer: int, prompt_lens: List[int], cached_lens: List[int], pg_name: str | None = None) -> LayerEmission:
        return self._layer_for(layer).prefill(name=name, pg_name=pg_name, prompt_lens=prompt_lens, cached_lens=cached_lens)

    def decode(self, name: str, npu_id: int, layer: int, kv_lens: List[int], pg_name: str | None = None) -> LayerEmission:
        return self._layer_for(layer).decode(name=name, pg_name=pg_name, kv_lens=kv_lens)

    def _layer_for(self, idx: int) -> InferenceLayer:
        return self.layers[idx]

    def get_layers(self) -> list[InferenceLayer]:
        return self.layers

    @property
    def model_cfg(self) -> ModelConfig:
        return self._model_cfg

    @property
    def parallelism(self) -> ParallelismConfig:
        return self._parallelism

    @property
    def num_params(self) -> float:
        d, L, V = self._model_cfg.hidden_size, self._model_cfg.num_layers, self._model_cfg.vocab_size
        layer_weights = sum(layer.attn_weight_elems + layer.ffn_weight_elems for layer in self.layers)
        embedding=2*V*d # embedding + lm head
        norms = (2*L+1)*d
        return float(layer_weights + embedding + norms)

    def get_num_layers(self) -> int:
        return self._model_cfg.num_layers

    def get_hidden_size(self) -> int:
        return self._model_cfg.hidden_size

    def get_bytes_per_val(self) -> int:
        return self._model_cfg.bytes_per_val

    def get_scale(self) -> float:
        return self._model_cfg.scale

    def get_name(self) -> str:
        return self._model_cfg.name
