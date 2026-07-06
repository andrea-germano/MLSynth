from typing import List

from Model.InferenceModel import InferenceModel
from Layer.TransformerInferenceLayer import TransformerInferenceLayer
from Utils.parser import ModelConfig, ParallelismConfig
from Layer.InferenceLayer import LayerEmission

class TransformerInference(InferenceModel):
    def __init__(self, model_cfg: ModelConfig, parallelism: ParallelismConfig = ParallelismConfig()):
        self._model_cfg = model_cfg
        self._parallelism = parallelism
        self._layer = TransformerInferenceLayer(
            hidden_size=model_cfg.hidden_size,
            query_dim=model_cfg.query_dim,
            key_value_dim=model_cfg.key_value_dim,
            ffn_intermediate_size=model_cfg.ffn_intermediate_size,
            ffn_type=model_cfg.ffn_type,
            bytes_per_val=model_cfg.bytes_per_val,
            tp_size=parallelism.tp_size,
            scale=model_cfg.scale,
        )

    def with_parallelism(self, parallelism: ParallelismConfig) -> "TransformerInference":
        return TransformerInference(self._model_cfg, parallelism)

    def prefill(self, name: str, layer: int, prompt_lens: List[int], cached_lens: List[int], pg_name: str | None = None) -> LayerEmission:
        return self._layer_for(layer).prefill(name=name, pg_name=pg_name, prompt_lens=prompt_lens, cached_lens=cached_lens)

    def decode(self, name: str, layer: int, kv_lens: List[int], pg_name: str | None = None) -> LayerEmission:
        return self._layer_for(layer).decode(name=name, pg_name=pg_name, kv_lens=kv_lens)
    
    def _layer_for(self, idx: int) -> TransformerInferenceLayer:
        return self._layer # homogeneous: same instance for every layer
    
    @property
    def model_cfg(self) -> ModelConfig:
        return self._model_cfg

    @property
    def parallelism(self) -> ParallelismConfig:
        return self._parallelism

    @property
    def num_params(self) -> float:
        d, L, V = self._model_cfg.hidden_size, self._model_cfg.num_layers, self._model_cfg.vocab_size
        per_layer= self._layer.attn_weight_elems + self._layer.ffn_weight_elems
        embedding=2*V*d # embedding + lm head
        norms = (2*L+1)*d
        return float(L*per_layer + embedding + norms)

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