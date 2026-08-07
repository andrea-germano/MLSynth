from typing import List

from Model.Model import BaseInferenceModel
from Layer.Layer import LayerEmission, MoeEpContext
from Layer.MoeInferenceLayer import MoeInferenceLayer
from Utils.config import ModelConfig, MoeRoutingConfig, ParallelismConfig
from Utils.routing import RoutingPlan


class MoeInferenceModel(BaseInferenceModel):
    """Mixture-of-Experts transformer model for inference"""

    def __init__(self, model_cfg: ModelConfig, routing: MoeRoutingConfig,
                 parallelism: ParallelismConfig = ParallelismConfig(),
                 plan: RoutingPlan | None = None):
        self._model_cfg = model_cfg
        self._routing = routing
        self._parallelism = parallelism
        self.plan = plan if plan is not None else RoutingPlan(model_cfg.moe, routing)
        self.layers = [
            MoeInferenceLayer(model_cfg=model_cfg, moe=model_cfg.moe, plan=self.plan,
                              layer_idx=idx, tp_size=parallelism.tp_size)
            for idx in range(model_cfg.num_layers)
        ]

    def with_parallelism(self, parallelism: ParallelismConfig) -> "MoeInferenceModel":
        return MoeInferenceModel(self._model_cfg, self._routing, parallelism, plan=self.plan)

    def prefill(self, name: str, npu_id: int, layer: int, prompt_lens: List[int],
                cached_lens: List[int], pg_name: str | None = None,
                ep_ctx: MoeEpContext | None = None, key: tuple | None = None,
                step: int = 0) -> LayerEmission:
        return self._layer_for(layer).prefill(name=name, pg_name=pg_name, prompt_lens=prompt_lens,
                                              cached_lens=cached_lens, ep_ctx=ep_ctx,
                                              key=key, step=step)

    def decode(self, name: str, npu_id: int, layer: int, kv_lens: List[int],
               pg_name: str | None = None, ep_ctx: MoeEpContext | None = None,
               key: tuple | None = None, step: int = 0) -> LayerEmission:
        return self._layer_for(layer).decode(name=name, pg_name=pg_name, kv_lens=kv_lens,
                                             ep_ctx=ep_ctx, key=key, step=step)

    def _layer_for(self, idx: int) -> MoeInferenceLayer:
        return self.layers[idx]

    def get_layers(self) -> list[MoeInferenceLayer]:
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
        router = L * d * self._model_cfg.moe.num_experts
        embedding = 2 * V * d  # embedding + lm head
        norms = (2 * L + 1) * d
        return float(layer_weights + router + embedding + norms)

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