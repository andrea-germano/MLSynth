from Model.InferenceModel import InferenceModel
from Layer.TransformerInferenceLayer import TransformerInferenceLayer
from Evaluator.Evaluator import Evaluator
from Config.Config import ModelConfig

class TransformerInference(InferenceModel):
    def __init__(self, model_cfg: ModelConfig, evaluator: Evaluator, tp_size: int) -> None:
        self.cfg = model_cfg
        self.tp_size=tp_size

        #Identical to the formula for the number of parameters in the training model, since the inference model has the same architecture and we are not omitting any weight matrix
        self.num_params = 12 * self.cfg.num_layers * self.cfg.hidden_size * self.cfg.hidden_size * (1 + ((13)/(12*self.cfg.num_layers*self.cfg.hidden_size)) + ((self.cfg.vocab_size + self.cfg.sequence_length)/(12*self.cfg.num_layers*self.cfg.hidden_size)))

        self.layers = [
            TransformerInferenceLayer(
                hidden_size=self.cfg.hidden_size,
                bytes_per_val=self.cfg.bytes_per_val,
                tp_size=self.tp_size,
                evaluator=evaluator,
                scale=self.cfg.scale
            ) for _ in range(self.cfg.num_layers)
        ]

    def prefill(self, name, npu_id, layer, prompt_lens, pg_name = None):
        self._check_lens(prompt_lens)
        return self.layers[layer].prefill(name=name, pg_name=pg_name, prompt_lens=prompt_lens)

    def decode(self, name, npu_id, layer, kv_lens, pg_name = None):
        self._check_lens(kv_lens)
        return self.layers[layer].decode(name=name, pg_name=pg_name, kv_lens=kv_lens)
    
    def kv_bytes(self, layer: int, kv_len: int) -> int:
        """Total KV cache size, NOT SHARDED, for a given request on a single layer"""
        return 2*kv_len*self.cfg.hidden_size*self.cfg.bytes_per_val
    
    def pp_activation_bytes(self, num_tokens: int) -> int:
        """Total activation for inter PP communication, NOT SHARDED"""
        return num_tokens*self.cfg.hidden_size*self.cfg.bytes_per_val
    
    def get_num_layers(self) -> int:
        return self.cfg.num_layers
    
    def _check_lens(self, lens):
        #! Da vedere come implementare
        pass