from Model.InferenceModel import InferenceModel
from Layer.TransformerInferenceLayer import TransformerInferenceLayer
from config import ModelConfig, ParallelismConfig

class TransformerInference(InferenceModel):
    def __init__(self, model_cfg: ModelConfig, parallelism_cfg: ParallelismConfig):
        self.cfg = model_cfg
        self.par = parallelism_cfg
        self.layers = [
            TransformerInferenceLayer(hidden_size=self.cfg.hidden_size, 
                bytes_per_val=self.cfg.bytes_per_val, 
                scale=self.cfg.scale, 
                tp_size=self.par.tp_size
            ) 
            for _ in range(self.cfg.num_layers)
        ]

    def prefill(self, name, npu_id, layer, num_batches, prompt_len, pg_name = None):
        return self.layers[layer].prefill(name=name, pg_name=pg_name, num_batches=num_batches, prompt_len=prompt_len)

    def decode(self, name, npu_id, layer, num_batches, kv_len, pg_name = None):
        return self.layers[layer].decode(name=name, pg_name=pg_name, num_batches=num_batches, kv_len=kv_len)
    
    def get_name(self) -> str: 
        return self.cfg.name
    
    def get_num_params(self) -> int: 
        return self.cfg.num_params()
    
    def get_num_layers(self) -> int: 
        return self.cfg.num_layers
    
    def get_hidden_size(self) -> int: 
        return self.cfg.hidden_size
    
    def get_sequence_len(self) -> int: 
        return self.cfg.sequence_len
    
    def get_vocab_size(self) -> int: 
        return self.cfg.vocab_size
    
    def get_batch_size(self) -> int: 
        return self.cfg.batch_size
    
    def get_bytes_per_val(self) -> int: 
        return self.cfg.bytes_per_val
    
    def get_scale(self) -> float: 
        return self.cfg.scale

    def get_tp_size(self) -> int:
        return self.par.tp_size
    
    def get_pp_size(self) -> int:
        return self.par.pp_size
    
    def get_layers(self): 
        return self.layers