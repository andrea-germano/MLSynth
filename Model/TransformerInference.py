from Model.InferenceModel import InferenceModel
from Layer.TransformerInferenceLayer import TransformerInferenceLayer

class TransformerInference(InferenceModel):
    def __init__(self, config):
        m = config["model"]
        self.name = m["name"]
        self.num_layers = int(m["num_layers"])
        self.hidden_size = int(m["hidden_size"])
        self.sequence_len = int(m["sequence_len"])
        self.vocab_size = int(m["vocab_size"])
        self.batch_size = int(m["batch_size"])
        self.bytes_per_val = int(m["bytes_per_val"])
        self.scale = float(m.get("scale", 1.0))

        #parallelism degree (TP and PP)
        par=config.get("parallelism", {})
        self.tp_size = int(par.get("tp_size", 1))
        self.pp_size = int(par.get("pp_size", 1))

        #Identical to the formula for the number of parameters in the training model, since the inference model has the same architecture and we are not omitting any weight matrix
        self.num_params = 12 * self.num_layers * self.hidden_size * self.hidden_size * (1 + ((13)/(12*self.num_layers*self.hidden_size)) + ((self.vocab_size + self.sequence_len)/(12*self.num_layers*self.hidden_size)))

        self.layers = [
            TransformerInferenceLayer(hidden_size=self.hidden_size, 
                bytes_per_val=self.bytes_per_val, 
                scale=self.scale, 
                tp_size=self.tp_size
            ) 
            for _ in range(self.num_layers)
        ]

    def prefill(self, name, npu_id, layer, num_batches, prompt_len, pg_name = None):
        if prompt_len > self.sequence_len:
            raise ValueError(f"prompt_len={prompt_len} exceeds model sequence_len={self.sequence_len}")
        return self.layers[layer].prefill(name=name, pg_name=pg_name, num_batches=num_batches, prompt_len=prompt_len)

    def decode(self, name, npu_id, layer, num_batches, kv_len, pg_name = None):
        if kv_len > self.sequence_len:
            raise ValueError(f"kv_len={kv_len} exceeds model sequence_len={self.sequence_len}")
        return self.layers[layer].decode(name=name, pg_name=pg_name, num_batches=num_batches, kv_len=kv_len)
    
    def get_name(self) -> str: 
        return self.name
    
    def get_num_params(self) -> int: 
        return self.num_params
    
    def get_num_layers(self) -> int: 
        return self.num_layers
    
    def get_hidden_size(self) -> int: 
        return self.hidden_size
    
    def get_sequence_len(self) -> int: 
        return self.sequence_len
    
    def get_vocab_size(self) -> int: 
        return self.vocab_size
    
    def get_batch_size(self) -> int: 
        return self.batch_size
    
    def get_bytes_per_val(self) -> int: 
        return self.bytes_per_val
    
    def get_scale(self) -> float: 
        return self.scale

    def get_tp_size(self) -> int:
        return self.tp_size
    
    def get_pp_size(self) -> int:
        return self.pp_size
    
    def get_layers(self): 
        return self.layers