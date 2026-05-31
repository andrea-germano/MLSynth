from dataclasses import dataclass

@dataclass
class Request:
    """Represents a single inference request, containing all the information needed by the scheduler to make informed decisions about how to schedule it."""
    request_id: int
    prompt_len: int
    tokens_to_generate: int
    arrival_time: float = 0.0
    tokens_generated: int = 0

    @property
    def kv_len(self) -> int:
        """Returns the current length of the KV cache for this request, which is equal to the prompt length plus the number of tokens generated so far."""
        return self.prompt_len + self.tokens_generated
    
    @property
    def is_completed(self) -> bool:
        """Returns True if the request has finished generating all its tokens."""
        return self.tokens_generated >= self.tokens_to_generate
    
    def advance(self)-> None:
        """Advances the request by one token, incrementing the number of tokens generated."""
        if not self.is_completed:
            self.tokens_generated += 1