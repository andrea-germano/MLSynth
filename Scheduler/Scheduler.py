from collections import deque
from typing import List, Optional

from Scheduler.Request import Request

class Scheduler:
    """Implements a simple scheduler. For each iteration:
    1. Choose a batch of request to prefill (sequence packing)
    2. Form a decode batch of the requests tahta are alreaddy prefilled
    3. Ask the orchestrator to emit the corresponding chakra nodes
    4. Update the state of the requests in the batch, and repeat
    """
    def __init__(self, orchestrator, max_decode_batch_size: int, max_prefill_batch_size: int, max_kv: int = 10**9):
        # For now the kv size budget is unlimited
        self.orchestrator = orchestrator
        self.max_decode_batch_size = max_decode_batch_size
        self.max_prefill_batch_size = max_prefill_batch_size
        self.max_kv = max_kv

        self.waiting: deque[Request] = deque() # requests that have not been prefilled yet
        self.decoding: List[Request] =[] 
        self.completed: List[Request] = [] # requests that have been fully processed

    def add_request(self, request: Request) -> None:
        """Adds a new request to the waiting queue."""
        self.waiting.append(request)
    
    def run(self) -> dict:
        iteration = 0
        while self.waiting or self.decoding:
            prefill_batch = self._select_prefill_batch()
            decode_batch = self._select_decode_batch()

            if not prefill_batch and not decode_batch:
                raise RuntimeError("No requests selected for prefill or decode, but there are still requests waiting or decoding. This should not happen and may indicate a bug in the scheduling logic.")
            
            self.orchestrator.step(iteration, prefill_requests=prefill_batch, decode_requests=decode_batch)
            self._update_state(prefill_batch, decode_batch)
            iteration += 1
            
        return self.orchestrator.finalize()
    
    def _select_prefill_batch(self) -> List[Request]:
        batch: List[Request] = []
        projected_kv = self._current_kv_tokens()
        while self.waiting and len(batch) < self.max_prefill_batch_size:
            candidate = self.waiting[0]
            if projected_kv + candidate.kv_len > self.max_kv:
                break
            batch.append(self.waiting.popleft())
            projected_kv += candidate.prompt_len
        return batch
    
    def _select_decode_batch(self) -> List[Request]:
        batch: List[Request] = []
        kv_used = 0
        for req in self.decoding:
            if len(batch) >= self.max_decode_batch_size:
                break
            needed_kv = req.kv_len + 1
            if kv_used + needed_kv > self.max_kv:
                break
            batch.append(req)
            kv_used += needed_kv
        return batch

    def _update_state(self, prefill_batch: List[Request], decode_batch: List[Request]) -> None:
        for req in prefill_batch:
            self.decoding.append(req)
        
        for req in decode_batch:
            req.advance()
            if req.is_completed:
                self.decoding.remove(req)
                self.completed.append(req)
        
        still_decoding = []
        for req in self.decoding:
            if req.is_completed:
                self.completed.append(req)
            else:
                still_decoding.append(req)
        self.decoding = still_decoding

    def _current_kv_tokens(self) -> int:
        return sum(req.kv_len for req in self.decoding)