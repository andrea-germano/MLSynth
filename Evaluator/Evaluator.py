from abc import ABC, abstractmethod

class Evaluator(ABC):
    """Converts (FLOPs, tensor dimensions) to runtime in microseconds"""

    @abstractmethod
    def evaluate(self, flops: int, tensor_size: int) -> int:
        """Evaluate the runtime in microseconds for a given number of FLOPs and tensor size."""
        raise NotImplementedError
    
class RooflineEvaluator(Evaluator):
    def __init__(self, peak_flops: float, memory_bandwidth: float):
        if peak_flops <= 0:
            raise ValueError("Peak FLOPs must be greater than zero.")
        if memory_bandwidth <= 0:
            raise ValueError("Memory bandwidth must be greater than zero.")
        self.peak_flops = peak_flops
        self.memory_bandwidth = memory_bandwidth
    
    def evaluate(self, flops: int, tensor_size: int) -> int:
        t_compute = flops / self.peak_flops
        t_memory = tensor_size / self.memory_bandwidth
        return int(max(t_compute, t_memory) * 1e6)  # Convert seconds to microseconds