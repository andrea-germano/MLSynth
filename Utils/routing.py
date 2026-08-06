from __future__ import annotations
import numpy as np
from Utils.config import MoeConfig

PHASE_FWD, PHASE_PREFILL, PHASE_DECODE = 0, 1, 2
_POP_NS = 9 

class RoutingPlan:
    """Pool-agnostic (NO ep_size in the constructor: prefill and decode share this object by
    identity but have different EP sizes; the group size is derived from len(origin_tokens))."""

    def __init__(self, moe: MoeConfig):
        self.moe = moe
        self._cache: dict = {}

    def is_uniform(self) -> bool:
        return self.moe.routing_distribution == "uniform"

    # Level 1: persistent per-layer expert popularity
    def popularity(self, layer: int) -> np.ndarray:
        """Per-expert probability vector, persistent for the whole run.
            - uniform: exactly 1/E everywhere.
            - dirichlet: one draw per layer with concentration alpha (small alpha = skewed). 
            The popular block rotates from layer to layer, so hotspots are temporally decorrelated
        """
        E = self.moe.num_experts
        if self.moe.routing_distribution == "uniform":
            return np.full(E, 1.0 / E)
        rng = np.random.default_rng([self.moe.routing_seed, _POP_NS, layer])
        return rng.dirichlet(self.moe.routing_alpha * np.ones(E))

    # Level 2: per-key token draws (stochastic skew)
    def traffic_matrix(self, key: tuple, layer: int, origin_tokens: list[int]) -> np.ndarray:
        """[ep, ep] int matrix of routed token copies; M[s][d] = copies sent by EP rank s to
        experts hosted on EP rank d (diagonal included: local copies count for compute, are
        never emitted as traffic). ep is derived from len(origin_tokens).."""
        ep = len(origin_tokens)
        cache_key = (key, ep, tuple(origin_tokens))
        if cache_key in self._cache:
            return self._cache[cache_key]

        E, k = self.moe.num_experts, self.moe.top_k
        p = self.popularity(layer)
        rng = np.random.default_rng([self.moe.routing_seed, *key])

        routed = np.zeros(E, dtype=np.int64)
        M = np.zeros((ep, ep), dtype=np.int64)
        g = 0  # global token index (deterministic order: ranks, then tokens within the rank)
        for s in range(ep):
            for _ in range(origin_tokens[s]):
                if self.is_uniform():
                    # deterministic round-robin: k consecutive experts mod E => exact uniformity,
                    # distinct by construction (k <= E validated in config)
                    experts = [(g * k + j) % E for j in range(k)]
                else:
                    experts = rng.choice(E, size=k, replace=False, p=p)
                for e in experts:
                    routed[e] += 1
                    M[s][int(e) * ep // E] += 1
                g += 1

        self._check_marginals(M, origin_tokens, routed, ep)
        self._cache[cache_key] = M
        return M

    def tokens_on_rank(self, key: tuple, layer: int, origin_tokens: list[int], dst: int) -> int:
        """Routed copies computed by EP rank dst = column sum, DIAGONAL INCLUDED
        (local tokens cost FLOPs even though they generate no traffic)."""
        M = self.traffic_matrix(key, layer, origin_tokens)
        return int(M[:, dst].sum())

    def _check_marginals(self, M, origin_tokens, routed, ep) -> None:
        k = self.moe.top_k
        E = self.moe.num_experts
        for s in range(ep):
            row = int(M[s, :].sum())
            if row != origin_tokens[s] * k:
                raise AssertionError(f"RoutingPlan: row {s} sum {row} != origin*top_k "
                                     f"{origin_tokens[s] * k} — internal invariant violated")
        block = E // ep
        for d in range(ep):
            col = int(M[:, d].sum())
            load = int(routed[d * block:(d + 1) * block].sum())
            if col != load:
                raise AssertionError(f"RoutingPlan: column {d} sum {col} != routed rank load {load}")