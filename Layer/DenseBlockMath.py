# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple
from Utils.config import ModelConfig

# With tensor parallelism, each TP rank does 1/tp of the FLOPs and holds 1/tp of the weights, but 2 all reduce operations
# are required per layer (after output projection and FFW down projection) to reassemble the partial sums.

#! It is supposed that a method like flash attention is used

class DenseBlockMath:
    """Single source of truth for the FLOP/byte cost model of one dense transformer block (attention + FFN), shared by the training and inference layers.
    """

    # FFN: classic (2 matrix multiplications) or SwiGLU (3 matrix multiplications)
    def __init__(self, *, hidden_size: int, query_dim: int | None = None, key_value_dim: int | None = None, ffn_intermediate_size: int | None = None, ffn_type: str = "classic",
                 bytes_per_val: int = 2, tp_size: int = 1, scale: float = 1.0, qk_norm: bool = False):
        self.hidden_size = hidden_size
        self.query_dim = query_dim if query_dim is not None else hidden_size
        self.key_value_dim = key_value_dim if key_value_dim is not None else hidden_size
        self.ffn_intermediate_size = ffn_intermediate_size if ffn_intermediate_size is not None else 4 * hidden_size
        self.ffn_type = ffn_type
        self.bytes_per_val = bytes_per_val
        self.tp_size = tp_size
        self.scale = scale
        self.qk_norm = qk_norm

        self.attn_weight_elems = (
            2 * hidden_size * self.query_dim       # Q proj + O proj
            + 2 * hidden_size * self.key_value_dim # K proj + V proj
        )
        num_ffn_matrices = 3 if ffn_type == "swiglu" else 2
        self.ffn_weight_elems = num_ffn_matrices * hidden_size * self.ffn_intermediate_size
        self.ffn_flops_per_token = 2 * num_ffn_matrices * hidden_size * self.ffn_intermediate_size
        self.act_elems_per_token = (3 if ffn_type == "swiglu" else 2) * self.ffn_intermediate_size

    @classmethod
    def from_model_cfg(cls, model_cfg: ModelConfig, tp_size: int) -> "DenseBlockMath":
        return cls(
            hidden_size=model_cfg.hidden_size,
            query_dim=model_cfg.query_dim,
            key_value_dim=model_cfg.key_value_dim,
            ffn_intermediate_size=model_cfg.ffn_intermediate_size,
            ffn_type=model_cfg.ffn_type,
            bytes_per_val=model_cfg.bytes_per_val,
            tp_size=tp_size,
            scale=model_cfg.scale,
            qk_norm=model_cfg.qk_norm,
        )

    @staticmethod
    def prefill_counts(prompt_lens: List[int], cached_lens: List[int]) -> Tuple[int, int, int]:
        """(new_tokens, cached_tokens, score_entries) for a prefill over a batch of requests."""
        new_tokens = sum(prompt_len - cached_len for prompt_len, cached_len in zip(prompt_lens, cached_lens))
        cached_tokens = sum(cached_lens)
        # Causal: the query at absolute position p attends to p+1 keys; summed over the suffix
        score_entries = sum((prompt_len * (prompt_len + 1)) // 2 - (cached_len * (cached_len + 1)) // 2 for prompt_len, cached_len in zip(prompt_lens, cached_lens))
        return new_tokens, cached_tokens, score_entries

    def attn_costs(self, query_tokens, kv_read_tokens, kv_write_tokens, score_entries) -> Tuple[int, int]:
        """(flops, bytes) of the attention block.
        query_tokens:    tokens projected through Q/K/V/O (prefill: new tokens; decode: batch size; training: microbatch tokens)
        kv_read_tokens:  tokens whose K,V are read from the cache (0 when nothing is cached)
        kv_write_tokens: tokens whose K,V are produced/written
        score_entries:   query-key pairs in QKᵀ / scores·V
        """
        b = self.bytes_per_val

        flops = int(self.scale * (
            2 * query_tokens * self.hidden_size * self.query_dim  # Q projection
            + 4 * query_tokens * self.hidden_size * self.key_value_dim # K+V projections
            + 2 * query_tokens * self.query_dim * self.hidden_size # O projections
            + 4 * score_entries * self.query_dim # QKᵀ + (scores · V)
        ) // self.tp_size)

        mem = int(self.scale * (
            self.attn_weight_elems * b // self.tp_size  # Q,K,V,O weights (sharded)
            + query_tokens * self.hidden_size * b  # input activations
            + 2 * kv_write_tokens * self.key_value_dim * b // self.tp_size  # KV written to cache
            + 2 * kv_read_tokens * self.key_value_dim * b // self.tp_size  # KV cache already present read
            + query_tokens * self.hidden_size * b  # output activations
            + 4 * query_tokens * self.hidden_size * b  # RMSNorm x2: read+write h, replicated (tp_stable)
            + (2 * query_tokens * (self.query_dim + self.key_value_dim) * b // self.tp_size if self.qk_norm else 0)
        ))
        return flops, mem

    def ffn_costs(self, tokens, weight_copies: int = 1) -> Tuple[int, int]:
        """(flops, bytes) of the FFN block for the given number of tokens.
        tokens: how many tokens are processed by this device
        weight_copies: how many FFN weight sets this device READS"""
        b = self.bytes_per_val
        flops = int(self.scale * (tokens * self.ffn_flops_per_token) // self.tp_size)
        mem = int(self.scale * (
            weight_copies * self.ffn_weight_elems * b // self.tp_size  # FFN weights (sharded)
            + 2 * tokens * self.hidden_size * b  # input + output activations
            + self.act_elems_per_token * tokens * b // self.tp_size
        ))
        return flops, mem

    def allreduce_bytes(self, tokens) -> int:
        """Size of each of the two TP all-reduces"""
        return int(self.scale * tokens * self.hidden_size * self.bytes_per_val)
