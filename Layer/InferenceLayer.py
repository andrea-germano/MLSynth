from typing import List
from chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode

from Layer.Interfaces import BaseInferenceLayer, LayerEmission
from Layer.DenseBlockMath import DenseBlockMath
from Utils.nodes import compute, allreduce
from Utils.naming import comp_name, coll_name


class InferenceLayer(BaseInferenceLayer):
    """A single dense inference block, accounting for FLOPs and activations of a single
    transformer layer, for both prefill and decode phases.
    The cost model lives in DenseBlockMath (shared with the training layer)."""

    def __init__(self, hidden_size: int, query_dim: int | None = None,
                key_value_dim: int | None = None,
                ffn_intermediate_size: int | None = None,
                ffn_type: str = "classic",
                bytes_per_val: int = 2, tp_size: int = 1, scale: float = 1.0):
        self.math = DenseBlockMath(
            hidden_size=hidden_size,
            query_dim=query_dim,
            key_value_dim=key_value_dim,
            ffn_intermediate_size=ffn_intermediate_size,
            ffn_type=ffn_type,
            bytes_per_val=bytes_per_val,
            tp_size=tp_size,
            scale=scale,
        )
        self.hidden_size = hidden_size
        self.bytes_per_val = bytes_per_val
        self.tp_size = tp_size
        self.scale = scale

    @property
    def attn_weight_elems(self) -> int:
        return self.math.attn_weight_elems

    @property
    def ffn_weight_elems(self) -> int:
        return self.math.ffn_weight_elems

    def prefill(self, name: str, pg_name: str | None, prompt_lens: List[int], cached_lens: List[int]) -> LayerEmission:
        new_tokens, cached_tokens, score_entries = DenseBlockMath.prefill_counts(prompt_lens, cached_lens)

        attn_flops, attn_bytes = self.math.attn_costs(
            query_tokens=new_tokens, kv_read_tokens=cached_tokens,
            kv_write_tokens=new_tokens, score_entries=score_entries)
        ffn_flops, ffn_bytes = self.math.ffn_costs(new_tokens)

        return self._emit(name, pg_name, attn_flops, attn_bytes, ffn_flops, ffn_bytes,
                          allreduce_tokens=new_tokens)

    def decode(self, name: str, pg_name: str | None, kv_lens: List[int]) -> LayerEmission:
        """Emit a single decode step. `kv_lens[i]` is the length of request i's KV cache
        *including* the token produced in this step."""
        batch_size = len(kv_lens)
        total_kv_tokens = sum(kv_lens)

        attn_flops, attn_bytes = self.math.attn_costs(
            query_tokens=batch_size, kv_read_tokens=total_kv_tokens,
            kv_write_tokens=batch_size, score_entries=total_kv_tokens)
        ffn_flops, ffn_bytes = self.math.ffn_costs(batch_size)

        return self._emit(name, pg_name, attn_flops, attn_bytes, ffn_flops, ffn_bytes, allreduce_tokens=batch_size)


    def _emit(self, name: str, pg_name: str | None, attn_flops: int, attn_bytes: int, ffn_flops: int, ffn_bytes: int, allreduce_tokens: int) -> LayerEmission:
        allreduce_bytes = self.math.allreduce_bytes(allreduce_tokens)

        nodes: List[ChakraNode] = []

        attn = compute(attn_flops, attn_bytes, name=comp_name(name, "attn"))
        nodes.append(attn)
        attn_end = attn

        if self.tp_size > 1:
            attn_ar = allreduce(allreduce_bytes, pg_name=pg_name, parents=[attn], name=coll_name(name, "attn"))
            nodes.append(attn_ar)
            attn_end = attn_ar

        kv_ready = attn_end

        ffw = compute(ffn_flops, ffn_bytes, parents=[attn_end], name=comp_name(name, "ffw"))
        nodes.append(ffw)
        tail = ffw

        if self.tp_size > 1:
            ffn_ar = allreduce(allreduce_bytes, pg_name=pg_name, parents=[ffw], name=coll_name(name, "ffw"))
            nodes.append(ffn_ar)
            tail = ffn_ar

        return LayerEmission(nodes=nodes, tail=tail, kv_ready=kv_ready)
