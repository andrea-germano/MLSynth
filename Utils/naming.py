# Single source of truth for INFERENCE-side Chakra node names and comm tags.
import zlib

_ORDER = ("pl", # stands for pool --> prefill (p) or decode (d)
           "ss", # stands for scr pp stage
           "ds", # stands for dst pp stage
           "sh", # stands for tp shard id
           "ssh", # stands for src tp shard id (for kv, where src and dst differ)
           "dsh", # stands for dst tp shard id (for kv, where src and dst differ)
           "L", # stands for layer
           "seg", # stands for range of layers
           "op", # operation performed: attn/ffw/ecc
           "it" # stands for iteration within the decode steps
           )
_TAG_MOD = 500_000_000  # ASTRA-sim NATIVE tag range (fits int32)

_DEFAULT_PG = 3     # fallback for training edges
_PG_BY_CLASS = {
    # Highest priority to small numbers, available numbers are 1-7 (0 is reserved for ASTRA-sim internal use)
    # Collective communication have priority = 3 
    "KV": 4,      
    "FIRSTTOK": 5,  # handoff first token (gating decode): high priority
    "PP": 3,        # PP cross stage (gating next stage): medium priority
    "DECFB": 5,     # feedback decode
}

def pg_for_name(name: str) -> int:
    """Returns a deterministic priority group for a given node name"""
    return _PG_BY_CLASS.get(name.split("_", 1)[0], _DEFAULT_PG)

def _assemble(cls: str, fields: dict) -> str:
    pairs = "_".join(f"{k}={fields[k]}" for k in _ORDER if fields.get(k) is not None)
    return f"{cls}_{pairs}"

def comm_tag(name: str) -> int:
    """Deterministic tag from the node name. A SEND and its matching RECV carry the SAME name -> same tag"""
    return zlib.crc32(name.encode()) % _TAG_MOD

# --- compute graph inside a block (orchestrator -> layer) ------------------
def comp_base(*, pl, ss, sh, L, it) -> str:
    """Schedule position the orchestrator hands to the layer. NOT A FINAL NAME, but a base that the layer appends op + class via comp_name / coll_name."""
    f = dict(pl=pl, ss=ss, sh=sh, L=L, it=it)
    return "_".join(f"{k}={f[k]}" for k in _ORDER if f.get(k) is not None)

def comp_name(base: str, op: str) -> str:
    return f"COMP_{base}_op={op}"

def coll_name(base: str, op: str) -> str:
    return f"TP_{base}_op={op}"

# --- orchestrator point-to-point edges -------------------------------------
def pp_name(*, pl, src_stage, dst_stage, sh, it) -> str:
    return _assemble("PP", dict(pl=pl, ss=src_stage, ds=dst_stage, sh=sh, it=it))

def kv_name(*, src_stage, dst_stage, ssh, dsh, it, L=None, seg=None) -> str:
    return _assemble("KV", dict(ss=src_stage, ds=dst_stage, ssh=ssh, dsh=dsh, L=L, seg=seg, it=it))

def firsttok_name(*, src_stage, dst_stage, dsh, it) -> str:
    return _assemble("FIRSTTOK", dict(ss=src_stage, ds=dst_stage, dsh=dsh, it=it))

def decfb_name(*, pl, src_stage, dst_stage, sh, it) -> str:
    # Decode feedback edge. Similar to pp edge but goes back from dst_stage to src_stage, and carries the token just produced at dst_stage back to src_stage for the next decode iteration.
    return _assemble("DECFB", dict(pl=pl, ss=src_stage, ds=dst_stage, sh=sh, it=it))