from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()

_TP = None
_TP_RANK = None
_TP_WORLD_SIZE = None

def set_tp_context_info(tp_group, tp_rank, tp_world_size):
    global _TP, _TP_RANK, _TP_WORLD_SIZE
    _TP = tp_group
    _TP_RANK = tp_rank
    _TP_WORLD_SIZE = tp_world_size

def get_tp_group():
    return _TP

def get_tp_rank():
    return _TP_RANK

def get_tp_world_size():
    return _TP_WORLD_SIZE
