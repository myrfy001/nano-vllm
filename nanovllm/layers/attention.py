import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


# @triton.jit
# def store_kvcache_kernel(
#     key_ptr,
#     key_stride,
#     value_ptr,
#     value_stride,
#     k_cache_ptr,
#     v_cache_ptr,
#     slot_mapping_ptr,
#     D: tl.constexpr,
# ):
#     idx = tl.program_id(0)
#     key_offsets = idx * key_stride + tl.arange(0, D)
#     value_offsets = idx * value_stride + tl.arange(0, D)
#     key = tl.load(key_ptr + key_offsets)
#     value = tl.load(value_ptr + value_offsets)
#     slot = tl.load(slot_mapping_ptr + idx)
#     cache_offsets = slot * D + tl.arange(0, D)
#     tl.store(k_cache_ptr + cache_offsets, key)
#     tl.store(v_cache_ptr + cache_offsets, value)


# def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
#     N, num_heads, head_dim = key.shape
#     D = num_heads * head_dim
#     assert key.stride(-1) == 1 and value.stride(-1) == 1
#     assert key.stride(1) == head_dim and value.stride(1) == head_dim
#     assert k_cache.stride(1) == D and v_cache.stride(1) == D
#     assert slot_mapping.numel() == N
#     store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


@triton.jit
def store_kvcache_kernel(
    kv_pe_ptr,
    kv_pe_stride,
    kv_pe_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
    head_dim: tl.constexpr,
):
    idx = tl.program_id(0)

    element_offset = tl.arange(0, D)
    mask = element_offset < head_dim

    kv_pe_offsets = idx * kv_pe_stride + element_offset
    kv_pe = tl.load(kv_pe_ptr + kv_pe_offsets, mask=mask)
    slot = tl.load(slot_mapping_ptr + idx)

    
    cache_offsets = slot * head_dim + element_offset
    
    tl.store(kv_pe_cache_ptr + cache_offsets, kv_pe, mask=mask)


def store_kvcache(kv_pe: torch.Tensor, kv_pe_cache: torch.Tensor, slot_mapping: torch.Tensor):
    # import pdb; pdb.set_trace()
    N, num_heads, head_dim = kv_pe.shape
    assert N == 1
    assert num_heads == 1 # for MLA, the kv cache is compressed
    D = 1024
    assert D >= head_dim
    assert kv_pe.stride(-1) == 1 
    assert kv_pe.stride(1) == head_dim
    assert kv_pe_cache.stride(1) == head_dim
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](kv_pe, kv_pe.stride(0), kv_pe_cache, slot_mapping, D, head_dim)

class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    # def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    #     o: torch.Tensor
    #     q = q.view(-1, self.num_heads, self.head_dim)
    #     k = k.view(-1, self.num_kv_heads, self.head_dim)
    #     v = v.view(-1, self.num_kv_heads, self.head_dim)
    #     context = get_context()
    #     k_cache = self.k_cache
    #     v_cache = self.v_cache
    #     store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
    #     if context.is_prefill:
    #         if context.block_tables is not None:    # prefix cache
    #             k, v = k_cache, v_cache
    #         o = flash_attn_varlen_func(q, k, v,
    #                                    max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
    #                                    max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
    #                                    softmax_scale=self.scale, causal=True, block_table=context.block_tables)
    #     else:    # decode
    #         o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
    #                                     cache_seqlens=context.context_lens, block_table=context.block_tables, 
    #                                     softmax_scale=self.scale, causal=True)
    #     o = o.view(-1, self.num_heads * self.head_dim)
    #     return o
