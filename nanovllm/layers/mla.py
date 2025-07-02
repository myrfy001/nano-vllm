import torch
import triton
import triton.language as tl

from typing import Any, Dict, Optional, Tuple
from torch.profiler import ProfilerActivity

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_H": h, "BLOCK_SIZE_N": n, "BLOCK_SIZE_D1": d1, "BLOCK_SIZE_D2": d2}, num_warps=4, num_stages=1)
        for h in [1]
        # for h in [1, 16]
        for n in [1]
        # for n in [16, 32, 64]
        for d1 in [512]
        # for d1 in [16, 32, 64, 128, 256, 512]
        for d2 in [64]
        # for d2 in [16, 32, 64]
    ],
    key=['seq_len_kv', 'split_k']
)
@triton.jit
def mla_decode_split(
    seq_len_kv,
    split_k,
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    Req_to_tokens,
    B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b,
    stride_qbs, stride_qh,
    stride_buf_kbs, stride_buf_kh,
    stride_buf_vbs, stride_buf_vh,
    stride_mid_ob, stride_mid_oh, stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D1: tl.constexpr,
    BLOCK_SIZE_D2: tl.constexpr,
    SPLIT_K: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_SIZE_H)
    split_kv_id = tl.program_id(2)

    if kv_group_num > BLOCK_SIZE_H:
        VALID_BLOCK_H: tl.constexpr = BLOCK_SIZE_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_SIZE_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    # offs_d = tl.arange(0, BLOCK_DMODEL)
    # mask_d = offs_d < Lk

    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dv = offs_dv < Lv
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = cur_batch

    # offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[ None, :]
    # q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)

    # if BLOCK_DPE > 0:
    #     offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
    #     mask_dpe = offs_dpe < Lk
    #     off_qpe = (cur_batch * stride_qbs + cur_head[:, None] * stride_qh +
    #                offs_dpe[None, :])
    #     qpe = tl.load(Q + off_qpe,
    #                   mask=(mask_h[:, None]) & (mask_dpe[None, :]),
    #                   other=0.0)

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, SPLIT_K)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split,
                              cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_SIZE_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_SIZE_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_SIZE_H, BLOCK_DV], dtype=tl.float32)

    NUM_DIM_SPLIT_NOPE: tl.constexpr = BLOCK_DMODEL // BLOCK_SIZE_D1
    NUM_DIM_SPLIT_PE: tl.constexpr = BLOCK_DPE // BLOCK_SIZE_D2
    if split_kv_end > split_kv_start:
        for start_n in range(split_kv_start, split_kv_end, BLOCK_SIZE_N):
            offs_n = start_n + tl.arange(0, BLOCK_SIZE_N)
            kv_page_number = tl.load(
                Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n // PAGE_SIZE,
                mask=offs_n < split_kv_end,
                other=0,
            )
            kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE

            if BLOCK_SIZE_H == 1 and BLOCK_SIZE_D1 != BLOCK_DMODEL:
                qk1 = tl.zeros([BLOCK_SIZE_D1, BLOCK_SIZE_N], dtype=tl.float32)
            else:
                qk = tl.zeros([BLOCK_SIZE_H, BLOCK_SIZE_N], dtype=tl.float32)

            for d in range(0, NUM_DIM_SPLIT_NOPE):
                offs_d = d * BLOCK_SIZE_D1 + tl.arange(0, BLOCK_SIZE_D1)
                mask_d = offs_d < Lk

                offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[ None, :]
                q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)

                offs_buf_k = (kv_loc[None, :] * stride_buf_kbs + cur_kv_head * stride_buf_kh + offs_d[:, None])
                k = tl.load(K_Buffer + offs_buf_k, mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]), other=0.0)

                if BLOCK_SIZE_H == 1:
                    if BLOCK_SIZE_D1 != BLOCK_DMODEL:
                        qk1 += tl.trans(q) * k.to(q.dtype)
                    else:
                        qk += tl.sum(tl.trans(q) * k.to(q.dtype), axis=0, keep_dims=True)
                else:
                    qk += tl.dot(q, k.to(q.dtype))

            if BLOCK_SIZE_H == 1 and BLOCK_SIZE_D1 != BLOCK_DMODEL:
                qk = tl.sum(qk1, axis=0, keep_dims=True)

            if BLOCK_DPE > 0:
                if BLOCK_SIZE_H == 1 and BLOCK_SIZE_D2 != BLOCK_DPE:
                    qk2 = tl.zeros([BLOCK_SIZE_D2, BLOCK_SIZE_N], dtype=tl.float32)

                for d in range(0, NUM_DIM_SPLIT_PE):
                    offs_dpe = BLOCK_DMODEL + d * BLOCK_SIZE_D2 + tl.arange(0, BLOCK_SIZE_D2)
                    mask_dpe = offs_dpe < Lk

                    off_qpe = (cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :])
                    qpe = tl.load(Q + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0)
                    offs_buf_kpe = (kv_loc[None, :] * stride_buf_kbs + cur_kv_head * stride_buf_kh + offs_dpe[:, None])
                    kpe = tl.load(
                        K_Buffer + offs_buf_kpe,
                        mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]),
                        other=0.0,
                    )
                    if BLOCK_SIZE_H == 1:
                        if BLOCK_SIZE_D2 != BLOCK_DPE:
                            qk2 += tl.trans(qpe) * kpe.to(qpe.dtype)
                        else:
                            qk += tl.sum(tl.trans(qpe) * kpe.to(qpe.dtype), axis=0, keep_dims=True)
                    else:
                        qk += tl.dot(qpe, kpe.to(qpe.dtype))

                if BLOCK_SIZE_H == 1 and BLOCK_SIZE_D2 != BLOCK_DPE:
                    qk += tl.sum(qk2, axis=0, keep_dims=True)

            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf"))

            offs_buf_v = (kv_loc[:, None] * stride_buf_vbs +
                          cur_kv_head * stride_buf_vh + offs_dv[None, :])
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            if BLOCK_SIZE_H == 1:
                acc += tl.sum(tl.trans(p.to(v.dtype)) * v, axis=0, keep_dims=True)
            else:
                acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (cur_batch * stride_mid_ob +
                      cur_head[:, None] * stride_mid_oh +
                      split_kv_id * stride_mid_os + offs_dv[None, :])

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        offs_mid_o_1 = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh +
                        split_kv_id * stride_mid_os + Lv)

        tl.store(
            Att_Out + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


@triton.jit
def mla_decode_combine(
    Mid_O,
    o,
    B_Seqlen,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    SPLIT_K: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + Lv

    for split_kv_id in range(0, SPLIT_K):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, SPLIT_K)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split,
                                  cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(Mid_O + offs_v + split_kv_id * stride_mid_os,
                         mask=mask_d,
                         other=0.0)
            tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        o + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )


def mla_decode(
    q: torch.Tensor,            # [B, 128, (512 + 64)]
    kv_c_and_k_pe_cache: torch.Tensor,     # [Cache_Size // PAGE_SIZE, PAGE_SIZE, 1, 576]
    kv_c_cache: torch.Tensor,     # [Cache_Size // PAGE_SIZE, PAGE_SIZE, 1, 512]
    o: torch.Tensor,            # [B, 128, 512]
    req_to_tokens: torch.Tensor,
    b_seq_len: torch.Tensor,
    config: Dict[str, Any],
):
    _, page_size, num_kv_heads, qk_head_dim = kv_c_and_k_pe_cache.shape
    batch_size, num_q_heads, kv_lora_rank = o.shape
    qk_rope_head_dim = qk_head_dim - kv_lora_rank

    assert q.dtype == kv_c_and_k_pe_cache.dtype == kv_c_cache.dtype == torch.float16
    assert q.shape[1] == o.shape[1] == num_q_heads
    assert kv_c_and_k_pe_cache.shape[1] == kv_c_cache.shape[1] == page_size == 1, "Testing, Should Remove later"
    assert kv_c_and_k_pe_cache.shape[2] == kv_c_cache.shape[2] == num_kv_heads == 1, "number of mla kv head should be 1"
    assert kv_c_cache.shape[3] == o.shape[2] == kv_lora_rank
    assert q.shape[2] == qk_head_dim, f'{q.shape=}, {qk_head_dim=}, {kv_lora_rank=}, {qk_rope_head_dim=}'

    kv_group_num = num_q_heads // num_kv_heads

    # args
    BLOCK_DMODEL = kv_lora_rank
    BLOCK_DPE = qk_rope_head_dim
    BLOCK_DV = kv_lora_rank
    Lv = kv_lora_rank

    Lk = qk_head_dim

    # STAGE 1: split phase
    SPLIT_K = config['SPLIT_K']
    sm_scale = 1.0 / (qk_head_dim ** 0.5)

    # print("out", out.shape)
    Mid_O = torch.empty((batch_size, num_q_heads, SPLIT_K, kv_lora_rank + 1), device=q.device, dtype=torch.float32)
    # Logsumexp = torch.empty((batch_size, num_heads, SPLIT_K), device=q.device, dtype=torch.float32)

    grid = lambda META: (batch_size, triton.cdiv(num_q_heads, META['BLOCK_SIZE_H']), SPLIT_K)
    mla_decode_split[grid](
        req_to_tokens.shape[1],
        SPLIT_K,
        q,
        kv_c_and_k_pe_cache,
        kv_c_cache,
        sm_scale,
        req_to_tokens,
        b_seq_len,
        Mid_O,
        req_to_tokens.stride(0),
        q.stride(0), q.stride(1),
        kv_c_and_k_pe_cache.stride(1), kv_c_and_k_pe_cache.stride(2),
        kv_c_cache.stride(1), kv_c_cache.stride(2),
        Mid_O.stride(0), Mid_O.stride(1), Mid_O.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=num_q_heads,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        PAGE_SIZE=page_size,
        logit_cap=0.0,
        Lk=Lk,
        Lv=Lv,
        **config,
    )
    # print("MidO:", MidO)
    # print("L:", Logsumexp)

    # STAGE 2: combine phase
    grid = (batch_size, num_q_heads)
    mla_decode_combine[grid](
        Mid_O, o, b_seq_len,
        Mid_O.stride(0), Mid_O.stride(1), Mid_O.stride(2),
        o.stride(0), o.stride(1),
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        SPLIT_K=SPLIT_K,
        num_warps=4,
        num_stages=1,
    )


def generate_tokens(B: int, Skv: int, CACHE_SIZE = 16384, PAGE_SIZE = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    assert CACHE_SIZE % PAGE_SIZE == 0

    num_pages_per_batch = triton.cdiv(Skv, PAGE_SIZE)
    req_to_pages = torch.randint(0, CACHE_SIZE // PAGE_SIZE, (B, num_pages_per_batch, 1), device="cuda")
    req_to_tokens = req_to_pages * PAGE_SIZE
    req_to_tokens = req_to_tokens.expand(B, num_pages_per_batch, PAGE_SIZE)
    req_to_tokens = req_to_tokens + torch.arange(PAGE_SIZE, device="cuda").view(1, 1, -1)
    req_to_tokens = req_to_tokens.view(B, -1)
    req_to_tokens = req_to_tokens[:, :Skv].contiguous()

    b_seq_len = torch.full((B, ), Skv, device="cuda")

    return req_to_tokens, b_seq_len


def torch_ref(q, kv_c_and_k_pe_cache: torch.Tensor, Skv: int, sm_scale) -> torch.Tensor:
    bsz = 1
    Lkv, R = 512, 64
    start_pos = Skv - 1

    q = q.unsqueeze(0)
    q_nope, q_pe = q.split([Lkv, R], dim=-1)
    kv_c_normed = torch.randn([1, 1, Lkv])
    k_pe = torch.randn([1, 1, 1, R])
    kv_c_and_k_pe = torch.cat([kv_c_normed, k_pe.squeeze(2)], dim=-1)  # [1, 1, 512 + 64]

    kv_cache_view, pe_cache_view = kv_c_and_k_pe_cache.split([Lkv, R], dim=-1)
    kv_cache_view = kv_cache_view.view(1, -1, Lkv)
    pe_cache_view = pe_cache_view.view(1, -1, R)
    # print(f'{kv_cache[:Skv, :].shape=}')
    # print(f'{q_nope.shape=}')
    # print(f'{q_pe.shape=}')
    kv_cache = kv_cache_view.detach().clone()
    pe_cache = pe_cache_view.detach().clone()

    page_start, page_end = start_pos, Skv
    kv_c_and_k_pe_cache[page_start:page_end] = kv_c_and_k_pe

    kv_cache[:bsz, start_pos:Skv] = kv_c_normed
    pe_cache[:bsz, start_pos:Skv] = k_pe.squeeze(2)

    torch.testing.assert_close(kv_cache, kv_cache_view)
    torch.testing.assert_close(pe_cache, pe_cache_view)

    scores = (torch.einsum("bshc,btc->bsht", q_nope, kv_cache[:bsz, :Skv]) +
                torch.einsum("bshr,btr->bsht", q_pe, pe_cache[:bsz, :Skv])) * sm_scale

    scores = scores.softmax(dim=-1, dtype=torch.float32).to(torch.float16)
    x = torch.einsum("bsht,btc->bshc", scores, kv_cache[:bsz, :Skv]).squeeze(1)
    return x


def check_mla_decode(q, kv_c_and_k_pe_cache, kv_c_cache, o, Skv: int, split_k: int = 8) -> None:
    # from vllm.attention.ops.triton_decode_attention import decode_attention_fwd
    B, N, Lkv = o.shape
    _, _, Lq  = q.shape
    R = Lq - Lkv

    req_to_tokens, b_seq_len = generate_tokens(B, Skv, Lkv, R)
    req_to_tokens = torch.arange(0, Skv, dtype=torch.int32).broadcast_to(req_to_tokens.size())
    sm_scale = 1.0 / ((Lkv + R) ** 0.5)
    ref = torch.empty(B, N, Lkv)
    attn_logits = torch.empty(
        (B, N, split_k, Lkv + 1),
        dtype=torch.float32,
        device="cuda",
    )

    # decode_attention_fwd(
    #     q,
    #     kv_c_and_k_pe_cache,
    #     kv_c_cache,
    #     ref,
    #     req_to_tokens,
    #     b_seq_len,
    #     attn_logits,
    #     split_k,
    #     sm_scale,
    #     # best_config=None,
    # )

    ref = torch_ref(q, kv_c_and_k_pe_cache, Skv, sm_scale)
    # print(f"REF {ref=}")

    if False:
        configs = [
            {
                "BLOCK_SIZE_H": h,
                "BLOCK_SIZE_N": n,
                "BLOCK_SIZE_D1": d1,
                "BLOCK_SIZE_D2": d2,
                "SPLIT_K": k,
                "num_warps": 4,
                "num_stages": 1,
            }
            for h in [1]
            for n in [16, 32, 64, 128]
            for d1 in [16, 32, 64, 128, 256, 512]
            for d2 in [16, 32, 64]
            for k in [1, 2, 4, 8]
        ]
        for config in configs:
            mla_decode(
                q,
                kv_c_and_k_pe_cache,
                kv_c_cache,
                o,
                req_to_tokens,
                b_seq_len,
                sm_scale,
                config=config,
            )
            if torch.allclose(o, ref, atol=1e-2, rtol=1e-2):
                print(f'pass {config=}')
        return

    mla_decode(
        q,
        kv_c_and_k_pe_cache,
        kv_c_cache,
        o,
        req_to_tokens,
        b_seq_len,
        sm_scale,
        config={'SPLIT_K': split_k},
    )

    torch.testing.assert_close(o, ref, atol=1e-2, rtol=1e-2)
    print("PASS MLA CHECK")
    print(f'{o=}')
    print(f'{ref=}')


def test_mla_decode(check: bool = False, seq_len: int = -1, split_k = -1):
    CACHE_SIZE = 32768
    PAGE_SIZE = 1
    B = 1
    N = 128
    Lkv = 512
    R = 64
    Skv = seq_len if seq_len != -1 else 64
    SPLIT_K = split_k if split_k != -1 else 8
    sm_scale = 1 / ((Lkv + R) ** 0.5)

    kv_c_and_k_pe_cache = torch.randn([CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, 1, Lkv + R])
    kv_c_cache, pe_cache = kv_c_and_k_pe_cache.split([Lkv, R], dim=-1)

    q_nope = torch.randn([B, N, Lkv])
    q_pe = torch.randn([B, N, R])
    q = torch.cat([q_nope, q_pe], dim=-1)
    # q = q.contiguous()
    o = torch.empty([B, N, Lkv])

    if check:
        check_mla_decode(q, kv_c_and_k_pe_cache, kv_c_cache, o, Skv, SPLIT_K)
        return

    Skvs = [2 ** i for i in range(15)] if seq_len == -1 else [seq_len]
    split_ks = [2 ** i for i in range(0, 6, 1)] if split_k == -1 else [split_k]
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["Skv", "split_k_iters"],
            x_vals=[
                (Skv, split_k_iter)
                for Skv in Skvs
                for split_k_iter in split_ks
            ],
            line_arg="provider",
            line_vals=["mla"],
            line_names=["MLA"],
            ylabel="Time (ms)",
            plot_name=f"MLA Performance {N=}",
            args={},
        )
    )
    def benchmark_mla(Skv: int, split_k_iters: int, provider):
        quantiles = [0.5, 0.2, 0.8]

        config={'SPLIT_K': split_k_iters}
        req_to_tokens = torch.arange(0, Skv, device=q.device, dtype=torch.int32).broadcast_to([B, Skv])
        b_seq_len = torch.full([B], Skv, device=q.device, dtype=torch.int32)

        if provider == 'mla':
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: mla_decode(
                    q,
                    kv_c_and_k_pe_cache,
                    kv_c_cache,
                    o,
                    req_to_tokens,
                    b_seq_len,
                    sm_scale,
                    config=config,
                ),
                quantiles=quantiles,
            )
        return ms

    benchmark_mla.run(print_data=True, show_plots=False)
    return
    req_to_tokens = torch.arange(0, Skv, device=q.device, dtype=torch.int32).broadcast_to([B, Skv])
    b_seq_len = torch.full([B], Skv, device=q.device, dtype=torch.int32)

    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=5, active=80),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:
        for _ in range(100):
            mla_decode(
                q,
                kv_c_and_k_pe_cache,
                kv_c_cache,
                o,
                req_to_tokens,
                b_seq_len,
                sm_scale,
                config=config,
            )
            prof.step()
    prof.export_chrome_trace(f"mla_decode-{Skv=}-{SPLIT_K=}.json.gz")


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--check", "-c", action="store_true")
    parser.add_argument("--split", "-s", type=int, default=-1, help="Number of split K iterations for MoE")
    parser.add_argument("--seqlen", "-l", type=int, default=-1, help="Number of split K iterations for MoE")
    args = parser.parse_args()

    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float16)
    torch.manual_seed(0)

    test_mla_decode(
        check=args.check,
        seq_len=args.seqlen,
        split_k=2 ** args.split if args.split != -1 else -1
    )


if __name__ == '__main__':
    main()

