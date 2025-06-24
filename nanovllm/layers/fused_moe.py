import triton
import triton.language as tl
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.utils import set_weight_attrs
from typing import Any, Dict, List, Optional, Tuple
from torch.profiler import ProfilerActivity


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_X': x, 'BLOCK_SIZE_Y': y}, num_warps=4, num_stages=1)
        # for x in [16, 32, 64, 128]
        for x in [64]
        # for y in [16, 32, 64, 128]
        for y in [16]
    ],
    key=['num_cols', 'num_rows', 'group_size'],
)
@triton.jit
def awq_dequantize_kernel(
        qweight_ptr,  # quantized matrix
        scales_ptr,  # scales, per group
        zeros_ptr,  # zeros, per group
        group_size,  # Should always be one of the supported group sizes
        result_ptr,  # Output matrix
        num_cols,  # input num cols in qweight
        num_rows,  # input num rows in qweight
        BLOCK_SIZE_X: tl.constexpr,
        BLOCK_SIZE_Y: tl.constexpr):
    # Setup the pids.
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)

    # Compute offsets and masks for qweight_ptr.
    offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offsets = num_cols * offsets_y[:, None] + offsets_x[None, :]

    masks_y = offsets_y < num_rows
    masks_x = offsets_x < num_cols

    masks = masks_y[:, None] & masks_x[None, :]

    # Compute offsets and masks for result output ptr.
    result_offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    result_offsets_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(
        0, BLOCK_SIZE_X * 8)
    result_offsets = (8 * num_cols * result_offsets_y[:, None] +
                      result_offsets_x[None, :])

    result_masks_y = result_offsets_y < num_rows
    result_masks_x = result_offsets_x < num_cols * 8
    result_masks = result_masks_y[:, None] & result_masks_x[None, :]

    # Load the weights.
    iweights = tl.load(qweight_ptr + offsets, masks, 0.0)
    # iweights = tl.interleave(iweights, iweights)
    # iweights = tl.interleave(iweights, iweights)
    # iweights = tl.interleave(iweights, iweights)
    iweights = tl.reshape(iweights, (BLOCK_SIZE_Y * BLOCK_SIZE_X, 1))
    iweights = tl.broadcast_to(iweights, (BLOCK_SIZE_Y * BLOCK_SIZE_X, 8))
    iweights = tl.reshape(iweights, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    # Create reverse AWQ order as tensor: [0, 4, 1, 5, 2, 6, 3, 7]
    # that will map given indices to the correct order.
    reverse_awq_order_tensor = tl.reshape((tl.arange(0, 2) * 4)[None, :] +
                                tl.arange(0, 4)[:, None], [8])

    # Use this to compute a set of shifts that can be used to unpack and
    # reorder the values in iweights and zeros.
    shifts = reverse_awq_order_tensor * 4
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_Y * BLOCK_SIZE_X, 8))
    shifts = tl.reshape(shifts, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    # Unpack and reorder: shift out the correct 4-bit value and mask.
    iweights = (iweights >> shifts) & 0xF

    # Compute zero offsets and masks.
    zero_offsets_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(0, 1)
    zero_offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    zero_offsets = num_cols * zero_offsets_y[:, None] + zero_offsets_x[None, :]

    zero_masks_y = zero_offsets_y < num_rows // group_size
    zero_masks_x = zero_offsets_x < num_cols
    zero_masks = zero_masks_y[:, None] & zero_masks_x[None, :]

    # Load the zeros.
    zeros = tl.load(zeros_ptr + zero_offsets, zero_masks, 0.0)
    # zeros = tl.interleave(zeros, zeros)
    # zeros = tl.interleave(zeros, zeros)
    # zeros = tl.interleave(zeros, zeros)
    zeros = tl.reshape(zeros, (BLOCK_SIZE_X, 1))
    zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_X, 8))
    zeros = tl.reshape(zeros, (1, BLOCK_SIZE_X * 8))
    zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    # Unpack and reorder: shift out the correct 4-bit value and mask.
    zeros = (zeros >> shifts) & 0xF

    # Compute scale offsets and masks.
    scale_offsets_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(0, 1)
    scale_offsets_x = (pid_x * BLOCK_SIZE_X * 8 +
                       tl.arange(0, BLOCK_SIZE_X * 8))
    scale_offsets = (num_cols * 8 * scale_offsets_y[:, None] +
                     scale_offsets_x[None, :])
    scale_masks_y = scale_offsets_y < num_rows // group_size
    scale_masks_x = scale_offsets_x < num_cols * 8
    scale_masks = scale_masks_y[:, None] & scale_masks_x[None, :]

    # Load the scales.
    scales = tl.load(scales_ptr + scale_offsets, scale_masks, 0.0)
    scales = tl.broadcast_to(scales, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    # Dequantize.
    iweights = (iweights - zeros) * scales
    iweights = iweights.to(result_ptr.type.element_ty)

    # Finally, store.
    tl.store(result_ptr + result_offsets, iweights, result_masks)


# qweights - [K     , M // 8], int32
# scales   - [K // G, M     ], float16
# zeros    - [K // G, M // 8], int32
def awq_dequantize_triton(qweight: torch.Tensor,
                          scales: torch.Tensor,
                          zeros: torch.Tensor,
                          block_size_x: int = 32,
                          block_size_y: int = 32) -> torch.Tensor:
    K = qweight.shape[0]
    M = scales.shape[1]
    # print(f'DEBUG: {qweight.shape=}')
    # print(f'DEBUG: {scales.shape=}')
    # print(f'DEBUG: {zeros.shape=}')
    group_size = qweight.shape[0] // scales.shape[0]

    AWQ_TRITON_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]

    assert K > 0 and M > 0
    assert scales.shape[0] == K // group_size and scales.shape[1] == M
    assert zeros.shape[0] == K // group_size and zeros.shape[1] == M // 8
    assert group_size <= K
    assert group_size in AWQ_TRITON_SUPPORTED_GROUP_SIZES or group_size == K

    # Result tensor:
    # number of rows = same as input tensor
    # number of cols = 8 x input tensor num cols
    result = torch.empty(qweight.shape[0],
                         qweight.shape[1] * 8,
                         device=qweight.device,
                         dtype=scales.dtype)

    Y = qweight.shape[0]  # num rows
    X = qweight.shape[1]  # num cols

    grid = lambda META: (
        triton.cdiv(X, META['BLOCK_SIZE_X']),
        triton.cdiv(Y, META['BLOCK_SIZE_Y']),
    )
    awq_dequantize_kernel[grid](qweight,
                                scales,
                                zeros,
                                group_size,
                                result,
                                X,
                                Y)
    return result


def generate_expert_weight(
    K: int, N: int, E: int,
    use_int4_w4a16: bool = False,
    group_size: int = 128,
    dequantize: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    if use_int4_w4a16:
        w_quantized = torch.randint(0, torch.iinfo(torch.int32).max, (E, K, N // 8), dtype=torch.int32)
        w_quantized &= 0x11111111
        w_zeros = torch.randint(0, torch.iinfo(torch.int32).max, (E, K // group_size, N // 8), dtype=torch.int32)
        w_zeros &= 0x11111111
        w_scales = torch.rand((E, K // group_size, N))
        w_scales /= 1000
        w = w_quantized
        # w = awq_dequantize_triton(w_quantized, w_scales, w_zeros)
        if dequantize:
            w_list = []
            for e in range(E):
                # 提取当前 E 索引下的二维切片
                w_quantized_2d = w_quantized[e]  # 形状: (K, 2*N//8)
                w_zeros_2d = w_zeros[e]          # 形状: (K//group_size, 2*N//8)
                w_scales_2d = w_scales[e]        # 形状: (K//group_size, 2*N)

                # 调用二维解量化函数
                dequantized_2d = awq_dequantize_triton(w_quantized_2d, w_scales_2d, w_zeros_2d)
                w_list.append(dequantized_2d)
            w = torch.stack(w_list, dim=0)

        w_zeros = w_zeros.transpose(1, 2)
        w_scales = w_scales.transpose(1, 2)
    else:
        w_quantized = torch.randn((E, K, N))
        w_zeros = None
        w_scales = None
        w = w_quantized

    w = w.transpose(1, 2)
    w_quantized = w_quantized.transpose(1, 2)
    return (w, w_quantized, w_scales, w_zeros)


# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE_M': m, 'BLOCK_SIZE_N': n, 'BLOCK_SIZE_K': k, 'SUM_WAY': w}, num_warps=4, num_stages=1)
#         for m in [1]
#         # for n in [16]
#         for n in [16, 32, 64, 128]
#         # for k in [512]
#         for k in [16, 32, 64, 128]
#         for w in [0]
#     ],
#     key=['M', 'N', 'K', 'group_size'],
#     # key=['M', 'N', 'K', 'SPLIT_K'],
# )
@triton.jit
def gemv_fused_moe_kernel_awq_w4a16(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        topk_weights_ptr,
        expert_ids_ptr,
        # Matrix dimensions
        M, N, K, EM,
        # The stride variables represent how much to increase the ptr by when
        # moving by 1 element in a particular dimension. E.g. `stride_am` is
        # how much to increase `a_ptr` by to get the element one row down
        # (A has M rows).
        stride_be, stride_bk, stride_bn,
        # awq part
        b_scales_ptr,
        b_zeros_ptr,
        stride_bse, stride_bsk, stride_bsn,
        stride_bze, stride_bzk, stride_bzn,
        group_size: tl.constexpr,
        # parameters
        MUL_ROUTED_WEIGHT: tl.constexpr,
        top_k: tl.constexpr,
        compute_type: tl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        SPLIT_K: tl.constexpr,
        SUM_WAY: tl.constexpr,
    ):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`.
    """
    tl.static_assert(BLOCK_SIZE_M == 1)

    pid_expert = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_split = tl.program_id(axis=2)

    pid_m: tl.constexpr = 0

    # w[int32] -> 8 * int4
    # Create reverse AWQ order as tensor: [0, 4, 1, 5, 2, 6, 3, 7]
    # that will map given indices to the correct order.
    reverse_awq_order_tensor = tl.reshape((tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None], [8])

    # Create the necessary shifts to use to unpack.
    shifts = reverse_awq_order_tensor * 4 # (shift 4 bits)
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_K * (BLOCK_SIZE_N // 8), 8))
    shifts = tl.reshape(shifts, (BLOCK_SIZE_K, BLOCK_SIZE_N))

    offsets_am = (pid_expert // top_k) + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    masks_am = offsets_am < M

    offsets_bn = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
    masks_bn = offsets_bn < (N // 8)

    offsets_zn = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
    masks_zn = offsets_zn < (N // 8)

    offsets_sn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    masks_sn = offsets_sn < N

    offsets_k = pid_split * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offsets_a = offsets_am[:, None] * K + offsets_k[None, :]
    offsets_b = offsets_k[:, None] * stride_bk + offsets_bn[None, :] * stride_bn
    # offsets_b = offsets_k[:, None] * (N // 8) + offsets_bn[None, :]

    a_ptrs = a_ptr + offsets_a

    off_experts = tl.load(expert_ids_ptr + pid_expert)
    b_ptrs = b_ptr + off_experts * stride_be + offsets_b

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    if SUM_WAY == 0:
        accumulator = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
    if SUM_WAY == 1:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        masks_k = offsets_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        a = tl.load(a_ptrs, mask=masks_a, other=0.0) # [1, BK]

        masks_b = masks_k[:, None] & masks_bn[None, :]
        b = tl.load(b_ptrs, mask=masks_b, other=0.0) # [BK, (BN // 8)]
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)

        # Dequantize b.
        offsets_szk = (
            (BLOCK_SIZE_K * SPLIT_K * k + pid_split * BLOCK_SIZE_K) // group_size +
            tl.arange(0, 1))
        offsets_z = offsets_szk[:, None] * stride_bzk + offsets_zn[None, :] * stride_bzn
        # offsets_z = offsets_szk[:, None] * (N // 8) + offsets_zn[None, :]
        masks_zk = offsets_szk < (K // group_size)
        masks_z = masks_zk[:, None] & masks_zn[None, :]
        zeros_ptrs = b_zeros_ptr + off_experts * stride_bze + offsets_z
        zeros = tl.load(zeros_ptrs, mask=masks_z, other=0.0)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)

        offsets_s = offsets_szk[:, None] * stride_bsk + offsets_sn[None, :] * stride_bsn
        # offsets_s = offsets_szk[:, None] * N + offsets_sn[None, :]
        masks_sk = offsets_szk < (K // group_size)
        masks_s = masks_sk[:, None] & masks_sn[None, :]
        scales_ptrs = b_scales_ptr + off_experts * stride_bse + offsets_s
        scales = tl.load(scales_ptrs, mask=masks_s, other=0.0)
        scales = tl.broadcast_to(scales, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        b = (b >> shifts) & 0xF
        zeros = (zeros >> shifts) & 0xF
        b = (b - zeros) * scales
        b = b.to(c_ptr.type.element_ty)

        # We accumulate along the K dimension.
        if SUM_WAY == 0:
            accumulator += tl.trans(a) * b
        if SUM_WAY == 1:
            accumulator += tl.sum(tl.trans(a) * b, axis=0)[None, :]

        # Advance the ptrs to the next K block.
        offsets_k += BLOCK_SIZE_K * SPLIT_K

        a_ptrs += BLOCK_SIZE_K * SPLIT_K
        # b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk

    if SUM_WAY == 0:
        accumulator = tl.sum(accumulator, axis=0, keep_dims=True)

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offsets_am)
        accumulator *= moe_weight[:, None]

    c = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    c_ptrs = c_ptr + pid_split * EM * N + pid_expert * N + N * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE_M': m, 'BLOCK_SIZE_N': n, 'BLOCK_SIZE_K': k, 'SUM_WAY': w}, num_warps=4, num_stages=1)
#         for m in [1]
#         # for n in [16]
#         for n in [16, 32, 64, 128]
#         # for k in [512]
#         for k in [16, 32, 64, 128]
#         for w in [0]
#     ],
#     key=['M', 'N', 'K'],
#     # key=['M', 'N', 'K', 'SPLIT_K'],
# )
@triton.jit
def gemv_fused_moe_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        topk_weights_ptr,
        expert_ids_ptr,
        # Matrix dimensions
        M, N, K, EM,
        # The stride variables represent how much to increase the ptr by when
        # moving by 1 element in a particular dimension. E.g. `stride_am` is
        # how much to increase `a_ptr` by to get the element one row down
        # (A has M rows).
        stride_be, stride_bk, stride_bn,
        MUL_ROUTED_WEIGHT: tl.constexpr,
        top_k: tl.constexpr,
        compute_type: tl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        SPLIT_K: tl.constexpr,
        SUM_WAY: tl.constexpr,
    ):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`.
    """
    tl.static_assert(BLOCK_SIZE_M == 1)

    pid_expert = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_split = tl.program_id(axis=2)

    pid_m: tl.constexpr = 0

    offsets_am = (pid_expert // top_k) + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    masks_am = offsets_am < M

    offsets_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    masks_bn = offsets_bn < N

    offsets_k = pid_split * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offsets_a = K * offsets_am[:, None] + offsets_k[None, :]
    offsets_b = offsets_k[:, None] * stride_bk + offsets_bn[None, :] * stride_bn

    a_ptrs = a_ptr + offsets_a

    off_experts = tl.load(expert_ids_ptr + pid_expert)
    b_ptrs = b_ptr + off_experts * stride_be + offsets_b

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    if SUM_WAY == 0:
        accumulator = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
    if SUM_WAY == 1:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        masks_k = offsets_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        a = tl.load(a_ptrs, mask=masks_a, other=0.0)

        masks_b = masks_k[:, None] & masks_bn[None, :]
        b = tl.load(b_ptrs, mask=masks_b, other=0.0)

        # We accumulate along the K dimension.
        if SUM_WAY == 0:
            accumulator += tl.trans(a) * b
        if SUM_WAY == 1:
            accumulator += tl.sum(tl.trans(a) * b, axis=0)[None, :]

        # Advance the ptrs to the next K block.
        offsets_k += BLOCK_SIZE_K * SPLIT_K

        a_ptrs += BLOCK_SIZE_K * SPLIT_K
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk

    if SUM_WAY == 0:
        accumulator = tl.sum(accumulator, axis=0)[None, :]

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offsets_am)
        accumulator *= moe_weight[:, None]

    c = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    c_ptrs = c_ptr + pid_split * EM * N + pid_expert * N + N * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def invoke_gemv_fused_moe_kernel(
    A: torch.Tensor,                      # [1, 7168]          [8, 2048]
    B: torch.Tensor,                      # [256, 4096, 7168]  [256, 7168, 2048]
    C: torch.Tensor,                      # [1, 8, 4096]       [1, 8, 7168]
    topk_weights: torch.Tensor,           # [1, 8]
    topk_ids: torch.Tensor,               # [1, 8]
    mul_routed_weight: bool,
    top_k: int,
    compute_type: tl.dtype,
    config: Dict[str, Any],
    use_int4_w4a16: bool = False,
    B_scales: Optional[torch.Tensor] = None,
    B_zeros: Optional[torch.Tensor] = None,
) -> None:
    assert A.is_contiguous() and C.is_contiguous()
    assert topk_weights.stride(1) == 1
    assert topk_ids.stride(1) == 1
    assert C.shape[0] == 1

    if not hasattr(invoke_gemv_fused_moe_kernel, "cache"):
       invoke_gemv_fused_moe_kernel.cache = {}  # it doesn't exist yet, so initialize it
    cache = invoke_gemv_fused_moe_kernel.cache

    M = A.shape[0]
    EM = top_k * M
    E, N, K = B.shape
    SPLIT_K = config['SPLIT_K']

    if use_int4_w4a16:
        N *= 8
        group_size = K // B_scales.shape[2]
        assert group_size == 128, f'{group_size=}, {K=}, {B_scales.shape=}'

        assert K > 0 and N > 0
        assert B_scales.shape[2] == K // group_size and B_scales.shape[1] == N
        assert B_zeros.shape[2] == K // group_size and B_zeros.shape[1] == N // 8
        assert group_size <= K

    # print(f'Invoke GEMV fused MoE kernel.')
    # print(f'{A.shape=}, {B.shape=}, {C.shape=}')
    # print(f'{topk_weights=}, {topk_ids=}, {top_k=}, {SPLIT_K=}')

    # print(f'{sorted_token_ids=}')

    if SPLIT_K != 1:
        O = torch.zeros((SPLIT_K, C.shape[1], C.shape[2]), dtype=C.dtype, device=C.device)
    else:
        O = C

    grid = lambda META: (
        triton.cdiv(EM, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
        SPLIT_K,
    )
    # grid = (
    #     triton.cdiv(EM, config['BLOCK_SIZE_M']),
    #     triton.cdiv(N, config['BLOCK_SIZE_N']),
    #     SPLIT_K,
    # )
    kernel = cache.get((M, N, K, use_int4_w4a16))
    if kernel is not None and triton.__version__ == "2.1.0":
        args = [
            A, B, O,
            topk_weights,
            topk_ids,
            M, N, K, EM,
        ]
        args_expand = kernel.assemble_tensormap_to_arg(args)
        stream = torch.cuda.current_stream().cuda_stream
        kernel.c_wrapper(
           grid[0], grid[1], grid[2],
           kernel.num_warps, kernel.num_ctas,
           kernel.clusterDims[0], kernel.clusterDims[1], kernel.clusterDims[2],
           kernel.shared, stream, kernel.cu_function,
           CompiledKernel.launch_enter_hook, CompiledKernel.launch_exit_hook,
           kernel, *args_expand
       )
    else:
        if use_int4_w4a16:
            kernel = gemv_fused_moe_kernel_awq_w4a16[grid](
                A, B, O,
                topk_weights,
                topk_ids,
                M, N, K, EM,
                B.stride(0), B.stride(2), B.stride(1),
                B_scales, B_zeros,
                B_scales.stride(0), B_scales.stride(2), B_scales.stride(1),
                B_zeros.stride(0), B_zeros.stride(2), B_zeros.stride(1),
                group_size=group_size,
                MUL_ROUTED_WEIGHT=mul_routed_weight,
                top_k=top_k,
                compute_type=compute_type,
                # SPLIT_K=SPLIT_K,
                **config,
            )
        else:
            kernel = gemv_fused_moe_kernel[grid](
                A, B, O,
                topk_weights,
                topk_ids,
                M, N, K, EM,
                B.stride(0), B.stride(2), B.stride(1),
                MUL_ROUTED_WEIGHT=mul_routed_weight,
                top_k=top_k,
                compute_type=compute_type,
                # SPLIT_K=SPLIT_K,
                **config,
            )
        cache[(M, N, K, use_int4_w4a16)] = kernel

    if SPLIT_K != 1:
        torch.sum(O, dim=0, keepdim=True, out=C)


# @torch.compile(dynamic=True, backend="inductor")
def grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool = True,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "sigmoid",
    e_score_correction_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert hidden_states.shape[0] == gating_output.shape[0], ("Number of tokens mismatch")

    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    num_token = scores.shape[0]
    assert num_token == 1, "Testing Decode"

    if e_score_correction_bias is not None:
        # Store original scores before applying correction bias. We use biased
        # scores for expert selection but original scores for routing weights
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
        group_scores = (scores.view(num_token, num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1))
    else:
        group_scores = scores.view(num_token, num_expert_group, -1).max(dim=-1).values  # [n, n_group]

    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = group_mask \
        .unsqueeze(-1) \
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group) \
        .reshape(num_token, -1)  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e]

    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)[1]
        # Use original unbiased scores for the routing weights
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def normal_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool = True,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "sigmoid",
    bias: Optional[torch.Tensor] = None,
):
    if scoring_func == "softmax":
        gating_output = gating_output.softmax(dim=-1, dtype=torch.float32)
    else:
        gating_output = gating_output.sigmoid()
    # print(f'MoE: {gating_output=}')
    original_scores = gating_output
    if bias is not None:
        gating_output = gating_output + bias
    if num_expert_group > 1:
        gating_output = gating_output.view(hidden_states.size(0), num_expert_group, -1)
        if bias is None:
            group_scores = gating_output.amax(dim=-1)
        else:
            group_scores = gating_output.topk(2, dim=-1)[0].sum(dim=-1)
        indices = group_scores.topk(topk_group, dim=-1)[1]
        mask = gating_output.new_ones(hidden_states.size(0), num_expert_group, dtype=bool).scatter_(1, indices, False)
        gating_output = gating_output.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
    indices = torch.topk(gating_output, topk, dim=-1)[1]
    weights = original_scores.gather(1, indices)
    if scoring_func == "sigmoid":
        weights /= weights.sum(dim=-1, keepdim=True)
    return weights, indices


def fused_experts(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = True,
    use_int4_w4a16: bool = False,
    w13_scales: Optional[torch.Tensor] = None,
    w2_scales: Optional[torch.Tensor] = None,
    w13_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
):
    M, K = hidden_states.shape
    E, _, N = w2.shape
    top_k = topk_ids.shape[1]

    assert K == w13.shape[2], f"Hidden size mismatch {K=}, {w13.shape=}"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    # assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    # assert w2.stride(-1) == 1, "Stride of last dimension must be 1"

    assert M == 1, "Testing decode"

    cache13 = torch.empty(M * top_k * max(2 * N, K))
    cache1 = cache13[:M * top_k * 2 * N].view(M, top_k, 2 * N)
    cache3 = cache13[:M * top_k * K].view(M, top_k, K)
    cache2 = torch.empty((M * top_k, N))

    compute_type = tl.float16

    if inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    w13_awq_config = {
        'BLOCK_SIZE_M': 1,
        'BLOCK_SIZE_N': 128,
        'BLOCK_SIZE_K': 16,
        'SUM_WAY': 0,
        'SPLIT_K': 8,
        'num_warps': 4,
        'num_stages':1
    }
    invoke_gemv_fused_moe_kernel(
        hidden_states, w13, cache1,
        topk_weights,
        topk_ids,
        mul_routed_weight=False,
        top_k=top_k,
        compute_type=compute_type,
        config=w13_awq_config,
        use_int4_w4a16=use_int4_w4a16,
        B_scales=w13_scales,
        B_zeros=w13_zeros,
    )
    # print(f'{cache1=}')

    torch.ops._C.silu_and_mul(cache2, cache1.view(-1, 2 * N))
    # x = cache1.view(-1, 2 * N)
    # cache2 = torch.nn.functional.silu(x[..., :N]) * x[..., N:]
    # print(f'{cache2=}')

    w2_awq_config = {
        'BLOCK_SIZE_M': 1,
        'BLOCK_SIZE_N': 64,
        'BLOCK_SIZE_K': 32,
        'SUM_WAY': 0,
        'SPLIT_K': 1,
        'num_warps': 4,
        'num_stages':1
    }
    invoke_gemv_fused_moe_kernel(
        cache2, w2, cache3,
        topk_weights,
        topk_ids,
        mul_routed_weight=True,
        top_k=1,
        compute_type=compute_type,
        config=w2_awq_config,
        use_int4_w4a16=use_int4_w4a16,
        B_scales=w2_scales,
        B_zeros=w2_zeros,
    )
    # print(f'{cache3=}')

    ops.moe_sum(cache3.view(*cache3.shape), out_hidden_states)
    # torch.sum(cache3.view(*cache3.shape), dim=1, out=out_hidden_states)

    return out_hidden_states


class FusedMoE(torch.nn.Module):
    """FusedMoE layer for MoE models.

    This layer contains both MergedColumnParallel weights (gate_up_proj /
    w13) and RowParallelLinear weights (down_proj/ w2).

    Note: Mixtral uses w1, w2, and w3 for gate, up, and down_proj. We
    copy that naming convention here and handle any remapping in the
    load_weights function in each model implementation.

    Args:
        num_experts: Number of experts in the model
        top_k: Number of experts selected for each token
        hidden_size: Input hidden state size of the transformer
        intermediate_size: Intermediate size of the experts
        params_dtype: Data type for the parameters.
        reduce_results: Whether to all all_reduce on the output of the layer
        renomalize: Whether to renormalize the logits in the fused_moe kernel
        quant_config: Quantization configure.
    """

    def __init__(
        self,
        num_experts: int,  # Global number of experts
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = True,
        num_expert_group: int = None,
        topk_group: int = None,
        tp_size: Optional[int] = None,
        ep_size: Optional[int] = None,
        dp_size: Optional[int] = None,
        prefix: str = "",
        scoring_func: str = "sigmoid",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ):
        super().__init__()

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        # Note: here we guard against accessing the TP and DP groups when
        # uninitialized (this happens when testing)
        assert tp_size is not None, "Tensor parallel size must be set"
        self.tp_size = tp_size
        # tp_rank = 0 if self.tp_size == 1 else get_tensor_model_parallel_rank()
        # self.dp_size = (dp_size
        #                 if dp_size is not None else get_dp_group().world_size)
        # self.dp_rank = (0
        #                 if self.dp_size == 1 else get_dp_group().rank_in_group)
        self.global_num_experts = num_experts

        # Use expert parallelism instead of tensor parallelism?
        self.layer_name = prefix

        # Adjust TP size for DP attention
        # self.tp_rank = tp_rank + self.tp_size * self.dp_rank
        # self.ep_rank = 0
        # self.tp_size = self.tp_size * self.dp_size
        # self.ep_size = 1
        self.local_num_experts = self.global_num_experts
        # self.expert_map = None
        self.top_k = top_k
        self.global_num_experts = num_experts

        assert intermediate_size % self.tp_size == 0
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.scoring_func = scoring_func
        self.activation = activation

        self.apply_router_weight_on_input = apply_router_weight_on_input

        num_experts = self.local_num_experts
        intermediate_size_per_partition = self.intermediate_size_per_partition
        moe_quant_params = {
            "num_experts": num_experts,
            "hidden_size": hidden_size,
            "intermediate_size_per_partition":
            intermediate_size_per_partition,
            "params_dtype": params_dtype,
            # "weight_loader": self.weight_loader,
            "is_transposed": True,
            "quant_method": "awq",
            "tp_size": self.tp_size,
        }

        # create awq weight
        w13_qweight = torch.nn.Parameter(
            torch.empty(
                [
                    num_experts,
                    hidden_size,
                    2 * intermediate_size_per_partition // 8,
                ],
                dtype=torch.int32,
            ).transpose(1, 2),
            requires_grad=False,
        )
        self.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, moe_quant_params)

        w2_qweight = torch.nn.Parameter(
            torch.empty(
                [
                    num_experts,
                    intermediate_size_per_partition,
                    hidden_size // 8,
                ],
                dtype=torch.int32,
            ).transpose(1, 2),
            requires_grad=False,
        )
        self.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, moe_quant_params)

        group_size = 128
        num_groups_w13 = hidden_size // group_size
        num_groups_w2 = intermediate_size_per_partition // group_size

        # WEIGHT_SCALES
        # Allocate 2 scales for w1 and w3 respectively.
        w13_scales = torch.nn.Parameter(
            torch.randn(
                num_experts,
                num_groups_w13,
                intermediate_size_per_partition * 2,
                dtype=params_dtype,
            ).transpose(1, 2),
            requires_grad=False,
        )
        self.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, moe_quant_params)

        w2_scales = torch.nn.Parameter(
            torch.randn(
                num_experts,
                num_groups_w2,
                hidden_size,
                dtype=params_dtype,
            ).transpose(1, 2),
            requires_grad=False,
        )
        self.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, moe_quant_params)

        # WEIGHT_ZERO_POINT
        # Allocate 2 zero points for w1 and w3 respectively.
        w13_qzeros = torch.nn.Parameter(
            torch.empty(
                [
                    num_experts,
                    num_groups_w13,
                    2 * intermediate_size_per_partition // 8,
                ],
                dtype=torch.int32,
            ).transpose(1, 2),
            requires_grad=False,
        )
        self.register_parameter("w13_qzeros", w13_qzeros)
        set_weight_attrs(w13_qzeros, moe_quant_params)

        w2_qzeros = torch.nn.Parameter(
            torch.empty(
                [
                    num_experts,
                    num_groups_w2,
                    hidden_size // 8,
                ],
                dtype=torch.int32,
            ).transpose(1, 2),
            requires_grad=False,
        )

        self.register_parameter("w2_qzeros", w2_qzeros)
        set_weight_attrs(w2_qzeros, moe_quant_params)

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        # print(f'MOE: {router_logits=}')
        topk_weights, topk_ids = grouped_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            num_expert_group=self.num_expert_group,
            topk_group=self.topk_group,
            scoring_func=self.scoring_func,
        )
        # topk_weights, topk_ids = normal_topk(
        #     hidden_states=hidden_states,
        #     gating_output=router_logits,
        #     topk=self.top_k,
        #     renormalize=self.renormalize,
        #     num_expert_group=self.num_expert_group,
        #     topk_group=self.topk_group,
        #     scoring_func=self.scoring_func,
        # )
        # print(f'{topk_weights=}, {topk_ids=}')

        return fused_experts(
            hidden_states, self.w13_qweight, self.w2_qweight,
            topk_weights, topk_ids,
            inplace=True,
            use_int4_w4a16=True,
            w13_scales=self.w13_scales, w13_zeros=self.w13_qzeros,
            w2_scales=self.w2_scales, w2_zeros=self.w2_qzeros,
            block_shape=[0, 128],
        )
