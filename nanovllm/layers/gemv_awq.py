# # SPDX-License-Identifier: Apache-2.0
# # SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# import torch

# from vllm.triton_utils import tl, triton

# AWQ_TRITON_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]


# @triton.jit
# def awq_gemm_kernel(a_ptr, b_ptr, c_ptr, zeros_ptr, scales_ptr, M, N, K,
#                     group_size, BLOCK_SIZE_M: tl.constexpr,
#                     BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
#                     SPLIT_K: tl.constexpr):
#     pid = tl.program_id(axis=0)
#     pid_z = tl.program_id(1)

#     # NOTE: This doesn't work in TRITON_INTERPRET=1 mode.  Use below instead.
#     # num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
#     num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

#     pid_m = pid // num_pid_n
#     pid_n = pid % num_pid_n

#     accumulator_dtype = c_ptr.type.element_ty

#     # NOTE: This doesn't work in TRITON_INTERPRET=1 mode.  Use below instead.
#     # accumulator = tl.arange(0, BLOCK_SIZE_N)
#     # accumulator = tl.broadcast_to(accumulator[None, :],
#     # (BLOCK_SIZE_M, BLOCK_SIZE_N))
#     # accumulator = accumulator & 0x0
#     # accumulator = accumulator.to(accumulator_dtype)
#     accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N),
#                            dtype=accumulator_dtype)

#     # Create reverse AWQ order as tensor: [0, 4, 1, 5, 2, 6, 3, 7]
#     # that will map given indices to the correct order.
#     reverse_awq_order_tensor = ((tl.arange(0, 2) * 4)[None, :] +
#                                 tl.arange(0, 4)[:, None]).reshape(8)

#     # Create the necessary shifts to use to unpack.
#     shifts = reverse_awq_order_tensor * 4
#     shifts = tl.broadcast_to(shifts[None, :],
#                              (BLOCK_SIZE_K * (BLOCK_SIZE_N // 8), 8))
#     shifts = tl.reshape(shifts, (BLOCK_SIZE_K, BLOCK_SIZE_N))

#     # Offsets and masks.
#     offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     masks_am = offsets_am < M

#     offsets_bn = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
#     masks_bn = offsets_bn < N // 8

#     offsets_zn = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
#     masks_zn = offsets_zn < N // 8

#     offsets_sn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     masks_sn = offsets_sn < N

#     offsets_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
#     offsets_a = K * offsets_am[:, None] + offsets_k[None, :]
#     offsets_b = (N // 8) * offsets_k[:, None] + offsets_bn[None, :]

#     a_ptrs = a_ptr + offsets_a
#     b_ptrs = b_ptr + offsets_b

#     # NOTE: Use this in TRITON_INTERPRET=1 mode instead of tl.cdiv
#     # block_offset = BLOCK_SIZE_K * SPLIT_K
#     # for k in range(0, (K + block_offset - 1) // (block_offset)):
#     for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
#         masks_k = offsets_k < K
#         masks_a = masks_am[:, None] & masks_k[None, :]
#         a = tl.load(a_ptrs, mask=masks_a, other=0.0)

#         masks_b = masks_k[:, None] & masks_bn[None, :]
#         b = tl.load(b_ptrs, mask=masks_b, other=0.0)
#         b = tl.interleave(b, b)
#         b = tl.interleave(b, b)
#         b = tl.interleave(b, b)

#         # Dequantize b.
#         offsets_szk = (
#             (BLOCK_SIZE_K * SPLIT_K * k + pid_z * BLOCK_SIZE_K) // group_size +
#             tl.arange(0, 1))
#         offsets_z = (N // 8) * offsets_szk[:, None] + offsets_zn[None, :]
#         masks_zk = offsets_szk < K // group_size
#         masks_z = masks_zk[:, None] & masks_zn[None, :]
#         zeros_ptrs = zeros_ptr + offsets_z
#         zeros = tl.load(zeros_ptrs, mask=masks_z, other=0.0)
#         zeros = tl.interleave(zeros, zeros)
#         zeros = tl.interleave(zeros, zeros)
#         zeros = tl.interleave(zeros, zeros)
#         zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_K, BLOCK_SIZE_N))

#         offsets_s = N * offsets_szk[:, None] + offsets_sn[None, :]
#         masks_sk = offsets_szk < K // group_size
#         masks_s = masks_sk[:, None] & masks_sn[None, :]
#         scales_ptrs = scales_ptr + offsets_s
#         scales = tl.load(scales_ptrs, mask=masks_s, other=0.0)
#         scales = tl.broadcast_to(scales, (BLOCK_SIZE_K, BLOCK_SIZE_N))

#         b = (b >> shifts) & 0xF
#         zeros = (zeros >> shifts) & 0xF
#         b = (b - zeros) * scales
#         b = b.to(c_ptr.type.element_ty)

#         # Accumulate results.
#         accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)

#         offsets_k += BLOCK_SIZE_K * SPLIT_K
#         a_ptrs += BLOCK_SIZE_K * SPLIT_K
#         b_ptrs += BLOCK_SIZE_K * SPLIT_K * (N // 8)

#     c = accumulator.to(c_ptr.type.element_ty)
#     offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     c_ptrs = c_ptr + pid_z * N * M + N * offs_cm[:, None] + offs_cn[None, :]
#     c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
#     tl.store(c_ptrs, c, mask=c_mask)




# # input   - [M, K]
# # qweight - [K, N // 8]
# # qzeros  - [K // G, N // 8]
# # scales  - [K // G, N]
# # split_k_iters - parallelism along K-dimension, int, power of 2.
# def awq_gemm_triton(input: torch.Tensor,
#                     qweight: torch.Tensor,
#                     scales: torch.Tensor,
#                     qzeros: torch.Tensor,
#                     split_k_iters: int,
#                     block_size_m: int = 32,
#                     block_size_n: int = 32,
#                     block_size_k: int = 32) -> torch.Tensor:
#     M, K = input.shape
#     N = qweight.shape[1] * 8
#     group_size = qweight.shape[0] // qzeros.shape[0]

#     assert N > 0 and K > 0 and M > 0
#     assert qweight.shape[0] == K and qweight.shape[1] == N // 8
#     assert qzeros.shape[0] == K // group_size and qzeros.shape[1] == N // 8
#     assert scales.shape[0] == K // group_size and scales.shape[1] == N
#     assert split_k_iters & (split_k_iters - 1) == 0 and split_k_iters != 0
#     assert split_k_iters <= 32
#     assert group_size <= K
#     assert group_size in AWQ_TRITON_SUPPORTED_GROUP_SIZES or group_size == K

#     grid = lambda META: (
#         triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(
#             N, META['BLOCK_SIZE_N']),
#         split_k_iters,
#     )

#     result = torch.zeros((split_k_iters, M, N),
#                          dtype=scales.dtype,
#                          device=input.device)

#     # A = input, B = qweight, C = result
#     # A = M x K, B = K x N, C = M x N
#     awq_gemm_kernel[grid](input,
#                           qweight,
#                           result,
#                           qzeros,
#                           scales,
#                           M,
#                           N,
#                           K,
#                           group_size,
#                           BLOCK_SIZE_M=block_size_m,
#                           BLOCK_SIZE_N=block_size_n,
#                           BLOCK_SIZE_K=block_size_k,
#                           SPLIT_K=split_k_iters)

#     result = result.sum(0)

#     return result






import torch
from torch.profiler import ProfilerActivity
import torch.nn.functional as F

import torch
import triton
import triton.language as tl

@triton.jit
def gemv_kernel(
    w_ptr,
    x_ptr,
    b_ptr,
    y_ptr,
    K,
    # N,
    BLOCK_SIZE_X: tl.constexpr,
    # BLOCK_SIZE_B: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    rx = tl.arange(0, BLOCK_SIZE_X)

    mask_x = rx < K
    
    w = tl.load(
        w_ptr + pid * K + rx,
        mask=mask_x,
        other=0.0
    ).to(tl.float64)
    x = tl.load(
        x_ptr + rx,
        mask=mask_x,
        other=0.0
    ).to(tl.float64)
        
    acc = tl.sum(w * x)

    if b_ptr is not None:
        b = tl.load(
            b_ptr + pid,
        ).to(tl.float64)
        acc += b

    tl.store(
        y_ptr + pid,
        acc.to(y_ptr.dtype.element_ty)
    )

def next_power_of_two(n):
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return n

def gemv(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, y: torch.Tensor):
    # 检查维度
    assert w.dim() == 2
    assert x.dim() == 1
    N, K = w.shape
    assert x.shape[0] == K
    
    if b is not None:
        assert b.dim() == 1
        assert b.shape[0] == N

    BLOCK_SIZE_X = next_power_of_two(K)
    # BLOCK_SIZE_B = next_power_of_two(N)
    
    grid = (N,)
    
    # 调用kernel
    gemv_kernel[grid](
        w_ptr=w,
        x_ptr=x,
        b_ptr=b,
        y_ptr=y,
        K=K,
        # N=N,
        BLOCK_SIZE_X=BLOCK_SIZE_X,
        # BLOCK_SIZE_B=BLOCK_SIZE_B,
    )
    return y





def compare_tensors(tensor1, tensor2, rtol=1e-5, atol=1e-8, top_k=5):
    if torch.equal(tensor1, tensor2):
        print("✅ 张量完全相同！")
        return
    
    print("❌ 张量存在差异！")
    
    # 逐元素比较
    close_mask = torch.isclose(tensor1, tensor2, rtol=rtol, atol=atol)
    not_close_mask = ~close_mask
    
    print("\n=== 差异统计 ===")
    print("不匹配的元素数量:", not_close_mask.sum().item())
    print("不匹配比例:", not_close_mask.sum().item() / tensor1.numel())
    
    # 计算误差
    abs_diff = torch.abs(tensor1 - tensor2)
    rel_diff = abs_diff / torch.abs(tensor2)
    
    print("\n=== 误差统计 ===")
    print("最大绝对误差:", abs_diff.max().item())
    print("平均绝对误差:", abs_diff.mean().item())
    print("最大相对误差:", rel_diff.max().item())
    print("平均相对误差:", rel_diff.mean().item())
    
    # 打印前 top_k 差异
    if not_close_mask.any():
        abs_diff_flat = abs_diff.flatten()
        top_values, top_indices = torch.topk(abs_diff_flat, k=min(top_k, not_close_mask.sum()))
        
        print(f"\n=== 前 {top_k} 个最大差异 ===")
        for i in range(len(top_values)):
            idx = top_indices[i].item()
            print(f"位置 {idx}:")
            print(f"  tensor1 = {tensor1.flatten()[idx].item()}")
            print(f"  tensor2 = {tensor2.flatten()[idx].item()}")
            print(f"  绝对误差 = {top_values[i].item()}")
            print(f"  相对误差 = {rel_diff.flatten()[idx].item()}\n")




# 测试代码
if __name__ == "__main__":
    torch.manual_seed(0)
    M, N = 7168, 18432
    a = torch.randn((M, N), device='cuda', dtype=torch.float16)
    x = torch.randn(N, device='cuda', dtype=torch.float16)
    b = torch.randn(M, device='cuda', dtype=torch.float16)
    

    # a = torch.tensor([
    #     [1,2,3],
    #     [4,5,6],
    #     ], device='cuda', dtype=torch.float16)
    
    # x = torch.tensor([7,8,9], device='cuda', dtype=torch.float16)
    # y = torch.zeros(2, device='cuda', dtype=torch.float16)
    
    # 使用我们的Triton实现
    y_triton = torch.randn(M, device='cuda', dtype=torch.float16)
    gemv(x, a, b, y_triton)

  
    
    # 使用PyTorch的参考实现
    y_ref = F.linear(x, a, b)
    print(y_triton)
    print(y_ref)

    # for v in y_triton:
    #     print(v)
    
    # 验证结果
    print(f"最大误差: {torch.max(torch.abs(y_triton - y_ref))}")
    print(f"平均误差: {torch.mean(torch.abs(y_triton - y_ref))}")

    compare_tensors(y_triton, y_ref)


    # mat = torch.randn(18432, 7168, dtype=torch.float16).cuda()
    # vec = torch.randn(7168, dtype=torch.float16).cuda()

    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=10, active=300),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:
        for _ in range(50):
            # torch.mv(mat, vec)
            gemv(x, a, b, y_triton)
            torch.cuda.synchronize()
            prof.step()
            
    prof.export_chrome_trace(f"tracing_gemv.json.gz")

