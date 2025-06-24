import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.layers.fused_moe import FusedMoE
from nanovllm.layers.kernel import act_quant, weight_dequant, fp8_gemm


world_size = 1
rank = 0
block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"

# @dataclass
# class ModelArgs:
#     """
#     Data class for defining model arguments and hyperparameters.

#     Attributes:
#         max_batch_size (int): Maximum batch size.
#         max_seq_len (int): Maximum sequence length.
#         dtype (Literal["bf16", "fp8"]): Data type for computations.
#         vocab_size (int): Vocabulary size.
#         dim (int): Model dimension.
#         inter_dim (int): Intermediate dimension for MLP layers.
#         moe_inter_dim (int): Intermediate dimension for MoE layers.
#         n_layers (int): Number of transformer layers.
#         n_dense_layers (int): Number of dense layers in the model.
#         n_heads (int): Number of attention heads.
#         n_routed_experts (int): Number of routed experts for MoE layers.
#         n_shared_experts (int): Number of shared experts for MoE layers.
#         n_activated_experts (int): Number of activated experts in MoE layers.
#         n_expert_groups (int): Number of expert groups.
#         n_limited_groups (int): Number of limited groups for MoE routing.
#         score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
#         route_scale (float): Scaling factor for routing scores.
#         q_lora_rank (int): LoRA rank for query projections.
#         kv_lora_rank (int): LoRA rank for key-value projections.
#         qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
#         qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
#         v_head_dim (int): Dimension for value projections.
#         original_seq_len (int): Original sequence length.
#         rope_theta (float): Base for rotary positional encoding.
#         rope_factor (float): Scaling factor for extended sequence lengths.
#         beta_fast (int): Fast beta correction factor.
#         beta_slow (int): Slow beta correction factor.
#         mscale (float): Scaling factor for extended attention.
#     """
#     max_batch_size: int = 8
#     max_seq_len: int = 4096 * 4
#     dtype: Literal["bf16", "fp8"] = "bf16"
#     vocab_size: int = 102400
#     dim: int = 2048
#     inter_dim: int = 10944
#     moe_inter_dim: int = 1408
#     n_layers: int = 27
#     n_dense_layers: int = 1
#     n_heads: int = 16
#     # moe
#     n_routed_experts: int = 64
#     n_shared_experts: int = 2
#     n_activated_experts: int = 6
#     n_expert_groups: int = 1
#     n_limited_groups: int = 1
#     score_func: Literal["softmax", "sigmoid"] = "softmax"
#     route_scale: float = 1.
#     # mla
#     q_lora_rank: int = 0
#     kv_lora_rank: int = 512
#     qk_nope_head_dim: int = 128
#     qk_rope_head_dim: int = 64
#     v_head_dim: int = 128
#     # yarn
#     original_seq_len: int = 4096
#     rope_theta: float = 10000.0
#     rope_factor: float = 40
#     beta_fast: int = 32
#     beta_slow: int = 1
#     mscale: float = 1.


class ParallelEmbedding(nn.Module):
    """
    Embedding layer with parallelism support across distributed processes.

    Args:
        vocab_size (int): Vocabulary size.
        dim (int): Embedding dimension.
    """
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0, f"Vocabulary size must be divisible by world size (world_size={world_size})"
        self.part_vocab_size = (vocab_size // world_size)
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for parallel embedding layer.

        Args:
            x (torch.Tensor): Input tensor containing token indices.

        Returns:
            torch.Tensor: Embedded representations.

        Raises:
            ValueError: If `world_size` is not defined.
        """

        mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
        x = x - self.vocab_start_idx
        x[mask] = 0

        y = F.embedding(x, self.weight)


        y[mask] = 0
        dist.all_reduce(y)
        return y


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    return F.linear(x, weight, bias)



class Linear(nn.Module):
    """
    Custom linear layer with support for quantized weights and optional bias.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    dtype = torch.float16

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype), requires_grad=False)
        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32), requires_grad=False)
        else:
            self.register_parameter("scale", None)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    """
    Linear layer with column parallelism, splitting output features across distributed processes.

    Args:
        in_features (int): Number of input features.
        out_features (int): Total number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for column parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with column-parallel computation.
        """
        y = linear(x, self.weight, self.bias)
        return y


class RowParallelLinear(Linear):
    """
    Linear layer with row parallelism, splitting input features across distributed processes.

    Args:
        in_features (int): Total number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for row parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with row-parallel computation.
        """
        y = linear(x, self.weight)

        dist.all_reduce(y)

        if self.bias is not None:
            y += self.bias
        return y


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim), requires_grad=False)

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


def precompute_freqs_cis(args) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_position_embeddings
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)




class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        self.w2 = RowParallelLinear(inter_dim, dim)
        self.w3 = ColumnParallelLinear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Z100_MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """
    def __init__(self, args, nanovllm_config):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()

        self.args = args



        self.dim = args.hidden_size
        assert args.n_routed_experts % nanovllm_config.tensor_parallel_size == 0, f"Number of experts must be divisible by world size (nanovllm_config.tensor_parallel_size={nanovllm_config.tensor_parallel_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // nanovllm_config.tensor_parallel_size
        self.n_activated_experts = args.num_experts_per_tok
        self.experts_start_idx = nanovllm_config.local_rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Linear(args.hidden_size, args.n_routed_experts, bias=True)
        self.experts = FusedMoE(
            num_experts=args.n_routed_experts,
            top_k=args.num_experts_per_tok,
            hidden_size=args.hidden_size,
            intermediate_size=args.moe_intermediate_size ,
            params_dtype=Linear.dtype,
            renormalize=True,
            use_grouped_topk=True,
            num_expert_group=args.n_group,
            topk_group=args.topk_group,
            tp_size=nanovllm_config.tensor_parallel_size,
        )
        self.shared_experts = MLP(args.hidden_size, args.n_shared_experts * args.moe_intermediate_size )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        shape = x.size()
        x = x.view(-1, self.dim)

        weights = self.gate(x)
        y = self.experts(hidden_states=x, router_logits=weights)

        z = self.shared_experts(x)
        dist.all_reduce(y)
        return (y + z).view(shape)


class Z100_MLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA) Layer.

    Attributes:
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        qk_head_dim (int): Total dimensionality of query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
    """
    def __init__(self, args, nanovllm_config):
        super().__init__()
        self.args = args

        self.dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_local_heads = args.num_attention_heads // nanovllm_config.tensor_parallel_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        # q_a_proj [7168x1536]
        self.wq_a = Linear(self.dim, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank)
        # [1536x(128 * (128 + 64))]
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)

        # [7168x(512 + 64)]
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        # [512x(128 * (128 + 128))]
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        # [(128 * 128)x7168]
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_position_embeddings > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.cache_size = args.max_batch_size * args.max_position_embeddings
        self.page_size = 1
        self.register_buffer("kv_c_and_k_pe_cache", torch.randn(self.cache_size // self.page_size, self.page_size, 1, self.kv_lora_rank + self.qk_rope_head_dim), persistent=False)
        kv_cache, pe_cache = self.kv_c_and_k_pe_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        self.register_buffer("kv_cache", kv_cache.view(args.max_batch_size, args.max_position_embeddings, self.kv_lora_rank), persistent=False)
        self.register_buffer("pe_cache", pe_cache.view(args.max_batch_size, args.max_position_embeddings, self.qk_rope_head_dim), persistent=False)

        kv_b_proj_weight = self.wkv_b.weight.T

        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.n_local_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        W_UK, W_UV = kv_b_proj_weight.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Convert from (L, N, V) to (N, L, V) (128, 512, 128)
        self.W_UV = W_UV.transpose(0, 1)
        # Convert from (L, N, P) to (N, P, L) (128, 128, 512)
        self.W_UK_T = W_UK.permute(1, 2, 0)

    def _v_up_proj_and_o_proj(self, x):
        # Convert from (B, N, L) to (N, B, L)
        x = x.view(-1, self.n_local_heads, self.kv_lora_rank).transpose(0, 1)
        # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
        x = torch.bmm(x, self.W_UV)
        # Convert from (N, B, V) to (B, N * V)
        x = x.transpose(0, 1).reshape(-1, self.n_local_heads * self.v_head_dim)
        return self.wo(x)

    # Return `ql_nope`, `q_pe`
    def _q_proj_and_k_up_proj(self, x):
        q = self.wq_b(x) # [1x(128 * (128 + 64))]
        # print(f'MLA: {q=}')
        q = q.view(-1, self.n_local_heads, self.qk_head_dim) # [1, 128, 192]
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        # print(f'MLA: {q_nope=}, {q_pe=}')

        # Convert from (B, N, P) to (N, B, P)
        q_nope = q_nope.transpose(0, 1) # [128, 1, 128]
        # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
        ql_nope = torch.bmm(q_nope, self.W_UK_T) # [128, 1, 512]
        # Convert from (N, B, L) to (B, N, L)
        return ql_nope.transpose(0, 1), q_pe # [1, 128, 512] [1, 128, 64]

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()
        # x = x.squeeze(0) # [1x7168]

        # Get Q, K, V
        ckq = self.wq_a(x) # [1x1536]
        kv = self.wkv_a(x) # [1x576]
        kv_c, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        # Normalize Q and KV
        q_c = self.q_norm(ckq) # [1x1536]
        kv_c_normed = self.kv_norm(kv_c) # [1x512]

        q_nope, q_pe = self._q_proj_and_k_up_proj(q_c)

        # apply rope
        q_pe = apply_rotary_emb(q_pe.unsqueeze(0), freqs_cis)
        q_pe = q_pe.squeeze(0)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)

        # print(f'MLA: {q_nope=}, {q_pe=}')
        # update kvcache
        # print(f'MLA: {kv_c_normed=}, {k_pe=}')
        kv_c_and_k_pe = torch.cat([kv_c_normed, k_pe.squeeze(1)], dim=-1)  # [1, 1, 512 + 64]
        assert self.page_size == 1
        Skv = start_pos + seqlen
        page_start, page_end = start_pos // self.page_size, triton.cdiv(Skv, self.page_size)
        self.kv_c_and_k_pe_cache[page_start:page_end] = kv_c_and_k_pe

        # Attention
        q = torch.cat([q_nope, q_pe], dim=-1)
        o = torch.empty([bsz, self.n_local_heads, self.kv_lora_rank], dtype=q.dtype, device=q.device)

        kv_c_and_k_pe_cache = self.kv_c_and_k_pe_cache
        kv_c_cache = kv_c_and_k_pe_cache[..., :self.kv_lora_rank]
        # Testing arguments
        req_to_tokens = torch.arange(0, Skv, device=q.device, dtype=torch.int32).broadcast_to([bsz, Skv])
        b_seq_len = torch.full([bsz], Skv, device=q.device, dtype=torch.int32)

        config = {
            'SPLIT_K': 2,
        }

        mla_decode(
            q,
            kv_c_and_k_pe_cache,
            kv_c_cache,
            o,
            req_to_tokens,
            b_seq_len,
            config=config,
        )

        return self._v_up_proj_and_o_proj(o)


class Z100_Block(nn.Module):
    """
    Transformer block combining attention and feed-forward layers.

    Attributes:
        attn (nn.Module): Attention layer (MLA).
        ffn (nn.Module): Feed-forward network (MLP or MoE).
        attn_norm (nn.Module): Layer normalization for attention.
        ffn_norm (nn.Module): Layer normalization for feed-forward network.
    """
    def __init__(self, layer_id: int, args, nanovllm_config):
        """
        Initializes the Transformer block.

        Args:
            layer_id (int): Layer index in the transformer.
            args (ModelArgs): Model arguments containing block parameters.
        """
        super().__init__()
        self.layer_id = layer_id
        self.args = args
        self.attn = Z100_MLA(args, nanovllm_config)
        self.ffn = MLP(args.hidden_size, args.intermediate_size) if layer_id < args.first_k_dense_replace else Z100_MoE(args, nanovllm_config)
        self.attn_norm = RMSNorm(args.hidden_size)
        self.ffn_norm = RMSNorm(args.hidden_size)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position in the sequence.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor after block computation.
        """
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class DeepseekV3ForCausalLLM(nn.Module):
    """
    Transformer model with positional embeddings, multiple layers, and output projection.

    Attributes:
        max_seq_len (int): Maximum sequence length for the transformer.
        embed (nn.Module): Embedding layer for input tokens.
        layers (torch.nn.ModuleList): List of transformer blocks.
        norm (nn.Module): Layer normalization applied after all blocks.
        head (nn.Module): Output projection layer mapping to vocabulary size.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
    """

    packed_modules_mapping = {
        "embed_tokens": ("embed", 0),
        "input_layernorm": ("attn_norm", None),
        "post_attention_layernorm": ("ffn_norm", None),
        "q_proj": ("wq", 0),
        "q_a_proj": ("wq_a", None),
        "q_a_layernorm": ("q_norm", None),
        "q_b_proj": ("wq_b", 0),
        "kv_a_proj_with_mqa": ("wkv_a", None),
        "kv_a_layernorm": ("kv_norm", None),
        "kv_b_proj": ("wkv_b", 0),
        "o_proj": ("wo", 1),
        "gate": ("gate", None),
        "gate_proj": ("w1", 0),
        "down_proj": ("w2", 1),
        "up_proj": ("w3", 0),
        "norm": ("norm", None),
        "lm_head": ("head", 0),
        "scale": ("scale", None),
    }

    def __init__(self, args, nanovllm_config):
        """
        Initializes the Transformer model.

        Args:
            args (ModelArgs): Model arguments containing transformer parameters.
        """
        # hack: inject fields to args to make it visiable to all other parts of the model
        args.original_seq_len = 4096
        args.mscale = 1.0
        args.rope_factor = 40.0
        args.beta_fast = 32
        args.beta_slow = 1
        args.rope_theta = 10000
        args.max_batch_size = 1



        
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        # assert dist.get_world_size == torch.cuda.device_count(), f"World size ({dist.get_world_size()}) must match the number of GPUs ({torch.cuda.device_count()})" 
        nanovllm_config.local_rank = rank

        Linear.dtype = torch.float16
        super().__init__()
        self.nanovllm_config = nanovllm_config
        self.max_seq_len = args.max_position_embeddings
        self.embed = ParallelEmbedding(args.vocab_size, args.hidden_size)
        self.layers = torch.nn.ModuleList()
        args.num_hidden_layers = 4
        for layer_id in range(args.num_hidden_layers):
            self.layers.append(Z100_Block(layer_id, args, nanovllm_config))
        self.norm = RMSNorm(args.hidden_size)
        self.head = ColumnParallelLinear(args.hidden_size, args.vocab_size, dtype=torch.get_default_dtype())
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
            start_pos (int, optional): Starting position in the sequence for rotary embeddings. Defaults to 0.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        seqlen = tokens.size(1)
        h = self.embed(tokens)
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)[:, -1]
        logits = self.head(h)

        all_logits = [torch.empty_like(logits) for _ in range(world_size)]
        dist.all_gather(all_logits, logits)
        logits = torch.cat(all_logits, dim=-1)
        return logits



    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits