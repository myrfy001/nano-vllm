from  glob import glob
import math
from dataclasses import dataclass
import os
import time
from typing import Set, Tuple, Optional, Literal

from safetensors import safe_open
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.cuda.nvtx as nvtx

from nanovllm.layers.fused_moe import FusedMoE, awq_dequantize_triton
from nanovllm.layers.kernel import act_quant, weight_dequant, fp8_gemm
from nanovllm.utils.context import get_context
from nanovllm.layers.attention import store_kvcache
from nanovllm.config import PPNodeType
from nanovllm.utils.context import get_context, get_tp_rank, get_tp_world_size, get_tp_group
from nanovllm.layers.mla import mla_decode
from nanovllm.layers.gemv_awq import gemv

from safetensors.torch import save_file

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

def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor, tp_dim: int | None = None):
    # print(f'{param.dtype=}, {loaded_weight.dtype=}')
    param.data.copy_(loaded_weight)


def get_param_from_model(root: nn.Module, name: str) -> nn.Parameter:
    """
    torch's get_parameter doesn't support ModuleList or ModuleDict
    """
    parts = name.split(".")
    module = root
    for part in parts[:-1]:
        if isinstance(module, nn.ModuleList):
            if len(module) > int(part):
                module = module[int(part)]
            else:
                return None
        elif isinstance(module, nn.ModuleDict):
            if part in module:
                module = module[part]
            else:
                return None
        else:
            module = getattr(module, part)
    return getattr(module, parts[-1], None)


def load_read_value_to_parameters(name: str, target_tensor: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor):
    if qweight.dtype == torch.int32:
        dequant_tensor = awq_dequantize_triton(qweight, scale, zero)

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
        dist.all_reduce(y, group=get_tp_group())
        return y


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:

    if weight.dtype == torch.int32:
        weight = awq_dequantize_triton(weight.T, weight.scales.T, weight.zeros.T).T
        # print(f'REF: {weight.shape=}')
        # print(f'REF: {weight=}')
        return F.linear(x, weight, bias)
    if weight.element_size() > 1:
        return F.linear(x, weight, bias)
    elif gemm_impl == "bf16":
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)
    else:
        raise Exception("should not reach here")
        x, scale = act_quant(x, block_size)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        return y


    # try:
    #     assert weight.is_contiguous()
    #     ret = torch.empty(1, weight.size(0), device=x.device, dtype=x.dtype)
    #     x_shape = x.shape[:-1]
    #     tb = bias
    #     if bias is not None:
    #         tb = bias.reshape(bias.size(-1))
    #     tx = x.reshape(x.size(-1))
    #     gemv(tx, weight, tb, ret)
    #     ret = ret.reshape(*x_shape, weight.size(0))
        
    # except:
    #     import traceback
    #     traceback.print_exc()
    #     import pdb;pdb.set_trace()
    # return ret




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
        # print(f"{time.time()}, rank{get_tp_rank()}, in RowParallelLinear forward(), tp={world_size}")
        dist.all_reduce(y, group=get_tp_group())

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
    
    def load_from_safetensor(
        self,
        safetensor_file,
        original_weight_name: str,
        layer_id: int,
        tp_rank: int,
        already_loaded_set,
    ):
        
        # ['model', 'layers', '0', 'mlp', 'up_proj', 'qweight']
        # ['model', 'layers', '3', 'mlp', 'shared_experts', 'down_proj', 'qweight']
        original_weight_name_parts = original_weight_name.split('.')

        assert original_weight_name_parts[3] == 'mlp'
        assert int(original_weight_name_parts[2]) == layer_id
        assert (0 <= layer_id < 3  and len(original_weight_name_parts) == 6) or \
               (3 <= layer_id < 61 and len(original_weight_name_parts) == 7 and
                original_weight_name_parts[4] == 'shared_experts')

        load_dedup_key = '.'.join(original_weight_name_parts[:-1])
        if load_dedup_key in already_loaded_set:
            return

        original_qweight_name = ".".join(original_weight_name_parts[:-1] + ["qweight"])
        original_scale_name = ".".join(original_weight_name_parts[:-1] + ["scales"])
        original_zero_name = ".".join(original_weight_name_parts[:-1] + ["qzeros"])
        is_quantized = original_qweight_name in safetensor_file.keys() and \
                    original_scale_name in safetensor_file.keys() and \
                    original_zero_name in safetensor_file.keys()
        assert is_quantized

        loaded_tensor_weight = safetensor_file.get_tensor(original_qweight_name)
        loaded_tensor_scale = safetensor_file.get_tensor(original_scale_name)
        loaded_tensor_zero = safetensor_file.get_tensor(original_zero_name)

        dequant_tensor = awq_dequantize_triton(loaded_tensor_weight, loaded_tensor_scale, loaded_tensor_zero)

        NAME_TO_SHARDED_ID = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}
        SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}

        # ['up_proj', 'qweight']
        # ['down_proj', 'qweight']
        weight_name_parts = original_weight_name_parts[-2:]

        name = weight_name_parts[0]
        shard_id = NAME_TO_SHARDED_ID[name]
        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]

        param_path = f'{shard_id}.weight'
        # print(f"{param_path=}, {weight_name=}, {original_weight_name=}, {dequant_tensor.shape=}")
        expert_data = get_param_from_model(self, param_path)
        if expert_data is None:
            print(f"Warning: Parameter {param_path} not found in model.")
            return

        # tp=1 w1 [2048x7168] -> dim=0 -> Get 2048
        # tp=4 w1 [512x7168] -> dim=0 -> Get 512
        shard_size = expert_data.shape[shard_dim]
        assert dequant_tensor.shape[1 - shard_dim] == shard_size * world_size

        weight_shard = dequant_tensor.narrow(1 - shard_dim, shard_size * tp_rank, shard_size)
        expert_data.T.copy_(weight_shard)
        already_loaded_set.add(load_dedup_key)


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
        self.routed_scaling_factor = args.routed_scaling_factor
        self.n_routed_experts = args.n_routed_experts
        self.n_activated_experts = args.num_experts_per_tok

        # for fused MOE with TP, each card all experts' partial weight
        self.n_local_experts = args.n_routed_experts
        self.experts_start_idx = 0
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts

        self.gate = Linear(args.hidden_size, args.n_routed_experts, bias=True)
        self.gate.e_score_correction_bias = nn.Parameter(torch.empty([args.n_routed_experts], dtype=torch.float16), requires_grad=False)
        self.experts = FusedMoE(
            num_experts=args.n_routed_experts,
            top_k=args.num_experts_per_tok,
            hidden_size=args.hidden_size,
            intermediate_size=args.moe_intermediate_size ,
            params_dtype=Linear.dtype,
            renormalize=True,
            use_grouped_topk=True,
            num_expert_group=args.n_group,
            scoring_func=args.scoring_func,
            e_score_correction_bias=self.gate.e_score_correction_bias,
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

        z = self.shared_experts(x)
        weights = self.gate(x)
        y = self.experts(hidden_states=x, router_logits=weights) * self.routed_scaling_factor

        

        # if rank == 0 or rank == 3:
        #     print(f"{self.routed_scaling_factor=}")
        #     save_file({"x":x}, f"dumps/{time.time()}_rank{rank}-z100_moe_x.safetensor")
        #     save_file({"y":y}, f"dumps/{time.time()}_rank{rank}-z100_moe_y_before_allreduce.safetensor")
        #     save_file({"z":z}, f"dumps/{time.time()}_rank{rank}-z100_moe_z.safetensor")
        #     save_file({"weights":weights}, f"dumps/{time.time()}_rank{rank}-z100_moe_weights.safetensor")

        dist.all_reduce(y, group=get_tp_group())

        # if rank == 0 or rank == 3:
        #     save_file({"y":y}, f"dumps/{time.time()}_rank{rank}-z100_moe_y_after_allreduce.safetensor")

        return (y + z).view(shape)
    
    def _load_w13(self, expert_data: torch.Tensor, shard_dim: int,
                  shard_id: str, loaded_weight: torch.Tensor, tp_rank: int):
        # Index the loaded weight for tp sharding.
        # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
        shard_size = expert_data.shape[shard_dim] // 2
        loaded_weight = loaded_weight.narrow(1 - shard_dim, shard_size * tp_rank, shard_size)
        # Narrow parameter and load.
        # w1, gate_proj: Load into first logical weight of w13.
        if shard_id == "w1":
            expert_data = expert_data.narrow(shard_dim, 0, shard_size)
        # w3, up_proj: Load into second logical weight of w13.
        else:
            assert shard_id == "w3"
            expert_data = expert_data.narrow(shard_dim, shard_size, shard_size)
        # print(f'{expert_data.shape=}, {loaded_weight.shape=}')
        expert_data.T.copy_(loaded_weight)

    def _load_w2(self,
                 expert_data: torch.Tensor,
                 shard_dim: int,
                 loaded_weight: torch.Tensor,
                 tp_rank: int):

        # Index the loaded weight for tp sharding.
        # down_proj: "RowParallel" so tp sharding on input_dim
        # Narrow parameter and load.
        shard_size = expert_data.shape[shard_dim]
        loaded_weight = loaded_weight.narrow(1 - shard_dim,
                                                shard_size * tp_rank,
                                                shard_size)
        # w2, down_proj: Load into only logical weight of w2.
        # print(f'{expert_data.shape=}, {loaded_weight.shape=}')
        expert_data.T.copy_(loaded_weight)

    def load_weight(self, path: str, layer_id: int):
        assert 3 <= layer_id < 61
        packed_modules_mapping = getattr(self, "packed_modules_mapping", {})
        already_loaded_set = set()

        for file in sorted(glob(os.path.join(path, "*.safetensors"))):
            # print(f'Loading MLA weight from {file}')
            with safe_open(file, "pt", "cpu") as safetensor_file:
                for weight_name in safetensor_file.keys():
                    if not (".layers." in weight_name and ".mlp." in weight_name):
                        continue
                    layer = int(weight_name.split('.')[2])
                    if layer != layer_id:
                        continue
                    self.load_from_safetensor(safetensor_file, weight_name, layer_id, rank, already_loaded_set)

    def load_from_safetensor(
        self,
        safetensor_file,
        original_weight_name: str,
        layer_id: int,
        tp_rank: int,
        already_loaded_set: Set[str],
    ):
        # ['model', 'layers', '3', 'mlp', 'experts', '0', 'down_proj', 'qweight']
        original_weight_name_parts = original_weight_name.split('.')

        assert int(original_weight_name_parts[2]) == layer_id
        assert original_weight_name_parts[3] == 'mlp'

        load_dedup_key = '.'.join(original_weight_name_parts[:-1])
        if load_dedup_key in already_loaded_set:
            # print(f'Skip {original_weight_name=}')
            return
        # print(f'Load {original_weight_name=}')

        # ['experts', '0', 'down_proj', 'qweight']
        weight_name_parts = original_weight_name_parts[4:]
        weight_name = '.'.join(weight_name_parts)
        assert weight_name_parts[0] in ['gate', 'shared_experts', 'experts']

        NAME_TO_SHARDED_ID = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}
        SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}

        if weight_name_parts[0] == 'experts':
            expert_id = int(weight_name_parts[1])
            assert 0 <= expert_id < 256

            # currently map to same id
            map_global_expert_id_to_local_expert_id = lambda id: id
            expert_id = map_global_expert_id_to_local_expert_id(expert_id)

            name = weight_name_parts[2]
            shard_id = NAME_TO_SHARDED_ID[name]
            shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]

            if shard_id in ["w1", "w3"]:
                w13_qweight = get_param_from_model(self, f"experts.w13_qweight")[expert_id]
                w13_scales = get_param_from_model(self, f"experts.w13_scales")[expert_id]
                w13_qzeros = get_param_from_model(self, f"experts.w13_qzeros")[expert_id]
                if w13_qweight is None:
                    print(f"Warning: Parameter layers.{layer_id}.ffn.experts.w13_qweight not found in model.")
                    return
                original_weight_name = f"model.layers.{layer_id}.mlp.experts.{expert_id}.{name}.qweight"
                loaded_tensor = safetensor_file.get_tensor(original_weight_name)
                self._load_w13(w13_qweight, shard_dim, shard_id, loaded_tensor, tp_rank)

                original_weight_name = f"model.layers.{layer_id}.mlp.experts.{expert_id}.{name}.scales"
                loaded_tensor = safetensor_file.get_tensor(original_weight_name)
                self._load_w13(w13_scales, shard_dim, shard_id, loaded_tensor, tp_rank)

                original_weight_name = f"model.layers.{layer_id}.mlp.experts.{expert_id}.{name}.qzeros"
                loaded_tensor = safetensor_file.get_tensor(original_weight_name)
                self._load_w13(w13_qzeros, shard_dim, shard_id, loaded_tensor, tp_rank)
            else:
                assert shard_id == 'w2'

                w2_qweight = get_param_from_model(self, f"experts.w2_qweight")[expert_id]
                w2_scales = get_param_from_model(self, f"experts.w2_scales")[expert_id]
                w2_qzeros = get_param_from_model(self, f"experts.w2_qzeros")[expert_id]
                original_weight_name = f"model.layers.{layer_id}.mlp.experts.{expert_id}.{name}.qweight"
                loaded_tensor = safetensor_file.get_tensor(original_weight_name)
                self._load_w2(w2_qweight, shard_dim, loaded_tensor, tp_rank)

                original_weight_name = f"model.layers.{layer_id}.mlp.experts.{expert_id}.{name}.scales"
                loaded_tensor = safetensor_file.get_tensor(original_weight_name)
                self._load_w2(w2_scales, shard_dim, loaded_tensor, tp_rank)

                original_weight_name = f"model.layers.{layer_id}.mlp.experts.{expert_id}.{name}.qzeros"
                loaded_tensor = safetensor_file.get_tensor(original_weight_name)
                self._load_w2(w2_qzeros, shard_dim, loaded_tensor, tp_rank)
            already_loaded_set.add(load_dedup_key)
        elif weight_name_parts[0] == 'shared_experts':
            self.shared_experts.load_from_safetensor(
                safetensor_file,
                original_weight_name,
                layer_id,
                tp_rank,
                already_loaded_set,
            )
        elif weight_name_parts[0] == 'gate':
            param = get_param_from_model(self, weight_name)
            if param is not None:
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, safetensor_file.get_tensor(original_weight_name), None)
            else:
                raise Exception(f"Warning: Parameter {weight_name} not found in model.")


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
        self.q_a_proj = Linear(self.dim, self.q_lora_rank)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank)
        # [1536x(128 * (128 + 64))]
        self.q_b_proj = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)

        # [7168x(512 + 64)]
        self.kv_a_proj_with_mqa = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank)
        # [512x(128 * (128 + 128))]
        self.kv_b_proj = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        # [(128 * 128)x7168]
        self.o_proj = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5

        # print(f"args={args}, {args.max_position_embeddings=}, {args.original_seq_len=}, {args.mscale=}, {args.rope_factor=}, {self.softmax_scale=}")
        if args.max_position_embeddings > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale
        

        self.cache_size = args.max_batch_size * args.max_position_embeddings
        self.page_size = 1
        self.register_buffer("kv_c_and_k_pe_cache", torch.empty(self.cache_size // self.page_size, self.page_size, 1, self.kv_lora_rank + self.qk_rope_head_dim), persistent=False)
        # kv_cache, pe_cache = self.kv_c_and_k_pe_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        # self.register_buffer("kv_cache", kv_cache.view(args.max_batch_size, args.max_position_embeddings, self.kv_lora_rank), persistent=False)
        # self.register_buffer("pe_cache", pe_cache.view(args.max_batch_size, args.max_position_embeddings, self.qk_rope_head_dim), persistent=False)

        kv_b_proj_weight = self.kv_b_proj.weight.T

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
        return self.o_proj(x)

    # Return `ql_nope`, `q_pe`
    def _q_proj_and_k_up_proj(self, x):
        q = self.q_b_proj(x) # [1x(128 * (128 + 64))]
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

    def forward(self, x: torch.Tensor, positions: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None):
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
        # if rank == 0 or rank == 3:
        #     # print(f"MLA input x = {x}", flush=True)
        #     save_file({"x":x}, f"dumps/{time.time()}_rank{rank}-mla_input_x.safetensor")

        bsz, seqlen, _ = x.size()
        # x = x.squeeze(0) # [1x7168]

        # Get Q, K, V
        ckq = self.q_a_proj(x) # [1x1536]
        kv = self.kv_a_proj_with_mqa(x) # [1x576]

        # if rank == 0 or rank == 3:
        #     # print(f"MLA ckq= {ckq}, kv={kv}", flush=True)
        #     save_file({"ckq":ckq}, f"dumps/{time.time()}_rank{rank}-mla_ckq.safetensor")
        #     save_file({"kv":kv}, f"dumps/{time.time()}_rank{rank}-mla_kv.safetensor")
        kv_c, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        # Normalize Q and KV
        q_c = self.q_a_layernorm(ckq) # [1x1536]
        kv_c_normed = self.kv_a_layernorm(kv_c) # [1x512]

        q_nope, q_pe = self._q_proj_and_k_up_proj(q_c)

        # apply rope
        q_pe = apply_rotary_emb(q_pe.unsqueeze(0), freqs_cis)
        q_pe = q_pe.squeeze(0)
        # if rank == 0:
        #     print(f"freqs_cis = {freqs_cis}")

        # if rank == 0 or rank == 3:
        #     save_file({"k_pe":k_pe}, f"dumps/{time.time()}_rank{rank}-mla_k_pe_before_rope.safetensor")

        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        # if rank == 0 or rank == 3:
        #     save_file({"k_pe":k_pe}, f"dumps/{time.time()}_rank{rank}-mla_k_pe_after_rope.safetensor")
        #     save_file({"kv_c_normed":kv_c_normed}, f"dumps/{time.time()}_rank{rank}-mla_kv_c_normed.safetensor")

        # print(f'MLA: {q_nope=}, {q_pe=}')
        # update kvcache
        # print(f'MLA: {kv_c_normed=}, {k_pe=}')
        kv_c_and_k_pe = torch.cat([kv_c_normed, k_pe.squeeze(2)], dim=-1)  # [1, 1, 512 + 64]
        assert self.page_size == 1

        context = get_context()
        # if rank == 0 or rank == 3:
        #     # print(f"in decode forward, before update kvcache. self.kv_c_and_k_pe_cache={self.kv_c_and_k_pe_cache}")
        #     # print(f"in decode forward, before update kvcache. context.slot_mapping={context.slot_mapping}")
        #     # print(f"MLA kv_c_and_k_pe = {kv_c_and_k_pe}", flush=True)
        #     save_file({"kv_c_and_k_pe":kv_c_and_k_pe}, f"dumps/{time.time()}_rank{rank}-mla_kv_c_and_k_pe.safetensor")
        #     save_file({"positions":positions}, f"dumps/{time.time()}_rank{rank}-mla_positions.safetensor")
            
        

        
        store_kvcache(kv_c_and_k_pe, self.kv_c_and_k_pe_cache, context.slot_mapping)

        # if rank == 0:
        #     print(f"in decode forward, after update kvcache. self.kv_c_and_k_pe_cache={self.kv_c_and_k_pe_cache}", flush=True)

        # Attention
        q = torch.cat([q_nope, q_pe], dim=-1)
        o = torch.empty([bsz, self.n_local_heads, self.kv_lora_rank], dtype=q.dtype, device=q.device)

        kv_c_and_k_pe_cache = self.kv_c_and_k_pe_cache
        kv_c_cache = kv_c_and_k_pe_cache[..., :self.kv_lora_rank]

        config = {
            'SPLIT_K': 8,
        }

        # if rank == 0 or rank == 3:
        #     # print(f"MLA q = {q}, req_to_tokens={context.block_tables}, b_seq_len={context.context_lens}", flush=True)

        #     save_file({"q":q}, f"dumps/{time.time()}_rank{rank}-mla_q.safetensor")
        #     save_file({"kv_c_and_k_pe_cache":kv_c_and_k_pe_cache}, f"dumps/{time.time()}_rank{rank}-mla_kv_c_and_k_pe_cache.safetensor")
        #     save_file({"kv_c_cache":kv_c_cache.contiguous()}, f"dumps/{time.time()}_rank{rank}-mla_kv_c_cache.safetensor")
        #     save_file({"req_to_tokens":context.block_tables}, f"dumps/{time.time()}_rank{rank}-mla_req_to_tokens.safetensor")
        #     save_file({"b_seq_len":context.context_lens}, f"dumps/{time.time()}_rank{rank}-mla_b_seq_len.safetensor")

        mla_decode(
            q,
            kv_c_and_k_pe_cache,
            kv_c_cache,
            o,
            context.block_tables,
            context.context_lens,
            sm_scale=self.softmax_scale,
            config=config,
        )

        # if rank == 0 or rank == 3:
        #     # print(f"MLA o = {o}, sm_scale={self.softmax_scale}", flush=True)
        #     save_file({"o":o}, f"dumps/{time.time()}_rank{rank}-mla_o.safetensor")

        return self._v_up_proj_and_o_proj(o)
    
    def load_weight(self, path, layer_id):
        packed_modules_mapping = getattr(self, "packed_modules_mapping", {})
        already_loaded_set = set()

        for file in sorted(glob(os.path.join(path, "*.safetensors"))):
            # print(f'Loading MLA weight from {file}')
            with safe_open(file, "pt", "cpu") as safetensor_file:
                for original_weight_name in safetensor_file.keys():
                    original_weight_name_parts = original_weight_name.split(".")

                    # MLA layer
                    if not 'self_attn' in original_weight_name:
                        continue
                    cur_layer_id = int(original_weight_name_parts[2])
                    if cur_layer_id != layer_id:
                        continue

                    self.load_from_safetensor(
                        safetensor_file,
                        original_weight_name,
                        layer_id,
                        rank,
                        already_loaded_set,
                    )

    def load_from_safetensor(
        self,
        safetensor_file,
        original_weight_name: str,
        layer_id: int,
        tp_rank: int,
        already_loaded_set: Set[str],
    ):
        # ['model', 'layers', '1', 'self_attn', 'kv_b_proj', 'qweight']
        # ['model', 'layers', '1', 'self_attn', 'kv_a_proj_with_mqa', 'weight']
        original_weight_name_parts = original_weight_name.split('.')

        assert original_weight_name_parts[3] == 'self_attn'
        assert int(original_weight_name_parts[2]) == layer_id

        load_dedup_key = '.'.join(original_weight_name_parts[:-1])
        if load_dedup_key in already_loaded_set:
            return

        # ['kv_b_proj', 'qweight']
        # ['kv_a_proj_with_mqa', 'weight']
        weight_name_parts = original_weight_name_parts[4:]
        weight_name = '.'.join(weight_name_parts)
        assert weight_name_parts[0] in [
            'q_a_proj',
            'q_a_layernorm',
            'q_b_proj',
            'kv_a_proj_with_mqa',
            'kv_a_layernorm',
            'kv_b_proj',
            'o_proj',
        ]

        original_qweight_name = ".".join(original_weight_name_parts[:-1] + ["qweight"])
        original_scale_name = ".".join(original_weight_name_parts[:-1] + ["scales"])
        original_zero_name = ".".join(original_weight_name_parts[:-1] + ["qzeros"])
        is_quantized = original_qweight_name in safetensor_file.keys() and \
                    original_scale_name in safetensor_file.keys() and \
                    original_zero_name in safetensor_file.keys()

        if is_quantized:
            assert weight_name_parts[0] in ['q_a_proj', 'q_b_proj', 'kv_b_proj', 'o_proj']
            loaded_tensor_weight = safetensor_file.get_tensor(original_qweight_name)
            loaded_tensor_scale = safetensor_file.get_tensor(original_scale_name)
            loaded_tensor_zero = safetensor_file.get_tensor(original_zero_name)

            dequant_tensor = awq_dequantize_triton(loaded_tensor_weight, loaded_tensor_scale, loaded_tensor_zero)

            SHARD_ID_TO_SHARDED_DIM = {'q_a_proj': -1, 'q_b_proj': 0, 'kv_b_proj': 0, 'o_proj': 1}

            shard_id = weight_name_parts[0]
            shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]

            param_path = f'{shard_id}.weight'
            # print(f"{param_path=}, {weight_name=}, {original_weight_name=}, {dequant_tensor.shape=}")
            weight = get_param_from_model(self, param_path)
            if weight is None:
                print(f"Warning: Parameter {param_path} not found in model.")
                return

            if shard_dim != -1:
                shard_size = weight.shape[shard_dim]
                assert dequant_tensor.shape[1 - shard_dim] == shard_size * world_size
                dequant_tensor = dequant_tensor.narrow(1 - shard_dim, shard_size * tp_rank, shard_size)

            weight.T.copy_(dequant_tensor)
        else:
            assert weight_name_parts[0] in ['q_a_layernorm', 'kv_a_proj_with_mqa', 'kv_a_layernorm']
            param = get_param_from_model(self, weight_name)
            if param is not None:
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, safetensor_file.get_tensor(original_weight_name), None)
            else:
                raise Exception(f"Warning: Parameter {weight_name} not found in model.")

        already_loaded_set.add(load_dedup_key)


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
        self.self_attn = Z100_MLA(args, nanovllm_config)
        self.mlp = MLP(args.hidden_size, args.intermediate_size) if layer_id < args.first_k_dense_replace else Z100_MoE(args, nanovllm_config)
        self.input_layernorm = RMSNorm(args.hidden_size)
        self.post_attention_layernorm = RMSNorm(args.hidden_size)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
            
        context = get_context()
        if context.is_prefill:
            return x
        else:
            x = x + self.self_attn(self.input_layernorm(x * 0.2), positions, freqs_cis, mask)
            # if rank == 0 or rank == 3:
            #     save_file({"x":x}, f"dumps/{time.time()}_rank{rank}-z100_block_middle_x.safetensor")
            x = x + self.mlp(self.post_attention_layernorm(x * 0.2))
            # if rank == 0 or rank == 3:
            #     save_file({"x":x}, f"dumps/{time.time()}_rank{rank}-z100_block_final_x.safetensor")
        return x
    

    def load_weight(self, path: str, layer_id: int):
        assert 0 <= layer_id < 61
        already_loaded_set = set()

        for file in sorted(glob(os.path.join(path, "*.safetensors"))):
            # print(f'Loading MLA weight from {file}')
            with safe_open(file, "pt", "cpu") as safetensor_file:
                for original_weight_name in safetensor_file.keys():
                    # ['model', 'layers', '6', 'self_attn', 'kv_b_proj', 'qweight']
                    # ['model', 'layers', '6', 'mlp', 'experts', '0', 'down_proj', 'qweight']
                    original_weight_name_parts = original_weight_name.split(".")

                    # layer
                    if not "layers" in original_weight_name_parts:
                        continue

                    cur_layer_id = int(original_weight_name_parts[2])
                    if cur_layer_id != layer_id:
                        continue

                    # ['self_attn', 'kv_b_proj', 'qweight']
                    # ['mlp', 'experts', '0', 'down_proj', 'qweight']
                    weight_name_parts = original_weight_name_parts[3:]
                    weight_name = '.'.join(weight_name_parts)
                    assert weight_name_parts[0] in ['input_layernorm', 'self_attn', 'post_attention_layernorm', 'mlp']

                    if weight_name_parts[0] == 'self_attn':
                        self.self_attn.load_from_safetensor(
                            safetensor_file,
                            original_weight_name,
                            layer_id,
                            rank,
                            already_loaded_set,
                        )
                    elif weight_name_parts[0] == 'mlp':
                        self.mlp.load_from_safetensor(
                            safetensor_file,
                            original_weight_name,
                            layer_id,
                            rank,
                            already_loaded_set,
                        )
                    else:
                        assert weight_name_parts[0] in ['input_layernorm', 'post_attention_layernorm']
                        param = get_param_from_model(self, weight_name)
                        if param is not None:
                            weight_loader = getattr(param, "weight_loader", default_weight_loader)
                            weight_loader(param, safetensor_file.get_tensor(original_weight_name), None)
                        else:
                            raise Exception(f"Warning: Parameter {weight_name} not found in model.")


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

    def __init__(self, args, nanovllm_config):
        """
        Initializes the Transformer model.

        Args:
            args (ModelArgs): Model arguments containing transformer parameters.
        """
        # hack: inject fields to args to make it visiable to all other parts of the model
        # TODO:FIXME: all the following param should match with prefill. so, prefill should read from the same config.
        # but it seems not now. 
        args.original_seq_len = 4096
        args.max_position_embeddings = 4096  
        args.mscale = 1.0
        args.rope_factor = 40.0
        args.beta_fast = 32
        args.beta_slow = 1
        args.rope_theta = 10000.0
        args.max_batch_size = 1
        self.args = args

        pp_start_layer_id, pp_end_layer_id, pp_node_type = nanovllm_config.pp_schema
        self.pp_start_layer_id = pp_start_layer_id

        global world_size, rank
        world_size = nanovllm_config.tensor_parallel_size
        rank = nanovllm_config.local_rank

        # assert dist.get_world_size == torch.cuda.device_count(), f"World size ({dist.get_world_size()}) must match the number of GPUs ({torch.cuda.device_count()})" 

        Linear.dtype = torch.float16
        super().__init__()
        self.nanovllm_config = nanovllm_config
        self.max_seq_len = args.max_position_embeddings

        self.layers = torch.nn.ModuleList()

        for layer_id in range(pp_start_layer_id, pp_end_layer_id):
            self.layers.append(Z100_Block(layer_id, args, nanovllm_config))
        self.pp_node_type = pp_node_type

        
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

        # print(f"freqs_cis={self.freqs_cis}")

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, positions: torch.Tensor):
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
            start_pos (int, optional): Starting position in the sequence for rotary embeddings. Defaults to 0.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        raise NotImplementedError



    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return hidden_states
    

    def load_weight(self, path: str = None, start_layer=0, end_layer=0):
        self.path = path

        for idx, layer in enumerate(self.layers):
            print(f"loading layer weights: {start_layer+idx}")
            layer.load_weight(path, start_layer+idx)

        for file in sorted(glob(os.path.join(path, "*.safetensors"))):
            # print(f'Loading MLA weight from {file}')
            with safe_open(file, "pt", "cpu") as safetensor_file:
                for original_weight_name in safetensor_file.keys():
                    # ['model', 'embed_tokens', 'weight']
                    # ['model', 'lm_head', 'weight']
                    original_weight_name_parts = original_weight_name.split(".")

                    # Do not handle layer here
                    if "layers" in original_weight_name_parts:
                        continue

                    weight_name_parts = original_weight_name_parts[-2:]
                    weight_name = '.'.join(weight_name_parts)
                    assert weight_name_parts[0] in ['embed_tokens', 'lm_head', 'norm'], f'{original_weight_name=}'

                    if start_layer != 0 and weight_name_parts[0] in ['embed_tokens']:
                        continue
                    
                    if end_layer != 60 and weight_name_parts[0] in ['lm_head', 'norm']:
                        continue

                    SHARD_ID_TO_SHARDED_DIM = {'embed_tokens': 0, 'lm_head': 0, 'norm': -1}

                    shard_id = weight_name_parts[0]
                    shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]

                    weight = get_param_from_model(self, weight_name)
                    if weight is None:
                        print(f"Warning: Parameter {weight_name} not found in model.")
                        return

                    loaded_weight = safetensor_file.get_tensor(original_weight_name)
                    # print(f'{loaded_weight.shape=}')

                    if shard_dim != -1:
                        shard_size = weight.shape[shard_dim]
                        assert loaded_weight.shape[shard_dim] == shard_size * world_size
                        loaded_weight = loaded_weight.narrow(shard_dim, shard_size * rank, shard_size)

                    weight.copy_(loaded_weight)
    

class DeepseekV3ForCausalLLMFirst(DeepseekV3ForCausalLLM):
    def __init__(self, args, nanovllm_config):
        super().__init__(args, nanovllm_config)
        self.embed_tokens = ParallelEmbedding(args.vocab_size, args.hidden_size)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, positions: torch.Tensor):

        context = get_context()

        tokens = tokens.unsqueeze(0) # FIXME: nano-vllm use cumulated input, but our's doesn't support it yet. for now, batch size is 1, so we can add a dim to walk around.

        # print(f"{time.time()}, DSV3 forward rank={rank}, DeepseekV3ForCausalLLMFirst tokens={tokens}, positions={positions}")

        seqlen = tokens.size(1)
        mask = None
        if context.is_prefill:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
        freqs_cis = self.freqs_cis[positions]

        h = self.embed_tokens(tokens)

        for layer_idx, layer in enumerate(self.layers):
            # if rank == 0 or rank == 3:
            #     print(f"layer{self.pp_start_layer_id + layer_idx} input h at top most={h}", flush=True)
            # if self.pp_node_type == PPNodeType.PPNodeFirst:
            #     save_file({"h":h}, f"dumps/{time.time()}_rank{rank}-layer{self.pp_start_layer_id + layer_idx}-input_h_at_top_most.safetensor")

            h = layer(h, positions, freqs_cis, mask)
            # if rank == 0:
            #     print(f'layer{self.pp_start_layer_id + layer_idx} output h at top most={h}', flush=True)
            # if self.pp_node_type == PPNodeType.PPNodeFirst:
            #     save_file({"h":h}, f"dumps/{time.time()}_rank{rank}-layer{self.pp_start_layer_id + layer_idx}-output_h_at_top_most.safetensor")

        h = h.squeeze(0) # FIXME: nano-vllm use cumulated input, but our's doesn't support it yet. for now, batch size is 1, so we can remove a dim to walk around.
        return h

class DeepseekV3ForCausalLLMMiddle(DeepseekV3ForCausalLLM):
    def __init__(self, args, nanovllm_config):
        super().__init__(args, nanovllm_config)

    @torch.inference_mode()
    def forward(self, h: torch.Tensor, positions: torch.Tensor):
        h = h.unsqueeze(0) # FIXME: nano-vllm use cumulated input, but our's doesn't support it yet. for now, batch size is 1, so we can add a dim to walk around.

        # print(f"{time.time()}, rank={get_tp_rank()}, DeepseekV3ForCausalLLMMiddle forward() positions={positions}, h={h}, {h.size()}")
        assert h.dim() == 3  # [bs, seq_len, hidden]
        assert h.size()[0] == 1 # TODO: FIXME: only support bs=1

        context = get_context()
        seqlen = h.size(0)
        mask = None
        if context.is_prefill:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=h.device).triu_(1)

        # print(f"{time.time()}, rank={get_tp_rank()}, positions={positions}, self.freqs_cis={self.freqs_cis}, {positions.device}, {self.freqs_cis.device}")

        freqs_cis = self.freqs_cis[positions]
        for layer_idx, layer in enumerate(self.layers):
            # if rank == 0 or rank == 3:
            #     save_file({"h":h}, f"dumps/{time.time()}_rank{rank}-layer{self.pp_start_layer_id + layer_idx}-input_h_at_top_most.safetensor")
            h = layer(h, positions, freqs_cis, mask)

            # if rank == 0 or rank == 3:
            #     save_file({"h":h}, f"dumps/{time.time()}_rank{rank}-layer{self.pp_start_layer_id + layer_idx}-output_h_at_top_most.safetensor")
        
        h = h.squeeze(0) # FIXME: nano-vllm use cumulated input, but our's doesn't support it yet. for now, batch size is 1, so we can remove a dim to walk around.
        return h
        

class DeepseekV3ForCausalLLMLast(DeepseekV3ForCausalLLM):
    def __init__(self, args, nanovllm_config):
        super().__init__(args, nanovllm_config)
        self.norm = RMSNorm(args.hidden_size)
        self.lm_head = ColumnParallelLinear(args.hidden_size, args.vocab_size, dtype=torch.get_default_dtype())

    @torch.inference_mode()
    def forward(self, h: torch.Tensor, positions: torch.Tensor):
        h = h.unsqueeze(0) # FIXME: nano-vllm use cumulated input, but our's doesn't support it yet. for now, batch size is 1, so we can add a dim to walk around.

        context = get_context()
        seqlen = h.size(0)
        mask = None
        if context.is_prefill:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=h.device).triu_(1)

        freqs_cis = self.freqs_cis[positions]
        
        print(f"{time.time()}, rank={get_tp_rank()}, DeepseekV3ForCausalLLMLast's forward before layers", flush=True)

        for layer_idx, layer in enumerate(self.layers):
            # if rank == 0 or rank == 3:
            #     save_file({"h":h}, f"dumps/{time.time()}_rank{rank}-layer{self.pp_start_layer_id + layer_idx}-input_h_at_top_most.safetensor")
            h = layer(h, positions, freqs_cis, mask)

            # if rank == 0 or rank == 3:
            #     save_file({"h":h}, f"dumps/{time.time()}_rank{rank}-layer{self.pp_start_layer_id + layer_idx}-output_h_at_top_most.safetensor")

        print(f"{time.time()}, rank={get_tp_rank()}, DeepseekV3ForCausalLLMLast's forward before norm", flush=True)

        h = self.norm(h)[:, -1]

        # if rank == 0 or rank == 3:
        #     save_file({"h":h}, f"dumps/{time.time()}_rank{rank}-layer-last-output_h_after_norm_at_top_most.safetensor")

        if context.is_prefill:
            logits = torch.randn(list(h.shape[:-1]) + [self.args.vocab_size // world_size], device=h.device, dtype=h.dtype)
        else:
            logits = self.lm_head(h)

        # if rank == 0 or rank == 3:
        #     save_file({"logits":logits}, f"dumps/{time.time()}_rank{rank}-layer-last-output_logits.safetensor")

        all_logits = [torch.empty_like(logits) for _ in range(world_size)]
        dist.all_gather(all_logits, logits, group=get_tp_group())

        logits = torch.cat(all_logits, dim=-1)
        # if rank == 0 or rank == 3:
        #     save_file({"all_logits":logits}, f"dumps/{time.time()}_rank{rank}-layer-last-output_all_logits_after_gather.safetensor")


        logits = logits.squeeze(0) # FIXME: nano-vllm use cumulated input, but our's doesn't support it yet. for now, batch size is 1, so we can remove a dim to walk around.
        return logits
