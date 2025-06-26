import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open, SafetensorError
from nanovllm.layers.fused_moe import awq_dequantize_triton


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor, tp_dim: int | None = None):
    param.data.copy_(loaded_weight)


def get_param_from_model(root: nn.Module, name: str, pp_layer_offset: int) -> nn.Parameter:
    """
    torch's get_parameter doesn't support ModuleList or ModuleDict
    """
    parts = name.split(".")
    module = root
    for part in parts[:-1]:
        if isinstance(module, nn.ModuleList):
            adjusted_part_id = int(part) - pp_layer_offset
            if len(module) > adjusted_part_id:
                module = module[adjusted_part_id]
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
    

# some layer would cross two file.
def load_tensor_from_file_helper(f, next_file, name):
    try:
        loaded_tensor = f.get_tensor(name)
    except SafetensorError:
        with safe_open(next_file, "pt", "cpu") as f1:
            loaded_tensor = f1.get_tensor(name)
        
    return loaded_tensor



def load_model(model: nn.Module, path: str, tp_size: int, local_rank: int, start_layer: int, end_layer: int):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    already_loaded_set = set()

    file_list_a = sorted(glob(os.path.join(path, "*.safetensors")))
    file_list_b = sorted(glob(os.path.join(path, "*.safetensors")))
    del file_list_b[0]
    file_list_b.append("")

    for file, next_file in zip(file_list_a, file_list_b):
        print(f"Loading weights from {file}")
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # if ("embed_tokens" in weight_name):
                #     import pdb; pdb.set_trace()
            
                # print(f"origin weight_name = {weight_name}")
                original_weight_name = weight_name
                original_weight_name_parts = original_weight_name.split(".")

                layer_idx = None
                if ".layers." in original_weight_name:
                    layer_idx = int(original_weight_name_parts[2])
                    if not (start_layer <= layer_idx < end_layer):
                        continue
                else:
                    if ("model.lm_head" in original_weight_name or  "model.norm" in original_weight_name) and  end_layer != 60:
                        # some weights only belong to the last pp rank
                        continue
                    elif ("embed_tokens" in original_weight_name) and start_layer != 0:
                        # some weights only belong to the first pp rank
                        continue

                
                if weight_name.startswith("model."):
                    weight_name = weight_name[6:]
                
                weight_name = weight_name.replace("self_attn", "attn")
                weight_name = weight_name.replace("mlp", "ffn")
                weight_name = weight_name.replace("weight_scale_inv", "scale")
                weight_name = weight_name.replace("e_score_correction_bias", "bias")

                weight_name_parts = weight_name.split(".")
                tp_split_dim = None
                for i in range(len(weight_name_parts)):
                    if weight_name_parts[i] in packed_modules_mapping:
                        weight_name_parts[i], tp_split_dim = packed_modules_mapping[weight_name_parts[i]]

                weight_name = ".".join(weight_name_parts)

                # merge MOE TP params
                if ".experts." in weight_name:

                    # weight name looks like "layers.3.ffn.experts.0.w2.qweight"
                    
                    expert_idx = int(weight_name_parts[4])
                    expert_param_name = weight_name_parts[5]

                    load_dedup_key = f"{layer_idx}.{expert_idx}"
                    if load_dedup_key in already_loaded_set:
                        continue
                    
                    
                    # load w1 and w3
                    fused_moe_param_weight = get_param_from_model(model, f"layers.{layer_idx}.ffn.experts.w13_qweight", start_layer)
                    fused_moe_param_scale = get_param_from_model(model, f"layers.{layer_idx}.ffn.experts.w13_scales", start_layer)
                    fused_moe_param_zero = get_param_from_model(model, f"layers.{layer_idx}.ffn.experts.w13_qzeros", start_layer)
                    if fused_moe_param_weight is None:
                        print(f"Warning: Parameter layers.{layer_idx}.ffn.experts.w13_qweight not found in model.")
                        continue

                    intermediate_size_per_partition = fused_moe_param_weight.intermediate_size_per_partition

                    # handle w1

                    original_weight_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.qweight"
                    loaded_tensor = load_tensor_from_file_helper(f, next_file, original_weight_name)
                    fused_moe_param_weight.data[expert_idx, :intermediate_size_per_partition // 8, :].copy_(
                        loaded_tensor.narrow(1, local_rank * intermediate_size_per_partition // 8, intermediate_size_per_partition // 8).t())

                    original_weight_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.scales"
                    loaded_tensor = load_tensor_from_file_helper(f, next_file, original_weight_name)
                    fused_moe_param_scale.data[expert_idx, :intermediate_size_per_partition, :].copy_(
                        loaded_tensor.narrow(1, local_rank * intermediate_size_per_partition, intermediate_size_per_partition).t())

                    original_weight_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.qzeros"
                    loaded_tensor = load_tensor_from_file_helper(f, next_file, original_weight_name)
                    fused_moe_param_zero.data[expert_idx, :intermediate_size_per_partition // 8, :].copy_(
                        loaded_tensor.narrow(1, local_rank * intermediate_size_per_partition // 8, intermediate_size_per_partition // 8).t())

                    
                    # handle w3
                    original_weight_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.qweight"
                    loaded_tensor = load_tensor_from_file_helper(f, next_file, original_weight_name)
                    fused_moe_param_weight.data[expert_idx, intermediate_size_per_partition // 8:, :].copy_(
                        loaded_tensor.narrow(1, local_rank * intermediate_size_per_partition // 8, intermediate_size_per_partition // 8).t())

                    original_weight_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.scales"
                    loaded_tensor = load_tensor_from_file_helper(f, next_file, original_weight_name)
                    fused_moe_param_scale.data[expert_idx, intermediate_size_per_partition:, :].copy_(
                        loaded_tensor.narrow(1, local_rank * intermediate_size_per_partition, intermediate_size_per_partition).t())

                    original_weight_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.qzeros"
                    loaded_tensor = load_tensor_from_file_helper(f, next_file, original_weight_name)
                    fused_moe_param_zero.data[expert_idx, intermediate_size_per_partition // 8:, :].copy_(
                        loaded_tensor.narrow(1, local_rank * intermediate_size_per_partition // 8, intermediate_size_per_partition // 8).t())

                    # load w2
                    fused_moe_param_weight = get_param_from_model(model, f"layers.{layer_idx}.ffn.experts.w2_qweight", start_layer)
                    fused_moe_param_scale = get_param_from_model(model, f"layers.{layer_idx}.ffn.experts.w2_scales", start_layer)
                    fused_moe_param_zero = get_param_from_model(model, f"layers.{layer_idx}.ffn.experts.w2_qzeros", start_layer)

                    original_weight_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.qweight"
                    loaded_tensor = load_tensor_from_file_helper(f, next_file, original_weight_name)
                    fused_moe_param_weight.data[expert_idx, :, :].copy_(
                        loaded_tensor.narrow(0, local_rank * intermediate_size_per_partition, intermediate_size_per_partition).t())

                    original_weight_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.scales"
                    loaded_tensor = load_tensor_from_file_helper(f, next_file, original_weight_name)
                    shard_size = loaded_tensor.size(0) // tp_size
                    fused_moe_param_scale.data[expert_idx, :, :].copy_(
                        loaded_tensor.narrow(0, local_rank*shard_size, shard_size).t())

                    original_weight_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.qzeros"
                    loaded_tensor = load_tensor_from_file_helper(f, next_file, original_weight_name)
                    shard_size = loaded_tensor.size(0) // tp_size
                    fused_moe_param_zero.data[expert_idx, :, :].copy_(
                        loaded_tensor.narrow(0, local_rank*shard_size, shard_size).t())


                    already_loaded_set.add(load_dedup_key)
                    continue


                load_dedup_key = ".".join(weight_name_parts[:-1])
                if load_dedup_key in already_loaded_set:
                    continue
                

                
                original_qweight_name = ".".join(original_weight_name_parts[:-1] + ["qweight"])
                original_scale_name = ".".join(original_weight_name_parts[:-1] + ["scales"])
                original_zero_name = ".".join(original_weight_name_parts[:-1] + ["qzeros"])
                is_quantized =  original_qweight_name in f.keys() and \
                                original_scale_name in f.keys() and \
                                original_zero_name in f.keys()
                
                if is_quantized:
                    loaded_tensor_weight = load_tensor_from_file_helper(f, next_file, original_qweight_name)
                    loaded_tensor_scale = load_tensor_from_file_helper(f, next_file, original_scale_name)
                    loaded_tensor_zero = load_tensor_from_file_helper(f, next_file, original_zero_name)

                    dequant_tensor = awq_dequantize_triton(loaded_tensor_weight, loaded_tensor_scale, loaded_tensor_zero)


                    param_path = ".".join(weight_name_parts[:-1] + ["weight"])
                    # print(f"param_path = {param_path}, weight_name = {weight_name}, original_weight_name = {original_weight_name}, dequant_tensor.shape = {dequant_tensor.shape}")
                    dequant_param = get_param_from_model(model, param_path, start_layer)
                    if dequant_param is None:
                        print(f"Warning: Parameter {param_path} not found in model.")
                        continue
                    
                    if tp_split_dim is not None:
                        assert dequant_tensor.size(tp_split_dim) % tp_size == 0, f"Dimension {tp_split_dim} must be divisible by {tp_size}"
                        shard_size = dequant_tensor.size(tp_split_dim) // tp_size
                        
                        weight_shard = dequant_tensor.narrow(tp_split_dim, local_rank * shard_size, shard_size)
                        dequant_param.data.copy_(weight_shard.T)
                    else:
                        dequant_param.data.copy_(dequant_tensor.T)

                    already_loaded_set.add(load_dedup_key)
                else:

                    for k in packed_modules_mapping:
                        if k in original_weight_name_parts:
                            v, split_dim = packed_modules_mapping[k]
                            param_name = weight_name.replace(k, v)
                            # print(f"param_name = {param_name}, k= {k}, v = {v}, split_dim = {split_dim}")
                            param = get_param_from_model(model, param_name, start_layer)
                            if param is None:
                                raise Exception(f"Warning: Parameter {param_name} not found in model.")
                                

                            weight_loader = getattr(param, "weight_loader", default_weight_loader)
                                

                            if split_dim is None:
                                whole_weight = load_tensor_from_file_helper(f, next_file, original_weight_name)
                                if whole_weight.dtype == torch.bfloat16:
                                    whole_weight = whole_weight.to(torch.float16)

                                weight_loader(param, whole_weight, None)
                            else:
                                whole_weight = load_tensor_from_file_helper(f, next_file, original_weight_name)
                                if whole_weight.dtype == torch.bfloat16:
                                    whole_weight = whole_weight.to(torch.float16)
                                
                                assert whole_weight.size(tp_split_dim) % tp_size == 0, f"Dimension {tp_split_dim} must be divisible by {tp_size}"
                                shard_size = whole_weight.size(tp_split_dim) // tp_size
                                
                                weight_shard = whole_weight.narrow(tp_split_dim, local_rank * shard_size, shard_size)
                                try:
                                    weight_loader(param, weight_shard, None)
                                except Exception as e:
                                    if local_rank == 0:
                                        import pdb; pdb.set_trace()
                            break


                    else:
                        param = get_param_from_model(model, weight_name, start_layer)
                        if param is not None:
                            weight_loader = getattr(param, "weight_loader", default_weight_loader)
                            
                            whole_weight = load_tensor_from_file_helper(f, next_file, original_weight_name)
                            if whole_weight.dtype == torch.bfloat16:
                                whole_weight = whole_weight.to(torch.float16)
                            
                            weight_loader(param, whole_weight, None)
                        else:
                            raise Exception(f"Warning: Parameter {weight_name} not found in model.")
                            
