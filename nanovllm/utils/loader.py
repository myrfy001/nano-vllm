import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open
from nanovllm.layers.fused_moe import awq_dequantize_triton


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor, tp_dim: int | None = None):
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
    


def load_model(model: nn.Module, path: str, local_rank: int, start_layer: int, end_layer: int):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    already_loaded_set = set()
    for file in sorted(glob(os.path.join(path, "*.safetensors"))):
        # print(f"Loading weights from {file}")
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                print(f"origin weight_name = {weight_name}")
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
                    elif ("embed_tokens" in original_weight_name) and start_layer != 1:
                        # some weights only belong to the first pp rank
                        continue

                
                if weight_name.startswith("model."):
                    weight_name = weight_name[6:]
                
                weight_name = weight_name.replace("self_attn", "attn")
                weight_name = weight_name.replace("mlp", "ffn")
                weight_name = weight_name.replace("weight_scale_inv", "scale")
                weight_name = weight_name.replace("e_score_correction_bias", "bias")

                weight_name_parts = weight_name.split(".")
                for i in range(len(weight_name_parts)):
                    if weight_name_parts[i] in packed_modules_mapping:
                        weight_name_parts[i] = packed_modules_mapping[weight_name_parts[i]][0]

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
                    fused_moe_param_weight = get_param_from_model(model, f"layers.{layer_idx}.ffn.experts.w13_qweight")
                    fused_moe_param_scale = get_param_from_model(model, f"layers.{layer_idx}.ffn.experts.w13_scales")
                    fused_moe_param_zero = get_param_from_model(model, f"layers.{layer_idx}.ffn.experts.w13_qzeros")
                    if fused_moe_param_weight is None:
                        print(f"Warning: Parameter layers.{layer_idx}.ffn.experts.w13_qweight not found in model.")
                        continue

                    intermediate_size_per_partition = fused_moe_param_weight.intermediate_size_per_partition

                    # handle w1

                    original_weight_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.qweight"
                    loaded_tensor = f.get_tensor(original_weight_name)
                    fused_moe_param_weight.data[expert_idx, :intermediate_size_per_partition // 8, :].copy_(
                        loaded_tensor[:, local_rank * intermediate_size_per_partition // 8: (local_rank + 1) * intermediate_size_per_partition // 8].t())

                    original_weight_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.scales"
                    loaded_tensor = f.get_tensor(original_weight_name)
                    fused_moe_param_scale.data[expert_idx, :intermediate_size_per_partition, :].copy_(
                        loaded_tensor[:, local_rank * intermediate_size_per_partition: (local_rank + 1) * intermediate_size_per_partition].t())

                    original_weight_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.qzeros"
                    loaded_tensor = f.get_tensor(original_weight_name)
                    fused_moe_param_zero.data[expert_idx, :intermediate_size_per_partition // 8, :].copy_(
                        loaded_tensor[:, local_rank * intermediate_size_per_partition // 8: (local_rank + 1) * intermediate_size_per_partition // 8].t())

                    
                    # handle w3
                    original_weight_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.qweight"
                    loaded_tensor = f.get_tensor(original_weight_name)
                    fused_moe_param_weight.data[expert_idx, intermediate_size_per_partition // 8:, :].copy_(
                        loaded_tensor[:, local_rank * intermediate_size_per_partition // 8: (local_rank + 1) * intermediate_size_per_partition // 8].t())

                    original_weight_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.scales"
                    loaded_tensor = f.get_tensor(original_weight_name)
                    fused_moe_param_scale.data[expert_idx, intermediate_size_per_partition:, :].copy_(
                        loaded_tensor[:, local_rank * intermediate_size_per_partition: (local_rank + 1) * intermediate_size_per_partition].t())

                    original_weight_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.qzeros"
                    loaded_tensor = f.get_tensor(original_weight_name)
                    fused_moe_param_zero.data[expert_idx, intermediate_size_per_partition // 8:, :].copy_(
                        loaded_tensor[:, local_rank * intermediate_size_per_partition // 8: (local_rank + 1) * intermediate_size_per_partition // 8].t())

                    # load w2
                    fused_moe_param_weight = get_param_from_model(model, f"layers.{layer_idx}.ffn.experts.w2_qweight")
                    fused_moe_param_scale = get_param_from_model(model, f"layers.{layer_idx}.ffn.experts.w2_scales")
                    fused_moe_param_zero = get_param_from_model(model, f"layers.{layer_idx}.ffn.experts.w2_qzeros")

                    original_weight_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.qweight"
                    loaded_tensor = f.get_tensor(original_weight_name)
                    fused_moe_param_weight.data[expert_idx, :, :].copy_(loaded_tensor.t())

                    original_weight_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.scales"
                    loaded_tensor = f.get_tensor(original_weight_name)
                    fused_moe_param_scale.data[expert_idx, :, :].copy_(loaded_tensor.t())

                    original_weight_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.qzeros"
                    loaded_tensor = f.get_tensor(original_weight_name)
                    fused_moe_param_zero.data[expert_idx, :, :].copy_(loaded_tensor.t())


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
                    loaded_tensor_weight = f.get_tensor(original_qweight_name)
                    loaded_tensor_scale = f.get_tensor(original_scale_name)
                    loaded_tensor_zero = f.get_tensor(original_zero_name)

                    dequant_tensor = awq_dequantize_triton(loaded_tensor_weight, loaded_tensor_scale, loaded_tensor_zero)


                    param_path = ".".join(weight_name_parts[:-1] + ["weight"])
                    # print(f"param_path = {param_path}, weight_name = {weight_name}, original_weight_name = {original_weight_name}, dequant_tensor.shape = {dequant_tensor.shape}")
                    dequant_param = get_param_from_model(model, param_path)
                    if dequant_param is None:
                        print(f"Warning: Parameter {param_path} not found in model.")
                        continue
                    dequant_param.data.copy_(dequant_tensor.T)

                    already_loaded_set.add(load_dedup_key)
                else:

                    for k in packed_modules_mapping:
                        if k in original_weight_name_parts:
                            v, split_dim = packed_modules_mapping[k]
                            param_name = weight_name.replace(k, v)
                            # print(f"param_name = {param_name}, k= {k}, v = {v}, split_dim = {split_dim}")
                            param = get_param_from_model(model, param_name)
                            if param is None:
                                raise Exception(f"Warning: Parameter {param_name} not found in model.")
                                

                            weight_loader = getattr(param, "weight_loader", default_weight_loader)
                            weight_loader(param, f.get_tensor(original_weight_name), None)
                            break


                    else:
                        param = get_param_from_model(model, weight_name)
                        if param is not None:
                            weight_loader = getattr(param, "weight_loader", default_weight_loader)
                            weight_loader(param, f.get_tensor(original_weight_name), None)
                        else:
                            raise Exception(f"Warning: Parameter {weight_name} not found in model.")
                            
