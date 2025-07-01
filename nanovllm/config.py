import os
from dataclasses import dataclass, field
from transformers import AutoConfig


class PPNodeType:
    PPNodeFirst = 0
    PPNodeMiddle = 1
    PPNodeLast = 2


_node_id_to_layers_mapping = {
    0: (10,14, PPNodeType.PPNodeMiddle),
    # 1: (57,61, PPNodeType.PPNodeLast),
}

# _node_id_to_layers_mapping = {
#     0: (0,3, PPNodeType.PPNodeFirst),
#     1: (57,61, PPNodeType.PPNodeLast),
# }

# _node_id_to_layers_mapping = {
#     0: (0,3, PPNodeType.PPNodeFirst),
#     1: (3,7, PPNodeType.PPNodeMiddle),
#     2: (57,61, PPNodeType.PPNodeLast),
# }



_node_id_to_layers_mapping = {
    0: (0,3, PPNodeType.PPNodeFirst),
    1: (3,9, PPNodeType.PPNodeMiddle),
    2: (9,15, PPNodeType.PPNodeMiddle),
    3: (15,21, PPNodeType.PPNodeMiddle),
    4: (21,27, PPNodeType.PPNodeMiddle),
    5: (27,33, PPNodeType.PPNodeMiddle),
    6: (33,39, PPNodeType.PPNodeMiddle),
    7: (39,45, PPNodeType.PPNodeMiddle),
    8: (45,51, PPNodeType.PPNodeMiddle),
    9: (51,57, PPNodeType.PPNodeMiddle),
    10: (57,61, PPNodeType.PPNodeLast),
}



# _node_id_to_layers_mapping = {
#     0: (0,3, PPNodeType.PPNodeFirst),
#     1: (3,8, PPNodeType.PPNodeMiddle),
#     2: (8,13, PPNodeType.PPNodeMiddle),
#     3: (13,18, PPNodeType.PPNodeMiddle),
#     4: (18,23, PPNodeType.PPNodeMiddle),
#     5: (23,28, PPNodeType.PPNodeMiddle),
#     6: (28,33, PPNodeType.PPNodeMiddle),
#     7: (33,38, PPNodeType.PPNodeMiddle),
#     8: (38,43, PPNodeType.PPNodeMiddle),
#     9: (43,48, PPNodeType.PPNodeMiddle),
#     10: (48,53, PPNodeType.PPNodeMiddle),
#     11: (53,58, PPNodeType.PPNodeMiddle),
#     12: (58,61, PPNodeType.PPNodeLast),
# }




# _node_id_to_layers_mapping = {
#     0: (0,3, PPNodeType.PPNodeFirst),
#     1: (3,7, PPNodeType.PPNodeMiddle),
#     2: (7,11, PPNodeType.PPNodeMiddle),
#     3: (11,15, PPNodeType.PPNodeMiddle),
#     4: (15,19, PPNodeType.PPNodeMiddle),
#     5: (19,23, PPNodeType.PPNodeMiddle),
#     6: (23,27, PPNodeType.PPNodeMiddle),
#     7: (27,31, PPNodeType.PPNodeMiddle),
#     8: (31,35, PPNodeType.PPNodeMiddle),
#     9: (35,39, PPNodeType.PPNodeMiddle),
#     10: (39,43, PPNodeType.PPNodeMiddle),
#     11: (43,47, PPNodeType.PPNodeMiddle),
#     12: (47,51, PPNodeType.PPNodeMiddle),
#     13: (51,55, PPNodeType.PPNodeMiddle),
#     14: (55,59, PPNodeType.PPNodeMiddle),
#     15: (59,61, PPNodeType.PPNodeLast),
# }




@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 32768
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 1
    num_kvcache_blocks: int = -1
    local_rank: int = 0
    pp_schema: tuple = _node_id_to_layers_mapping[0]
    node_id: int = 0
    node_num: int = 1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        # assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        self.pp_schema = _node_id_to_layers_mapping[self.node_id]