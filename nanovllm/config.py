import os
from dataclasses import dataclass, field
from transformers import AutoConfig


_node_id_to_layers_mapping = {
    0: (0,3, "first"),
    1: (3,7, "middle"),
    2: (7,11, "middle"),
    3: (11,15, "middle"),
    4: (15,19, "middle"),
    5: (19,23, "middle"),
    6: (23,27, "middle"),
    7: (27,31, "middle"),
    8: (31,35, "middle"),
    9: (35,39, "middle"),
    10: (39,43, "middle"),
    11: (43,47, "middle"),
    12: (47,51, "middle"),
    13: (51,55, "middle"),
    14: (55,59, "middle"),
    15: (59,61, "last"),
}

@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 32768
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 1
    num_kvcache_blocks: int = -1
    local_rank: int = 0
    pp_schema: tuple = _node_id_to_layers_mapping[0]
    pp_rank: int = 0
    node_id: int = 0

    def __post_init__(self):
        assert os.path.isdir(self.model)
        # assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        self.pp_schema = _node_id_to_layers_mapping[self.node_id]