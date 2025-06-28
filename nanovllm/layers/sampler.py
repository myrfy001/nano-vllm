import time
import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # print(f"{time.time()}, Sampler enter")
        logits = logits.to(torch.float)
        # print(f"{time.time()}, Sampler 1")
        greedy_tokens = logits.argmax(dim=-1)
        # print(f"{time.time()}, Sampler 2")
        logits.div_(temperatures.unsqueeze(dim=1))
        # print(f"{time.time()}, Sampler 3")
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # print(f"{time.time()}, Sampler 4")
        # logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)
        # print(f"{time.time()}, Sampler 5")
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)
