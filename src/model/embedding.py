"""
t 사인파를 16차원으로 임베딩, t로 c를 생성
"""
from __future__ import annotations

import math

import torch
from torch import nn

# t 사인파를 16차원으로 임베딩(학습시 모든 페이지에 대한 t가 벡터로 입력)
def sinusoidal_embedding(t:torch.Tensor, dim:int=16, max_period:float=10000.0) -> torch.Tensor:
    assert dim%2 == 0
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t.float()[:,None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=1)

# t_feat([B, 16]) -> 입력, mlp([B, d]) -> AdaLN
class TimeEmbedding(nn.Module):
    def __init__(self, d:int, t_dim:int=16):
        super().__init__()
        self.t_dim = t_dim
        self.mlp = nn.Sequential(
            nn.Linear(t_dim, d),
            nn.SiLU(),
            nn.Linear(d, d)
        )

    def forward(self, t:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t_feat = sinusoidal_embedding(t, self.t_dim)
        return t_feat, self.mlp(t_feat)