"""
모델의 GNN 계층 : AdaLN -> 어텐션 -> AdaLN -> FFN(Linear -> GeLU -> Linear)
"""
from __future__ import annotations

import math

import torch
from torch import nn

# LayNorm * (1 + scale) + shift : t에 따라 노드를 다르게 출력
class AdaLN(nn.Module):
    def __init__(self, d:int):
        super().__init__()
        self.norm = nn.LayerNorm(d, elementwise_affine=False)

    def forward(self, h:torch.Tensor, shift:torch.Tensor, scale:torch.Tensor) -> torch.Tensor:
        return self.norm(h) * (1 + scale) + shift

# d / 4로 나눠서 그래프 가중치를 다르게 적용 : 각 차원마다 관계의 중요도를 다르게 부여
class EdgeAttention(nn.Module):
    def __init__(self, d:int, edge_dim:int, num_heads:int=4, dropout:float=0.1):
        super().__init__()
        assert d % num_heads == 0
        self.num_heads = num_heads
        self.d_h = d // num_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.edge_bias = nn.Linear(edge_dim, num_heads)
        self.out = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h:torch.Tensor, edge_mask:torch.Tensor, edge_attr:torch.Tensor) -> torch.Tensor:
        B, N, d = h.shape

        # q, k, v 각각 4개의 헤드를 보유
        q, k, v = (
            self.qkv(h).view(B, N, 3, self.num_heads, self.d_h).permute(2, 0, 3, 1, 4)
        )

        score = q @ k.transpose(-2, -1) / math.sqrt(self.d_h)
        score = score + self.edge_bias(edge_attr).permute(0, 3, 1, 2)

        eye = torch.eye(N, dtype=torch.bool, device=h.device)
        mask = edge_mask | eye
        score = score.masked_fill(~mask[:,None], float('-inf'))

        alpha = score.softmax(dim=-1)
        out = (alpha @ v).transpose(1, 2).reshape(B, N, d)
        return self.dropout(self.out(out))

# FFN : Linear -> GeLU -> Linear
class FeedForward(nn.Module):
    def __init__(self, d:int, expansion:int=4, dropout:float=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, expansion * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * d, d)
        )

    def forward(self, h:torch.Tensor) -> torch.Tensor:
        return self.net(h)

# AdaLN -> attention(그래프) -> *gate -> AdaLN -> FFN -> gate
class GNNLayer(nn.Module):
    def __init__(self, d:int, edge_dim:int, num_heads:int=4, dropout:float=0.1):
        super().__init__()
        # shift, scale, gate를 생성하는 W * x(t) + b를 초기화
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

        self.norm1 = AdaLN(d)
        self.attn = EdgeAttention(d, edge_dim, num_heads, dropout)
        self.norm2 = AdaLN(d)
        self.ffn = FeedForward(d, dropout=dropout)

    def forward(self, h:torch.Tensor, c:torch.Tensor, edge_mask:torch.Tensor, edge_attr:torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(c)[:, None].chunk(6, dim=-1)

        h2 = self.norm1(h, shift1, scale1)
        h2 = self.attn(h2, edge_mask, edge_attr)
        h2 = h + gate1 * h2
        
        h3 = self.norm2(h2, shift2, scale2)
        h3 = self.ffn(h3)
        h3 = h2 + gate2 * h3

        return h3