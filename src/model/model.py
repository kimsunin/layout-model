"""
입력(39) -> Encoder(39 -> d) -> GNNLayer(d -> d) * l(여러층) -> LayerNorm(정규화) -> Decoder(d -> 4) -> 출력
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from src.data.category import NUM_TYPES
from src.model.embedding import TimeEmbedding
from src.model.layer import GNNLayer

# 39차원 입력 -> 4차원 출력
class DenoiserModel(nn.Module):
    def __init__(self, d:int=128, num_layers:int=4, edge_dim:int=16, num_heads:int=4, t_dim:int=16, dropout:float=0.1):
        super().__init__()
        in_dim = NUM_TYPES + NUM_TYPES + 1 + 4 + t_dim
        self.time_emb = TimeEmbedding(d, t_dim)

        # endcoder
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, d),
            nn.GELU(),
            nn.Linear(d, d)
        )

        # GNN layer
        self.layers = nn.ModuleList(
            [GNNLayer(d, edge_dim, num_heads, dropout) for _ in range(num_layers)]
        )

        # 정규화
        self.final_norm = nn.LayerNorm(d)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 4)
        )
    
    # 입력생성 : 현재 노드 타입(9) + 페이지 내 타입별 개수(9) + 총 노드 수(1) + 노이즈가 섞인 boxes(4) + t_embedding(16)
    def build_feature(self, types, x_t, t_feat, node_mask):
        B, N = types.shape
        m = node_mask[..., None].float()

        type_onehot = F.one_hot(types, NUM_TYPES + 1)[...,:NUM_TYPES].float()
        type_onehot = type_onehot * m

        type_count = type_onehot.sum(dim=1)
        num_nodes = node_mask.sum(dim=1, keepdim=True).float()

        glob = torch.cat([torch.log1p(type_count), torch.log1p(num_nodes), t_feat], dim=1)
        glob = glob[:,None].expand(B, N, glob.size(-1))
        
        return torch.cat([type_onehot, x_t * m, glob], dim=-1)

    def forward(self, types, x_t, t, node_mask, edge_mask, edge_attr):
        t_feat, c = self.time_emb(t)
        if t_feat.size(0) != types.size(0):
            t_feat = t_feat.expand(types.size(0), -1)

        edge_mask = edge_mask & node_mask[:, :, None] & node_mask[:,None,:]

        h = self.encoder(self.build_feature(types, x_t, t_feat, node_mask))

        for layer in self.layers:
            h = layer(h, c, edge_mask, edge_attr)

        h = self.final_norm(h)

        x_hat = self.decoder(h)
        return x_hat * node_mask[..., None]

# test
if __name__ == "__main__":
    from src.data.category import PAD_TYPE

    B, N = 3, 7
    model = DenoiserModel(d=128, num_layers=6)

    types = torch.randint(0, NUM_TYPES, (B, N))
    node_mask = torch.ones(B, N, dtype=torch.bool)
    node_mask[1, 5:] = False

    x_t = torch.rand(B, N, 4)
    t = torch.randint(1, 1001, (B, ))

    edge_mask = torch.ones(B, N, N, dtype=torch.bool)
    edge_attr = torch.zeros(B, N, N, 16)

    out = model(types, x_t, t, node_mask, edge_mask, edge_attr)
    print(out[0])
