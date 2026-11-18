"""
예측된 좌표로 손실 계산
현재 복원 + 겹침
"""
from __future__ import annotations

import torch

from src.diffusion.schedule import to_page

# 페이지별 mse
def recon_loss(x0_hat:torch.Tensor, x0:torch.Tensor, node_mask:torch.Tensor) -> torch.Tensor:
    m = node_mask.unsqueeze(-1).to(x0.dtype)
    se = ((x0_hat - x0) ** 2 * m).sum(dim=(1, 2))
    cnt = (m.sum(dim=(1, 2)) * x0.size(-1)).clamp_min(1)
    return se / cnt

# [-1, 1] -> [0, 1]
def to_xyxy(x0: torch.Tensor) -> torch.Tensor:
    x, y, w, h = to_page(x0).unbind(-1)
    return torch.stack([x, y, x + w, y + h], dim=-1)

# 페이지별 겹침 면적 합 / 노드 수 -> 페이지별 겹침 면적 평균
def overlap_loss(x0_hat: torch.Tensor, node_mask: torch.Tensor):
    b = to_xyxy(x0_hat)
    x1, y1, x2, y2 = b.unbind(-1)
    # x, y겹침계산
    iw  = (torch.minimum(x2[:, :, None], x2[:, None]) - torch.maximum(x1[:, :, None], x1[:, None])).relu()
    ih  = (torch.minimum(y2[:, :, None], y2[:, None]) - torch.maximum(y1[:, :, None], y1[:, None])).relu()

    inter = iw * ih

    N = node_mask.size(1)
    pair = node_mask[:, :, None] & node_mask[:, None, :]

    pair &= ~torch.eye(N, dtype=torch.bool, device=node_mask.device)
    inter = inter * pair

    n = node_mask.sum(1).clamp_min(1)
    return inter.sum((1, 2)) / n

class DiffusionLoss:
    def __init__(self, c_overlap:float = 0.0, schedule_weight: bool = True):
        self.c_overlap = c_overlap
        self.schedule_weight = schedule_weight

    def __call__(self, x0_hat, x0, ab, node_mask) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        recon = recon_loss(x0_hat, x0, node_mask)
        overlap = overlap_loss(x0_hat, node_mask)

        if self.schedule_weight:
            w = ab
        else:
            w = torch.ones_like(ab)
        loss = (recon + self.c_overlap * w * overlap).mean()

        return loss, {"recon":recon.detach(), "overlap":overlap.detach()}