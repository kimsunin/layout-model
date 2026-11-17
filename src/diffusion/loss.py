"""
예측된 좌표로 손실 계산
현재 y-pred(복원)로만 계산 -> 추후에 기하 손실 추가
"""
from __future__ import annotations

import torch

# 페이지별 mse
def recon_loss(x0_hat:torch.Tensor, x0:torch.Tensor, node_mask:torch.Tensor) -> torch.Tensor:
    m = node_mask.unsqueeze(-1).to(x0.dtype)
    se = ((x0_hat - x0) ** 2 * m).sum(dim=(1, 2))
    cnt = (m.sum(dim=(1, 2)) * x0.size(-1)).clamp_min(1)
    return se / cnt

class DiffusionLoss:
    def __call__(self, x0_hat, x0, t, node_mask) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        recon = recon_loss(x0_hat, x0, node_mask)
        # 페이지별 mse의 배치 평균
        loss = recon.mean()
        return loss, {"recon":recon.detach()}