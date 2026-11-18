"""
ddim step 반복 : x_t -> x_t-1 -> .. -> x0 생성
"""
from __future__ import annotations

import torch

from src.diffusion.schedule import NoiseSchedule, sample_noise, to_page
from src.graph.build import GraphBuilder

# T를 steps개로 분할
def ddim_timesteps(T:int, steps:int) -> list[int]:
    assert 1 <= steps <= T
    return torch.linspace(T, 0, steps + 1).round().long().tolist()

class DDIMSampler:
    def __init__(self, model, schedule:NoiseSchedule, builder:GraphBuilder, steps:int=50):
        self.model, self.schedule, self.builder = model, schedule, builder
        self.timesteps = ddim_timesteps(schedule.T, steps)

    @torch.no_grad()
    def sample(self, types:torch.Tensor, node_mask:torch.Tensor, generator:torch.Generator | None = None, return_traj:bool = False):
        self.model.eval()
        B, N = types.shape
        device = types.device

        noise = sample_noise((B, N, 4), device, generator)
        # ᾱ(T) = 0, 각 단계를 거치면서 x_t에 x0_hat을 섞어 x_t 업데이트
        # x_T = √ᾱ(T)·x0 + √(1−ᾱ(T))·ε = ε
        x_t = noise
        traj = []

        for t_cur, t_prev in zip(self.timesteps[:-1], self.timesteps[1:]):
            t = torch.full((B,), t_cur, dtype=torch.long, device=device)
            tp = torch.full((B,), t_prev, dtype=torch.long, device=device)

            edge_mask, edge_attr = self.builder(types, x_t, t, node_mask, ab=self.schedule.ab(t))
            x0_hat = self.model(types, x_t, t, node_mask, edge_mask, edge_attr).clamp(-1, 1)
            
            x_t = self.schedule.ddim_step(x_t, x0_hat, t, tp)

            if return_traj:
                traj.append(to_page(x0_hat).cpu())

        boxes = to_page(x_t) * node_mask[...,None]
        if return_traj:
            return (boxes, traj)
        else:
            return boxes

