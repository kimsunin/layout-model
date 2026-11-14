"""
ᾱ(t) 생성, x0에 노이즈를 섞어 x_t 생성
"""
from __future__ import annotations

import math

import torch

# 각 t에 대한 ᾱ(t)를 생성(선형으로 할때보다 t에 따라 일정하게 변화 -> 일정한 간격으로 정답에 근접)
def cosine_alpha_bar(T:int, s:float = 0.008) -> torch.Tensor:
    t = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    return f / f[0]

# last(t=T) -> 0, first(t=0) -> 1
def rescale_zero_terminal(alpha_bar:torch.Tensor) -> torch.Tensor:
    sab = alpha_bar.sqrt()
    first, last = sab[0].clone(), sab[-1].clone()
    sab = (sab - last) / (first - last)
    return sab ** 2

def sample_noise(shape, device=None) -> torch.Tensor:
    return torch.randn(shape, device=device)

class NoiseSchedule:
    def __init__(self, T:int, s:float=0.008):
        self.T = T
        cosine_alpha_bar = cosine_alpha_bar(T, s)
        self.alpha_bar = rescale_zero_terminal(cosine_alpha_bar).float()

    # ᾱ(t)반환
    def ab(self, t:torch.Tensor) -> torch.Tensor:
        return self.alpha_bar.to(t.device)[t]
    
    # t 랜덤 샘플링
    def sample_t(self, B:int, device=None) -> torch.Tensor:
        return torch.randint(1, self.T + 1, (B,), device=device)
    
    # t에 따라 x0에 노이즈를 섞어 학습을 위한 x_t반환
    def q_sample(self, x0:torch.Tensor, t:torch.Tensor) -> torch.Tensor:
        eps = sample_noise(x0.shape, x0.device)
        a = self.ab(t).view(-1, *([1] * (x0.dim() - 1)))
        return a.sqrt() * x0 + (1 - a).sqrt() * eps

    # x_t와 x0를 받아 x_t-1을 반환(다음 입력으로 사용)
    def ddim_step(self, x_t:torch.Tensor, x0_hat:torch.Tensor, t:torch.Tensor, t_prev:torch.Tensor) -> torch.Tensor:
        shape = (-1,) + (1,) * (x_t.dim() - 1)
        a_t = self.ab(t).view(shape)
        a_prev = self.ab(t_prev).view(shape)
        
        eps_hat = (x_t - a_t.sqrt() * x0_hat) / (1 - a_t).sqrt()
        x_t_prev = a_prev.sqrt() * x0_hat + (1 - a_prev).sqrt() * eps_hat
        return x_t_prev
