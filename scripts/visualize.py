"""
카테고리별 개수를 하나씩 입력받아 1페이지 생성 후 시각화
  python -m scripts.visualize
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Patch, Rectangle

from src.data.category import DOCLAYNET_CATEGORIES, RAW_TO_TYPE
from src.diffusion.sampler import DDIMSampler
from src.diffusion.schedule import NoiseSchedule
from src.graph.build import EDGE_DIM, GraphBuilder
from src.model.model import DenoiserModel

# 모델 타입 인덱스 <-> 이름 (Page-header/footer 제외 9종)
TYPE_NAMES = {t: DOCLAYNET_CATEGORIES[raw - 1] for raw, t in RAW_TO_TYPE.items() if t is not None}
COLORS = dict(zip(sorted(TYPE_NAMES), plt.get_cmap("tab10").colors))
PAGE_ASPECT = 1.29  # 페이지 세로/가로 비율


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="outputs/exp2/best.pt")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# 체크포인트의 args로 모델·스케줄·그래프 설정을 복원
def load_sampler(ckpt_path: str, steps: int) -> DDIMSampler:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    a = argparse.Namespace(**ckpt["args"])
    model = DenoiserModel(a.d, a.layers, EDGE_DIM, a.heads, dropout=a.dropout)
    model.load_state_dict(ckpt["model"])
    return DDIMSampler(model, NoiseSchedule(a.T), GraphBuilder(a.T, mode=a.graph, k=a.k), steps=steps)


# 타입마다 개수를 하나씩 입력 (엔터 = 0) -> types [1, N], mask [1, N]
def read_input() -> tuple[torch.Tensor, torch.Tensor]:
    print("타입별 개수를 입력하세요 (엔터 = 0)")
    types = []
    for t, name in sorted(TYPE_NAMES.items()):
        while True:
            s = input(f"{name} : ").strip()
            if s == "":
                n = 0
                break
            if s.isdigit():
                n = int(s)
                break
            print("  0 이상의 정수를 입력하세요")
        types += [t] * n

    if not types:
        raise SystemExit("요소가 하나도 없습니다.")
    types = torch.tensor([types], dtype=torch.long)
    mask = torch.ones_like(types, dtype=torch.bool)
    return types, mask


# boxes : [N, 4], [0,1] 좌상단 (x, y, w, h)
def draw(types: torch.Tensor, boxes: torch.Tensor, title: str):
    fig, ax = plt.subplots(figsize=(4, 4 * PAGE_ASPECT + 0.5))
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec="0.6", lw=1))
    for t, (x, y, w, h) in zip(types.tolist(), boxes.tolist()):
        c = COLORS[t]
        ax.add_patch(Rectangle((x, y), w, h, fc=(*c, 0.25), ec=c, lw=1.5))

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(1.05, -0.05)   # y축 아래 방향 (좌상단 원점)
    ax.set_aspect(PAGE_ASPECT)
    ax.axis("off")
    ax.set_title(title, fontsize=10)

    handles = [Patch(fc=(*COLORS[t], 0.25), ec=COLORS[t], label=TYPE_NAMES[t]) for t in sorted(set(types.tolist()))]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    plt.show()


def main():
    args = parse_args()
    sampler = load_sampler(args.ckpt, args.steps)
    types, mask = read_input()

    g = torch.Generator().manual_seed(args.seed)
    boxes = sampler.sample(types, mask, generator=g)   # [1, N, 4]

    for t, b in zip(types[0].tolist(), boxes[0].tolist()):
        print(f"{TYPE_NAMES[t]:15s} x={b[0]:.3f} y={b[1]:.3f} w={b[2]:.3f} h={b[3]:.3f}")

    draw(types[0], boxes[0], f"steps={args.steps}, seed={args.seed}")


if __name__ == "__main__":
    main()
