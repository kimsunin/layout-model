"""
카테고리별 개수를 하나씩 입력받아 1페이지 생성 후 step/10마다 x0_hat 시각화
python -m scripts.visualize

엔터만 누르면 [ ] 안의 기본값 사용 (0개는 0 입력)
Caption [1] :
Footnote [0] :
Formula [0] :
List-item [0] :
Picture [1] :
Section-header [2] :
Table [0] :
Text [5] :
Title [1] :
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

# 엔터 입력 시 사용할 타입별 기본 개수 (없는 타입은 0)
DEFAULT_COUNTS = {"Caption": 1, "Picture": 1, "Section-header": 2, "Text": 5, "Title": 1}


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


# 타입마다 개수를 하나씩 입력 (엔터 = 기본값) -> types [1, N], mask [1, N]
def read_input() -> tuple[torch.Tensor, torch.Tensor]:
    print("타입별 개수를 입력하세요 (엔터 = [ ] 안의 기본값)")
    types = []
    for t, name in sorted(TYPE_NAMES.items()):
        default = DEFAULT_COUNTS.get(name, 0)
        while True:
            s = input(f"{name} [{default}] : ").strip()
            if s == "":
                n = default
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


# 하나의 축에 페이지 1장 그리기 / boxes : [N, 4], [0,1] 좌상단 (x, y, w, h)
def draw_page(ax, types: torch.Tensor, boxes: torch.Tensor, title: str):
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec="0.6", lw=1))
    for t, (x, y, w, h) in zip(types.tolist(), boxes.tolist()):
        c = COLORS[t]
        ax.add_patch(Rectangle((x, y), w, h, fc=(*c, 0.25), ec=c, lw=1.2))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(1.05, -0.05)   # y축 아래 방향 (좌상단 원점)
    ax.set_aspect(PAGE_ASPECT)
    ax.axis("off")
    ax.set_title(title, fontsize=9)


# step/10 간격의 스냅샷들을 한 figure에 (5열)
def draw_traj(types: torch.Tensor, snaps: list[tuple[str, torch.Tensor]], suptitle: str):
    cols = 5
    rows = (len(snaps) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.4 * PAGE_ASPECT * rows + 0.6), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (title, boxes) in zip(axes.flat, snaps):
        draw_page(ax, types, boxes, title)

    handles = [Patch(fc=(*COLORS[t], 0.25), ec=COLORS[t], label=TYPE_NAMES[t]) for t in sorted(set(types.tolist()))]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    plt.show()


def main():
    args = parse_args()
    sampler = load_sampler(args.ckpt, args.steps)
    types, mask = read_input()

    g = torch.Generator().manual_seed(args.seed)
    boxes, traj = sampler.sample(types, mask, generator=g, return_traj=True)   # traj : steps개 [1, N, 4]

    for t, b in zip(types[0].tolist(), boxes[0].tolist()):
        print(f"{TYPE_NAMES[t]:15s} x={b[0]:.3f} y={b[1]:.3f} w={b[2]:.3f} h={b[3]:.3f}")

    # step/10 간격 (steps < 10이면 매 스텝), 마지막 스텝은 항상 포함
    every = max(1, args.steps // 10)
    idx = list(range(every - 1, args.steps, every))
    if idx[-1] != args.steps - 1:
        idx.append(args.steps - 1)

    snaps = [(f"step {i + 1}  (t={sampler.timesteps[i]})", traj[i][0]) for i in idx]
    draw_traj(types[0], snaps, f"x0_hat trajectory  steps={args.steps}, seed={args.seed}")


if __name__ == "__main__":
    main()
