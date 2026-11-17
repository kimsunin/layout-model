"""
학습 구조 : batch -> to_diffusion(좌표를 -1, 1로 변경) -> sample_t(페이지별 t 샘플링) -> q_sample(t를 통해 좌표에 노이즈 추가)
          -> GraphBuilder(노이즈가 추가된 좌표로 그래프 생성) -> model(예측) -> loss(손실계산) -> backward
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.batch import LayoutBatch
from src.data.dataset import LayoutDataset
from src.data.source import DataSource
from src.diffusion.loss import DiffusionLoss
from src.diffusion.schedule import NoiseSchedule, to_diffusion
from src.graph.build import EDGE_DIM, GraphBuilder
from src.model.model import DenoiserModel

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # 데이터
    p.add_argument("--limit", type=int, default=None, help="split마다 읽을 페이지 수 (None이면 전체)")
    p.add_argument("--max_elements", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=64)
    # 디퓨전, 그래프
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--graph", default="both", choices=["complete", "vis", "hier", "both"])
    p.add_argument("--k", type=int, default=0)
    # 모델
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    # 학습
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="cuda / mps / cpu (None이면 자동)")
    p.add_argument("--out", default="outputs/exp")
    return p.parse_args()

def pick_device(name:str|None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_seed(seed:int):
    random.seed(seed)
    torch.manual_seed(seed)

def fork_rng(device:torch.device):
    if device.type in ("cuda", "mps"):
        return torch.random.fork_rng(devices=[device], device_type=device.type)
    return torch.random.fork_rng(devices=[])

# T 구간별 손실 평균 계산
class TBucket:
    def __init__(self, T:int, n:int=10):
        self.T, self.n = T, n
        self.sum = torch.zeros(n)        
        self.cnt = torch.zeros(n)

    def add(self, t:torch.Tensor, losses:torch.Tensor):
        idx = ((t - 1) * self.n // self.T).clamp(max=self.n - 1).cpu()
        self.sum.index_add_(0, idx, losses.float().cpu())
        self.cnt.index_add_(0, idx, torch.ones(idx.size()))

    def result(self) -> list[float]:
        return (self.sum / self.cnt.clamp_min(1)).tolist()

class Trainer:
    def __init__(self, args:argparse.Namespace, device:torch.device):
        self.args, self.device = args, device
        self.schedule = NoiseSchedule(args.T)
        self.builder = GraphBuilder(args.T, mode=args.graph, k=args.k)
        self.model = DenoiserModel(args.d, args.layers, EDGE_DIM, args.heads, dropout=args.dropout).to(device)
        self.loss_fn = DiffusionLoss()
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # forward -> loss
    def forward(self, batch:LayoutBatch):
        types, boxes, mask = batch.to(self.device)
        x0 = to_diffusion(boxes)
        t = self.schedule.sample_t(x0.size(0), self.device)
        x_t = self.schedule.q_sample(x0, t) 
        edge_mask, edge_attr = self.builder(types, x_t, t, mask, ab=self.schedule.ab(t))
        x0_hat = self.model(types, x_t, t, mask, edge_mask, edge_attr)
        loss, log = self.loss_fn(x0_hat, x0, t, mask)  
        return loss, log, t
    
    def train_epoch(self, loader:DataLoader) -> tuple[float, list[float]]:
        self.model.train()
        bucket, total, n = TBucket(self.args.T), 0.0, 0
        for batch in loader:
            loss, log, t = self.forward(batch)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.opt.step()

            bucket.add(t, log["recon"])
            total += loss.item() * t.size(0)
            n += t.size(0)
        return total / max(n, 1), bucket.result()
    
    @torch.no_grad()
    def evaluate(self, loader:DataLoader) -> tuple[float, list[float]]:
        self.model.eval()
        bucket, total, n = TBucket(self.args.T), 0.0, 0
        with fork_rng(self.device):
            torch.manual_seed(1234)
            for batch in loader:
                loss, log, t = self.forward(batch)
                bucket.add(t, log["recon"])
                total += loss.item() * t.size(0)
                n += t.size(0)
        return total / max(n, 1), bucket.result()

    def save(self, path:Path, epoch:int, best_val:float):
        torch.save({
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "args": vars(self.args),
            "epoch":epoch,
            "best_val":best_val
        }, path)

def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False))

    # 데이터 불러오기
    source = DataSource()
    train_ds = LayoutDataset(source, "train", max_elements=args.max_elements, limit=args.limit)
    val_ds = LayoutDataset(source, "val", max_elements=args.max_elements, limit=args.limit)
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=LayoutBatch.collate, generator=g)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=LayoutBatch.collate)

    # 학습 
    trainer = Trainer(args, device)
    num_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"device:{device}, 학습데이터수:{len(train_ds)}, 검증데이터수:{len(val_ds)}, 파라미터수:{num_params:,}")
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_t = trainer.train_epoch(train_loader)
        val_loss, val_t = trainer.evaluate(val_loader)

        record = {"epoch": epoch, "train": train_loss, "val": val_loss,
                  "train_t": train_t, "val_t": val_t, "sec": time.time() - start}
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[{epoch:3d}] train {train_loss:.4f}  val {val_loss:.4f}  ({record['sec']:.1f}s)")

        trainer.save(out/"last.pt", epoch, best)
        if val_loss < best:
            best = val_loss
            trainer.save(out / "best.pt", epoch, best)
    
if __name__ == "__main__":
    main()
    

