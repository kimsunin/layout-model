"""
여러 페이지의 LayoutSample을 패딩해 하나의 LayoutBatch로 합침
"""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch.nn.utils.rnn import pad_sequence

from src.data.category import PAD_TYPE
from src.data.type import LayoutSample

class LayoutBatch(NamedTuple):
    types: torch.Tensor  # [B, N]
    boxes: torch.Tensor  # [B, N, 4]
    mask: torch.Tensor   # [B, N]

    @staticmethod
    def collate(samples: list[LayoutSample]) -> LayoutBatch:
        types = pad_sequence([s.types for s in samples], batch_first=True, padding_value=PAD_TYPE)
        boxes = pad_sequence([s.boxes for s in samples], batch_first=True, padding_value=0.0)
        lengths = torch.tensor([len(s.types) for s in samples])
        mask = torch.arange(types.size(1)) < lengths[:, None]
        return LayoutBatch(types, boxes, mask)

    def to(self, device) -> LayoutBatch:
        return LayoutBatch(*(x.to(device) for x in self))

# test
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from src.data.dataset import LayoutDataset
    from src.data.source import DataSource

    ds = LayoutDataset(DataSource(), "val", limit=10)
    loader = DataLoader(ds, batch_size=4, collate_fn=LayoutBatch.collate)

    types, boxes, mask = next(iter(loader))
    print(f"배치 차원(B, N):{types.shape}")
    print(f"첫페이지:{types[0].tolist()}, {boxes[0].tolist()}")
