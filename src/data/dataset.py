"""
DocLayNet 페이지들을 텐서로 변환하여 하나의 리스트로 반환
"""

from typing import NamedTuple

import torch
from torch.utils.data import Dataset

from src.data.category import RAW_TO_TYPE
from src.data.type import Page, LayoutSample
from src.data.source import DataSource

class LayoutDataset(Dataset):
    def __init__(
        self,
        source: DataSource,
        split: str,
        min_elements: int = 1,
        max_elements: int | None = None,
        limit: int | None = None
    ):
        self.samples: list[LayoutSample] = []
        self.num_skipped = 0

        for page in source.pages(split, limit=limit):
            sample = self.to_sample(page)
            if self.keep(len(sample.types), min_elements, max_elements):
                self.samples.append(sample)
            else:
                self.num_skipped += 1

    @staticmethod
    def to_sample(page: Page) -> LayoutSample:
        types, boxes = [], []
        for cid, box in zip(page.category_ids, page.boxes):
            t = RAW_TO_TYPE[cid]
            if t is None:
                continue
            types.append(t)
            boxes.append(box)
        return LayoutSample(
            types=torch.tensor(types, dtype=torch.long),
            boxes=torch.tensor(boxes, dtype=torch.float32)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> LayoutSample:
        return self.samples[i]


    @staticmethod
    def keep(n: int, min_elements: int, max_elements: int | None) -> bool:
        if n < min_elements:
            return False
        if max_elements is not None and n > max_elements:
            return False
        return True

if __name__ == "__main__":
    ds = LayoutDataset(DataSource(), "val", limit=10)
    print(f"가져온 데이터 수:{len(ds)}, 건너뛴 데이터:{ds.num_skipped}")
    types, boxes = ds[0]
    print(f"첫번 페이지:{types}, {boxes}")