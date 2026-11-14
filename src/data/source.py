"""
DocLayNet 데이터를 페이지 단위로 로드
"""

from collections.abc import Iterator
from itertools import islice

from datasets import load_dataset

from src.data.type import Page


class DataSource:
    REPO = "docling-project/DocLayNet-v1.1"
    SPLITS = ("train", "val", "test")
    COLUMNS = ["bboxes", "category_id", "metadata"]

    def __init__(self, revision: str | None = None):
        self.revision = revision

    def pages(self, split: str, limit: int | None = None) -> Iterator[Page]:
        if split not in self.SPLITS:
            raise ValueError(f"split은 {self.SPLITS} 중 하나여야 합니다.")

        ds = load_dataset(
            self.REPO,
            split=split,
            revision=self.revision,
            streaming=True,
            columns=self.COLUMNS,
        )

        for row in islice(ds, limit):
            yield self.convert_row(row)

    @staticmethod
    def convert_row(row: dict) -> Page:
        metadata = row["metadata"]
        page_w = float(metadata["coco_width"])
        page_h = float(metadata["coco_height"])

        category_ids, boxes = [], []
        for (x, y, w, h), cid in zip(row["bboxes"], row["category_id"]):
            x0, y0 = max(x, 0.0), max(y, 0.0)
            x1, y1 = min(x + w, page_w), min(y + h, page_h)

            if x1 <= x0 or y1 <= y0:
                continue
            
            boxes.append([x0 / page_w, y0 / page_h, (x1 - x0) / page_w, (y1 - y0) / page_h])
            category_ids.append(int(cid))
        
        return Page(
            category_ids,
            boxes,
            metadata["doc_category"],
            metadata["page_hash"],
        )

# test
if __name__ == "__main__":
    source = DataSource()
    for page in source.pages("val", limit=3):
        print(f"노드 수:{len(page.category_ids)}, 페이지 종류:{page.doc_category}, 첫번째 노드 label:{page.category_ids[:1]}, 첫번째 노드 box:{page.boxes[:1]}")