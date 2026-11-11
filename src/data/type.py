"""
Page 클래스
"""

from typing import NamedTuple
import torch

class Page(NamedTuple):
    category_ids: list[int]
    boxes: list[list[float]]
    doc_category: str
    page_hash: str

# B = 배치크기, N = 한 페이지의 노드 수
# 1 페이지
class LayoutSample(NamedTuple):
    types: torch.Tensor # [N]
    boxes: torch.Tensor # [N, 4]