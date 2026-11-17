"""
x_t와 t로 그래프 생성(t == T -> 완전 그래프)
"""
from __future__ import annotations

import torch

from src.diffusion.schedule import to_page

EDGE_ATTR_DIM = 15 # 엣지 종류
EDGE_DIM = EDGE_ATTR_DIM + 1 # + 1(완전 그래프)

TYPE_LEVEL = [3, 3, 2, 2, 2, 1, 2, 2, 0]
CAPTION = 0 # 캡션이 위에 있는 경우를 검사하기 위함
EPS = 1e-3

# max(w, EPS) -> w, h가 0 이하인 경우 "매우 작은 박스"로 취급 (크기 순서 유지)
def sanitize(boxes:torch.Tensor) -> torch.Tensor:
    return torch.cat([boxes[...,:2], boxes[...,2:].clamp_min(EPS)], dim=-1)

def cross(o, a, b):
    cross_product = (a[...,0] - o[...,0]) * (b[...,1] - o[...,1]) - (a[...,1] - o[...,1]) * (b[...,0] - o[...,0])
    return cross_product

# 가시성 그래프 생성 : 두 중심을 잇는 선이 다른 노드의 4개의 변과 교차하는지 확인 -> 연결
def visibility(boxes:torch.Tensor, max_dx:float=0.5, max_dy:float=0.7) -> torch.Tensor:
    n = boxes.size(0)
    x, y, w, h = boxes.unbind(-1)
    c = torch.stack([x + w / 2, y + h / 2], dim=-1) # [n, 2]
    corners = torch.stack([
        torch.stack([x, y], -1), torch.stack([x + w, y], -1),
        torch.stack([x + w, y + h], -1), torch.stack([x, y + h], -1)
    ], dim=1)

    A = c[:, None, None, None] # 선분시작
    B = c[None, :, None, None] # 선분종료
    C = corners[None, None]
    D = corners.roll(-1, dims=1)[None, None]

    hit = (cross(A, B, C) * cross(A, B, D) < 0) & (cross(C, D, A) * cross(C, D, B) < 0) # [i, j, k, 4]
    
    blocked = hit.any(-1) #[i, j, k]
    idx = torch.arange(n)
    blocked[idx, :, idx] = False
    blocked[:, idx, idx] = False
    blocked = blocked.any(-1) # [i, j]

    d = (c[:, None] - c[None]).abs()
    near = (d[...,0] <= max_dx) & (d[...,1] <= max_dy)
    eye = torch.eye(n, dtype=torch.bool)
    return near & ~blocked & ~eye

# 계층 그래프 생성 : 현재 노드보다 앞선 노드중 상위 레벨이 있는 경우 연결
def hierarchy(boxes:torch.Tensor, types:list[int], vis:torch.Tensor) -> list[tuple[int, int]]:
    n = len(types)
    cx = (boxes[:,0] + boxes[:,2] / 2).tolist()
    cy = (boxes[:,1] + boxes[:,3] / 2).tolist()
    
    level = [TYPE_LEVEL[t] for t in types]
    adj = vis.tolist()

    # connected component 생성 -> 하나의 컴포넌트가 보통 하나의 컬럼
    comp_of = [-1] * n
    comps = []
    for s in range(n):
        if comp_of[s] != -1:
            continue
        comp_of[s] = len(comps)
        comp, stack = [s], [s]
        while stack:
            u = stack.pop()
            for v in range(n):
                if adj[u][v] and comp_of[v] == -1:
                    comp_of[v] = len(comps)
                    comp.append(v)
                    stack.append(v)
        comps.append(comp)

    # 컴포넌트끼리는 x를 기준으로 정렬
    comps.sort(key=lambda comp:sum(cx[i] for i in comp) / len(comp))
    # 컴포넌트 내부 노드들은 y를 기준으로 정렬
    order = [i for comp in comps for i in sorted(comp, key=lambda i:cy[i])]

    edges = []
    parent = [None] * 4
    orphan_caption = None

    for i in order:
        lv = level[i]
        # 자기보다 상위 레벨 중 가장 가까운 레벨의 최근 노드
        p = next((parent[l] for l in range(lv-1, -1, -1) if parent[l] is not None), None) 

        if p is not None:
            edges.append((i, p))
        if lv == 2 and orphan_caption is not None:
            edges.append((orphan_caption, i)) 
        if types[i] == CAPTION and parent[2] is None:
            orphan_caption = i
        else:
            orphan_caption = None
        parent[lv] = i
        for l in range(lv + 1, 4):
            parent[l] = None
    return edges

def edge_attributes(boxes: torch.Tensor) -> torch.Tensor:
    ux, uy, uw, uh = boxes[:, :, None].unbind(-1)
    vx, vy, vw, vh = boxes[:, None, :].unbind(-1)

    # 거리 (3)
    dx = (vx + vw / 2) - (ux + uw / 2)
    dy = (vy + vh / 2) - (uy + uh / 2)
    dist = torch.sqrt(dx ** 2 + dy ** 2)

    # 크기 비율 (3) : 노이즈 박스는 비율이 수천 배까지 튀므로 log
    w_ratio = torch.log(vw / uw)
    h_ratio = torch.log(vh / uh)
    a_ratio = w_ratio + h_ratio

    # 방향 (2)
    safe = dist.clamp_min(1e-8)
    cos = torch.where(dist > 0, dx / safe, torch.ones_like(dx))
    sin = torch.where(dist > 0, dy / safe, torch.zeros_like(dy))

    # 정렬 오차 (4) : u의 너비/높이로 정규화, 상한 10
    l2l = ((ux - vx).abs() / uw).clamp_max(10)
    r2r = (((ux + uw) - (vx + vw)).abs() / uw).clamp_max(10)
    t2t = ((uy - vy).abs() / uh).clamp_max(10)
    b2b = (((uy + uh) - (vy + vh)).abs() / uh).clamp_max(10)

    # 겹침 (3)
    x_ov = (torch.minimum(ux + uw, vx + vw) - torch.maximum(ux, vx)).clamp_min(0)
    y_ov = (torch.minimum(uy + uh, vy + vh) - torch.maximum(uy, vy)).clamp_min(0)
    x_ov_r = x_ov / torch.minimum(uw, vw)
    y_ov_r = y_ov / torch.minimum(uh, vh)
    inter = x_ov * y_ov
    iou = inter / (uw * uh + vw * vh - inter)

    # 각 엣지에 대해서 15차원의 정보 반환
    return torch.stack([
        dx.abs(), dy.abs(), dist,
        w_ratio, h_ratio, a_ratio,
        cos, sin,
        l2l, r2r, t2t, b2b,
        x_ov_r, y_ov_r, iou,
    ], dim=-1)

class GraphBuilder:
    def __init__(self, T:int, mode:str="both", k:int=0, max_dx:float=0.5, max_dy:float=0.7):
        assert mode in ("complete", "vis", "hier", "both")
        self.T, self.mode, self.k = T, mode, k
        self.max_dx, self.max_dy = max_dx, max_dy

    @torch.no_grad()
    def __call__(self, types, x_t, t, node_mask, ab=None) -> tuple[torch.Tensor, torch.Tensor]:
        B, N = types.shape
        device = x_t.device
        pair = node_mask[:, :, None] & node_mask[:, None, :]
        eye = torch.eye(N, dtype=torch.bool, device=device)

        edge_mask = pair & ~eye
        edge_attr = torch.zeros(B, N, N, EDGE_DIM, device=device)

        rel = t < self.T - self.k
        if self.mode == "complete" or not rel.any():
            return edge_mask, edge_attr
        boxes = sanitize(to_page(x_t.detach().float())) # 디퓨전 좌표 [-1, 1] -> 페이지 좌표 [0, 1]
        rel_mask = self.relation_mask(types, boxes, node_mask, rel, ab) & pair
        attr = torch.cat([edge_attributes(boxes), torch.ones(B, N, N, 1, device=device)], dim=-1)

        r = rel[:, None, None]
        edge_mask = torch.where(r, rel_mask, edge_mask)
        edge_attr = torch.where((r & rel_mask)[..., None], attr, edge_attr)
        return edge_mask, edge_attr

    def relation_mask(self, types, boxes, node_mask, rel, ab=None) -> torch.Tensor:
        B, N = types.shape
        out = torch.zeros(B, N, N, dtype=torch.bool)
        types_c, boxes_c, n_c = types.cpu(), boxes.cpu(), node_mask.sum(1).cpu()
        scale = ab.sqrt().cpu().tolist() if ab is not None else [1.0] * B

        for b in rel.nonzero().flatten().tolist():
            n = int(n_c[b])
            bx, ty = boxes_c[b, :n], types_c[b, :n].tolist()
            vis = visibility(bx, self.max_dx * scale[b], self.max_dy * scale[b])

            m = torch.zeros(n, n, dtype=torch.bool)
            if self.mode in ("vis", "both"):
                m |= vis
            if self.mode in ("hier", "both"):
                for c, p in hierarchy(bx, ty, vis):
                    m[c, p] = m[p, c] = True
            out[b, :n, :n] = m
        return out.to(boxes.device)