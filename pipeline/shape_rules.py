"""TactileCheck M3 꺾임·분기점 점형 누락 *후보* 판정.

입력은 기존 ``extract_path(label, PhotoKind.CLOSE)`` 결과와 YOLO 라벨 마스크.
- 점형 노드는 연결된 영역을 대표하며, 점형 내부를 골격화하지 않음.
- 경로의 단순 굴곡을 오류라고 판단하지 않도록 '긴 팔의 안정된 방향' 검증.
- 후보가 없거나 모호하면 정상이라고 확정하지 않고 HOLD.
- 설치 기준의 적법성 판정이 아니라 오류 의심 휴리스틱.

본 모듈은 기존 schema.py/skeleton.py 수정을 요구하지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import cv2

from .schema import Finding, MaskClass, NodeType, PathNode, PathStructure, PhotoKind, Point, Verdict


@dataclass(frozen=True)
class M3Config:
    min_arm_px: float = 50.0         # 실제 연결 경로 길이 최소 조건
    sample_arm_px: float = 110.0     # 분기/꺾임 기준점으로부터 방향 추정 거리
    min_straightness: float = 0.82  # 표본 경로의 직선성 (양끝 거리 / 누적 길이)
    min_turn_deg: float = 55.0      # 180° 직진 대비 방향 변화량
    min_branch_separation_deg: float = 30.0
    dot_near_px: float = 15.0      # 이 안에 점형 마스크가 있으면 존재 확인
    dot_ambiguous_px: float = 45.0 # 그 이상 떨어지면 누락 의심; 중간은 보류


@dataclass(frozen=True)
class M3Candidate:
    node_id: int
    kind: str                        # "turn" 또는 "branch"
    location: Point                  # 원본 이미지 좌표계
    direction_change_deg: float      # turn: 180° 대비 굴절, branch: 최소 팔 각도
    arm_lengths_px: tuple[float, ...]
    dot_distance_px: float | None
    verdict: Verdict
    reason: str


def _arm_outward_points(path: PathStructure, node: PathNode, edge_id: int) -> np.ndarray:
    edge = next(e for e in path.edges if e.id == edge_id)
    coords = np.array([[p.x, p.y] for p in edge.polyline], dtype=np.float64)
    if edge.v == node.id and edge.u != node.id:
        coords = coords[::-1]
    elif edge.u != node.id:
        raise ValueError('해당 노드와 연결되지 않은 엣지입니다')
    return coords


def _vector_and_length(coords: np.ndarray, cfg: M3Config):
    if len(coords) < 2:
        return None
    segment = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(segment)]
    total = float(cumulative[-1])
    if total < cfg.min_arm_px:
        return None
    distance = min(total, cfg.sample_arm_px)
    i = min(int(np.searchsorted(cumulative, distance, side="left")), len(cumulative)-1)
    if i == 0:
        return None
    previous = i - 1
    span = cumulative[i] - cumulative[previous]
    alpha = (distance - cumulative[previous]) / span if span > 1e-9 else 0.0
    sample = coords[previous] * (1 - alpha) + coords[i] * alpha
    v = sample - coords[0]
    displacement = float(np.linalg.norm(v))
    if displacement < 1e-6 or displacement / distance < cfg.min_straightness:
        return None
    return v / displacement, total


def _linear_arm_from_dot(
    label: np.ndarray, path: PathStructure, node: PathNode, edge_id: int, cfg: M3Config
):
    """점형 중심→선형 끝점의 *가상 연결선*을 제외한 선형 실제 진행 방향.

    PathEdge.polyline은 DOT 중심까지 확장되므로 기존 첫 110px 기준
    각도는 넓은 점형 영역에서 실제 유도블록의 방향이 아니다.
    각 구간을 2px 간격으로 샘플링하고, 가장 가까운 충분히 긴 선형
    연속 영역에서 진행 방향을 구한다. 점형 내부의 대각선 연결은 무시.
    """
    points = _arm_outward_points(path, node, edge_id)
    if len(points) < 2:
        return None
    samples = [points[0]]
    for p, q in zip(points[:-1], points[1:]):
        length = float(np.linalg.norm(q-p))
        num = max(1, int(math.ceil(length / 2.0)))
        for i in range(1, num + 1):
            samples.append(p + (q-p) * (i / num))
    samples = np.asarray(samples, dtype=np.float64)
    h, w = label.shape
    xi = np.rint(samples[:, 0]).astype(np.int32)
    yi = np.rint(samples[:, 1]).astype(np.int32)
    valid = (0 <= xi) & (xi < w) & (0 <= yi) & (yi < h)
    is_linear = np.zeros(len(samples), dtype=bool)
    is_linear[valid] = label[yi[valid], xi[valid]] == MaskClass.LINEAR

    # 선형 영역의 가장 가까운 연속 구간 중 길이가 충분한 것을 사용.
    starts = np.flatnonzero(is_linear & ~np.r_[False, is_linear[:-1]])
    ends = np.flatnonzero(is_linear & ~np.r_[is_linear[1:], False])
    for start, end in zip(starts, ends):
        segment = samples[start:end+1]
        arm = _vector_and_length(segment, cfg)
        if arm is not None:
            return arm
    return None


def _angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(a @ b, -1.0, 1.0))))


def _dot_distance(label: np.ndarray, pt: Point, radius: float) -> float | None:
    """후보 주변의 가장 가까운 dot 픽셀 거리. ROI 바깥이면 None.

    전체 원본 해상도에 대한 distanceTransform 대신 후보별 ROI만 계산한다.
    """
    h, w = label.shape
    x, y = float(pt.x), float(pt.y)
    r = int(math.ceil(radius))
    x0, x1 = max(0, int(math.floor(x))-r), min(w, int(math.ceil(x))+r+1)
    y0, y1 = max(0, int(math.floor(y))-r), min(h, int(math.ceil(y))+r+1)
    yy, xx = np.where(label[y0:y1, x0:x1] == MaskClass.DOT)
    if not len(xx):
        return None
    distance = float(np.hypot(xx+x0-x, yy+y0-y).min())
    return distance if distance <= radius else None


def _mask_branch_exits(linear_components: np.ndarray, pt: Point, local_radius: float) -> int | None:
    """Count substantial mask arms outside the junction, independently of its skeleton.

    A rectangular strip can skeletonize into a Y near its end. A circle beyond
    the strip's half-width intersects only one/two arms, unlike a genuine T/Y.
    Cropped neighborhoods cannot establish the number of arms.
    """
    h, w = linear_components.shape
    x, y = round(pt.x), round(pt.y)
    if not (0 <= x < w and 0 <= y < h) or local_radius < 3:
        return None
    component = linear_components[y, x]
    available_radius = min(x, y, w - 1 - x, h - 1 - y)
    radius = min(max(12.0, 2.5 * local_radius), available_radius)
    if not component or radius < max(12.0, 1.8 * local_radius):
        return None
    count = max(180, int(math.ceil(2 * math.pi * radius)))
    angles = np.arange(count) * (2 * math.pi / count)
    xx = np.rint(x + radius * np.cos(angles)).astype(int)
    yy = np.rint(y + radius * np.sin(angles)).astype(int)
    inside = linear_components[yy, xx] == component
    if inside.all():
        return 0  # A solid patch has no separately supported branches.
    # Break the circular array at a background sample before counting runs.
    inside = np.roll(inside, -int(np.flatnonzero(~inside)[0]))
    transitions = np.diff(np.r_[False, inside, False].astype(np.int8))
    lengths = np.flatnonzero(transitions == -1) - np.flatnonzero(transitions == 1)
    min_arc = max(3.0, 0.5 * local_radius)
    return int(np.count_nonzero(lengths * (2 * math.pi * radius / count) >= min_arc))


def find_m3_candidates(
    label: np.ndarray, path: PathStructure, cfg: M3Config | None = None
) -> list[M3Candidate]:
    cfg = cfg or M3Config()
    if label.dtype != np.uint8 or label.ndim != 2:
        raise ValueError("label은 (H,W) uint8 0/1/2/... 라벨 마스크여야 합니다")
    if label.shape != (path.height, path.width):
        raise ValueError("path와 label의 해상도가 서로 다릅니다")
    if cfg.dot_near_px < 0 or cfg.dot_ambiguous_px < cfg.dot_near_px:
        raise ValueError("dot 거리 임계값이 올바르지 않습니다")
    result: list[M3Candidate] = []
    linear = (label == MaskClass.LINEAR).astype(np.uint8)
    radius_map = cv2.distanceTransform(linear, cv2.DIST_L2, 5)
    _, linear_components = cv2.connectedComponents(linear, connectivity=8)
    _, dot_components = cv2.connectedComponents((label == MaskClass.DOT).astype(np.uint8), connectivity=8)
    for node in path.nodes:
        related = path.edges_of(node.id)
        if len(related) < 2:
            continue
        # 단순 polyline의 끝/노드를 모두 turn으로 취급하지 않는다.
        # 구조상 2-arm 또는 3-arm 이상인 노드만 검증한다.
        arms = []
        for e in related:
            # 점형 영역의 중심까지 이어진 가상 대각선이 각도를 왜곡하지
            # 않도록 실제 LINEAR 픽셀 위의 진행 방향을 우선 사용.
            arm = (
                _linear_arm_from_dot(label, path, node, e.id, cfg)
                if node.block == MaskClass.DOT
                else _vector_and_length(_arm_outward_points(path, node, e.id), cfg)
            )
            if arm is not None:
                arms.append(arm)
        # 짧거나 심하게 휘어진 팔은 연결의 신뢰할 만한 증거가 아니다.
        if len(arms) != len(related):
            continue
        dirs = [v for v, _ in arms]
        lengths = tuple(round(total, 1) for _, total in arms)
        if len(arms) == 2:
            change = 180.0 - _angle_degrees(dirs[0], dirs[1])
            if change < cfg.min_turn_deg:
                continue
            kind = "turn"
        else:
            pair_angles = [
                _angle_degrees(dirs[i], dirs[j])
                for i in range(len(dirs)) for j in range(i+1, len(dirs))
            ]
            if min(pair_angles) < cfg.min_branch_separation_deg:
                continue
            change = min(pair_angles)
            kind = "branch"
        # node.block은 골격화 단계의 추정치라 직접 마스크도 확인한다.
        # DOT 노드는 점형 연결 성분을 대표하므로 정상 점형이 확인된 상태.
        x, y = round(node.pt.x), round(node.pt.y)
        dot_node = (node.block == MaskClass.DOT and
                    0 <= x < path.width and 0 <= y < path.height and
                    label[y, x] == MaskClass.DOT)
        # A wide block can push a medial-axis junction away from the true
        # contact. Nearby dot evidence makes the result uncertain, not normal.
        local_radius = float(radius_map[y, x]) if 0 <= x < path.width and 0 <= y < path.height else 0.0
        search_radius = max(cfg.dot_ambiguous_px, min(1.5 * local_radius, 0.15 * min(path.width, path.height)))
        d = 0.0 if dot_node else _dot_distance(label, node.pt, search_radius)
        if d is not None and not dot_node:
            # Proximity alone is insufficient: an unrelated dot on the other
            # side of a background gap must not hide a missing-dot junction.
            component = linear_components[y, x] if 0 <= x < path.width and 0 <= y < path.height else 0
            touching = cv2.dilate((linear_components == component).astype(np.uint8),
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))) != 0 if component else np.zeros(label.shape, bool)
            ids = np.unique(dot_components[touching])
            ids = ids[ids > 0]
            nearby_connected = np.where(np.isin(dot_components, ids), int(MaskClass.DOT), 0).astype(np.uint8)
            d = _dot_distance(nearby_connected, node.pt, search_radius) if len(ids) else None
        # A wide linear region cut by the image border can skeletonize into
        # two arms ending on that same border. It is not reliable evidence of
        # a real branch; keep the candidate but request a wider photo.
        boundary_counts = [0, 0, 0, 0]
        margin = max(2.0, min(path.width, path.height) * 0.02)
        if kind == "branch":
            for edge in related:
                other_id = edge.v if edge.u == node.id else edge.u
                other = path.node(other_id)
                if other.type != NodeType.END:
                    continue
                for i, near in enumerate((other.pt.x <= margin, other.pt.x >= path.width - 1 - margin,
                                          other.pt.y <= margin, other.pt.y >= path.height - 1 - margin)):
                    boundary_counts[i] += int(near)
        if max(boundary_counts) >= 2 and not dot_node:
            verdict = Verdict.HOLD
            reason = "사진 경계에서 선형 영역이 잘려 분기 여부를 확인할 수 없음"
        elif dot_node or (d is not None and d <= cfg.dot_near_px):
            verdict = Verdict.NORMAL
            reason = "블록이 꺾이거나 갈라지는 위치에서 연결된 점형 블록을 확인했어요."
        elif kind == "branch" and (exits := _mask_branch_exits(linear_components, node.pt, local_radius)) is not None and exits < 3:
            verdict = Verdict.HOLD
            reason = "중심선은 갈라져 보이지만 블록 영역에서 세 방향의 경로가 확인되지 않아, 점형 누락 판정을 보류했어요."
        elif d is not None:
            verdict = Verdict.HOLD
            reason = "연결된 점형 블록은 보이지만 꺾임·분기 위치와 맞닿는지 확인하기 어려워요."
        else:
            # missing dot is *suspected*, not a definitive legal violation.
            verdict = Verdict.SUSPECTED
            reason = f"선형 블록이 {'갈라지는' if kind == 'branch' else '꺾이는'} 위치에서 연결된 점형 블록을 확인하지 못했어요."
        result.append(M3Candidate(
            node_id=node.id, kind=kind, location=node.pt,
            direction_change_deg=round(change, 1), arm_lengths_px=lengths,
            dot_distance_px=None if d is None else round(d, 1),
            verdict=verdict, reason=reason,
        ))
    return result


def analyze_m3(
    label: np.ndarray,
    path: PathStructure,
    cfg: M3Config | None = None,
) -> list[Finding]:
    """기존 schema.Finding 호환형 결과. 후보가 없으면 HOLD 1개를 반환."""
    candidates = find_m3_candidates(label, path, cfg)
    if not candidates:
        return [Finding(
            code="M3", verdict=Verdict.HOLD, photo=path.photo,
            message="이 사진에서는 신뢰할 수 있는 꺾임·분기 후보가 확인되지 않았습니다 (M3 대상 불명확).",
            suggestion=None,
            confidence=0.0,
        )]
    output = []
    for c in candidates:
        output.append(Finding(
            code="M3", verdict=c.verdict, photo=path.photo,
            location=c.location, message=c.reason,
            suggestion=("현장 확인 및 재촬영이 필요합니다."
                        if c.verdict != Verdict.NORMAL else None),
            confidence=0.0,  # 규칙의 점수는 검증 전이므로 신뢰도 수치를 임의 부여하지 않음
        ))
    return output
