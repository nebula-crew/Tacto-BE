"""M 계열: 근접 사진 모양 판정. (담당: A)"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..schema import Finding, MaskClass, NodeType, PhotoAnalysis, PhotoKind, Point, Verdict
from .clauses import CLAUSES, SUGGESTIONS


def check_m3(close: PhotoAnalysis) -> list[Finding]:
    """M3: 중심선의 꺾임·분기점 위치가 점형인가."""
    targets = [n for n in close.path.nodes if n.type in (NodeType.TURN, NodeType.BRANCH) and n.block is not None]
    if not targets:
        return []
    bad = [n for n in targets if n.block != MaskClass.DOT]
    if not bad:
        return [Finding(code="M3", verdict=Verdict.NORMAL, photo=PhotoKind.CLOSE,
                        message="꺾임·분기점이 점형으로 설치됨", confidence=0.7)]
    return [
        Finding(
            code="M3",
            verdict=Verdict.SUSPECTED,
            photo=PhotoKind.CLOSE,
            message=f"선형 블록이 {'갈라지는' if n.type == NodeType.BRANCH else '꺾이는'} 곳에서 점형 블록을 확인하지 못했어요.",
            location=n.pt,
            clause=CLAUSES["M3"],
            suggestion=SUGGESTIONS["M3"],
            confidence=0.7,  # TODO(A): ROI 분류 신뢰도 반영
        )
        for n in bad
    ]


def check_m4(close: PhotoAnalysis, label: Optional[np.ndarray] = None) -> list[Finding]:
    """서로 마주 보는 다른 선형 성분의 끝점을 단절 후보로 확인한다."""
    path = close.path
    scale = min(path.width, path.height)
    ends = []
    for node in path.nodes:
        edges = path.edges_of(node.id)
        if node.type != NodeType.END or len(edges) != 1 or edges[0].block != MaskClass.LINEAR:
            continue
        edge = edges[0]
        if len(edge.polyline) < 2 or edge.length_px < scale * 0.03:
            continue
        pts = edge.polyline if edge.u == node.id else list(reversed(edge.polyline))
        origin = np.array([node.pt.x, node.pt.y], dtype=float)
        other = np.array([pts[-1].x, pts[-1].y], dtype=float)
        direction = origin - other
        length = np.linalg.norm(direction)
        if length:
            ends.append((node, origin, direction / length))
    candidates = []
    for i, (a, p, da) in enumerate(ends):
        for b, q, db in ends[i + 1:]:
            if a.component == b.component:
                continue
            delta = q - p
            distance = float(np.linalg.norm(delta))
            if not scale * 0.005 <= distance <= scale * 0.15:
                continue
            unit = delta / distance
            alignment = min(float(np.dot(da, unit)), float(np.dot(db, -unit)))
            if alignment < 0.85:
                continue
            candidates.append((distance, alignment, a, b, p, q))
    findings, used = [], set()
    for distance, alignment, a, b, p, q in sorted(candidates, key=lambda item: item[0]):
        if a.id in used or b.id in used:
            continue
        used.update((a.id, b.id))
        mid = (p + q) / 2
        margin = scale * 0.04
        near_boundary = any(x < margin or x > limit - margin for pt in (p, q) for x, limit in zip(pt, (path.width, path.height)))
        unknown = label is None or label.shape != (path.height, path.width)
        if not unknown:
            # A dot block in the gap is a valid connector, not a missing linear block.
            samples = np.linspace(p, q, max(2, int(distance) + 1)).round().astype(int)
            samples[:, 0] = np.clip(samples[:, 0], 0, path.width - 1)
            samples[:, 1] = np.clip(samples[:, 1], 0, path.height - 1)
            if np.any(label[samples[:, 1], samples[:, 0]] == int(MaskClass.DOT)):
                continue
        hold = near_boundary or unknown
        findings.append(Finding(code="M4", verdict=Verdict.HOLD if hold else Verdict.SUSPECTED,
            photo=path.photo, location=Point(x=float(mid[0]), y=float(mid[1])),
            message="사진 끝에 걸리거나 블록이 선명하지 않아 연결 상태를 확인하기 어려워요." if hold else "선형 블록이 이어져야 할 구간에 끊긴 부분이 있는 것으로 보여요.",
            confidence=alignment if not hold else 0.0,
            clause=None if hold else CLAUSES["M4"], suggestion=None if hold else SUGGESTIONS["M4"]))
    return findings


def check_m(photo: PhotoAnalysis, label: Optional[np.ndarray]) -> list[Finding]:
    """사진 한 장용 M 계열. 시경의 shape_rules.analyze_m3(main)가 있으면 그걸 쓰고, 없으면 위의 check_m3."""
    try:
        from ..shape_rules import analyze_m3
    except ImportError:
        analyze_m3 = None
    if analyze_m3 is not None and label is not None:
        return analyze_m3(label, photo.path) + check_m4(photo, label)
    return check_m3(photo) + check_m4(photo, label)
