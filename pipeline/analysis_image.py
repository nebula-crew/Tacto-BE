"""관리자용 분석 이미지: 판정에 쓴 영역·경로·연장선·판정 위치를 원본 사진 위에 한 장으로 겹쳐 그린다.

시민 앱에는 쓰지 않는다 (원본 + 설명만). 관리자가 "AI가 어디를 보고 이렇게 판단했는지" 확인하는 용도.
판정 로직은 건드리지 않고, run_photo_detailed가 돌려준 라벨·경로 분석·결과를 그대로 그림으로 옮긴다.

색 (관리자 화면 범례와 같음, RGB):
  횡단보도 #E63CC8 · 차도 #1E8CE6 · 선형 블록 #FFC814 · 점형 블록 #FF641E
  선형 중심선 #3CE650 · 점형 범위 #00C8E6(점선) · 노드: 끝점 흰색 / 꺾임 노랑 / 분기 보라 / 점형 위 주황
  연장선: 차도에 닿음 빨강 / 횡단보도 입구에 닿음 초록 · 판정 위치: 원 + 규칙 코드 (오류 의심 빨강 / 정상 초록 / 보류 회색)
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from .schema import InspectionResult, MaskClass, NodeType, PhotoAnalysis

log = logging.getLogger(__name__)

# BGR
_C = {
    "crosswalk": (200, 60, 230), "road": (230, 140, 30), "linear": (20, 200, 255), "dot": (30, 100, 255),
    "line": (80, 230, 60), "extent": (230, 200, 0), "end": (255, 255, 255), "turn": (0, 220, 255),
    "branch": (255, 0, 200), "hit_road": (40, 40, 235), "hit_cw": (80, 200, 60),
    "suspected": (40, 40, 220), "normal": (60, 170, 60), "hold": (130, 130, 130),
}


def _tint(img: np.ndarray, mask: np.ndarray, color, alpha: float) -> None:
    if mask.any():
        img[mask] = ((1 - alpha) * img[mask] + alpha * np.array(color)).astype(np.uint8)


def _dashed(img, p, q, color, t, dash):
    length = math.dist(p, q)
    n = max(1, int(length // dash))
    for i in range(0, n, 2):
        a = (p[0] + (q[0] - p[0]) * i / n, p[1] + (q[1] - p[1]) * i / n)
        b = (p[0] + (q[0] - p[0]) * min(i + 1, n) / n, p[1] + (q[1] - p[1]) * min(i + 1, n) / n)
        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, t, cv2.LINE_AA)


def draw_analysis(image_bgr: np.ndarray, label: np.ndarray, analysis: PhotoAnalysis,
                  result: InspectionResult) -> np.ndarray:
    """분석 이미지 (원본과 같은 크기, BGR)."""
    h, w = image_bgr.shape[:2]
    t = max(2, round(max(h, w) / 400))
    # 배경은 살짝 흐리게 해서 색칠한 영역이 잘 보이게
    vis = (image_bgr.astype(np.float32) * 0.75).astype(np.uint8)

    _tint(vis, label == int(MaskClass.ROAD), _C["road"], 0.40)
    _tint(vis, label == int(MaskClass.CROSSWALK), _C["crosswalk"], 0.45)
    _tint(vis, label == int(MaskClass.LINEAR), _C["linear"], 0.60)
    _tint(vis, label == int(MaskClass.DOT), _C["dot"], 0.60)

    path = analysis.path
    try:  # 점형 범위·점형 경로 (디버그 화면·서비스 오버레이와 같은 그림용 경로)
        from .dot_paths import dot_pad_extents, expand_dot_paths

        for ext in dot_pad_extents(path, label):
            for line in ext["polylines"]:
                for a, b in zip(line, line[1:]):
                    _dashed(vis, (a["x"], a["y"]), (b["x"], b["y"]), _C["extent"], max(1, t // 2 + 1), 4 * t)
        layout = expand_dot_paths(path, label)
    except Exception as e:  # 그림용 경로가 실패해도 판정 경로는 그린다
        log.warning("점형 범위 그리기 실패: %s", e)
        layout = path

    for e in layout.edges:
        pts = np.array([[p.x, p.y] for p in e.polyline], np.int32)
        if len(pts) < 2:
            continue
        if e.block == MaskClass.DOT:
            for a, b in zip(pts, pts[1:]):
                _dashed(vis, tuple(a), tuple(b), _C["dot"], t + 1, 5 * t)
        else:
            cv2.polylines(vis, [pts], False, _C["line"], t + 1, cv2.LINE_AA)
    for n in layout.nodes:
        color = {NodeType.END: _C["end"], NodeType.TURN: _C["turn"], NodeType.BRANCH: _C["branch"]}.get(n.type)
        if n.block == MaskClass.DOT and n.type != NodeType.ISOLATED:
            color = _C["dot"]
        if color is None:
            continue
        c = (int(n.pt.x), int(n.pt.y))
        cv2.circle(vis, c, 3 * t, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(vis, c, int(2.3 * t), color, -1, cv2.LINE_AA)

    try:  # 끝점 연장선 (C 계열 판정에 쓰는 선)
        from .rules.c_rules import ContextConfig, cast_rays

        for ray in cast_rays(analysis, label, ContextConfig()):
            if ray.hit is None or ray.hit_pt is None:
                continue
            color = _C["hit_road"] if ray.hit == MaskClass.ROAD else _C["hit_cw"]
            _dashed(vis, ray.tip, ray.hit_pt, color, t + 1, 6 * t)
            hp = (int(ray.hit_pt[0]), int(ray.hit_pt[1]))
            cv2.circle(vis, hp, int(3.5 * t), color, -1, cv2.LINE_AA)
            cv2.circle(vis, hp, int(3.5 * t), (255, 255, 255), max(1, t // 2), cv2.LINE_AA)
    except Exception as e:
        log.warning("연장선 그리기 실패: %s", e)

    # 판정 위치: 원 + 규칙 코드 (관리자 화면 AI 판정 목록의 코드와 같음)
    for f in result.findings:
        if f.location is None:
            continue
        color = _C.get(f.verdict.value, _C["hold"])
        c = (int(f.location.x), int(f.location.y))
        cv2.circle(vis, c, 11 * t, color, t + 1, cv2.LINE_AA)
        cv2.circle(vis, c, 6 * t, color, -1, cv2.LINE_AA)
        label_txt = f.code
        (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45 * t, t)
        org = (min(max(0, c[0] + 12 * t), w - tw - 2), max(th + 2, c[1] - 12 * t))
        cv2.putText(vis, label_txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45 * t, (0, 0, 0), t + 2, cv2.LINE_AA)
        cv2.putText(vis, label_txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45 * t, (255, 255, 255), t, cv2.LINE_AA)
    return vis
