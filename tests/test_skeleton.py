"""Path structure regression tests (dot-tile junctions are PASS nodes)."""

import pytest
import cv2
import numpy as np

from pipeline import synth
from pipeline.schema import MaskClass, NodeType, PhotoKind
from pipeline.skeleton import end_direction, extract_path


def _count(path, t):
    return sum(n.type == t for n in path.nodes)


@pytest.mark.parametrize(
    "scene, dot_corner",
    [
        (synth.straight, False),
        (synth.l_turn, True),
        (lambda: synth.l_turn(False), False),
        (synth.t_branch, False),
        (lambda: synth.t_branch(False), False),
    ],
)
def test_node_counts(scene, dot_corner):
    label, exp = scene()
    path = extract_path(label, PhotoKind.CLOSE)
    assert _count(path, NodeType.END) == exp["ends"]
    assert _count(path, NodeType.BRANCH) == exp["branches"]
    assert len(path.component_area_px) == 1

    if dot_corner:
        # 점형이 L자 모서리에 있으면 선형 끝점 2개가 점형 노드에 연결된다.
        # 점형 노드는 degree=2 이므로 PASS이고, TURN 노드가 아니다.
        assert _count(path, NodeType.TURN) == 0
        dot_passes = [
            n for n in path.nodes
            if n.block == MaskClass.DOT and n.type == NodeType.PASS
        ]
        assert len(dot_passes) == 1
        assert len(path.edges_of(dot_passes[0].id)) == 2
    else:
        assert _count(path, NodeType.TURN) == exp["turns"]


@pytest.mark.parametrize(
    "with_dot, expected_type, expected_block",
    [
        (True, NodeType.PASS, MaskClass.DOT),
        (False, NodeType.TURN, MaskClass.LINEAR),
    ],
)
def test_turn_location(with_dot, expected_type, expected_block):
    label, _ = synth.l_turn(with_dot)
    path = extract_path(label, PhotoKind.CLOSE)
    candidates = [
        n for n in path.nodes
        if n.type == expected_type and n.block == expected_block
    ]
    assert len(candidates) == 1
    corner = candidates[0]
    assert abs(corner.pt.x - 200) < 15
    assert abs(corner.pt.y - 250) < 15


def test_end_direction_points_outward():
    label, _ = synth.straight()
    path = extract_path(label, PhotoKind.CLOSE)
    top = min((n for n in path.nodes if n.type == NodeType.END), key=lambda n: n.pt.y)
    dx, dy = end_direction(path, top.id)
    assert dy < -0.95  # 위쪽(바깥)을 향함


def test_path_json_serializable():
    label, _ = synth.t_branch()
    path = extract_path(label, PhotoKind.CLOSE)
    assert path.model_dump_json()


def test_clipped_convex_corridor_does_not_create_y_branch():
    label = np.zeros((360, 640), dtype=np.uint8)
    cv2.fillPoly(label, [np.array([[205, 145], [315, 145], [325, 359], [130, 359]])], 1)
    cv2.rectangle(label, (195, 95), (325, 144), 2, -1)
    path = extract_path(label, PhotoKind.WIDE)
    assert _count(path, NodeType.BRANCH) == 0
    assert _count(path, NodeType.END) == 2


def test_wide_short_strip_uses_opposite_dot_contacts():
    label = np.zeros((360, 640), dtype=np.uint8)
    label[140:220, 180:440] = 1
    label[100:140, 100:520] = 2
    label[220:260, 180:440] = 2
    path = extract_path(label, PhotoKind.WIDE)
    assert len(path.edges) == 1
    assert len(path.nodes) == 2
    assert all(n.block == MaskClass.DOT for n in path.nodes)
    pts = path.edges[0].polyline
    assert max(p.x for p in pts) - min(p.x for p in pts) < 10
