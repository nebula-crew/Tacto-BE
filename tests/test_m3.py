import cv2
import numpy as np
import pytest

from pipeline.schema import PhotoKind, Verdict
from pipeline.shape_rules import analyze_m3, find_m3_candidates
from pipeline.skeleton import extract_path


def make_mask(kind, dot=False):
    m = np.zeros((480, 480), dtype=np.uint8)
    if kind == 'L':
        cv2.line(m, (120, 400), (120, 120), 1, 33)
        cv2.line(m, (120, 120), (400, 120), 1, 33)
        if dot: cv2.rectangle(m, (82, 82), (158, 158), 2, -1)
    elif kind == 'T':
        cv2.line(m, (80, 120), (400, 120), 1, 33)
        cv2.line(m, (240, 120), (240, 400), 1, 33)
        if dot: cv2.rectangle(m, (202, 82), (278, 158), 2, -1)
    else:
        cv2.line(m, (240, 70), (240, 400), 1, 33)
    return m


def check(kind, dot, expected):
    mask = make_mask(kind, dot)
    path = extract_path(mask, PhotoKind.CLOSE)
    findings = analyze_m3(mask, path)
    assert any(f.verdict == expected for f in findings), [f.model_dump() for f in findings]


def test_L_missing_dot(): check('L', False, Verdict.SUSPECTED)
def test_L_with_dot(): check('L', True, Verdict.NORMAL)
def test_T_missing_dot(): check('T', False, Verdict.SUSPECTED)
def test_T_with_dot(): check('T', True, Verdict.NORMAL)
def test_straight_no_turn(): check('straight', False, Verdict.HOLD)


def test_wide_branch_with_nearby_dot_is_held_not_declared_missing():
    from pipeline.schema import MaskClass, NodeType, PathEdge, PathNode, PathStructure, Point
    label = np.zeros((800, 800), dtype=np.uint8)
    cv2.rectangle(label, (200, 200), (600, 600), int(MaskClass.LINEAR), -1)
    cv2.rectangle(label, (200, 601), (600, 700), int(MaskClass.DOT), -1)
    coords = [(400, 500), (400, 260), (220, 500), (580, 500)]
    nodes = [PathNode(id=i, type=NodeType.BRANCH if i == 0 else NodeType.END,
                      pt=Point(x=x, y=y), block=MaskClass.LINEAR, component=1)
             for i, (x, y) in enumerate(coords)]
    edges = [PathEdge(id=i-1, u=0, v=i, polyline=[nodes[0].pt, nodes[i].pt],
                      length_px=180, block=MaskClass.LINEAR) for i in range(1, 4)]
    path = PathStructure(photo=PhotoKind.CLOSE, width=800, height=800, nodes=nodes, edges=edges)
    candidates = find_m3_candidates(label, path)
    assert len(candidates) == 1
    assert candidates[0].verdict == Verdict.HOLD
    assert candidates[0].dot_distance_px == 101
    # A disconnected dot is not accepted as evidence. The rectangular mask
    # still lacks a real three-arm junction, so its artificial Y is held.
    label[601:701, :] = 0
    cv2.rectangle(label, (200, 625), (600, 700), int(MaskClass.DOT), -1)
    path.nodes[0].pt.y = 525
    candidates = find_m3_candidates(label, path)
    assert candidates[0].dot_distance_px is None
    assert candidates[0].verdict == Verdict.HOLD


@pytest.mark.parametrize('kind', ['L', 'T'])
@pytest.mark.parametrize('rotation', [0, 1, 2, 3])
@pytest.mark.parametrize('dot', [False, True])
def test_real_junctions_preserved_at_all_right_angle_rotations(kind, rotation, dot):
    label = np.rot90(make_mask(kind, dot), rotation).copy()
    candidates = find_m3_candidates(label, extract_path(label, PhotoKind.CLOSE))
    expected = Verdict.NORMAL if dot else Verdict.SUSPECTED
    assert any(c.verdict == expected for c in candidates)
    if dot:
        assert all(c.verdict != Verdict.SUSPECTED for c in candidates)


def test_rectangular_mask_does_not_support_synthetic_y_branch():
    from pipeline.schema import MaskClass, NodeType, PathEdge, PathNode, PathStructure, Point
    label = np.zeros((600, 600), np.uint8)
    cv2.rectangle(label, (200, 80), (400, 450), int(MaskClass.LINEAR), -1)
    coords = [(300, 350), (300, 90), (210, 440), (390, 440)]
    nodes = [PathNode(id=i, type=NodeType.BRANCH if i == 0 else NodeType.END,
                      pt=Point(x=x, y=y), block=MaskClass.LINEAR, component=1)
             for i, (x, y) in enumerate(coords)]
    edges = [PathEdge(id=i-1, u=0, v=i, polyline=[nodes[0].pt, nodes[i].pt],
                      length_px=180, block=MaskClass.LINEAR) for i in range(1, 4)]
    path = PathStructure(photo=PhotoKind.CLOSE, width=600, height=600, nodes=nodes, edges=edges)
    candidate, = find_m3_candidates(label, path)
    assert candidate.verdict == Verdict.HOLD
    assert '세 방향' in candidate.reason


def test_disconnected_dot_does_not_hide_real_missing_junction():
    label = make_mask('T')
    # Nearby in Euclidean distance but separated from the guide by >8 pixels.
    cv2.rectangle(label, (225, 75), (255, 90), 2, -1)
    candidates = find_m3_candidates(label, extract_path(label, PhotoKind.CLOSE))
    assert any(c.verdict == Verdict.SUSPECTED and c.dot_distance_px is None for c in candidates)


def test_dot_outside_circular_search_radius_is_not_returned():
    from pipeline.schema import Point
    from pipeline.shape_rules import _dot_distance
    label = np.zeros((100, 100), np.uint8)
    label[80, 80] = 2
    assert _dot_distance(label, Point(x=50, y=50), 35) is None


def test_two_branch_arms_at_same_image_border_are_held():
    from pipeline.schema import MaskClass, NodeType, PathEdge, PathNode, PathStructure, Point
    coords = [(200, 260), (200, 100), (100, 397), (300, 394)]
    nodes = [PathNode(id=i, type=NodeType.BRANCH if i == 0 else NodeType.END,
                      pt=Point(x=x, y=y), block=MaskClass.LINEAR, component=1)
             for i, (x, y) in enumerate(coords)]
    edges = [PathEdge(id=i-1, u=0, v=i, polyline=[nodes[0].pt, nodes[i].pt],
                      length_px=170, block=MaskClass.LINEAR) for i in range(1, 4)]
    path = PathStructure(photo=PhotoKind.CLOSE, width=400, height=400, nodes=nodes, edges=edges)
    label = np.ones((400, 400), dtype=np.uint8)
    candidates = find_m3_candidates(label, path)
    assert len(candidates) == 1 and candidates[0].verdict == Verdict.HOLD
    assert '경계' in candidates[0].reason


def test_mismatch_rejected():
    m = make_mask('L')
    path = extract_path(m, PhotoKind.CLOSE)
    try:
        find_m3_candidates(m[:300], path)
    except ValueError:
        pass
    else:
        raise AssertionError('해상도 불일치를 검사하지 못함')
