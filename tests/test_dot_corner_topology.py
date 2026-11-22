"""Dot-corner diagonal bridges must not become several missing-dot junctions."""
import cv2
import numpy as np
import pytest

from pipeline.schema import MaskClass, NodeType, PhotoKind, Verdict
from pipeline.shape_rules import find_m3_candidates
from pipeline.skeleton import extract_path, prepare_linear_mask


@pytest.mark.parametrize('angle', [0, 30, 90, 150, 225])
@pytest.mark.parametrize('scale', [.7, 1., 1.5])
def test_corner_bridge_becomes_one_real_dot_turn(angle, scale):
    label = np.zeros((640, 640), np.uint8)
    label[100:300, 240:300] = MaskClass.LINEAR
    label[300:360, 300:500] = MaskClass.LINEAR
    label[300:360, 240:300] = MaskClass.DOT
    label[60:100, 240:300] = MaskClass.DOT
    matrix = cv2.getRotationMatrix2D((320, 320), angle, 1)
    label = cv2.warpAffine(label, matrix, (640, 640), flags=cv2.INTER_NEAREST)
    label = cv2.resize(label, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    original = label.copy()
    path = extract_path(label, PhotoKind.WIDE)
    candidates = find_m3_candidates(label, path)
    assert len(candidates) == 1, candidates
    assert candidates[0].verdict == Verdict.NORMAL
    assert not any(n.type == NodeType.BRANCH and n.block == MaskClass.LINEAR for n in path.nodes)
    assert np.array_equal(label, original)


@pytest.mark.parametrize('angle', [0, 25, 75, 140])
@pytest.mark.parametrize('dot', [False, True])
def test_oblique_true_y_junction_preserves_missing_dot_detection(angle, dot):
    label = np.zeros((720, 720), np.uint8)
    for tip in [(180, 150), (540, 150), (360, 610)]:
        cv2.line(label, (360, 360), tip, 1, 35)
    if dot:
        cv2.circle(label, (360, 360), 45, 2, -1)
    label = cv2.warpAffine(label, cv2.getRotationMatrix2D((360, 360), angle, 1),
                           (720, 720), flags=cv2.INTER_NEAREST)
    candidates = find_m3_candidates(label, extract_path(label, PhotoKind.CLOSE))
    assert any(c.kind == 'branch' and c.verdict == (Verdict.NORMAL if dot else Verdict.SUSPECTED)
               for c in candidates), candidates
    if dot:
        assert not any(c.verdict == Verdict.SUSPECTED for c in candidates)


def test_cleanup_never_fills_a_real_gap():
    label = np.zeros((480, 480), np.uint8)
    label[80:200, 220:260] = 1
    label[230:420, 220:260] = 1
    label[420:460, 200:280] = 2
    cleaned = prepare_linear_mask(label)
    assert not np.any(cleaned[200:230])
    assert np.all(cleaned <= (label == 1))


def test_dot_at_one_corner_does_not_hide_another_missing_corner():
    label = np.zeros((720, 720), np.uint8)
    cv2.line(label, (180, 80), (180, 330), 1, 35)
    cv2.line(label, (180, 330), (480, 330), 1, 35)
    cv2.line(label, (480, 330), (480, 640), 1, 35)
    cv2.rectangle(label, (140, 290), (220, 370), 2, -1)
    candidates = find_m3_candidates(label, extract_path(label, PhotoKind.CLOSE))
    assert any(c.verdict == Verdict.NORMAL for c in candidates)
    assert any(c.verdict == Verdict.SUSPECTED and abs(c.location.x - 480) < 35
               for c in candidates)


def test_substantial_thin_real_path_is_not_erased_by_corner_cleanup():
    label = np.zeros((480, 480), np.uint8)
    label[200:203, 70:410] = 1
    label[190:215, 410:450] = 2
    cleaned = prepare_linear_mask(label)
    assert np.all(cleaned[label == 1] == 1)
