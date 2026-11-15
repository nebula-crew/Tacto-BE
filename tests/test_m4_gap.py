import cv2
import numpy as np

from pipeline.run import analyze_label
from pipeline.rules.m_rules import check_m4
from pipeline.schema import MaskClass, PhotoKind, Verdict


def mask(gap=True, dot=False, boundary=False):
    label = np.zeros((400, 400), np.uint8)
    y = 12 if boundary else 200
    cv2.rectangle(label, (50, y - 10), (185, y + 10), int(MaskClass.LINEAR), -1)
    cv2.rectangle(label, (215 if gap else 186, y - 10), (350, y + 10), int(MaskClass.LINEAR), -1)
    if dot:
        cv2.rectangle(label, (186, y - 10), (214, y + 10), int(MaskClass.DOT), -1)
    return label


def check(label):
    return check_m4(analyze_label(label, PhotoKind.CLOSE), label)


def test_gap_is_suspected_without_joining_components():
    label = mask()
    result = check(label)
    assert len(result) == 1 and result[0].verdict == Verdict.SUSPECTED
    assert label[200, 200] == MaskClass.BACKGROUND


def test_continuous_or_dot_connection_is_not_gap():
    assert check(mask(gap=False)) == []
    assert check(mask(dot=True)) == []


def test_gap_near_image_edge_is_held():
    assert [f.verdict for f in check(mask(boundary=True))] == [Verdict.HOLD]
