"""SceneTypeSegmenter (scene4 + v11): 모델 없이 합치는 규칙을, 가중치가 있으면 실제 로딩까지 확인."""

from pathlib import Path

import numpy as np
import pytest

from pipeline.dual_segment import MergeConfig
from pipeline.schema import MaskClass, PhotoKind
from pipeline.scene_type_segment import SceneTypeSegmenter, TypeConfig, is_type_checkpoint

ROOT = Path(__file__).resolve().parents[1]
H, W = 200, 300


def _stub(tactile, probs, scene=()):
    """모델을 불러오지 않고 장면 결과·종류 확률만 정해 둔 분할기."""
    seg = SceneTypeSegmenter.__new__(SceneTypeSegmenter)
    seg.cfg, seg.type_cfg = MergeConfig(), TypeConfig()
    seg.type_classes = [MaskClass.LINEAR, MaskClass.DOT]
    seg._scene = lambda image: (list(scene), tactile)
    seg._type_probs = lambda rgb, region: (0, H, 0, W, probs)
    return seg


def _image():
    return np.full((H, W, 3), 128, np.uint8)


def test_tactile_region_is_split_into_linear_and_dot():
    region = np.zeros((H, W), bool)
    region[40:160, 20:280] = True
    probs = np.zeros((2, H, W), np.float32)
    probs[0, :, :150], probs[1, :, :150] = 0.9, 0.1  # 왼쪽 선형
    probs[0, :, 150:], probs[1, :, 150:] = 0.2, 0.8  # 오른쪽 점형
    label, seg = _stub([(region, 0.8)], probs).predict(_image(), PhotoKind.WIDE)

    assert (label[50:150, 30:140] == MaskClass.LINEAR).all()
    assert (label[50:150, 160:270] == MaskClass.DOT).all()
    assert (label[~region] == MaskClass.BACKGROUND).all()  # tactile 밖은 블록이 아님
    assert seg.mean_confidence == pytest.approx(0.8)  # 블록 신뢰도 = tactile 덩어리 신뢰도
    assert seg.unknown_block_area_px == 0


def test_uncertain_pixels_become_unknown_blocks():
    region = np.zeros((H, W), bool)
    region[40:160, 20:280] = True
    probs = np.zeros((2, H, W), np.float32)
    probs[0, :, :100], probs[1, :, :100] = 0.9, 0.1
    probs[0, :, 100:], probs[1, :, 100:] = 0.55, 0.45  # 확신 낮음 → 종류를 정하지 않음
    label, seg = _stub([(region, 0.8)], probs).predict(_image(), PhotoKind.WIDE)

    assert (label[50:150, 200:270] == MaskClass.BACKGROUND).all()
    assert seg.unknown_block_area_px > 0  # 규칙에서 C4·C5 의심을 보류로 돌리는 값


def test_no_tactile_means_no_blocks():
    label, seg = _stub([], np.zeros((2, H, W), np.float32)).predict(_image(), PhotoKind.WIDE)
    assert not np.isin(label, [MaskClass.LINEAR, MaskClass.DOT]).any()
    assert seg.unknown_block_area_px is None


def test_is_type_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    good, bad = tmp_path / "type.pt", tmp_path / "other.pt"
    torch.save({"state_dict": {}, "config": {"size": 640}}, good)
    torch.save({"model": "yolo"}, bad)
    assert is_type_checkpoint(good)
    assert not is_type_checkpoint(bad)
    assert not is_type_checkpoint(tmp_path / "missing.pt")


@pytest.mark.skipif(not (ROOT / "weights/v11_best.pt").is_file() or not (ROOT / "weights/scene4_v1_best.pt").is_file(),
                    reason="v11·scene4 가중치 없음")
def test_real_weights_load_and_predict():
    pytest.importorskip("ultralytics")
    assert is_type_checkpoint(ROOT / "weights/v11_best.pt")
    assert not is_type_checkpoint(ROOT / "weights/scene4_v1_best.pt")
    seg = SceneTypeSegmenter(ROOT / "weights/v11_best.pt", ROOT / "weights/scene4_v1_best.pt")
    assert seg.available_classes == ["crosswalk", "dot", "linear", "road"]
    label, info = seg.predict(_image(), PhotoKind.WIDE)
    assert label.shape == (H, W) and info.width == W
