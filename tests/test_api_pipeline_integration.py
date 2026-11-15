import numpy as np

from pipeline.run import run_inspection, run_photo
from pipeline.schema import PhotoKind, Verdict
from pipeline.segment import summarize


class StubSegmenter:
    available_classes = ['linear', 'dot']

    def predict(self, image, photo):
        label = np.zeros(image.shape[:2], dtype=np.uint8)
        label[30:170, 75:125] = 1
        label[2:4, 2:4] = 1
        return label, summarize(label, photo)


def test_api_pipeline_refines_noise_and_holds_unavailable_rules():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    result = run_inspection(StubSegmenter(), image, image)

    assert result.close.segmentation.class_area_px['linear'] == 140 * 50
    assert result.wide.segmentation.class_area_px['linear'] == 140 * 50
    assert not any(f.code == 'M4' and f.verdict == Verdict.SUSPECTED for f in result.findings)
    assert any(f.code == 'H3' and f.photo == PhotoKind.WIDE for f in result.findings)
    assert not any(f.code.startswith('C') for f in result.findings)
    assert result.overall == Verdict.HOLD


def test_missing_close_photo_is_not_an_overall_normal_result():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    result = run_inspection(StubSegmenter(), None, image)
    assert any(f.code == 'M3' and f.verdict == Verdict.HOLD for f in result.findings)
    assert result.overall == Verdict.HOLD


def test_single_model_does_not_claim_scene_is_absent():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    result = run_photo(StubSegmenter(), image)
    assert result.scope['C'].startswith('보류')
    assert any(f.code == 'H3' and f.verdict == Verdict.HOLD for f in result.findings)
    assert result.overall != Verdict.NORMAL
