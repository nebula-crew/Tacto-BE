import json
import cv2
import numpy as np

from pipeline.dual_segment import _to_masks, class_mapping
from pipeline.schema import MaskClass
from scripts.run_mask_extract import export_detections


class Tensor:
    def __init__(self, data):
        self.data = np.asarray(data)

    def cpu(self): return self
    def int(self): return self
    def numpy(self): return self.data


def test_separate_model_export_preserves_size_classes_and_confidence(tmp_path):
    result = type('Result', (), {})()
    result.masks = type('Masks', (), {'data': Tensor([[[1, 0], [0, 0]]])})()
    result.boxes = type('Boxes', (), {'cls': Tensor([0]), 'conf': Tensor([0.87])})()
    mapping = class_mapping({0: 'road'}, {MaskClass.ROAD, MaskClass.CROSSWALK})
    detections, _ = _to_masks(result, mapping, 8, 10)
    image = np.zeros((8, 10, 3), np.uint8)
    export_detections(image, detections, tmp_path)
    label = cv2.imdecode(np.fromfile(str(tmp_path / 'label_mask.png'), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert label.shape == (8, 10) and label.dtype == np.uint8
    assert set(np.unique(label)) == {0, MaskClass.ROAD}
    assert (tmp_path / 'overlay.png').is_file()
    assert json.loads((tmp_path / 'detections.json').read_text(encoding='utf-8'))[0]['confidence'] == 0.87
