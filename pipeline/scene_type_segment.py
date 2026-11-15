"""② 영역 분할 (장면 모델 + 블록 종류 모델): scene4 + v11 → MaskClass 라벨 하나. (담당: 은재, 모델: 시경)

    장면 모델 (scene4_v1)   crosswalk, road, sidewalk, tactile  (YOLOv8-seg)
    블록 종류 모델 (v11)    tactile 영역 안의 픽셀마다 linear / dot  (YOLOv8n 백본 + 분할 헤드)

블록 모델이 사진 전체에서 선형·점형을 직접 찾던 방식(DualSegmenter) 대신,
장면 모델이 점자블록 영역(tactile)을 먼저 찾고 v11이 그 영역 안을 선형·점형으로 나눈다.
v11 학습·검토 코드(scripts/review_scene4_v11.py, outputs/v11_bundle_inspect/…/train_v11.py)와 같은 방식이다.

순서
1. 장면 모델: 횡단보도·차도 검출, tactile 덩어리마다 (마스크, 신뢰도)
2. tactile 덩어리마다 10% 여유를 두고 잘라 정사각형(size)으로 맞춘 뒤 v11로 픽셀별 확률을 구한다.
   덩어리가 겹치면 확률을 평균한다
3. 확률이 uncertain_prob 미만인 픽셀은 종류를 정하지 않는다 → tactile인데 블록 라벨이 없는 곳
4. 나머지는 DualSegmenter와 같은 merge_detections로 합친다
   (줄무늬 필터, 횡단보도·차도 안 블록 제거, 신뢰도 지도, 종류 모름 블록 면적)
   블록 신뢰도는 그 블록이 나온 tactile 덩어리의 장면 모델 신뢰도를 쓴다 (H1)

segment.Segmenter·DualSegmenter와 같은 predict(image, photo) -> (label, Segmentation) 형식이라 run.py에 그대로 꽂힌다.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .dual_segment import SCENE_EXTRA_NAMES, TACTILE_NAME, Detection, MergeConfig, _names, class_mapping, merge_detections
from .schema import MaskClass, PhotoKind, Segmentation

TYPE_NAME_TO_CLASS = {"linear": MaskClass.LINEAR, "dot": MaskClass.DOT}


@dataclass
class TypeConfig:
    pad_ratio: float = 0.1  # tactile 덩어리를 자를 때 여유 (덩어리 크기 대비)
    min_pad_px: int = 8
    uncertain_prob: float = 0.6  # 이보다 확신이 낮은 픽셀은 종류를 정하지 않음
    fill: int = 114  # 정사각형으로 맞출 때 빈칸 색 (학습과 같음)


def is_type_checkpoint(path: str | Path) -> bool:
    """v11처럼 {'state_dict', 'config'}로 저장된 블록 종류 모델인가 (YOLO 체크포인트는 False)."""
    import torch

    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:  # YOLO 체크포인트는 weights_only로 열리지 않는다
        return False
    return isinstance(ckpt, dict) and "state_dict" in ckpt and "config" in ckpt


def _type_model():
    """v11 구조 (train_v11.TypeModel과 같음). 가중치 없이 yolov8n-seg.yaml로 백본 모양만 만든다."""
    import torch
    from torch import nn
    from torch.nn import functional as F
    from ultralytics import YOLO

    class TypeModel(nn.Module):
        def __init__(self):
            super().__init__()
            source = YOLO("yolov8n-seg.yaml").model  # 패키지에 들어 있는 설정 파일 (다운로드 없음)
            self.encoder = copy.deepcopy(source.model[:10]).float()
            self.lateral = nn.ModuleList([nn.Conv2d(c, 64, 1) for c in (64, 128, 256)])
            self.head = nn.Sequential(nn.Conv2d(192, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.SiLU(),
                                      nn.Conv2d(128, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
                                      nn.Conv2d(64, 2, 1))

        def forward(self, x):
            size, features = x.shape[-2:], []
            for i, layer in enumerate(self.encoder):
                x = layer(x)
                if i in (4, 6, 9):  # stride 8 / 16 / 32
                    features.append(x)
            target = features[0].shape[-2:]
            xs = [F.interpolate(lat(f), size=target, mode="bilinear", align_corners=False)
                  for lat, f in zip(self.lateral, features)]
            return F.interpolate(self.head(torch.cat(xs, 1)), size=size, mode="bilinear", align_corners=False)

    return TypeModel()


class SceneTypeSegmenter:
    """장면 모델(tactile 포함) + 블록 종류 모델. DualSegmenter와 같은 predict 형식."""

    def __init__(self, type_weights: str | Path, scene_weights: str | Path,
                 cfg: Optional[MergeConfig] = None, type_cfg: Optional[TypeConfig] = None):
        import torch
        from ultralytics import YOLO

        for p in (type_weights, scene_weights):
            if not Path(p).is_file():
                raise FileNotFoundError(f"가중치 파일이 없음: {p}")
        self.cfg = cfg or MergeConfig()
        self.type_cfg = type_cfg or TypeConfig()
        self.scene_model = YOLO(str(scene_weights))
        self.scene_map = class_mapping(self.scene_model.names, {MaskClass.CROSSWALK, MaskClass.ROAD},
                                       skip=SCENE_EXTRA_NAMES)
        self.tactile_ids = frozenset(i for i, n in _names(self.scene_model.names) if n == TACTILE_NAME)
        if len(self.tactile_ids) != 1:
            raise ValueError("장면 모델에 tactile 클래스가 정확히 하나 있어야 블록 종류 모델을 쓸 수 있습니다.")

        ckpt = torch.load(str(type_weights), map_location="cpu", weights_only=True)
        names = [str(n).lower() for n in ckpt["config"].get("classes", ["linear", "dot"])]
        if sorted(names) != ["dot", "linear"]:
            raise ValueError(f"블록 종류 모델 클래스가 linear·dot이 아님: {names}")
        self.type_classes = [TYPE_NAME_TO_CLASS[n] for n in names]  # 출력 채널 순서
        self.size = int(ckpt["config"]["size"])
        self.type_model = _type_model()
        self.type_model.load_state_dict(ckpt["state_dict"], strict=True)
        self.type_model.eval()
        self.available_classes = sorted({"linear", "dot", *(c.name.lower() for c in self.scene_map.values())})

    @property
    def dual_detection(self) -> bool:
        return True  # tactile 영역 = 블록 후보, 종류를 못 정한 곳은 '종류 모름 블록'

    def _scene(self, image_bgr: np.ndarray):
        h, w = image_bgr.shape[:2]
        r = self.scene_model.predict(image_bgr, conf=self.cfg.scene_conf, imgsz=self.cfg.imgsz,
                                     retina_masks=True, verbose=False)[0]
        scene: list[Detection] = []
        tactile: list[tuple[np.ndarray, float]] = []
        if r.masks is None or r.boxes is None:
            return scene, tactile
        for m, c, s in zip(r.masks.data.cpu().numpy(), r.boxes.cls.int().cpu().numpy(), r.boxes.conf.cpu().numpy()):
            c = int(c)
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            if c in self.tactile_ids:
                tactile.append((m > 0.5, float(s)))
            elif c in self.scene_map:
                scene.append((m > 0.5, self.scene_map[c], float(s)))
        return scene, tactile

    def _type_probs(self, rgb: np.ndarray, region: np.ndarray):
        """tactile 덩어리 하나 → (y1, y2, x1, x2, 확률[2, ch, cw]). 학습·검토 코드와 같은 자르기·맞추기."""
        import torch
        from torch.nn import functional as F

        t = self.type_cfg
        h, w = region.shape
        ys, xs = np.nonzero(region)
        if not len(xs):
            return None
        dx = max(t.min_pad_px, round((xs.max() - xs.min() + 1) * t.pad_ratio))
        dy = max(t.min_pad_px, round((ys.max() - ys.min() + 1) * t.pad_ratio))
        x1, x2 = max(0, int(xs.min()) - dx), min(w, int(xs.max()) + 1 + dx)
        y1, y2 = max(0, int(ys.min()) - dy), min(h, int(ys.max()) + 1 + dy)
        crop = rgb[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        scale = self.size / max(cw, ch)
        nw, nh = max(1, round(cw * scale)), max(1, round(ch * scale))
        left, top = (self.size - nw) // 2, (self.size - nh) // 2
        arr = np.full((self.size, self.size, 3), t.fill, np.uint8)
        arr[top:top + nh, left:left + nw] = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(arr.transpose(2, 0, 1).copy()).float()[None] / 255
        with torch.inference_mode():
            logits = self.type_model(x)[:, :, top:top + nh, left:left + nw]
            prob = F.interpolate(logits, size=(ch, cw), mode="bilinear", align_corners=False).softmax(1)[0].numpy()
        return y1, y2, x1, x2, prob

    def predict(self, image_bgr: np.ndarray, photo: PhotoKind = PhotoKind.WIDE) -> tuple[np.ndarray, Segmentation]:
        h, w = image_bgr.shape[:2]
        scene, tactile = self._scene(image_bgr)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        scores = np.zeros((2, h, w), np.float32)
        counts = np.zeros((h, w), np.float32)
        for region, _ in tactile:
            out = self._type_probs(rgb, region)
            if out is None:
                continue
            y1, y2, x1, x2, prob = out
            local = region[y1:y2, x1:x2]
            scores[:, y1:y2, x1:x2] += prob * local
            counts += region
        scores /= np.maximum(counts, 1)[None]
        valid = counts > 0
        sure = valid & (scores.max(0) >= self.type_cfg.uncertain_prob)
        pred = scores.argmax(0)

        blocks: list[Detection] = []
        for region, conf in tactile:
            for ch, cls in enumerate(self.type_classes):
                m = region & sure & (pred == ch)
                if m.any():
                    blocks.append((m, cls, conf))
        return merge_detections(image_bgr, scene, blocks, photo, self.cfg, valid if tactile else None)
