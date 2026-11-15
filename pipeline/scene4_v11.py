"""Scene segmentation followed by v11 tactile type classification."""
import copy

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from ultralytics import YOLO

from .dual_segment import _names, _to_masks, class_mapping, SCENE_EXTRA_NAMES
from .schema import MaskClass, PhotoKind
from .segment import summarize


class TypeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = copy.deepcopy(YOLO('yolov8n-seg.yaml').model.model[:10]).float()
        self.lateral = nn.ModuleList([nn.Conv2d(c, 64, 1) for c in (64, 128, 256)])
        self.head = nn.Sequential(nn.Conv2d(192, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.SiLU(),
                                  nn.Conv2d(128, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU(), nn.Conv2d(64, 2, 1))

    def forward(self, x):
        size = x.shape[-2:]
        features = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in (4, 6, 9):
                features.append(x)
        target = features[0].shape[-2:]
        xs = [F.interpolate(layer(f), size=target, mode='bilinear', align_corners=False)
              for layer, f in zip(self.lateral, features)]
        return F.interpolate(self.head(torch.cat(xs, 1)), size=size, mode='bilinear', align_corners=False)


class Scene4V11Segmenter:
    def __init__(self, scene_weights, type_weights):
        self.scene_model = YOLO(str(scene_weights))
        self.scene_map = class_mapping(self.scene_model.names, {MaskClass.CROSSWALK, MaskClass.ROAD}, skip=SCENE_EXTRA_NAMES)
        self.tactile_ids = frozenset(i for i, n in _names(self.scene_model.names) if n == 'tactile')
        if len(self.tactile_ids) != 1:
            raise ValueError('Scene model must contain one tactile class')
        checkpoint = torch.load(str(type_weights), map_location='cpu', weights_only=True)
        self.model = TypeModel()
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.model.eval()
        self.size = checkpoint['config']['size']
        self.available_classes = ['linear', 'dot'] + [c.name.lower() for c in self.scene_map.values()]
        self.dual_detection = True

    def predict(self, image_bgr, photo=PhotoKind.WIDE):
        h, w = image_bgr.shape[:2]
        result = self.scene_model.predict(image_bgr, imgsz=640, conf=.25, retina_masks=True, verbose=False)[0]
        scene, _ = _to_masks(result, self.scene_map, h, w)
        label = np.zeros((h, w), np.uint8)
        for mask, cls, score in sorted(scene, key=lambda d: d[2]):
            label[mask] = int(cls)
        scores = np.zeros((2, h, w), np.float32)
        counts = np.zeros((h, w), np.float32)
        rgb = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if result.masks is not None and result.boxes is not None:
            for mask, cls in zip(result.masks.data.cpu().numpy(), result.boxes.cls.int().cpu().numpy()):
                if int(cls) not in self.tactile_ids:
                    continue
                if mask.shape != (h, w):
                    mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                region = mask > .5
                yy, xx = np.where(region)
                if not len(xx):
                    continue
                dx = max(8, round((xx.max()-xx.min()+1)*.1)); dy = max(8, round((yy.max()-yy.min()+1)*.1))
                x1=max(0,int(xx.min())-dx); x2=min(w,int(xx.max())+1+dx)
                y1=max(0,int(yy.min())-dy); y2=min(h,int(yy.max())+1+dy)
                crop=rgb.crop((x1,y1,x2,y2)); cw,ch=crop.size; scale=self.size/max(cw,ch)
                nw=max(1,round(cw*scale)); nh=max(1,round(ch*scale)); left=(self.size-nw)//2; top=(self.size-nh)//2
                arr=np.full((self.size,self.size,3),114,np.uint8)
                arr[top:top+nh,left:left+nw]=crop.resize((nw,nh),Image.Resampling.BILINEAR)
                x=torch.from_numpy(arr.transpose(2,0,1).copy()).float()[None]/255
                with torch.inference_mode():
                    logits=self.model(x)[:,:,top:top+nh,left:left+nw]
                    prob=F.interpolate(logits,size=(ch,cw),mode='bilinear',align_corners=False).softmax(1)[0].numpy()
                scores[:,y1:y2,x1:x2]+=prob*region[y1:y2,x1:x2]
                counts+=region
        scores/=np.maximum(counts,1)[None]
        valid=counts>0; uncertain=valid & (scores.max(0)<.6); pred=scores.argmax(0)
        label[valid]=0
        label[valid & ~uncertain & (pred==0)]=int(MaskClass.LINEAR)
        label[valid & ~uncertain & (pred==1)]=int(MaskClass.DOT)
        seg=summarize(label, photo, scores.max(0))
        seg.unknown_block_area_px=int(uncertain.sum())
        return label, seg
