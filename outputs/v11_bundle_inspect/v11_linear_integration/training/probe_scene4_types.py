"""Read-only model probe: scene4 tactile ROIs -> v8/v9 segmentation (not a trained classifier)."""
import argparse
import hashlib
import html
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from ultralytics import YOLO


COLORS = {0: (40, 210, 90), 1: (255, 130, 30)}


def masks(result, height, width):
    items = []
    if result.masks is not None:
        for b, polygon in zip(result.boxes, result.masks.xy):
            mask = np.zeros((height, width), dtype=np.uint8)
            if len(polygon) >= 3:
                cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
            items.append((int(b.cls.item()), float(b.conf.item()), mask))
    return items


def overlay(image, layers):
    arr = np.asarray(image).copy()
    for mask, color, label in layers:
        active = mask.astype(bool)
        arr[active] = (.65 * arr[active] + .35 * np.array(color)).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(arr, contours, -1, color, 2)
        yy, xx = np.where(mask)
        if len(xx):
            cv2.putText(arr, label, (int(xx.min()), max(16, int(yy.min()))),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
    return Image.fromarray(arr)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path, default=Path('/home/nebula/runs/scene4_v1/weights/best.pt'))
    p.add_argument('--source', type=Path, default=Path('datasets/test_data'))
    p.add_argument('--out', type=Path, default=Path('runs/scene4_type_probe'))
    p.add_argument('--device', default='0')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--conf', type=float, default=.25)
    p.add_argument('--margin', type=float, default=.10)
    a = p.parse_args()
    if a.out.exists():
        p.error('Output already exists; use a new --out')
    files = sorted(f for f in a.source.rglob('*') if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'})
    if not files:
        p.error('No source images')
    torch.set_num_threads(2)
    scene = YOLO(str(a.scene))
    assert scene.names[3] == 'tactile', scene.names
    types = {name: YOLO(f'{name}_best.pt') for name in ('v8', 'v9')}
    for model in types.values():
        assert model.names == {0: 'linear', 1: 'dot'}, model.names
    a.out.mkdir(parents=True)
    rows = []
    cards = []
    for index, file in enumerate(files):
        with Image.open(file) as raw:
            image = ImageOps.exif_transpose(raw).convert('RGB')
        w, h = image.size
        kw = dict(imgsz=a.imgsz, conf=a.conf, device=a.device, verbose=False, retina_masks=True)
        regions = masks(scene.predict(image, classes=[3], **kw)[0], h, w)
        scene_union = np.zeros((h, w), dtype=np.uint8)
        for _, _, mask in regions:
            scene_union |= mask
        scene_layers = [(m, (30, 200, 255), f'tactile {s:.2f}') for _, s, m in regions]
        panels = [('original', image), ('scene4', overlay(image, scene_layers))]
        row = dict(file=str(file.relative_to(a.source)), size=[w, h], scene_regions=len(regions), models={})
        for name, model in types.items():
            full = masks(model.predict(image, **kw)[0], h, w)
            full_layers = [(m, COLORS[c], f'{model.names[c]} {s:.2f}') for c, s, m in full]
            full_support = np.zeros((h, w), dtype=np.uint8)
            for _, _, mask in full:
                full_support |= mask
            roi_support = np.zeros((h, w), dtype=np.uint8)
            roi_layers = []
            detail = []
            for region_index, (_, score, region) in enumerate(regions):
                yy, xx = np.where(region)
                if not len(xx):
                    continue
                x1, x2, y1, y2 = int(xx.min()), int(xx.max())+1, int(yy.min()), int(yy.max())+1
                dx, dy = max(8, round((x2-x1)*a.margin)), max(8, round((y2-y1)*a.margin))
                x1, y1, x2, y2 = max(0, x1-dx), max(0, y1-dy), min(w, x2+dx), min(h, y2+dy)
                crop = image.crop((x1, y1, x2, y2))
                predictions = masks(model.predict(crop, **kw)[0], y2-y1, x2-x1)
                used = []
                local_support = np.zeros_like(region)
                for c, s, small in predictions:
                    mapped = np.zeros_like(region)
                    mapped[y1:y2, x1:x2] = small
                    mapped &= region
                    if not mapped.any():
                        continue
                    local_support |= mapped
                    roi_layers.append((mapped, COLORS[c], f'{model.names[c]} {s:.2f}'))
                    used.append(dict(type=model.names[c], confidence=s, pixels=int(mapped.sum())))
                roi_support |= local_support
                detail.append(dict(region=region_index, scene_confidence=score, crop=[x1, y1, x2, y2],
                                   classes=sorted({u['type'] for u in used}), predictions=used,
                                   support_fraction=float(local_support.sum()/region.sum()),
                                   crop_input_scale=a.imgsz/max(x2-x1, y2-y1)))
            unknown = scene_union & (1-roi_support)
            roi_layers.insert(0, (unknown, (180, 180, 180), 'type unknown'))
            denom = int(scene_union.sum())
            row['models'][name] = dict(
                full_instances=len(full), regions=detail, scene_pixels=denom,
                full_supported_scene_pixels=int((full_support & scene_union).sum()),
                roi_supported_scene_pixels=int(roi_support.sum()),
                unknown_scene_pixels=int(unknown.sum()),
                full_outside_scene_pixels=int((full_support & (1-scene_union)).sum()))
            panels.extend([(f'{name}_full', overlay(image, full_layers)),
                           (f'{name}_roi', overlay(image, roi_layers))])
        cells = []
        for label, panel in panels:
            asset = f'{index:03d}_{label}.jpg'
            panel.save(a.out/asset, quality=93)
            cells.append(f'<figure><figcaption>{html.escape(label)}</figcaption><a href="{asset}" target="_blank"><img loading="lazy" src="{asset}"></a></figure>')
        cards.append(f'<section><h2>{html.escape(row["file"])}</h2><div>{"".join(cells)}</div></section>')
        rows.append(row)
        print(f'{index+1}/{len(files)} {file.name}: scene regions={len(regions)}', flush=True)
    summary = dict(images=len(rows), images_with_scene_tactile=sum(r['scene_regions']>0 for r in rows),
                   total_scene_regions=sum(r['scene_regions'] for r in rows), models={})
    for name in types:
        vals = [r['models'][name] for r in rows]
        regions = [d for v in vals for d in v['regions']]
        denominator = sum(v['scene_pixels'] for v in vals)
        summary['models'][name] = dict(
            roi_regions_with_no_type=sum(not d['classes'] for d in regions),
            roi_regions_with_both_types=sum(len(d['classes'])==2 for d in regions),
            full_scene_pixel_support=sum(v['full_supported_scene_pixels'] for v in vals)/max(1,denominator),
            roi_scene_pixel_support=sum(v['roi_supported_scene_pixels'] for v in vals)/max(1,denominator))
    metadata = dict(scene_weights=str(a.scene.resolve()),
                    weight_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [a.scene, Path('v8_best.pt'), Path('v9_best.pt')]},
                    imgsz=a.imgsz, confidence=a.conf, crop_margin=a.margin,
                    retina_masks=True, ground_truth_available=False,
                    caveat='Coverage is prediction agreement with scene4, NOT accuracy or recall. '
                           'ROI models still perform detection; uncovered pixels are unknown, never forced to a type.')
    (a.out/'report.json').write_text(json.dumps(dict(metadata=metadata, summary=summary, images=rows), ensure_ascii=False, indent=2))
    (a.out/'index.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>scene4 → v8/v9 영역 진단</title>
<style>body{background:#16191f;color:#eee;font:16px sans-serif;margin:20px}section{border-top:1px solid #555;margin:28px 0}section>div{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%}h2{font-size:16px;overflow-wrap:anywhere}figcaption{padding:8px}a{color:#8df}@media(max-width:800px){section>div{grid-template-columns:1fr}}</style>
<h1>scene4 검출 영역 → 선형·점형 진단</h1><p>초록: 선형 / 주황: 점형 / 하늘색: scene4 점자블록 / 회색: 점자블록은 검출했지만 유형 미정.</p>
<p>full: 전체 사진 추론. roi: scene4 영역을 여유 10%로 잘라 추론하고 원래 마스크 안으로 제한. 기존 v8/v9를 사용한 사전 실험이며 분류 전용 학습 결과가 아닙니다. 둘 다 640, conf 0.25. 사진을 클릭하면 크게 볼 수 있습니다.</p>
<p>정답 없는 사진입니다. 검출 수·마스크 일치율은 정확도나 재현율이 아닙니다. scene4가 놓친 영역은 roi 결과에 나오지 않습니다.</p>'''+''.join(cards)+'</html>')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
