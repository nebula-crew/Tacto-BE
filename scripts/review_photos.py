"""테스트 폴더의 모델별 마스크와 중심선을 사진마다 저장한다.

프로젝트 루트: python -m scripts.review_photos
"""
from pathlib import Path
import argparse
import html
import json
import traceback
import cv2
import numpy as np
from pipeline.dual_segment import DualSegmenter, merge_detections
from pipeline.mask_refine import refine_mask
from pipeline.run import analyze_label
from pipeline.rules import judge_photo
from pipeline.schema import MaskClass, PhotoKind
from pipeline.segment import summarize
from scripts.run_mask_extract import read_image, save_png
from scripts.run_path_debug import draw_path

parser = argparse.ArgumentParser(description='두 모델의 분할 결과와 중심선 일괄 검토')
parser.add_argument('--input', default='datasets/test')
parser.add_argument('--out', default='outputs/review_v9_scene5')
parser.add_argument('--block-weights', default='weights/v9_best.pt')
parser.add_argument('--scene-weights', default='weights/scene5_best.pt')
args = parser.parse_args()
source = Path(args.input)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
model = DualSegmenter(args.block_weights, args.scene_weights)
colors = {1:(255,0,0), 2:(0,0,255), 3:(0,255,255), 4:(0,200,0)}
summary = []

def label_for(detections, shape):
    label = np.zeros(shape, np.uint8)
    for m, cls, score in sorted(detections, key=lambda item:item[2]):
        label[m] = int(cls)
    return label

def preview(image, label):
    scale = min(1, 640 / max(image.shape[:2]))
    size = (max(1,round(image.shape[1]*scale)), max(1,round(image.shape[0]*scale)))
    view = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    lab = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
    colored = view.copy()
    for cls, color in colors.items(): colored[lab == cls] = color
    return cv2.addWeighted(view, .60, colored, .40, 0)

for path in sorted(source.iterdir()):
    if not path.is_file() or path.suffix.lower() not in ('.jpg', '.jpeg', '.png') or path.name.startswith('._'):
        continue
    target = out / path.stem
    target.mkdir(exist_ok=True)
    try:
        image = read_image(path)
        blocks, _ = model._run(model.block_model, model.block_map, image, model.cfg.block_conf)
        scene, tactile = model._run(model.scene_model, model.scene_map, image, model.cfg.scene_conf, model.tactile_ids)
        raw_blocks, raw_scene = label_for(blocks, image.shape[:2]), label_for(scene, image.shape[:2])
        merged, seg = merge_detections(image, scene, blocks, PhotoKind.WIDE, model.cfg, tactile)
        refined, audit = refine_mask(merged)
        seg.class_area_px = summarize(refined, PhotoKind.WIDE).class_area_px
        analysis = analyze_label(refined, PhotoKind.WIDE, image, seg)
        result = judge_photo(analysis, refined)
        for name, mask in [('block',raw_blocks), ('scene',raw_scene), ('merged',merged), ('refined',refined)]:
            save_png(target / (name + '_mask.png'), mask)
            save_png(target / (name + '_preview.png'), preview(image, mask))
        if tactile is not None:
            save_png(target / 'scene_tactile_mask.png', tactile.astype(np.uint8) * 255)
        path_img, graph = draw_path(image, refined, analysis.path)
        save_png(target / 'skeleton_overlay.png', path_img)
        save_png(target / 'skeleton_graph.png', graph)
        (target / 'path_structure.json').write_text(analysis.path.model_dump_json(indent=2), encoding='utf-8')
        scale = min(1, 1000 / max(path_img.shape[:2]))
        save_png(target / 'path_preview.png', cv2.resize(path_img, (max(1, round(path_img.shape[1]*scale)), max(1, round(path_img.shape[0]*scale)))))
        info = {'file':path.name, 'weights': {'block':args.block_weights, 'scene':args.scene_weights}, 'block_detections':[{'class':c.name.lower(),'confidence':float(s),'area_px':int(m.sum())} for m,c,s in blocks],
                'scene_detections':[{'class':c.name.lower(),'confidence':float(s),'area_px':int(m.sum())} for m,c,s in scene],
                'scene_tactile_px':int(tactile.sum()) if tactile is not None else None,
                'block_pixels_removed_by_merge':int(np.count_nonzero((raw_blocks > 0) & ~np.isin(merged,[1,2]))),
                'refine_changed_px':audit['pixels_changed'], 'result':result.model_dump(mode='json')}
        (target / 'result.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
        summary.append({'file':path.name,'status':'ok','overall':result.overall.value,'findings':[(f.code,f.verdict.value) for f in result.findings], 'removed_px':info['block_pixels_removed_by_merge']})
        print('OK', path.name, result.overall.value, flush=True)
    except Exception as exc:
        (target / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        summary.append({'file':path.name,'status':'error','error':str(exc)})
        print('ERROR', path.name, type(exc).__name__, str(exc), flush=True)
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

cards = []
for item in summary:
    name = item['file']
    folder = Path(name).stem
    if item['status'] == 'ok':
        cards.append(f'<section><h2>{html.escape(name)}</h2><p>자동 판정: {html.escape(item["overall"])}</p>'
                     f'<a href="{html.escape(folder)}/skeleton_overlay.png"><img src="{html.escape(folder)}/path_preview.png" alt="중심선"></a>'
                     f'<p><a href="{html.escape(folder)}/block_preview.png">블록</a> · '
                     f'<a href="{html.escape(folder)}/scene_preview.png">장면</a> · '
                     f'<a href="{html.escape(folder)}/merged_preview.png">통합</a> · '
                     f'<a href="{html.escape(folder)}/skeleton_graph.png">중심선만</a> · '
                     f'<a href="{html.escape(folder)}/result.json">판정 결과</a></p></section>')
    else:
        cards.append(f'<section><h2>{html.escape(name)}</h2><p>분석 실패: {html.escape(item["error"])}</p></section>')
page = ('<!doctype html><html lang="ko"><meta charset="utf-8"><title>중심선 검토</title>'
        '<style>body{font-family:sans-serif;margin:24px;background:#f4f5f5}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}'
        'section{background:white;padding:16px;border-radius:12px}img{max-width:100%;max-height:480px}h2{font-size:14px;overflow-wrap:anywhere}</style>'
        '<h1>모델 분석과 중심선 검토</h1><p>사진을 누르면 원본 크기 중심선 이미지가 열립니다. 흰 선: 경로, 노랑 원: 점형, 빨강 점: 분기, 초록 점: 끝점.</p><main>'
        + ''.join(cards) + '</main></html>')
(out / 'index.html').write_text(page, encoding='utf-8')
