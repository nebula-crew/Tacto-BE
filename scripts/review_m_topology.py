"""Compare M topology on identical model masks; no ground-truth accuracy claim.

python -m scripts.review_m_topology --images datasets/test --out outputs/m_topology
"""
import argparse
import html
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from pipeline.mask_refine import refine_mask
from pipeline.rules.m_rules import check_m, check_m4
from pipeline.run import analyze_label
from pipeline.scene_type_segment import SceneTypeSegmenter
from pipeline.schema import MaskClass, PhotoKind
from pipeline.shape_rules import find_m3_candidates
from pipeline.skeleton import SkeletonConfig
from pipeline.dot_paths import expand_dot_paths, dot_pad_extents, draw_dot_pad_extents
from pipeline.schema import PathStructure


def render_review(report, out):
    valid = [r for r in report if 'error' not in r]
    branches = [sum(r[mode]['nodes'].get('branch', 0) for r in valid) for mode in ('before', 'after')]
    cards = []
    for row in report:
        name = html.escape(Path(row['file']).name)
        if 'error' in row:
            cards.append(f'<article><h2>{name}</h2><p>{html.escape(row["error"])}</p></article>')
            continue
        folder = row['folder']
        changed = row['before'] != row['after']
        figures = ''.join(f'<figure><a href="{folder}/{file}_route.png" target="_blank"><img loading="lazy" src="{folder}/{file}_route.png" data-route="{folder}/{file}_route.png" data-coverage="{folder}/{file}.png"></a><figcaption>{caption}</figcaption></figure>' if file in ('before','after') else f'<figure><a href="{folder}/{file}.png" target="_blank"><img loading="lazy" src="{folder}/{file}.png"></a><figcaption>{caption}</figcaption></figure>'
                          for file, caption in [('original', '원본'), ('mask', '분할 마스크'), ('before', '보정 전 중심선'), ('after', '보정 후 중심선')])
        findings = ''.join(f'<li><b>{html.escape(f["code"])} · {dict(normal="정상", suspected="오류 의심", hold="보류").get(f["verdict"], f["verdict"])}</b> {html.escape(f["message"])}</li>' for f in row['findings'])
        before, after = row['before'], row['after']
        stats = f'전체 경로 분기 {before.get("layout_branches",0)} → {after.get("layout_branches",0)} · M3 후보 {sum(before["m3"].values())} → {sum(after["m3"].values())}'
        cards.append(f'<article data-changed="{str(changed).lower()}"><h2>{name} {"<small>변경 있음</small>" if changed else ""}</h2><p>{stats}</p><div class="grid">{figures}</div><ul>{findings}</ul><details><summary>검토 수치와 좌표</summary><pre>{html.escape(json.dumps(row, ensure_ascii=False, indent=2))}</pre></details></article>')
    doc = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Test 사진 · M계열 검토</title>
<style>body{font-family:system-ui,'Malgun Gothic',sans-serif;background:#f3f4f5;color:#202724;margin:0}main{max-width:1600px;margin:auto;padding:28px}header,article{background:#fff;border-radius:18px;padding:24px;margin-bottom:24px}h1{color:#188563}h2{font-size:19px;overflow-wrap:anywhere}small{background:#e7f6ee;padding:5px;color:#188563}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}figure{margin:0}img{width:100%;height:430px;object-fit:contain;background:#edf1ef}figcaption{text-align:center;padding:10px;font-weight:600}li{margin:10px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}label{display:inline-block;padding:12px;background:#e7f6ee;border-radius:10px}a{color:#188563}@media(max-width:900px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}img{height:300px}}@media(max-width:500px){main{padding:10px}header,article{padding:14px}img{height:230px}}</style><main><header><h1>Test 사진 · M계열 검토</h1>'''
    doc += f'<p>{len(report)}장 처리 · 성공 {len(valid)}장 · 실행 오류 {len(report)-len(valid)}장 · 선형 분기 합계 {branches[0]} → {branches[1]}</p>'
    doc += '<p>마스크: 보라=선형, 주황=점형, 파랑=횡단보도, 회색=차도. 중심선: 흰 선=선형, 노란 선=점형 범위, 노란 원=점형 끝, 하늘색 원=선형 접점. 판정 위치: 초록=정상, 빨강=의심, 노랑=보류.</p><p>동일한 모델 마스크에서 구조 보정 전후를 비교합니다. 점형 범위선은 M3 꺾임 후보나 선형 진행 방향으로 사용하지 않습니다. 표시된 결과는 M계열 구조 검토이며 정답 라벨에 대한 정확도나 서비스 종합 판정이 아닙니다. 후보 없음은 정상 판정을 의미하지 않습니다.</p><label><input type="checkbox" id="changed"> 변경된 사진만 보기</label> <a href="contact_sheet.jpg">전체 사진 모아보기</a></header>'
    doc += '<p><label><input type="checkbox" id="coverage"> 넓은 점형의 범위 보조선 표시</label> · 점선은 범위 정보이며 경로·분기에는 포함되지 않습니다.</p>'
    doc += ''.join(cards) + '</main><script>document.querySelector("#changed").onchange=e=>document.querySelectorAll("article[data-changed]").forEach(a=>a.hidden=e.target.checked&&a.dataset.changed!=="true");document.querySelector("#coverage").onchange=e=>document.querySelectorAll("img[data-route]").forEach(i=>{i.src=e.target.checked?i.dataset.coverage:i.dataset.route;i.parentElement.href=i.src})</script></html>'
    (out / 'index.html').write_text(doc, encoding='utf-8')


def contact_sheet(report, out):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype('C:/Windows/Fonts/malgun.ttf', 17)
    valid = [r for r in report if 'error' not in r]
    sheet = Image.new('RGB', (1200, 350 * ((len(valid) + 3) // 4)), '#f3f4f5')
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(valid):
        image = Image.open(out / row['folder'] / 'after_route.png').convert('RGB')
        image.thumbnail((280, 290))
        x, y = (i % 4) * 300, (i // 4) * 350
        sheet.paste(image, (x + (300-image.width)//2, y))
        short = Path(row['file']).name
        if len(short) > 26:
            short = short[:23] + '…'
        draw.text((x+8, y+293), f'{i+1:02d} {short}', fill='#202724', font=font)
        draw.text((x+8, y+319), f'분기 {row["before"]["nodes"].get("branch",0)} → {row["after"]["nodes"].get("branch",0)}', fill='#188563', font=font)
    if valid:
        sheet.save(out / 'contact_sheet.jpg', quality=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, default=Path('datasets/test'))
    parser.add_argument('--out', type=Path, default=Path('outputs/m_topology'))
    parser.add_argument('--reuse-masks', type=Path, help='같은 사진의 이전 summary.json·label.npy 사용 (모델 재추론 없음)')
    parser.add_argument('--comparison-layouts', type=Path, help='수정 전 경로 JSON 폴더 (00.json 등)')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(min(4, torch.get_num_threads()))
    cached = {}
    if args.reuse_masks:
        cached = {str(Path(row['file']).resolve()): row for row in json.loads(
            (args.reuse_masks / 'summary.json').read_text(encoding='utf-8')) if 'error' not in row}
    segmenter = None if args.reuse_masks else SceneTypeSegmenter('weights/v11_best.pt', 'weights/scene4_v1_best.pt')
    files = sorted(p for p in args.images.rglob('*') if p.is_file()
                   and p.suffix.lower() in {'.jpg', '.jpeg', '.png'} and '__MACOSX' not in p.parts)
    report = []
    for index, file in enumerate(files):
        entry = {'file': str(file)}
        try:
            image = cv2.imdecode(np.fromfile(file, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('Unreadable image')
            factor = min(1., 1400 / max(image.shape[:2]))
            if factor < 1:
                image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
            if args.reuse_masks:
                row = cached[str(file.resolve())]
                label = np.load(args.reuse_masks / row['folder'] / 'label.npy', allow_pickle=False)
                if label.shape != image.shape[:2]:
                    raise ValueError('저장된 마스크와 사진의 크기가 다릅니다')
                seg = None
            else:
                label, seg = segmenter.predict(image, PhotoKind.WIDE)
                label, _ = refine_mask(label)
            folder = args.out / f'{index:02d}'
            folder.mkdir(exist_ok=True)
            entry['folder'] = folder.name
            np.save(folder / 'label.npy', label)
            cv2.imencode('.png', image)[1].tofile(str(folder / 'original.png'))
            palette = np.array([(240,240,240), (180,90,70), (20,130,250), (220,150,30), (110,115,110)], np.uint8)
            cv2.imencode('.png', palette[label])[1].tofile(str(folder / 'mask.png'))
            for mode, cfg in [('before', SkeletonConfig(dot_corner_cleanup_px=0)), ('after', SkeletonConfig())]:
                if args.comparison_layouts:
                    cfg = SkeletonConfig()
                analysis = analyze_label(label, PhotoKind.WIDE, segmentation=seg, skel_cfg=cfg)
                path = analysis.path
                candidates = find_m3_candidates(label, path)
                entry[mode] = {
                    'nodes': dict(Counter(n.type.value for n in path.nodes if n.block == MaskClass.LINEAR)),
                    'm3': dict(Counter(c.verdict.value for c in candidates)),
                    'm4': dict(Counter(f.verdict.value for f in check_m4(analysis, label))),
                }
                if mode == 'after':
                    entry['findings'] = [f.model_dump(mode='json') for f in check_m(analysis, label)]
                (folder / f'{mode}.json').write_text(path.model_dump_json(indent=2), encoding='utf-8')
                overlay = image.copy()
                for cls, color in [(1, (180, 90, 70)), (2, (20, 130, 250))]:
                    selected = label == cls
                    overlay[selected] = (overlay[selected] * .7 + np.array(color) * .3).astype(np.uint8)
                layout = expand_dot_paths(path, label)
                extents = dot_pad_extents(path, label)
                if mode == 'before' and args.comparison_layouts:
                    layout = PathStructure.model_validate_json((args.comparison_layouts / (folder.name+'.json')).read_text(encoding='utf-8'))
                    extents = []
                entry[mode]['layout_branches'] = sum(n.type.value == 'branch' for n in layout.nodes)
                (folder / f'{mode}_layout.json').write_text(layout.model_dump_json(indent=2), encoding='utf-8')
                (folder / f'{mode}_coverage.json').write_text(json.dumps(extents,ensure_ascii=False,indent=2),encoding='utf-8')
                for edge in layout.edges:
                    points = np.array([[round(p.x), round(p.y)] for p in edge.polyline], np.int32)
                    cv2.polylines(overlay, [points], False, (0, 230, 255) if edge.block == MaskClass.DOT else (255, 255, 255), 2, cv2.LINE_AA)
                for node in layout.nodes:
                    if node.block == MaskClass.DOT:
                        color = (255, 210, 30) if any(e.block == MaskClass.LINEAR for e in layout.edges_of(node.id)) else (0, 230, 255)
                        cv2.circle(overlay, (round(node.pt.x), round(node.pt.y)), 6, color, 2)
                for number, candidate in enumerate(candidates, 1):
                    xy = (round(candidate.location.x), round(candidate.location.y))
                    color = {'normal': (30, 200, 30), 'suspected': (0, 0, 255), 'hold': (0, 200, 255)}[candidate.verdict.value]
                    cv2.circle(overlay, xy, 9, color, 2)
                    cv2.putText(overlay, f'M3-{number}', (xy[0]+12, max(20,xy[1]-12)), cv2.FONT_HERSHEY_SIMPLEX,.5,color,2,cv2.LINE_AA)
                cv2.imencode('.png', overlay)[1].tofile(str(folder / f'{mode}_route.png'))
                draw_dot_pad_extents(overlay, extents)
                cv2.imencode('.png', overlay)[1].tofile(str(folder / f'{mode}.png'))
        except Exception as exc:
            entry['error'] = f'{type(exc).__name__}: {exc}'
        report.append(entry)
        print(json.dumps(entry, ensure_ascii=True), flush=True)
        (args.out / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        render_review(report, args.out)
    contact_sheet(report, args.out)


if __name__ == '__main__':
    main()
