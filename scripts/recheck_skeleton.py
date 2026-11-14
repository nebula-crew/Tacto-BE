"""Recheck saved masks without rerunning YOLO; preserve the baseline outputs."""
from pathlib import Path
import json
from collections import Counter
from pipeline.skeleton import extract_path
from pipeline.dot_paths import expand_dot_paths
from pipeline.schema import PhotoKind
from scripts.run_path_debug import read_image, read_label, save_png, draw_path

def main():
    source = Path('outputs/review_v9_scene5')
    out = Path('outputs/skeleton_recheck')
    out.mkdir(parents=True, exist_ok=True)
    rows, report = [], []
    for folder in sorted(source.iterdir()):
        if not folder.is_dir():
            continue
        try:
            info = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
            image = read_image(Path('datasets/test') / info['file'])
            label = read_label(folder / 'refined_mask.png')
            path = extract_path(label, PhotoKind.WIDE)
            path = expand_dot_paths(path, label)
            overlay, graph = draw_path(image, label, path)
            target = out / folder.name
            save_png(target / 'overlay.png', overlay)
            save_png(target / 'graph.png', graph)
            (target / 'path.json').write_text(path.model_dump_json(indent=2), encoding='utf-8')
            old = json.loads((folder / 'path_structure.json').read_text(encoding='utf-8'))
            entry = dict(file=info['file'], before=dict(Counter(n['type'] for n in old['nodes'])), after=dict(Counter(n.type.value for n in path.nodes)))
            report.append(entry)
            rows.append(f'<h3>{folder.name}</h3><img width="45%" src="../review_v9_scene5/{folder.name}/skeleton_overlay.png"><a href="{folder.name}/overlay.png"><img width="45%" src="{folder.name}/overlay.png"></a>')
        except Exception as exc:
            report.append(dict(file=folder.name, error=str(exc)))
    (out / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'index.html').write_text('<meta charset="utf-8"><h1>Before / After</h1>' + ''.join(rows), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))

if __name__ == '__main__':
    main()
