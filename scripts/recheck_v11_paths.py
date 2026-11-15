"""Rebuild v11 overlays from saved masks, preserving a before image."""
from pathlib import Path
import json
import shutil
from collections import Counter
import cv2
import numpy as np
from PIL import Image, ImageOps
from pipeline.schema import PhotoKind
from pipeline.skeleton import extract_path
from pipeline.dot_paths import expand_dot_paths
from pipeline.run import analyze_label
from pipeline.segment import summarize
from pipeline.rules import judge_photo
from scripts.run_path_debug import read_label, save_png, draw_path

def main():
    root = Path('outputs/review_v11_scene4')
    report = []
    for row in json.loads((root/'summary.json').read_text(encoding='utf-8')):
        if row['status'] != 'ok': continue
        folder = root/Path(row['file']).stem
        label = read_label(folder/'refined_mask.png')
        before = json.loads((folder/'path.json').read_text(encoding='utf-8'))
        if not (folder/'before_overlay.png').exists():
            shutil.copyfile(folder/'skeleton_overlay.png',folder/'before_overlay.png')
        with Image.open(Path('datasets/test')/row['file']) as raw:
            image=cv2.cvtColor(np.asarray(ImageOps.exif_transpose(raw).convert('RGB')),cv2.COLOR_RGB2BGR)
        path=expand_dot_paths(extract_path(label,PhotoKind.WIDE),label)
        segmentation = summarize(label, PhotoKind.WIDE)
        uncertain = cv2.imdecode(np.fromfile(str(folder/'uncertain.png'),dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
        segmentation.unknown_block_area_px = int(np.count_nonzero(uncertain))
        analysis = analyze_label(label,PhotoKind.WIDE,image,segmentation)
        result = judge_photo(analysis,label)
        (folder/'rule_result.json').write_text(result.model_dump_json(indent=2),encoding='utf-8')
        overlay,graph=draw_path(image,label,path)
        save_png(folder/'skeleton_overlay.png',overlay); save_png(folder/'skeleton_graph.png',graph)
        (folder/'path.json').write_text(path.model_dump_json(indent=2),encoding='utf-8')
        report.append(dict(file=row['file'],before=dict(Counter(n['type'] for n in before['nodes'])),after=dict(Counter(n.type.value for n in path.nodes))))
    (root/'path_changes.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__': main()
