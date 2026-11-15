from pathlib import Path
import cv2
import numpy as np

root = Path('outputs/photo_review')
dirs = sorted(p for p in root.iterdir() if p.is_dir())
for kind in ('block', 'scene', 'merged', 'path'):
    canvas = np.full((len(dirs) // 4 * 190 + (190 if len(dirs) % 4 else 0), 4 * 250, 3), 245, np.uint8)
    for i, folder in enumerate(dirs):
        file = folder / (kind + '_preview.png')
        if not file.exists(): continue
        im = cv2.imdecode(np.fromfile(str(file), np.uint8), cv2.IMREAD_COLOR)
        s = min(240 / im.shape[1], 160 / im.shape[0])
        im = cv2.resize(im, (round(im.shape[1]*s), round(im.shape[0]*s)))
        y, x = (i // 4)*190, (i % 4)*250
        canvas[y:y+im.shape[0], x:x+im.shape[1]] = im
        cv2.putText(canvas, str(i)+' '+folder.name[:26], (x+2,y+178), cv2.FONT_HERSHEY_SIMPLEX, .36, (0,0,0), 1)
    cv2.imencode('.jpg', canvas)[1].tofile(str(root / (kind + '_sheet.jpg')))
(root / 'index.txt').write_text('\n'.join(f'{i} {p.name}' for i,p in enumerate(dirs)), encoding='utf-8')
