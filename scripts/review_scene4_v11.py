"""Review scene4 tactile regions and v11 type masks at original resolution."""
from pathlib import Path
import html, json, sys, traceback
import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F
from ultralytics import YOLO
from pipeline.schema import MaskClass, PhotoKind
from pipeline.skeleton import extract_path
from pipeline.dot_paths import expand_dot_paths
from pipeline.mask_refine import refine_mask
from scripts.run_path_debug import draw_path, save_png

BUNDLE = Path('outputs/v11_bundle_inspect/v11_linear_integration')
sys.path.insert(0, str(BUNDLE / 'training'))
from train_v11 import TypeModel
from probe_scene4_types import masks

def main():
    torch.set_num_threads(2)
    scene = YOLO('weights/scene4_v1_best.pt')
    model = TypeModel()
    ckpt = torch.load('weights/v11_best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()
    size = ckpt['config']['size']
    tactile_ids = [i for i, n in scene.names.items() if n == 'tactile']
    if len(tactile_ids) != 1:
        raise ValueError('scene model must have exactly one tactile class')
    out = Path('outputs/review_v11_scene4')
    out.mkdir(parents=True, exist_ok=True)
    rows, cards = [], []
    for file in sorted(Path('datasets/test').iterdir()):
        if not file.is_file() or file.name.startswith('._') or file.suffix.lower() not in ('.jpg','.jpeg','.png','.webp'):
            continue
        try:
            with Image.open(file) as raw:
                im = ImageOps.exif_transpose(raw).convert('RGB')
            w,h = im.size
            result = scene.predict(im, imgsz=640, conf=.25, retina_masks=True, verbose=False)[0]
            detections = masks(result,h,w)
            scores = np.zeros((2,h,w),np.float32)
            counts = np.zeros((h,w),np.float32)
            label = np.zeros((h,w),np.uint8)
            # Scene masks are assigned by confidence, then tactile takes priority.
            for cls, conf, region in sorted(detections,key=lambda x:x[1]):
                name = scene.names[cls]
                if name in ('road','crosswalk'):
                    label[region != 0] = int(MaskClass.ROAD if name == 'road' else MaskClass.CROSSWALK)
            for cls, conf, region in detections:
                if cls not in tactile_ids:
                    continue
                yy,xx = np.where(region)
                if not len(xx): continue
                dx=max(8,round((xx.max()-xx.min()+1)*.1)); dy=max(8,round((yy.max()-yy.min()+1)*.1))
                x1=max(0,int(xx.min())-dx); x2=min(w,int(xx.max())+1+dx)
                y1=max(0,int(yy.min())-dy); y2=min(h,int(yy.max())+1+dy)
                crop=im.crop((x1,y1,x2,y2)); cw,ch=crop.size; scale=size/max(cw,ch)
                nw=max(1,round(cw*scale)); nh=max(1,round(ch*scale)); left=(size-nw)//2; top=(size-nh)//2
                arr=np.full((size,size,3),114,np.uint8); arr[top:top+nh,left:left+nw]=crop.resize((nw,nh),Image.Resampling.BILINEAR)
                x=torch.from_numpy(arr.transpose(2,0,1).copy()).float()[None]/255
                with torch.inference_mode():
                    logits=model(x)[:,:,top:top+nh,left:left+nw]
                    prob=F.interpolate(logits,size=(ch,cw),mode='bilinear',align_corners=False).softmax(1)[0].numpy()
                local=region[y1:y2,x1:x2]
                scores[:,y1:y2,x1:x2]+=prob*local; counts+=region
            scores/=np.maximum(counts,1)[None]
            valid=counts>0; uncertain=valid & (scores.max(0)<.6); pred=scores.argmax(0)
            label[valid] = 0
            label[valid & ~uncertain & (pred==0)] = int(MaskClass.LINEAR)
            label[valid & ~uncertain & (pred==1)] = int(MaskClass.DOT)
            refined, _ = refine_mask(label)
            path = expand_dot_paths(extract_path(refined,PhotoKind.WIDE),refined)
            bgr=cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)
            overlay,graph=draw_path(bgr,refined,path)
            target=out/file.stem
            save_png(target/'label_mask.png',label); save_png(target/'refined_mask.png',refined)
            save_png(target/'uncertain.png',uncertain.astype(np.uint8)*255)
            save_png(target/'skeleton_overlay.png',overlay); save_png(target/'skeleton_graph.png',graph)
            (target/'path.json').write_text(path.model_dump_json(indent=2),encoding='utf-8')
            rows.append(dict(file=file.name,status='ok',linear=int((label==1).sum()),dot=int((label==2).sum()),uncertain=int(uncertain.sum())))
            cards.append(f'<section><h2>{html.escape(file.name)}</h2><a href="{html.escape(file.stem)}/skeleton_overlay.png"><img src="{html.escape(file.stem)}/skeleton_overlay.png"></a><p><a href="{html.escape(file.stem)}/label_mask.png">마스크</a> · <a href="{html.escape(file.stem)}/uncertain.png">불확실 영역</a> · <a href="{html.escape(file.stem)}/skeleton_graph.png">중심선</a></p></section>')
            print('OK',file.name,flush=True)
        except Exception as exc:
            rows.append(dict(file=file.name,status='error',error=str(exc),traceback=traceback.format_exc()))
            print('ERROR',file.name,str(exc),flush=True)
    (out/'summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'index.html').write_text('<meta charset="utf-8"><title>scene4 + v11</title><style>body{font-family:sans-serif;background:#f3f5f4}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}section{background:white;padding:16px}img{max-width:100%;max-height:500px}h2{font-size:14px}</style><h1>scene4 + v11 중심선</h1><p>사진을 누르면 원본 크기로 열립니다. 결과는 정답 평가가 아닙니다.</p><main>'+''.join(cards)+'</main>',encoding='utf-8')

if __name__ == '__main__': main()
