"""Render scene4 localization with v11 conditional pixel types."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image, ImageOps
from ultralytics import YOLO
from train_v11 import TypeModel, ROOT
from probe_scene4_types import masks, overlay


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=ROOT/'datasets/test_data')
    p.add_argument('--weights',type=Path,default=ROOT/'v11_type_best.pt')
    p.add_argument('--out',type=Path,default=ROOT/'runs/v11_preview')
    p.add_argument('--location-weights',type=Path,default=None)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);device='cuda:0' if torch.cuda.is_available() else 'cpu'
    ckpt=torch.load(a.weights,map_location='cpu',weights_only=False)
    model=TypeModel();model.load_state_dict(ckpt['state_dict']);model.to(device).eval()
    size=ckpt['config']['size'];scene=YOLO(str(a.location_weights or ckpt['config']['location_weights']))
    tactile_ids=[i for i,name in scene.names.items() if name=='tactile']
    if len(tactile_ids)!=1: raise ValueError(f'Expected one tactile class, got {scene.names}')
    for file in sorted(a.source.rglob('*')):
        if file.suffix.lower() not in {'.jpg','.jpeg','.png','.webp'}: continue
        with Image.open(file) as raw: im=ImageOps.exif_transpose(raw).convert('RGB')
        w,h=im.size;result=scene.predict(im,imgsz=640,conf=.25,classes=tactile_ids,retina_masks=True,device=device,verbose=False)[0]
        scores=np.zeros((2,h,w),np.float32);counts=np.zeros((h,w),np.float32)
        for _,_,region in masks(result,h,w):
            yy,xx=np.where(region)
            if not len(xx): continue
            dx=max(8,round((xx.max()-xx.min()+1)*.1));dy=max(8,round((yy.max()-yy.min()+1)*.1))
            x1=max(0,int(xx.min())-dx);x2=min(w,int(xx.max())+1+dx)
            y1=max(0,int(yy.min())-dy);y2=min(h,int(yy.max())+1+dy)
            crop=im.crop((x1,y1,x2,y2));cw,ch=crop.size;scale=size/max(cw,ch)
            nw=max(1,round(cw*scale));nh=max(1,round(ch*scale));left=(size-nw)//2;top=(size-nh)//2
            arr=np.full((size,size,3),114,np.uint8);arr[top:top+nh,left:left+nw]=crop.resize((nw,nh),Image.Resampling.BILINEAR)
            x=torch.from_numpy(arr.transpose(2,0,1).copy()).float()[None].to(device)/255
            with torch.inference_mode():
                logits=model(x)[:,:,top:top+nh,left:left+nw]
                prob=F.interpolate(logits,size=(ch,cw),mode='bilinear',align_corners=False).softmax(1)[0].cpu().numpy()
            local=region[y1:y2,x1:x2]
            scores[:,y1:y2,x1:x2]+=prob*local;counts+=region
        scores/=np.maximum(counts,1)[None];valid=counts>0;pred=scores.argmax(0)
        uncertain=valid&(scores.max(0)<.6)
        layers=[(uncertain.astype(np.uint8),(180,180,180),'type uncertain')]
        for c,color,name in [(0,(40,210,90),'linear'),(1,(255,130,30),'dot')]:
            layers.append(((valid&~uncertain&(pred==c)).astype(np.uint8),color,name))
        overlay(im,layers).save(a.out/file.name)
        print(file.name,flush=True)


if __name__=='__main__':main()
