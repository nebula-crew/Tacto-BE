"""V11: fixed scene4 localization + independent dense linear/dot classifier.

Train only on existing GT tactile pixels; background/padding is ignored.
Encoder starts from base COCO YOLO, never from v8/v9/v10.
"""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps, ImageEnhance
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'datasets/selected_data/incremental_improvements/linear'
OUT = ROOT/'runs/v11_type'


class TypeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # YOLOv8n backbone layers 0..9 are sequential, with outputs at strides 8/16/32.
        source = YOLO(str(ROOT/'yolov8n-seg.pt')).model
        self.encoder = copy.deepcopy(source.model[:10]).float()
        self.encoder.requires_grad_(True)
        assert all(m.f == -1 for m in self.encoder)
        self.lateral = nn.ModuleList([nn.Conv2d(c, 64, 1) for c in (64,128,256)])
        self.head = nn.Sequential(nn.Conv2d(192,128,3,padding=1),nn.GroupNorm(8,128),nn.SiLU(),
                                  nn.Conv2d(128,64,3,padding=1),nn.GroupNorm(8,64),nn.SiLU(),nn.Conv2d(64,2,1))

    def forward(self, x):
        size=x.shape[-2:]; features=[]
        for i, layer in enumerate(self.encoder):
            x=layer(x)
            if i in (4,6,9): features.append(x)
        target=features[0].shape[-2:]
        xs=[F.interpolate(layer(f),size=target,mode='bilinear',align_corners=False)
            for layer,f in zip(self.lateral,features)]
        return F.interpolate(self.head(torch.cat(xs,1)),size=size,mode='bilinear',align_corners=False)


class Regions(Dataset):
    def __init__(self, split, size):
        self.train=split=='train'; self.size=size; self.rows=[]
        for image in sorted((BASE/'images'/split).iterdir()):
            lab=BASE/'labels'/split/(image.stem+'.txt')
            anns=[]
            for line in lab.read_text().splitlines():
                vals=list(map(float,line.split()))
                if len(vals)>=7: anns.append((int(vals[0]),np.array(vals[1:],dtype=np.float32).reshape(-1,2)))
            if anns: self.rows.append((image,anns))

    def __len__(self): return len(self.rows)

    def __getitem__(self,i):
        path,anns=self.rows[i]
        # Training labels are in stored raster coordinates (no additional EXIF rotation).
        with Image.open(path) as raw: im=raw.convert('RGB')
        w,h=im.size; mask=np.full((h,w),255,np.uint8)
        per_class=[]
        for cls in (0,1):
            m=np.zeros((h,w),np.uint8)
            for c,p in anns:
                if c==cls: cv2.fillPoly(m,[np.rint(p*[w,h]).astype(np.int32)],1)
            per_class.append(m); mask[m>0]=cls
        mask[(per_class[0]>0)&(per_class[1]>0)]=255
        points=np.concatenate([p for _,p in anns])*[w,h]
        lo=points.min(0);hi=points.max(0)
        margin=random.uniform(.05,.25) if self.train else .10
        pad=np.maximum(8,(hi-lo)*margin)
        x1,y1=np.maximum(0,np.floor(lo-pad)).astype(int)
        x2,y2=np.minimum([w,h],np.ceil(hi+pad)).astype(int)
        im=im.crop((x1,y1,x2,y2));mask=mask[y1:y2,x1:x2]
        if self.train:
            if random.random()<.4: im=ImageOps.grayscale(im).convert('RGB')
            im=ImageEnhance.Brightness(im).enhance(random.uniform(.85,1.15))
            if random.random()<.5: im=ImageOps.mirror(im);mask=np.fliplr(mask)
            k=random.randrange(4);im=im.rotate(90*k,expand=True);mask=np.rot90(mask,k)
        w,h=im.size;scale=self.size/max(w,h);nw=max(1,round(w*scale));nh=max(1,round(h*scale))
        arr=np.full((self.size,self.size,3),114,np.uint8);target=np.full((self.size,self.size),255,np.uint8)
        left=(self.size-nw)//2;top=(self.size-nh)//2
        arr[top:top+nh,left:left+nw]=np.asarray(im.resize((nw,nh),Image.Resampling.BILINEAR))
        target[top:top+nh,left:left+nw]=cv2.resize(mask,(nw,nh),interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(arr.transpose(2,0,1).copy()).float()/255,torch.from_numpy(target.astype(np.int64))


def loss_fn(logits,y):
    valid=y!=255
    if not valid.any(): return logits.sum()*0
    ce=F.cross_entropy(logits,y,ignore_index=255)
    prob=logits.softmax(1); dice=[]
    for c in (0,1):
        gt=(y==c).float()
        if gt.sum()>0:
            p=prob[:,c]*valid
            dice.append(1-(2*(p*gt).sum()+1)/(p.sum()+gt.sum()+1))
    return ce+torch.stack(dice).mean()


def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--size',type=int,default=640);p.add_argument('--batch',type=int,default=4)
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--location-weights',type=Path,default=Path('/home/nebula/runs/scene4_v1/weights/best.pt'))
    p.add_argument('--export',type=Path,default=None)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    output=a.out.resolve()
    export=a.export.resolve() if a.export else (ROOT/'v11_type_best.pt' if output==OUT else output/'type_best.pt')
    if not a.smoke and export.exists(): p.error(f'Export already exists: {export}')
    torch.set_num_threads(2);random.seed(47);np.random.seed(47);torch.manual_seed(47)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train,val=Regions('train',a.size),Regions('val',a.size)
    model=TypeModel().to(device)
    if a.smoke:
        x,y=train[0];pred=model(x[None].to(device));loss=loss_fn(pred,y[None].to(device));loss.backward()
        assert pred.shape==(1,2,a.size,a.size) and torch.isfinite(loss)
        print('SMOKE PASS',len(train),len(val),float(loss),flush=True);return
    output.mkdir(parents=True,exist_ok=False)
    config=dict(initialization='base COCO yolov8n-seg encoder, new dense type head',
                location_weights=str(a.location_weights.resolve()),
                source=str(BASE),train_rois=len(train),val_rois=len(val),gray_probability=.4,
                size=a.size,batch=a.batch,epochs=a.epochs,seed=47,
                ignore_index=255,classes=['linear','dot'],validation='GT ROI conditional type mIoU; not end-to-end location+type accuracy')
    (output/'config.json').write_text(json.dumps(config,indent=2))
    (output/'split.json').write_text(json.dumps({k:[str(r[0]) for r in ds.rows] for k,ds in [('train',train),('val',val)]},indent=2))
    loaders=[DataLoader(ds,batch_size=a.batch,shuffle=ds is train,num_workers=2,pin_memory=True) for ds in (train,val)]
    opt=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':1e-4},
                          {'params':list(model.lateral.parameters())+list(model.head.parameters()),'lr':5e-4}],weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs,eta_min=1e-6)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    best=-1.;best_epoch=0;history=[]
    for epoch in range(1,a.epochs+1):
        start=time.time();stats={}
        for phase,loader in zip(('train','val'),loaders):
            model.train(phase=='train');total=0.;matrix=torch.zeros((2,2),device=device,dtype=torch.long)
            for x,y in loader:
                x,y=x.to(device),y.to(device);opt.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase=='train'),torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                    logits=model(x);loss=loss_fn(logits,y)
                if phase=='train':
                    scaler.scale(loss).backward();scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(),5)
                    scaler.step(opt);scaler.update()
                total+=loss.item()*len(x)
                valid=y!=255;pred=logits.detach().argmax(1)
                matrix+=torch.bincount(2*y[valid]+pred[valid],minlength=4).reshape(2,2)
            m=matrix.cpu().numpy();iou=np.diag(m)/np.maximum(1,m.sum(0)+m.sum(1)-np.diag(m))
            stats[phase]=dict(loss=total/len(loader.dataset),iou=iou.tolist(),miou=float(iou.mean()),confusion=m.tolist())
        scheduler.step();row=dict(epoch=epoch,seconds=time.time()-start,**stats);history.append(row)
        if stats['val']['miou']>best:
            best=stats['val']['miou'];best_epoch=epoch
            checkpoint=dict(state_dict=model.state_dict(),config=config,epoch=epoch,val=stats['val'])
            torch.save(checkpoint,output/'best.pt')
        torch.save(dict(state_dict=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),config=config,epoch=epoch),output/'last.pt')
        (output/'history.json').write_text(json.dumps(history,indent=2))
        status=dict(state='training',epoch=epoch,best_epoch=best_epoch,best_miou=best,latest=row)
        (output/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
        if epoch>=30 and epoch-best_epoch>=20: break
    import shutil
    shutil.copy2(output/'best.pt',export)
    status['state']='complete';(output/'status.json').write_text(json.dumps(status,indent=2))
    print(f'SAVED {export} (custom TypeModel, not a YOLO checkpoint)',flush=True)


if __name__=='__main__': main()
