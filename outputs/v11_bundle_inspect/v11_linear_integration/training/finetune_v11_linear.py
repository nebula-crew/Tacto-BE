"""Fine-tune v11 with mild linear/mixed oversampling; preserve v11 and evaluate both classes."""
import json, random, time, subprocess, sys, html
from pathlib import Path
from urllib.parse import quote
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch import nn
from train_v11 import ROOT, Regions, TypeModel, loss_fn

out=ROOT/'runs/v11_linear_focus'
out.mkdir(exist_ok=False,parents=True)
torch.set_num_threads(2);torch.manual_seed(123);random.seed(123);np.random.seed(123)
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train=Regions('train',640);val=Regions('val',640)
weights=[];counts={'linear':0,'dot':0,'mixed':0}
for _,anns in train.rows:
 classes={a[0] for a in anns};kind='mixed' if len(classes)==2 else 'linear' if 0 in classes else 'dot'
 counts[kind]+=1;weights.append({'linear':1.3,'mixed':1.4,'dot':1.0}[kind])
sampler=WeightedRandomSampler(weights,len(weights),replacement=True,generator=torch.Generator().manual_seed(123))
loaders=[DataLoader(train,batch_size=4,sampler=sampler,num_workers=2,pin_memory=True),DataLoader(val,batch_size=4,shuffle=False,num_workers=2,pin_memory=True)]
model=TypeModel().to(device);ckpt=torch.load(ROOT/'v11_type_best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ckpt['state_dict'])
config=dict(ckpt['config']);config.update(initialization=str(ROOT/'v11_type_best.pt'),linear_sampling=1.3,mixed_sampling=1.4,dot_sampling=1.0,seed=123,epochs=20,encoder_lr=0.00001,head_lr=0.00005)
(out/'config.json').write_text(json.dumps(dict(config=config,counts=counts),indent=2))
opt=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':1e-5},{'params':list(model.lateral.parameters())+list(model.head.parameters()),'lr':5e-5}],weight_decay=.01)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,20,eta_min=1e-6)
scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda');history=[];best=-1;best_epoch=0
for epoch in range(1,21):
 stats={};start=time.time()
 for phase,loader in zip(('train','val'),loaders):
  model.train(phase=='train');total=0;matrix=torch.zeros((2,2),device=device,dtype=torch.long)
  for x,y in loader:
   x,y=x.to(device),y.to(device);opt.zero_grad(set_to_none=True)
   with torch.set_grad_enabled(phase=='train'),torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
    logits=model(x);loss=loss_fn(logits,y)
   if phase=='train':
    scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),5);scaler.step(opt);scaler.update()
   total+=loss.item()*len(x);valid=y!=255;pred=logits.detach().argmax(1)
   matrix+=torch.bincount(2*y[valid]+pred[valid],minlength=4).reshape(2,2)
  m=matrix.cpu().numpy();iou=np.diag(m)/np.maximum(1,m.sum(0)+m.sum(1)-np.diag(m))
  stats[phase]=dict(loss=total/len(loader.dataset),iou=iou.tolist(),miou=float(iou.mean()),confusion=m.tolist())
 scheduler.step();row=dict(epoch=epoch,seconds=time.time()-start,**stats);history.append(row)
 if stats['val']['miou']>best:
  best=stats['val']['miou'];best_epoch=epoch
  torch.save(dict(state_dict=model.state_dict(),config=config,epoch=epoch,val=stats['val']),out/'best.pt')
 torch.save(dict(state_dict=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),config=config,epoch=epoch),out/'last.pt')
 status=dict(state='training',epoch=epoch,best_epoch=best_epoch,best_miou=best,latest=row)
 (out/'history.json').write_text(json.dumps(history,indent=2));(out/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
 if epoch>=10 and epoch-best_epoch>=8: break
status['state']='generating_comparison';(out/'status.json').write_text(json.dumps(status,indent=2))
subprocess.run([sys.executable,str(ROOT/'training/predict_v11.py'),'--weights',str(out/'best.pt'),'--out',str(out/'preview')],check=True,cwd=ROOT)
rows=json.loads((ROOT/'runs/scene4_type_probe/report.json').read_text())['images']
for name,target in [('probe','../scene4_type_probe'),('v11','../v11_preview'),('new','preview')]:
 (out/name).symlink_to(target,target_is_directory=True)
cards=[]
for i,r in enumerate(rows):
 items=[('v9',f'probe/{i:03d}_v9_full.jpg'),('v11', 'v11/'+quote(r['file'])),('v11 선형 보강','new/'+quote(r['file']))]
 cells=''.join(f'<figure><figcaption>{html.escape(label)}</figcaption><a href="{url}" target="_blank"><img loading="lazy" src="{url}"></a></figure>' for label,url in items)
 cards.append(f'<section><h2>{html.escape(r["file"])}</h2><div>{cells}</div></section>')
(out/'index.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>v9 v11 선형 보강 비교</title><style>body{background:#16191f;color:#eee;font:16px sans-serif;margin:20px}section{border-top:1px solid #555;margin:28px 0}section>div{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%}h2{font-size:16px;overflow-wrap:anywhere}figcaption{padding:8px}@media(max-width:800px){section>div{grid-template-columns:1fr}}</style><h1>v9 · v11 · v11 선형 보강</h1><p>초록: 선형 · 주황: 점형 · 회색: 유형 불확실. 이미지를 클릭하면 확대됩니다.</p><p>v9은 전체 사진 검출, v11 두 모델은 동일한 scene4 검출 영역에서 유형을 분류합니다. test_data는 정답이 없으므로 시각 비교입니다.</p>'''+''.join(cards)+'</html>')
status['state']='complete';(out/'status.json').write_text(json.dumps(status,indent=2));print('COMPLETE',flush=True)
