"""Real VideoSAUR inference. Microbatch only the independent frame encoder;
run the recurrent slot processor on the complete video to preserve continuity.
"""
import argparse,sys,time,json,pickle,os
from pathlib import Path
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'cjepa'))
from src.third_party.videosaur.videosaur import configuration,models
p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=0);p.add_argument('--device',default='cpu');p.add_argument('--batch',type=int,default=2);a=p.parse_args()
torch.set_num_threads(2);torch.manual_seed(42)
conf=configuration.load_config(str(ROOT.parent/'cjepa/src/third_party/videosaur/configs/videosaur/clevrer_dinov2_hf.yml'))
conf.globals.NUM_SLOTS=4
model=models.build(conf.model,conf.optimizer)
ckpt=torch.load(ROOT.parent/'.cache/clevrer_videosaur_model.ckpt',map_location='cpu',weights_only=False)
print(model.load_state_dict(ckpt['state_dict'],strict=True),flush=True);del ckpt
model.eval().to(a.device)
mean=torch.tensor([.485,.456,.406],device=a.device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=a.device)[None,:,None,None]
OUT=ROOT/'four_slots'; OUT.mkdir(exist_ok=True)
outpath=OUT/('smoke_slots.pkl' if a.limit else 'pendulum_videosaur_4slots.pkl')
slots={'train':{},'val':{}}
if outpath.exists():
 with outpath.open('rb') as f:slots=pickle.load(f)
t0=time.time()
with torch.inference_mode():
 for split in slots:
  files=sorted((ROOT/'cjepa_data_root'/f'pendulum_{split}'/'videos').glob('*.mp4'))
  if a.limit:files=files[:a.limit]
  for idx,path in enumerate(files):
   if path.name in slots[split]:continue
   # Stable per-video seed makes resumed inference reproducible.
   torch.manual_seed(42+int(path.stem.split('_')[-1]))
   cap=cv2.VideoCapture(str(path));frames=[]
   while True:
    ok,im=cap.read()
    if not ok:break
    frames.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
   cap.release();assert frames
   video=torch.from_numpy(np.stack(frames)).permute(0,3,1,2)
   features=[]
   for b in video.split(a.batch):
    b=b.to(a.device).float()/255
    b=torch.nn.functional.interpolate(b,size=(196,196),mode='bilinear',align_corners=False,antialias=True)
    features.append(model.encoder(((b-mean)/std).unsqueeze(0))['features'])
   f=torch.cat(features,dim=1)
   result=model.processor(model.initializer(batch_size=1).to(a.device),f)
   arr=result['state'][0].cpu().numpy().astype('float32')
   assert arr.shape==(len(frames),4,128) and np.isfinite(arr).all()
   masks=result['corrector']['masks'][0].cpu().numpy().astype('float32')
   assert masks.shape==(len(frames),4,196)
   assert np.allclose(masks.sum(axis=1),1,atol=1e-5)
   if idx in [0,1,2]:
    np.savez_compressed(OUT/f'{split}_{path.stem}_attention.npz',attention=masks.reshape(-1,4,14,14),slots=arr)
   # Summary diagnostics for every video, without storing large full mask archives.
   import csv
   stats=OUT/'diagnostics.csv'; exists=stats.exists()
   with stats.open('a',newline='') as fh:
    w=csv.writer(fh)
    if not exists:w.writerow(['split','video','slot','mean_soft_area','mean_hard_area','mean_l2_norm','mean_frame_delta_l2','mean_patch_entropy'])
    labels=masks.argmax(axis=1)
    entropy=-(masks*np.log(masks.clip(1e-10))).sum(axis=1).mean()
    for k in range(4):w.writerow([split,path.name,k+1,float(masks[:,k].mean()),float((labels==k).mean()),float(np.linalg.norm(arr[:,k],axis=-1).mean()),float(np.linalg.norm(np.diff(arr[:,k],axis=0),axis=-1).mean()),float(entropy)])
   slots[split][path.name]=arr
   with open(str(outpath)+'.tmp','wb') as fh:pickle.dump(slots,fh)
   os.replace(str(outpath)+'.tmp',outpath)
   print(split,path.name,arr.shape,'elapsed',round(time.time()-t0,1),flush=True)
print('COMPLETE',outpath,{k:len(v) for k,v in slots.items()},flush=True)
