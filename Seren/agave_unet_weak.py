#!/usr/bin/env python
"""Weakly supervised multispectral U-Net for Agave Sentinel-2 64x64 patches."""
import argparse, json, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

VERSION='1.0-weak-mil-unet'
BANDS=9; SIZE=64; BAND_NAMES=['B2','B3','B4','B5','B6','B7','B8','B11','B12']

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def load_ckpt(path, device='cpu'):
    return torch.load(path, map_location=device, weights_only=False)

def standardize(df):
    df=df.copy()
    if 'image_path' not in df: raise ValueError('manifest needs image_path')
    if 'target' not in df:
        if 'standardized_label' in df: df['target']=df['standardized_label'].map({'agave':1,'not_agave':0})
        elif 'label' in df: df['target']=df['label'].map({'yes':1,'no':0,'agave':1,'not_agave':0,'1':1,'0':0})
        else: raise ValueError('manifest needs target, standardized_label, or label')
    if 'base_id' not in df:
        if 'id' in df: df['base_id']=df['id'].astype(str).str.replace(r'_\d{4}$','',regex=True)
        else: raise ValueError('manifest needs base_id or id')
    if 'year' not in df:
        if 'target_year' in df: df['year']=df['target_year']
        else: raise ValueError('manifest needs year or target_year')
    df=df[df.target.isin([0,1])].copy()
    df['target']=df.target.astype(int); df['year']=df.year.astype(int); df['base_id']=df.base_id.astype(str); df['image_path']=df.image_path.astype(str)
    return df.reset_index(drop=True)

def split_groups(df, seed=42):
    if 'split' in df:
        s=df['split'].astype(str).str.lower().replace({'val':'validation'})
        tr=df[s=='train'].copy(); va=df[s=='validation'].copy(); te=df[s=='test'].copy()
        if len(tr) and len(va) and len(te):
            a,b,c=set(tr.base_id),set(va.base_id),set(te.base_id)
            if a&b or a&c or b&c: raise RuntimeError('base_id leakage in supplied splits')
            return tr,va,te
    g1=GroupShuffleSplit(n_splits=1,train_size=.70,random_state=seed)
    tri,ri=next(g1.split(df,groups=df.base_id)); tr=df.iloc[tri].copy(); rest=df.iloc[ri].copy()
    g2=GroupShuffleSplit(n_splits=1,train_size=.50,random_state=seed+1)
    vi,ti=next(g2.split(rest,groups=rest.base_id)); va=rest.iloc[vi].copy(); te=rest.iloc[ti].copy()
    tr['split']='train'; va['split']='validation'; te['split']='test'
    return tr,va,te

def check_raster(p):
    with rasterio.open(p) as s:
        if s.count!=BANDS or s.width!=SIZE or s.height!=SIZE: raise ValueError(f'{p}: expected 9x64x64, got {s.count}x{s.height}x{s.width}')

def normalization(df,n,seed):
    sm=df.sample(min(n,len(df)),random_state=seed); sums=np.zeros(BANDS); sums2=np.zeros(BANDS); counts=np.zeros(BANDS,dtype=np.int64)
    for p in tqdm(sm.image_path,desc='Computing normalization'):
        with rasterio.open(p) as s:
            a=s.read(out_dtype='float32'); v=np.isfinite(a)
            if s.nodata is not None: v &= a!=s.nodata
        for b in range(BANDS):
            z=a[b][v[b]]
            if z.size: sums[b]+=z.sum(dtype=np.float64); sums2[b]+=(z.astype(np.float64)**2).sum(); counts[b]+=z.size
    m=sums/counts; sd=np.sqrt(np.maximum(sums2/counts-m*m,1e-12)); return m.astype('float32'),sd.astype('float32')

class DS(Dataset):
    def __init__(self,df,m,sd,aug=False): self.df=df.reset_index(drop=True); self.m=np.array(m,dtype='float32')[:,None,None]; self.sd=np.maximum(np.array(sd,dtype='float32'),1e-6)[:,None,None]; self.aug=aug
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        with rasterio.open(r.image_path) as s: a=s.read(out_dtype='float32'); valid=s.read_masks(1)>0
        a[:,~valid]=np.nan
        for b in range(BANDS):
            bad=~np.isfinite(a[b]); a[b,bad]=self.m[b,0,0]
        a=(a-self.m)/self.sd
        if self.aug:
            if random.random()<.5: a=np.flip(a,2).copy()
            if random.random()<.5: a=np.flip(a,1).copy()
            k=random.randint(0,3)
            if k: a=np.rot90(a,k,axes=(1,2)).copy()
        return torch.from_numpy(a.astype('float32')),torch.tensor(float(r.target),dtype=torch.float32),i

class DC(nn.Module):
    def __init__(self,ci,co): super().__init__(); self.n=nn.Sequential(nn.Conv2d(ci,co,3,padding=1,bias=False),nn.BatchNorm2d(co),nn.ReLU(),nn.Conv2d(co,co,3,padding=1,bias=False),nn.BatchNorm2d(co),nn.ReLU())
    def forward(self,x): return self.n(x)
class UNet(nn.Module):
    def __init__(self,base=32):
        super().__init__(); self.e1=DC(9,base); self.e2=DC(base,base*2); self.e3=DC(base*2,base*4); self.e4=DC(base*4,base*8); self.p=nn.MaxPool2d(2); self.b=DC(base*8,base*16)
        self.u4=nn.ConvTranspose2d(base*16,base*8,2,2); self.d4=DC(base*16,base*8); self.u3=nn.ConvTranspose2d(base*8,base*4,2,2); self.d3=DC(base*8,base*4); self.u2=nn.ConvTranspose2d(base*4,base*2,2,2); self.d2=DC(base*4,base*2); self.u1=nn.ConvTranspose2d(base*2,base,2,2); self.d1=DC(base*2,base); self.o=nn.Conv2d(base,1,1)
    def forward(self,x):
        e1=self.e1(x); e2=self.e2(self.p(e1)); e3=self.e3(self.p(e2)); e4=self.e4(self.p(e3)); b=self.b(self.p(e4)); d4=self.d4(torch.cat([self.u4(b),e4],1)); d3=self.d3(torch.cat([self.u3(d4),e3],1)); d2=self.d2(torch.cat([self.u2(d3),e2],1)); d1=self.d1(torch.cat([self.u1(d2),e1],1)); return self.o(d1).squeeze(1)

def pool_topk(logits,frac):
    f=logits.flatten(1); k=max(1,int(round(f.shape[1]*frac))); return torch.topk(f,k,dim=1).values.mean(1)
def tv(prob): return (prob[:,:,1:]-prob[:,:,:-1]).abs().mean()+(prob[:,1:,:]-prob[:,:-1,:]).abs().mean()
def loss_fn(px,y,args):
    pl=pool_topk(px,args.topk_fraction); patch=F.binary_cross_entropy_with_logits(pl,y); prob=torch.sigmoid(px); neg=y<.5; pos=y>.5
    negloss=F.binary_cross_entropy_with_logits(px[neg],torch.zeros_like(px[neg])) if neg.any() else px.sum()*0
    sparse=prob[pos].mean() if pos.any() else prob.sum()*0
    return patch+.5*negloss+args.tv_weight*tv(prob)+args.sparsity_weight*sparse,pl

def met(y,p,t):
    y=np.asarray(y).astype(int); p=np.asarray(p); pr=(p>=t).astype(int)
    d={'n':int(len(y)),'positives':int(y.sum()),'threshold':float(t),'accuracy':float(accuracy_score(y,pr)),'balanced_accuracy':float(balanced_accuracy_score(y,pr)),'precision':float(precision_score(y,pr,zero_division=0)),'recall':float(recall_score(y,pr,zero_division=0)),'f1':float(f1_score(y,pr,zero_division=0)),'confusion_matrix':confusion_matrix(y,pr,labels=[0,1]).tolist()}
    if len(np.unique(y))==2: d['roc_auc']=float(roc_auc_score(y,p)); d['average_precision']=float(average_precision_score(y,p))
    else: d['roc_auc']=None; d['average_precision']=None
    return d
def threshold(y,p):
    ts=np.arange(.05,.951,.01); fs=[f1_score(y,(p>=t).astype(int),zero_division=0) for t in ts]; return float(ts[int(np.argmax(fs))])
def epoch(model,loader,opt,dev,train,args):
    model.train(train); losses=[]; ys=[]; ps=[]; ids=[]
    with (torch.enable_grad() if train else torch.no_grad()):
        for x,y,i in loader:
            x=x.to(dev); y=y.to(dev)
            if train: opt.zero_grad(set_to_none=True)
            px=model(x); loss,pl=loss_fn(px,y,args)
            if train: loss.backward(); opt.step()
            losses.append(loss.item()); ys.append(y.cpu().numpy()); ps.append(torch.sigmoid(pl).detach().cpu().numpy()); ids.append(np.asarray(i))
    return float(np.mean(losses)),np.concatenate(ys),np.concatenate(ps),np.concatenate(ids)

def loader(ds,b,w,shuffle=False,sampler=None): return DataLoader(ds,batch_size=b,shuffle=shuffle if sampler is None else False,sampler=sampler,num_workers=w,pin_memory=torch.cuda.is_available())

def train(args):
    seed_all(args.seed); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); df=standardize(pd.read_csv(args.manifest)); tr,va,te=split_groups(df,args.seed)
    print(f'Train: {len(tr)} / {tr.base_id.nunique()} base_ids'); print(f'Validation: {len(va)} / {va.base_id.nunique()}'); print(f'Test: {len(te)} / {te.base_id.nunique()}')
    if args.validate_rasters:
        for p in tqdm(df.image_path.unique(),desc='Validating rasters'): check_raster(p)
    tr.to_csv(out/'train_manifest.csv',index=False); va.to_csv(out/'validation_manifest.csv',index=False); te.to_csv(out/'test_manifest.csv',index=False)
    m,sd=normalization(tr,args.normalization_images,args.seed); (out/'normalization.json').write_text(json.dumps({'bands':BAND_NAMES,'means':m.tolist(),'stds':sd.tolist()},indent=2))
    dtr,dva,dte=DS(tr,m,sd,True),DS(va,m,sd),DS(te,m,sd); sampler=None
    if args.balanced_sampler:
        c=tr.target.value_counts().to_dict(); weights=tr.target.map(lambda z:1/c[z]).values; sampler=WeightedRandomSampler(weights,len(weights),replacement=True)
    ltr,lva,lte=loader(dtr,args.batch_size,args.workers,True,sampler),loader(dva,args.batch_size,args.workers),loader(dte,args.batch_size,args.workers)
    dev=torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')); print('Device:',dev); model=UNet(args.base_channels).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay); sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=3)
    best=-math.inf; bad=0; hist=[]
    for ep in range(1,args.epochs+1):
        tl,_,_,_=epoch(model,ltr,opt,dev,True,args); vl,vy,vp,_=epoch(model,lva,opt,dev,False,args); vm=met(vy,vp,.5); score=vm['average_precision']; sched.step(score); hist.append({'epoch':ep,'train_loss':tl,'val_loss':vl,'val_f1_at_0.5':vm['f1'],'val_roc_auc':vm['roc_auc'],'val_average_precision':vm['average_precision'],'learning_rate':opt.param_groups[0]['lr']}); print(f'Epoch {ep:03d} train_loss={tl:.4f} val_loss={vl:.4f} val_f1={vm["f1"]:.4f} val_AP={score:.4f}')
        if score>best+1e-6:
            best=score; bad=0; torch.save({'state_dict':model.state_dict(),'means':m.tolist(),'stds':sd.tolist(),'base_channels':args.base_channels,'topk_fraction':args.topk_fraction,'epoch':ep,'script_version':VERSION},out/'best_model.pt')
        else: bad+=1
        if args.patience>0 and bad>=args.patience: print('Early stopping'); break
    pd.DataFrame(hist).to_csv(out/'training_history.csv',index=False); ck=load_ckpt(out/'best_model.pt',dev); model.load_state_dict(ck['state_dict']); _,vy,vp,vi=epoch(model,lva,opt,dev,False,args); th=threshold(vy,vp); _,ty,tp,ti=epoch(model,lte,opt,dev,False,args); pred=te.iloc[ti].copy(); pred['probability_agave_patch']=tp; pred['prediction']=(tp>=th).astype(int); pred.to_csv(out/'test_predictions.csv',index=False)
    report={'script_version':VERSION,'supervision':'weak_patch_level_MIL','warning':'Pixel outputs are pseudo-segmentation maps; no true pixel masks were used.','device':str(dev),'best_model_epoch':int(ck['epoch']),'selected_patch_threshold':th,'topk_fraction':args.topk_fraction,'train_n':len(tr),'validation_n':len(va),'test_n':len(te),'test_metrics':met(ty,tp,th),'test_by_year':{str(y):met(g.target.values,g.probability_agave_patch.values,th) for y,g in pred.groupby('year')}}; (out/'metrics.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))

def write_tif(srcp,outp,arr,dtype,nodata=None):
    with rasterio.open(srcp) as s:
        prof=s.profile.copy(); prof.update(count=1,dtype=dtype,compress='deflate',nodata=nodata)
        with rasterio.open(outp,'w',**prof) as d: d.write(arr.astype(dtype),1)
def predict(args):
    ck=load_ckpt(args.model); m=np.array(ck['means'],dtype='float32'); sd=np.maximum(np.array(ck['stds'],dtype='float32'),1e-6); model=UNet(int(ck.get('base_channels',32))); model.load_state_dict(ck['state_dict']); dev=torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')); model.to(dev).eval(); df=standardize(pd.read_csv(args.manifest)); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); rows=[]
    for _,r in tqdm(df.iterrows(),total=len(df),desc='Writing pseudo-masks'):
        with rasterio.open(r.image_path) as s: a=s.read(out_dtype='float32'); valid=s.read_masks(1)>0
        a[:,~valid]=np.nan
        for b in range(BANDS): a[b,~np.isfinite(a[b])]=m[b]
        a=(a-m[:,None,None])/sd[:,None,None]; x=torch.from_numpy(a[None]).float().to(dev)
        with torch.no_grad(): p=torch.sigmoid(model(x))[0].cpu().numpy()
        stem=Path(r.image_path).stem; pp=out/f'{stem}_weak_unet_probability.tif'; mp=out/f'{stem}_weak_unet_pseudomask.tif'; write_tif(r.image_path,pp,p,'float32'); binary=(p>=args.pixel_threshold).astype('uint8'); write_tif(r.image_path,mp,binary,'uint8',255); rows.append({'image_path':r.image_path,'base_id':r.base_id,'year':int(r.year),'target':int(r.target),'mean_pixel_probability':float(p.mean()),'max_pixel_probability':float(p.max()),'fraction_pixels_above_threshold':float(binary.mean()),'pixel_threshold':args.pixel_threshold,'probability_raster':str(pp),'pseudomask_raster':str(mp)})
    pd.DataFrame(rows).to_csv(out/'pseudo_mask_summary.csv',index=False); print('Pseudo-masks:',out.resolve())

def cli():
    p=argparse.ArgumentParser(description='Weakly supervised 9-band U-Net for Agave Sentinel-2 patches.'); p.add_argument('--version',action='version',version=VERSION); sp=p.add_subparsers(dest='command',required=True)
    t=sp.add_parser('train'); t.add_argument('--manifest',required=True); t.add_argument('--output-dir',required=True); t.add_argument('--epochs',type=int,default=60); t.add_argument('--batch-size',type=int,default=16); t.add_argument('--workers',type=int,default=0); t.add_argument('--seed',type=int,default=42); t.add_argument('--learning-rate',type=float,default=1e-3); t.add_argument('--weight-decay',type=float,default=1e-4); t.add_argument('--base-channels',type=int,default=32); t.add_argument('--topk-fraction',type=float,default=.10); t.add_argument('--tv-weight',type=float,default=.02); t.add_argument('--sparsity-weight',type=float,default=.02); t.add_argument('--normalization-images',type=int,default=2000); t.add_argument('--balanced-sampler',action='store_true'); t.add_argument('--validate-rasters',action='store_true'); t.add_argument('--patience',type=int,default=10); t.add_argument('--device',default=None); t.set_defaults(func=train)
    q=sp.add_parser('predict'); q.add_argument('--model',required=True); q.add_argument('--manifest',required=True); q.add_argument('--output-dir',required=True); q.add_argument('--pixel-threshold',type=float,default=.5); q.add_argument('--device',default=None); q.set_defaults(func=predict); return p

if __name__=='__main__':
    a=cli().parse_args(); print(f'Agave Weak U-Net script version: {VERSION}'); a.func(a)