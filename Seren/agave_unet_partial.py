#!/usr/bin/env python
"""
agave_unet_partial.py

Pipeline
--------
1) build-pseudolabels:
   Uses the weak U-Net probability GeoTIFFs plus the original patch labels.

   For known non-agave patches:
       every valid pixel -> 0

   For known agave patches:
       probability >= positive_threshold -> 1
       probability <= negative_threshold -> 0
       otherwise -> 255 (IGNORE)

2) train:
   Trains a normal 9-band U-Net using masked BCE + masked Dice loss.
   Pixels with value 255 contribute nothing to the loss.

3) predict:
   Writes probability maps and binary masks for any manifest.

Mask values
-----------
0   confident non-agave
1   confident agave
255 unknown / ignore
"""

import argparse, json, math, random
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_auc_score,
    average_precision_score
)
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

VERSION = "1.0-partial-pseudolabel-unet"
IGNORE = 255
N_BANDS = 9
SIZE = 64
BANDS = ["B2","B3","B4","B5","B6","B7","B8","B11","B12"]


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_ckpt(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def norm_manifest(df):
    df = df.copy()
    if "target" not in df.columns:
        if "standardized_label" in df.columns:
            df["target"] = df["standardized_label"].map({"agave":1,"not_agave":0})
        elif "label" in df.columns:
            df["target"] = df["label"].map({"yes":1,"no":0,"agave":1,"not_agave":0,1:1,0:0,"1":1,"0":0})
        else:
            raise ValueError("Need target, standardized_label, or label column.")
    if "image_path" not in df.columns:
        raise ValueError("Need image_path column.")
    if "base_id" not in df.columns:
        raise ValueError("Need base_id column.")
    if "year" not in df.columns:
        raise ValueError("Need year column.")
    df = df[df.target.isin([0,1])].copy()
    df["target"] = df.target.astype(int)
    df["year"] = df.year.astype(int)
    df["base_id"] = df.base_id.astype(str)
    return df.reset_index(drop=True)


def find_probability_raster(prob_dir, image_path):
    stem = Path(image_path).stem
    candidates = [
        Path(prob_dir) / f"{stem}_weak_unet_probability.tif",
        Path(prob_dir) / f"{stem}_probability.tif",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def write_mask_like(src_path, out_path, mask):
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=IGNORE, compress="deflate")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mask.astype("uint8"), 1)


def command_build(args):
    df = norm_manifest(pd.read_csv(args.manifest))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = 0

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Building partial pseudo-labels"):
        prob_path = find_probability_raster(args.probability_dir, r.image_path)
        if prob_path is None:
            missing += 1
            continue

        with rasterio.open(r.image_path) as src:
            valid = src.read_masks(1) > 0
            transform = src.transform
            crs = src.crs
            width, height = src.width, src.height

        with rasterio.open(prob_path) as src:
            prob = src.read(1).astype(np.float32)
            if src.width != width or src.height != height:
                raise ValueError(f"Grid size mismatch: {prob_path}")

        mask = np.full((height, width), IGNORE, dtype=np.uint8)

        if int(r.target) == 0:
            # Strong negative supervision.
            mask[valid] = 0
        else:
            # Positive patch: retain only high-confidence pseudo-labels.
            mask[valid & (prob <= args.negative_threshold)] = 0
            mask[valid & (prob >= args.positive_threshold)] = 1

        name = Path(r.image_path).stem + "_partial_mask.tif"
        mask_path = out / name
        write_mask_like(r.image_path, mask_path, mask)

        n_valid = int((mask != IGNORE).sum())
        n_pos = int((mask == 1).sum())
        n_neg = int((mask == 0).sum())
        rows.append({
            **r.to_dict(),
            "mask_path": str(mask_path.resolve()),
            "weak_probability_path": str(prob_path.resolve()),
            "confident_pixels": n_valid,
            "confident_positive_pixels": n_pos,
            "confident_negative_pixels": n_neg,
            "ignored_pixels": int((mask == IGNORE).sum()),
            "fraction_confident": n_valid / mask.size,
            "fraction_positive": n_pos / mask.size,
        })

    result = pd.DataFrame(rows)
    result.to_csv(out / "partial_unet_manifest.csv", index=False)

    report = {
        "script_version": VERSION,
        "source_rows": int(len(df)),
        "masks_created": int(len(result)),
        "missing_probability_rasters": int(missing),
        "positive_threshold": float(args.positive_threshold),
        "negative_threshold": float(args.negative_threshold),
        "mask_values": {"non_agave":0,"agave":1,"ignore":255},
    }
    if len(result):
        pos = result[result.target == 1]
        neg = result[result.target == 0]
        report["positive_patch_summary"] = {
            "n": int(len(pos)),
            "mean_fraction_confident": float(pos.fraction_confident.mean()) if len(pos) else None,
            "mean_fraction_positive": float(pos.fraction_positive.mean()) if len(pos) else None,
            "median_fraction_positive": float(pos.fraction_positive.median()) if len(pos) else None,
        }
        report["negative_patch_summary"] = {
            "n": int(len(neg)),
            "mean_fraction_confident": float(neg.fraction_confident.mean()) if len(neg) else None,
        }

    (out / "partial_unet_manifest.report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Manifest: {(out/'partial_unet_manifest.csv').resolve()}")


class DoubleConv(nn.Module):
    def __init__(self, a,b):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(a,b,3,padding=1,bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
            nn.Conv2d(b,b,3,padding=1,bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True)
        )
    def forward(self,x): return self.net(x)


class UNet(nn.Module):
    def __init__(self, inc=9, base=32):
        super().__init__()
        self.e1=DoubleConv(inc,base); self.e2=DoubleConv(base,base*2)
        self.e3=DoubleConv(base*2,base*4); self.e4=DoubleConv(base*4,base*8)
        self.pool=nn.MaxPool2d(2); self.b=DoubleConv(base*8,base*16)
        self.u4=nn.ConvTranspose2d(base*16,base*8,2,2); self.d4=DoubleConv(base*16,base*8)
        self.u3=nn.ConvTranspose2d(base*8,base*4,2,2); self.d3=DoubleConv(base*8,base*4)
        self.u2=nn.ConvTranspose2d(base*4,base*2,2,2); self.d2=DoubleConv(base*4,base*2)
        self.u1=nn.ConvTranspose2d(base*2,base,2,2); self.d1=DoubleConv(base*2,base)
        self.out=nn.Conv2d(base,1,1)
    def forward(self,x):
        e1=self.e1(x); e2=self.e2(self.pool(e1)); e3=self.e3(self.pool(e2)); e4=self.e4(self.pool(e3))
        b=self.b(self.pool(e4))
        d4=self.d4(torch.cat([self.u4(b),e4],1))
        d3=self.d3(torch.cat([self.u3(d4),e3],1))
        d2=self.d2(torch.cat([self.u2(d3),e2],1))
        d1=self.d1(torch.cat([self.u1(d2),e1],1))
        return self.out(d1).squeeze(1)


def compute_norm(df, max_images, seed):
    sample=df.sample(min(max_images,len(df)),random_state=seed)
    sums=np.zeros(N_BANDS); sums2=np.zeros(N_BANDS); counts=np.zeros(N_BANDS,dtype=np.int64)
    for p in tqdm(sample.image_path,desc="Normalization"):
        with rasterio.open(p) as src:
            a=src.read(out_dtype="float32"); valid=np.isfinite(a)
            if src.nodata is not None: valid &= a != src.nodata
        for b in range(N_BANDS):
            v=a[b][valid[b]]
            if v.size:
                sums[b]+=v.sum(dtype=np.float64); sums2[b]+=(v.astype(np.float64)**2).sum(); counts[b]+=v.size
    means=sums/counts; var=np.maximum(sums2/counts-means**2,1e-12)
    return means.astype("float32"),np.sqrt(var).astype("float32")


class PartialDS(Dataset):
    def __init__(self,df,means,stds,augment=False):
        self.df=df.reset_index(drop=True); self.m=means[:,None,None]; self.s=np.maximum(stds,1e-6)[:,None,None]
        self.augment=augment
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        with rasterio.open(r.image_path) as src:
            x=src.read(out_dtype="float32"); valid=src.read_masks(1)>0
        x[:,~valid]=np.nan
        for b in range(N_BANDS):
            bad=~np.isfinite(x[b]); x[b,bad]=self.m[b,0,0]
        x=(x-self.m)/self.s
        with rasterio.open(r.mask_path) as src: y=src.read(1).astype(np.uint8)

        if self.augment:
            if random.random()<.5: x=np.flip(x,2).copy(); y=np.flip(y,1).copy()
            if random.random()<.5: x=np.flip(x,1).copy(); y=np.flip(y,0).copy()
            k=random.randint(0,3)
            if k: x=np.rot90(x,k,axes=(1,2)).copy(); y=np.rot90(y,k).copy()

        return torch.from_numpy(x.astype("float32")), torch.from_numpy(y.astype("int64")), i


def masked_loss(logits, mask):
    valid = mask != IGNORE
    if not valid.any():
        return logits.sum()*0.0, 0.0, 0.0
    y = mask.float()
    bce = F.binary_cross_entropy_with_logits(logits[valid], y[valid])

    p = torch.sigmoid(logits)
    pv = p[valid]; yv=y[valid]
    inter=(pv*yv).sum()
    dice=(2*inter+1.0)/(pv.sum()+yv.sum()+1.0)
    dl=1-dice
    return bce+dl, float(bce.detach().cpu()), float(dl.detach().cpu())


def run_epoch(model,loader,opt,device,training):
    model.train(training); losses=[]; bces=[]; dices=[]
    ctx=torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x,y,_ in loader:
            x=x.to(device); y=y.to(device)
            if training: opt.zero_grad(set_to_none=True)
            z=model(x); loss,b,d=masked_loss(z,y)
            if training: loss.backward(); opt.step()
            losses.append(float(loss.detach().cpu())); bces.append(b); dices.append(d)
    return float(np.mean(losses)), float(np.mean(bces)), float(np.mean(dices))


def pixel_eval(model,loader,device):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for x,y,_ in loader:
            x=x.to(device); z=model(x); p=torch.sigmoid(z).cpu().numpy(); y=y.numpy()
            valid=y!=IGNORE
            ys.extend(y[valid].astype(int).tolist()); ps.extend(p[valid].tolist())
    return np.asarray(ys),np.asarray(ps)


def metric(y,p,t=.5):
    pred=(p>=t).astype(int)
    o={
        "n_pixels":int(len(y)),"positive_pixels":int(y.sum()),"threshold":float(t),
        "accuracy":float(accuracy_score(y,pred)),
        "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
        "precision":float(precision_score(y,pred,zero_division=0)),
        "recall":float(recall_score(y,pred,zero_division=0)),
        "f1":float(f1_score(y,pred,zero_division=0)),
        "confusion_matrix":confusion_matrix(y,pred,labels=[0,1]).tolist(),
    }
    if len(np.unique(y))==2:
        o["roc_auc"]=float(roc_auc_score(y,p)); o["average_precision"]=float(average_precision_score(y,p))
    return o


def command_train(args):
    seed_all(args.seed)
    df=pd.read_csv(args.manifest)
    need={"image_path","mask_path","target","base_id","year"}
    if not need.issubset(df.columns): raise ValueError(f"Missing columns: {sorted(need-set(df.columns))}")
    if "split" not in df.columns:
        raise ValueError("Use a manifest carrying the train/validation/test split from the weak U-Net run.")

    s=df.split.astype(str).str.lower().replace({"val":"validation"})
    tr=df[s=="train"].copy(); va=df[s=="validation"].copy(); te=df[s=="test"].copy()
    if not len(tr) or not len(va) or not len(te): raise ValueError("Need train, validation, and test rows.")

    # Verify no base_id leakage.
    A,B,C=set(tr.base_id),set(va.base_id),set(te.base_id)
    if A&B or A&C or B&C: raise RuntimeError("base_id leakage detected.")

    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    tr.to_csv(out/"train_manifest.csv",index=False); va.to_csv(out/"validation_manifest.csv",index=False); te.to_csv(out/"test_manifest.csv",index=False)

    means,stds=compute_norm(tr,args.normalization_images,args.seed)
    (out/"normalization.json").write_text(json.dumps({"bands":BANDS,"means":means.tolist(),"stds":stds.tolist()},indent=2))

    dtr=PartialDS(tr,means,stds,True); dva=PartialDS(va,means,stds,False); dte=PartialDS(te,means,stds,False)
    sampler=None
    if args.balanced_sampler:
        counts=tr.target.value_counts().to_dict()
        w=tr.target.map(lambda z:1.0/counts[z]).values
        sampler=WeightedRandomSampler(w,len(w),replacement=True)
    def loader(ds,shuffle=False,sampler=None):
        return DataLoader(ds,batch_size=args.batch_size,shuffle=shuffle if sampler is None else False,sampler=sampler,num_workers=args.workers,pin_memory=torch.cuda.is_available())
    ltr=loader(dtr,True,sampler); lva=loader(dva); lte=loader(dte)

    device=torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Device:",device)
    model=UNet(base=args.base_channels).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=.5,patience=3)

    hist=[]; best=-math.inf; bad=0
    for epoch in range(1,args.epochs+1):
        tl,tb,td=run_epoch(model,ltr,opt,device,True)
        vl,vb,vd=run_epoch(model,lva,opt,device,False)
        vy,vp=pixel_eval(model,lva,device)
        vm=metric(vy,vp,.5)
        score=vm.get("average_precision",vm["f1"]); sched.step(score)
        hist.append({"epoch":epoch,"train_loss":tl,"val_loss":vl,"val_f1":vm["f1"],"val_AP":vm.get("average_precision"),"val_roc_auc":vm.get("roc_auc"),"lr":opt.param_groups[0]["lr"]})
        print(f"Epoch {epoch:03d} train_loss={tl:.4f} val_loss={vl:.4f} val_f1={vm['f1']:.4f} val_AP={vm.get('average_precision')}")
        if score>best+1e-6:
            best=score; bad=0
            torch.save({"state_dict":model.state_dict(),"means":means.tolist(),"stds":stds.tolist(),"base_channels":args.base_channels,"epoch":epoch,"script_version":VERSION},out/"best_model.pt")
        else: bad+=1
        if args.patience>0 and bad>=args.patience:
            print("Early stopping"); break

    pd.DataFrame(hist).to_csv(out/"training_history.csv",index=False)
    ck=load_ckpt(out/"best_model.pt",device); model.load_state_dict(ck["state_dict"])
    ty,tp=pixel_eval(model,lte,device)

    rep={
        "script_version":VERSION,
        "supervision":"partial_high_confidence_pseudolabels",
        "warning":"Evaluation is against pseudo-label pixels, not human field masks.",
        "device":str(device),
        "best_model_epoch":int(ck["epoch"]),
        "train_n":int(len(tr)),"validation_n":int(len(va)),"test_n":int(len(te)),
        "test_confident_pixel_metrics":metric(ty,tp,.5),
    }
    (out/"metrics.json").write_text(json.dumps(rep,indent=2))
    print(json.dumps(rep,indent=2))


def save_like(src_path,out_path,a,dtype,nodata=None):
    with rasterio.open(src_path) as src:
        prof=src.profile.copy(); prof.update(count=1,dtype=dtype,nodata=nodata,compress="deflate")
        with rasterio.open(out_path,"w",**prof) as dst: dst.write(a.astype(dtype),1)


def command_predict(args):
    ck=load_ckpt(args.model,"cpu"); device=torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model=UNet(base=int(ck.get("base_channels",32))).to(device); model.load_state_dict(ck["state_dict"]); model.eval()
    means=np.asarray(ck["means"],dtype="float32")[:,None,None]; stds=np.maximum(np.asarray(ck["stds"],dtype="float32"),1e-6)[:,None,None]
    df=pd.read_csv(args.manifest); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for _,r in tqdm(df.iterrows(),total=len(df),desc="Predicting"):
        with rasterio.open(r.image_path) as src:
            x=src.read(out_dtype="float32"); valid=src.read_masks(1)>0
        x[:,~valid]=np.nan
        for b in range(N_BANDS):
            bad=~np.isfinite(x[b]); x[b,bad]=means[b,0,0]
        x=(x-means)/stds
        with torch.no_grad(): p=torch.sigmoid(model(torch.from_numpy(x[None].astype("float32")).to(device)))[0].cpu().numpy()
        stem=Path(r.image_path).stem
        pp=out/f"{stem}_partial_unet_probability.tif"; mp=out/f"{stem}_partial_unet_mask.tif"
        save_like(r.image_path,pp,p,"float32",None); save_like(r.image_path,mp,(p>=args.threshold).astype(np.uint8),"uint8",255)
        rows.append({"image_path":r.image_path,"base_id":r.base_id,"year":int(r.year),"target":int(r.target),"mean_probability":float(p.mean()),"fraction_pixels_positive":float((p>=args.threshold).mean()),"probability_raster":str(pp),"mask_raster":str(mp)})
    pd.DataFrame(rows).to_csv(out/"prediction_summary.csv",index=False)
    print("Predictions:",out.resolve())


def parser():
    p=argparse.ArgumentParser(description="Partial-supervision U-Net from weak U-Net pseudo-labels.")
    p.add_argument("--version",action="version",version=VERSION)
    sp=p.add_subparsers(dest="command",required=True)

    b=sp.add_parser("build-pseudolabels")
    b.add_argument("--manifest",required=True)
    b.add_argument("--probability-dir",required=True)
    b.add_argument("--output-dir",required=True)
    b.add_argument("--positive-threshold",type=float,default=.80)
    b.add_argument("--negative-threshold",type=float,default=.10)
    b.set_defaults(func=command_build)

    t=sp.add_parser("train")
    t.add_argument("--manifest",required=True)
    t.add_argument("--output-dir",required=True)
    t.add_argument("--epochs",type=int,default=60)
    t.add_argument("--batch-size",type=int,default=16)
    t.add_argument("--workers",type=int,default=0)
    t.add_argument("--seed",type=int,default=42)
    t.add_argument("--learning-rate",type=float,default=1e-3)
    t.add_argument("--weight-decay",type=float,default=1e-4)
    t.add_argument("--base-channels",type=int,default=32)
    t.add_argument("--normalization-images",type=int,default=2000)
    t.add_argument("--balanced-sampler",action="store_true")
    t.add_argument("--patience",type=int,default=10)
    t.add_argument("--device",default=None)
    t.set_defaults(func=command_train)

    q=sp.add_parser("predict")
    q.add_argument("--model",required=True)
    q.add_argument("--manifest",required=True)
    q.add_argument("--output-dir",required=True)
    q.add_argument("--threshold",type=float,default=.5)
    q.add_argument("--device",default=None)
    q.set_defaults(func=command_predict)
    return p


if __name__=="__main__":
    a=parser().parse_args()
    print(f"Agave Partial U-Net script version: {VERSION}")
    a.func(a)