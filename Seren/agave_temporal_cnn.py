#!/usr/bin/env python
"""Three-year spatiotemporal CNN for 9-band, 64x64 Agave Sentinel-2 patches.

Uses an existing single-year CNN manifest with columns:
  image_path, image_name, image_id, base_id, year, target

Commands:
  build-sequences  Create chronological rolling sequences ending in a target year.
  train            Train shared spatial encoder + 1D temporal CNN.
  predict          Predict from an existing sequence manifest.

Default sequence: three consecutive years [t-2, t-1, t], target = label at t.
All sequences from one base_id remain in the same split.
"""
from __future__ import annotations

import argparse, json, math, random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import rasterio
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
except ImportError:
    torch = None
    nn = None
    Dataset = object

VERSION = "1.2-resume-training"
BANDS, HEIGHT, WIDTH = 9, 64, 64
BAND_NAMES = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12"]


def load_checkpoint(path, map_location):
    """Load a checkpoint created by this script.

    PyTorch 2.6 changed torch.load() to weights_only=True by default. Our
    checkpoint also stores normalization arrays and model configuration, so
    it must be loaded with weights_only=False. Only use this for checkpoints
    you created or otherwise trust.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Compatibility with older PyTorch releases that lack weights_only.
        return torch.load(path, map_location=map_location)


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def validate_image(path: str):
    with rasterio.open(path) as src:
        if src.count != BANDS or src.height != HEIGHT or src.width != WIDTH:
            raise ValueError(f"Expected 9x64x64, found {src.count}x{src.height}x{src.width}: {path}")


def command_build_sequences(args):
    src = pd.read_csv(args.manifest)
    required = {"image_path", "base_id", "year", "target"}
    missing = required - set(src.columns)
    if missing: raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    src = src[src.target.isin([0,1])].copy()
    src["year"] = src.year.astype(int); src["target"] = src.target.astype(int); src["base_id"] = src.base_id.astype(str)
    dup = src.duplicated(["base_id","year"], keep=False)
    if dup.any():
        examples = src.loc[dup, ["base_id","year","image_path"]].head(20).to_dict("records")
        raise ValueError(f"Duplicate base_id/year rows found. Examples: {examples}")

    rows, gap_counts = [], 0
    L = args.sequence_length
    for base_id, g in tqdm(src.groupby("base_id"), desc="Building temporal sequences"):
        by_year = {int(r.year): r for r in g.itertuples(index=False)}
        years = sorted(by_year)
        for target_year in years:
            seq_years = list(range(target_year - L + 1, target_year + 1))
            if not all(y in by_year for y in seq_years):
                gap_counts += 1; continue
            target_row = by_year[target_year]
            row = {"sequence_id": f"{base_id}_{target_year}_L{L}", "base_id": base_id,
                   "target_year": target_year, "target": int(target_row.target), "sequence_length": L}
            for i, year in enumerate(seq_years):
                r = by_year[year]
                row[f"year_{i}"] = year
                row[f"image_path_{i}"] = r.image_path
                if hasattr(r, "image_id"): row[f"image_id_{i}"] = r.image_id
            rows.append(row)
    out = pd.DataFrame(rows).sort_values(["base_id","target_year"]).reset_index(drop=True)
    if args.validate_rasters:
        errors=[]
        for i,r in tqdm(out.iterrows(), total=len(out), desc="Validating sequence rasters"):
            try:
                for j in range(L): validate_image(r[f"image_path_{j}"])
            except Exception as e: errors.append((i,str(e)))
        if errors:
            bad={i for i,_ in errors}; out=out.drop(index=list(bad)).reset_index(drop=True)
        else: errors=[]
    else: errors=[]
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv,index=False)
    transitions = int((src.groupby("base_id").target.nunique()>1).sum())
    report={"script_version":VERSION,"source_rows":len(src),"sequence_length":L,"sequences_created":len(out),
            "unique_base_ids":int(out.base_id.nunique()) if len(out) else 0,
            "target_counts":out.target.value_counts().sort_index().to_dict() if len(out) else {},
            "target_year_counts":out.target_year.value_counts().sort_index().to_dict() if len(out) else {},
            "candidate_windows_skipped_for_missing_years":gap_counts,
            "base_ids_with_label_transitions":transitions,"invalid_sequences":len(errors),
            "invalid_examples":errors[:20]}
    Path(args.output_csv).with_suffix(".report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2)); print(f"Sequence manifest written to: {Path(args.output_csv).resolve()}")


def compute_normalization(frame, L, max_sequences, seed):
    sample=frame.sample(min(max_sequences,len(frame)),random_state=seed) if len(frame)>max_sequences else frame
    sums=np.zeros(BANDS); sums2=np.zeros(BANDS); counts=np.zeros(BANDS,dtype=np.int64)
    for _,r in tqdm(sample.iterrows(),total=len(sample),desc="Computing normalization"):
        for j in range(L):
            with rasterio.open(r[f"image_path_{j}"]) as src:
                a=src.read(out_dtype="float32"); valid=np.isfinite(a)
                if src.nodata is not None: valid &= a != src.nodata
                for b in range(BANDS):
                    v=a[b][valid[b]]
                    if v.size: sums[b]+=v.sum(dtype=np.float64); sums2[b]+=(v.astype(np.float64)**2).sum(); counts[b]+=v.size
    means=sums/counts; var=np.maximum(sums2/counts-means**2,1e-12)
    return means.astype("float32"),np.sqrt(var).astype("float32")


class TemporalDataset(Dataset):
    def __init__(self, frame, means, stds, augment=False):
        self.f=frame.reset_index(drop=True); self.L=int(frame.sequence_length.iloc[0]); self.m=means[:,None,None]; self.s=np.maximum(stds,1e-6)[:,None,None]; self.augment=augment
    def __len__(self): return len(self.f)
    def __getitem__(self,idx):
        r=self.f.iloc[idx]; seq=[]
        for j in range(self.L):
            with rasterio.open(r[f"image_path_{j}"]) as src:
                a=src.read(out_dtype="float32"); mask=src.read_masks(1)>0
            a[:,~mask]=np.nan
            for b in range(BANDS): a[b][~np.isfinite(a[b])]=float(self.m[b,0,0])
            seq.append((a-self.m)/self.s)
        x=np.stack(seq).astype("float32")
        if self.augment:
            # identical spatial transform across all years preserves alignment
            if random.random()<.5: x=np.flip(x,axis=3).copy()
            if random.random()<.5: x=np.flip(x,axis=2).copy()
            k=random.randint(0,3)
            if k: x=np.rot90(x,k,axes=(2,3)).copy()
        return torch.from_numpy(x), torch.tensor(float(r.target),dtype=torch.float32), idx


class SpatialEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        def block(ci,co): return nn.Sequential(nn.Conv2d(ci,co,3,padding=1,bias=False),nn.BatchNorm2d(co),nn.ReLU(),nn.Conv2d(co,co,3,padding=1,bias=False),nn.BatchNorm2d(co),nn.ReLU())
        self.net=nn.Sequential(block(9,32),nn.MaxPool2d(2),block(32,64),nn.MaxPool2d(2),block(64,128),nn.MaxPool2d(2),block(128,256),nn.AdaptiveAvgPool2d(1))
        self.proj=nn.Linear(256,out_dim)
    def forward(self,x): return self.proj(self.net(x).flatten(1))


class TemporalCNN(nn.Module):
    def __init__(self, embed_dim=128, temporal_channels=128, dropout=.3):
        super().__init__(); self.encoder=SpatialEncoder(embed_dim)
        self.temporal=nn.Sequential(nn.Conv1d(embed_dim,temporal_channels,3,padding=1,bias=False),nn.BatchNorm1d(temporal_channels),nn.ReLU(),nn.Conv1d(temporal_channels,temporal_channels,3,padding=1,bias=False),nn.BatchNorm1d(temporal_channels),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(temporal_channels,64),nn.ReLU(),nn.Dropout(dropout),nn.Linear(64,1))
    def forward(self,x):
        B,T,C,H,W=x.shape; z=self.encoder(x.reshape(B*T,C,H,W)).reshape(B,T,-1).transpose(1,2); return self.head(self.temporal(z)).squeeze(1)


def split_grouped(f,seed,val_fraction,test_fraction):
    s1=GroupShuffleSplit(1,test_size=test_fraction,random_state=seed); tv_i,te_i=next(s1.split(f,f.target,f.base_id)); tv=f.iloc[tv_i].reset_index(drop=True); te=f.iloc[te_i].reset_index(drop=True)
    s2=GroupShuffleSplit(1,test_size=val_fraction/(1-test_fraction),random_state=seed+1); tr_i,va_i=next(s2.split(tv,tv.target,tv.base_id)); return tv.iloc[tr_i].reset_index(drop=True),tv.iloc[va_i].reset_index(drop=True),te


def split_future(f,test_year,val_year):
    te=f[f.target_year==test_year].copy(); va=f[f.target_year==val_year].copy(); hold=set(te.base_id)|set(va.base_id); tr=f[(f.target_year<val_year)&(~f.base_id.isin(hold))].copy()
    if min(len(tr),len(va),len(te))==0: raise ValueError("Empty train/validation/test split in future-year mode")
    return tr.reset_index(drop=True),va.reset_index(drop=True),te.reset_index(drop=True)


def metrics(y,p,t=.5):
    pr=(p>=t).astype(int); o={"n":len(y),"positives":int(y.sum()),"threshold":float(t),"accuracy":accuracy_score(y,pr),"balanced_accuracy":balanced_accuracy_score(y,pr),"precision":precision_score(y,pr,zero_division=0),"recall":recall_score(y,pr,zero_division=0),"f1":f1_score(y,pr,zero_division=0),"confusion_matrix":confusion_matrix(y,pr,labels=[0,1]).tolist()}
    o.update({"roc_auc":roc_auc_score(y,p) if len(np.unique(y))==2 else None,"average_precision":average_precision_score(y,p) if len(np.unique(y))==2 else None}); return {k:(float(v) if isinstance(v,(np.floating,float)) else v) for k,v in o.items()}


def choose_threshold(y,p):
    ts=np.linspace(.05,.95,91); return float(ts[np.argmax([f1_score(y,p>=t,zero_division=0) for t in ts])])


def run_epoch(model,loader,criterion,opt,device,train):
    model.train(train); losses=[]; ys=[]; ps=[]; ids=[]
    with (torch.enable_grad() if train else torch.no_grad()):
        for x,y,idx in loader:
            x=x.to(device); y=y.to(device)
            if train: opt.zero_grad(set_to_none=True)
            logit=model(x); loss=criterion(logit,y)
            if train: loss.backward(); opt.step()
            losses.append(loss.item()); ys.append(y.cpu().numpy()); ps.append(torch.sigmoid(logit).detach().cpu().numpy()); ids.append(np.asarray(idx))
    return float(np.mean(losses)),np.concatenate(ys),np.concatenate(ps),np.concatenate(ids)


def command_train(args):
    if torch is None: raise SystemExit("PyTorch required: pip install torch")
    seed_everything(args.seed); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    f=pd.read_csv(args.sequence_manifest); f=f[f.target.isin([0,1])].copy(); f.target=f.target.astype(int); f.target_year=f.target_year.astype(int); f.base_id=f.base_id.astype(str)
    if args.test_year is not None:
        val_year=args.val_year if args.val_year is not None else args.test_year-1; tr,va,te=split_future(f,args.test_year,val_year); split_mode=f"future_year_{val_year}_{args.test_year}"
    else: tr,va,te=split_grouped(f,args.seed,args.val_fraction,args.test_fraction); split_mode="grouped_location"
    if set(tr.base_id)&set(va.base_id) or set(tr.base_id)&set(te.base_id) or set(va.base_id)&set(te.base_id): raise RuntimeError("base_id leakage")
    for name,df in [("train",tr),("validation",va),("test",te)]: df.assign(split=name).to_csv(out/f"{name}_sequences.csv",index=False)
    means,stds=compute_normalization(tr,int(f.sequence_length.iloc[0]),args.normalization_sequences,args.seed)
    (out/"normalization.json").write_text(json.dumps({"bands":BAND_NAMES,"means":means.tolist(),"stds":stds.tolist()},indent=2))
    ds_tr=TemporalDataset(tr,means,stds,True); ds_va=TemporalDataset(va,means,stds); ds_te=TemporalDataset(te,means,stds)
    sampler=None
    if args.balanced_sampler:
        counts=tr.target.value_counts().to_dict(); w=tr.target.map(lambda z:1/counts[z]).values; sampler=WeightedRandomSampler(w,len(w),replacement=True)
    loaders=lambda ds,shuffle=False,sampler=None: DataLoader(ds,batch_size=args.batch_size,shuffle=shuffle if sampler is None else False,sampler=sampler,num_workers=args.workers,pin_memory=torch.cuda.is_available())
    ltr=loaders(ds_tr,True,sampler); lva=loaders(ds_va); lte=loaders(ds_te)
    device=torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")); model=TemporalCNN(args.embed_dim,args.temporal_channels,args.dropout).to(device)
    pos=tr.target.sum(); neg=len(tr)-pos; pw=torch.tensor([neg/max(pos,1)],device=device); criterion=nn.BCEWithLogitsLoss(pos_weight=None if args.balanced_sampler else pw)
    opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay); sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=.5,patience=3)
    best=-1; bad=0; history=[]
    for epoch in range(1,args.epochs+1):
        tl,ty,tp,_=run_epoch(model,ltr,criterion,opt,device,True); vl,vy,vp,_=run_epoch(model,lva,criterion,opt,device,False); vm=metrics(vy,vp,.5); score=vm["average_precision"] if vm["average_precision"] is not None else vm["f1"]; sched.step(score)
        history.append({"epoch":epoch,"train_loss":tl,"val_loss":vl,"val_f1":vm["f1"],"val_average_precision":vm["average_precision"],"lr":opt.param_groups[0]["lr"]})
        print(f"Epoch {epoch:03d} train_loss={tl:.4f} val_loss={vl:.4f} val_f1={vm['f1']:.4f} val_AP={vm['average_precision']}")
        if score>best+1e-6:
            best=score; bad=0; torch.save({"state_dict":model.state_dict(),"means":means.tolist(),"stds":stds.tolist(),"sequence_length":int(f.sequence_length.iloc[0]),"embed_dim":args.embed_dim,"temporal_channels":args.temporal_channels,"dropout":args.dropout,"script_version":VERSION},out/"best_model.pt")
        else:
            bad+=1
            if bad>=args.patience: print("Early stopping"); break
    pd.DataFrame(history).to_csv(out/"training_history.csv",index=False)
    ck=load_checkpoint(out/"best_model.pt",map_location=device); model.load_state_dict(ck["state_dict"])
    _,vy,vp,_=run_epoch(model,lva,criterion,opt,device,False); threshold=choose_threshold(vy,vp)
    _,y,p,idx=run_epoch(model,lte,criterion,opt,device,False); pred=te.iloc[idx].copy(); pred["probability_agave"]=p; pred["prediction"]=(p>=threshold).astype(int); pred.to_csv(out/"test_predictions.csv",index=False)
    per_year={str(year):metrics(g.target.values,g.probability_agave.values,threshold) for year,g in pred.groupby("target_year")}
    report={"script_version":VERSION,"split_mode":split_mode,"device":str(device),"sequence_length":int(f.sequence_length.iloc[0]),"train_n":len(tr),"validation_n":len(va),"test_n":len(te),"train_base_ids":tr.base_id.nunique(),"validation_base_ids":va.base_id.nunique(),"test_base_ids":te.base_id.nunique(),"selected_threshold":threshold,"test_metrics":metrics(y,p,threshold),"per_target_year_metrics":per_year}
    (out/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2)); print(f"Outputs written to: {out.resolve()}")




def command_resume(args):
    """Continue a prior run through a requested total epoch count.

    The model resumes from the run's saved best_model.pt. Existing geographic
    splits and normalization are reused. Optimizer/scheduler state was not
    stored by v1.0/v1.1, so they are reinitialized while model weights are
    preserved. By default, all remaining epochs run; set --patience above 0
    only if renewed early stopping is desired.
    """
    if torch is None:
        raise SystemExit("PyTorch required")

    out = Path(args.output_dir)
    required = {
        "checkpoint": out / "best_model.pt",
        "train": out / "train_sequences.csv",
        "validation": out / "validation_sequences.csv",
        "test": out / "test_sequences.csv",
    }
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"Required {name} file not found: {path}")

    seed_everything(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    ck = load_checkpoint(required["checkpoint"], map_location=device)
    means = np.asarray(ck["means"], dtype="float32")
    stds = np.asarray(ck["stds"], dtype="float32")

    tr = pd.read_csv(required["train"])
    va = pd.read_csv(required["validation"])
    te = pd.read_csv(required["test"])
    for frame in (tr, va, te):
        frame["target"] = frame.target.astype(int)
        frame["target_year"] = frame.target_year.astype(int)
        frame["base_id"] = frame.base_id.astype(str)

    history_path = out / "training_history.csv"
    if history_path.exists() and history_path.stat().st_size > 0:
        history_df = pd.read_csv(history_path)
        history = history_df.to_dict("records")
        completed_epoch = int(history_df["epoch"].max()) if len(history_df) else 0
        score_series = history_df["val_average_precision"].where(
            history_df["val_average_precision"].notna(), history_df["val_f1"]
        )
        best_score = float(score_series.max()) if len(score_series) else -1.0
    else:
        history = []
        completed_epoch = int(ck.get("completed_epoch", 0))
        best_score = -1.0

    if args.total_epochs <= completed_epoch:
        raise ValueError(
            f"Run already contains {completed_epoch} completed epochs; "
            f"--total-epochs must be greater than that."
        )

    ds_tr = TemporalDataset(tr, means, stds, augment=True)
    ds_va = TemporalDataset(va, means, stds)
    ds_te = TemporalDataset(te, means, stds)

    sampler = None
    if args.balanced_sampler:
        counts = tr.target.value_counts().to_dict()
        weights = tr.target.map(lambda z: 1 / counts[z]).values
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    def make_loader(ds, shuffle=False, sampler=None):
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
        )

    ltr = make_loader(ds_tr, shuffle=True, sampler=sampler)
    lva = make_loader(ds_va)
    lte = make_loader(ds_te)

    model = TemporalCNN(ck["embed_dim"], ck["temporal_channels"], ck["dropout"]).to(device)
    model.load_state_dict(ck["state_dict"])

    pos = int(tr.target.sum())
    neg = len(tr) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=None if args.balanced_sampler else pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.scheduler_patience
    )

    bad_epochs = 0
    latest_path = out / "last_model.pt"
    print(
        f"Resuming from saved best weights after epoch {completed_epoch}; "
        f"running epochs {completed_epoch + 1} through {args.total_epochs}."
    )
    print("Optimizer and scheduler are reinitialized because older checkpoints did not store their state.")

    for epoch in range(completed_epoch + 1, args.total_epochs + 1):
        train_loss, _, _, _ = run_epoch(model, ltr, criterion, optimizer, device, True)
        val_loss, val_y, val_p, _ = run_epoch(model, lva, criterion, None, device, False)
        val_metrics = metrics(val_y, val_p, 0.5)
        score = (
            val_metrics["average_precision"]
            if val_metrics["average_precision"] is not None
            else val_metrics["f1"]
        )
        scheduler.step(score)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_f1": val_metrics["f1"],
            "val_average_precision": val_metrics["average_precision"],
            "lr": optimizer.param_groups[0]["lr"],
            "resumed": True,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)

        latest_ck = {
            "state_dict": model.state_dict(),
            "means": means.tolist(),
            "stds": stds.tolist(),
            "sequence_length": int(ck["sequence_length"]),
            "embed_dim": int(ck["embed_dim"]),
            "temporal_channels": int(ck["temporal_channels"]),
            "dropout": float(ck["dropout"]),
            "completed_epoch": epoch,
            "script_version": VERSION,
        }
        torch.save(latest_ck, latest_path)

        print(
            f"Epoch {epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_AP={val_metrics['average_precision']}"
        )

        if score > best_score + 1e-6:
            best_score = score
            bad_epochs = 0
            torch.save(latest_ck, required["checkpoint"])
        else:
            bad_epochs += 1
            if args.patience > 0 and bad_epochs >= args.patience:
                print("Early stopping during resumed training")
                break

    # Evaluate the best checkpoint after resumed training.
    best_ck = load_checkpoint(required["checkpoint"], map_location=device)
    model.load_state_dict(best_ck["state_dict"])
    _, val_y, val_p, _ = run_epoch(model, lva, criterion, None, device, False)
    threshold = choose_threshold(val_y, val_p)
    _, test_y, test_p, test_idx = run_epoch(model, lte, criterion, None, device, False)
    predictions = te.iloc[test_idx].copy()
    predictions["probability_agave"] = test_p
    predictions["prediction"] = (test_p >= threshold).astype(int)
    predictions.to_csv(out / "test_predictions.csv", index=False)

    per_year = {
        str(year): metrics(group.target.values, group.probability_agave.values, threshold)
        for year, group in predictions.groupby("target_year")
    }
    report = {
        "script_version": VERSION,
        "resumed_training": True,
        "device": str(device),
        "requested_total_epochs": int(args.total_epochs),
        "epochs_recorded": int(pd.DataFrame(history)["epoch"].max()),
        "sequence_length": int(best_ck["sequence_length"]),
        "train_n": len(tr),
        "validation_n": len(va),
        "test_n": len(te),
        "selected_threshold": threshold,
        "test_metrics": metrics(test_y, test_p, threshold),
        "per_target_year_metrics": per_year,
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Resumed training outputs written to: {out.resolve()}")

def command_finalize(args):
    """Finish validation/test evaluation from a completed training run.

    Use this when training reached early stopping but the old PyTorch 2.6
    checkpoint-loading behavior caused a crash before metrics were written.
    """
    if torch is None:
        raise SystemExit("PyTorch required")
    out = Path(args.output_dir)
    model_path = out / "best_model.pt"
    val_path = out / "validation_sequences.csv"
    test_path = out / "test_sequences.csv"
    for path in (model_path, val_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Required run file not found: {path}")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    ck = load_checkpoint(model_path, map_location=device)
    means = np.asarray(ck["means"], dtype="float32")
    stds = np.asarray(ck["stds"], dtype="float32")
    va = pd.read_csv(val_path)
    te = pd.read_csv(test_path)
    va["target"] = va.target.astype(int); te["target"] = te.target.astype(int)
    va["target_year"] = va.target_year.astype(int); te["target_year"] = te.target_year.astype(int)

    ds_va = TemporalDataset(va, means, stds)
    ds_te = TemporalDataset(te, means, stds)
    loader = lambda ds: DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=torch.cuda.is_available())
    lva, lte = loader(ds_va), loader(ds_te)

    model = TemporalCNN(ck["embed_dim"], ck["temporal_channels"], ck["dropout"]).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    criterion = nn.BCEWithLogitsLoss()
    _, vy, vp, _ = run_epoch(model, lva, criterion, None, device, False)
    threshold = choose_threshold(vy, vp)
    _, y, p, idx = run_epoch(model, lte, criterion, None, device, False)
    pred = te.iloc[idx].copy()
    pred["probability_agave"] = p
    pred["prediction"] = (p >= threshold).astype(int)
    pred.to_csv(out / "test_predictions.csv", index=False)
    per_year = {str(year): metrics(g.target.values, g.probability_agave.values, threshold) for year, g in pred.groupby("target_year")}
    report = {
        "script_version": VERSION,
        "recovered_from_existing_checkpoint": True,
        "device": str(device),
        "sequence_length": int(ck["sequence_length"]),
        "validation_n": len(va),
        "test_n": len(te),
        "selected_threshold": threshold,
        "test_metrics": metrics(y, p, threshold),
        "per_target_year_metrics": per_year,
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Recovered evaluation outputs written to: {out.resolve()}")


def command_predict(args):
    if torch is None: raise SystemExit("PyTorch required")
    ck=load_checkpoint(args.model,map_location="cpu"); f=pd.read_csv(args.sequence_manifest); means=np.asarray(ck["means"],dtype="float32"); stds=np.asarray(ck["stds"],dtype="float32")
    ds=TemporalDataset(f,means,stds); loader=DataLoader(ds,batch_size=args.batch_size,num_workers=args.workers); device=torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")); model=TemporalCNN(ck["embed_dim"],ck["temporal_channels"],ck["dropout"]).to(device); model.load_state_dict(ck["state_dict"]); model.eval()
    ps=[]; ids=[]
    with torch.no_grad():
        for x,_,idx in loader: ps.append(torch.sigmoid(model(x.to(device))).cpu().numpy()); ids.append(np.asarray(idx))
    ids=np.concatenate(ids); ps=np.concatenate(ps); out=f.iloc[ids].copy(); out["probability_agave"]=ps; out.to_csv(args.output_csv,index=False); print(f"Predictions written to: {Path(args.output_csv).resolve()}")


def parser():
    p=argparse.ArgumentParser(description="Three-year spatiotemporal CNN for Agave Sentinel-2 patches.")
    sub=p.add_subparsers(dest="command",required=True)
    b=sub.add_parser("build-sequences"); b.add_argument("--manifest",required=True); b.add_argument("--output-csv",required=True); b.add_argument("--sequence-length",type=int,default=3); b.add_argument("--validate-rasters",action="store_true"); b.set_defaults(func=command_build_sequences)
    t=sub.add_parser("train"); t.add_argument("--sequence-manifest",required=True); t.add_argument("--output-dir",required=True); t.add_argument("--epochs",type=int,default=60); t.add_argument("--batch-size",type=int,default=8); t.add_argument("--workers",type=int,default=0); t.add_argument("--learning-rate",type=float,default=1e-3); t.add_argument("--weight-decay",type=float,default=1e-4); t.add_argument("--dropout",type=float,default=.3); t.add_argument("--embed-dim",type=int,default=128); t.add_argument("--temporal-channels",type=int,default=128); t.add_argument("--patience",type=int,default=10); t.add_argument("--seed",type=int,default=42); t.add_argument("--val-fraction",type=float,default=.15); t.add_argument("--test-fraction",type=float,default=.15); t.add_argument("--test-year",type=int); t.add_argument("--val-year",type=int); t.add_argument("--normalization-sequences",type=int,default=500); t.add_argument("--balanced-sampler",action="store_true"); t.add_argument("--device"); t.set_defaults(func=command_train)
    r=sub.add_parser("resume", help="Continue an existing run through a requested total epoch count"); r.add_argument("--output-dir",required=True); r.add_argument("--total-epochs",type=int,default=60); r.add_argument("--batch-size",type=int,default=8); r.add_argument("--workers",type=int,default=0); r.add_argument("--learning-rate",type=float,default=1e-4); r.add_argument("--weight-decay",type=float,default=1e-4); r.add_argument("--scheduler-patience",type=int,default=3); r.add_argument("--patience",type=int,default=0,help="0 disables early stopping so all remaining epochs run"); r.add_argument("--balanced-sampler",action="store_true"); r.add_argument("--seed",type=int,default=42); r.add_argument("--device"); r.set_defaults(func=command_resume)
    e=sub.add_parser("finalize", help="Finish evaluation from an existing training output directory"); e.add_argument("--output-dir",required=True); e.add_argument("--batch-size",type=int,default=8); e.add_argument("--workers",type=int,default=0); e.add_argument("--device"); e.set_defaults(func=command_finalize)
    q=sub.add_parser("predict"); q.add_argument("--model",required=True); q.add_argument("--sequence-manifest",required=True); q.add_argument("--output-csv",required=True); q.add_argument("--batch-size",type=int,default=8); q.add_argument("--workers",type=int,default=0); q.add_argument("--device"); q.set_defaults(func=command_predict)
    return p

if __name__=="__main__":
    print(f"Agave Temporal CNN script version: {VERSION}")
    a=parser().parse_args(); a.func(a)