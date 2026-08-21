#!/usr/bin/env python
"""
agave_cnn_paired_test.py

Purpose
-------
Train a single-year 9-band CNN on the EXACT SAME target samples and EXACT SAME
train/validation/test base_id splits used by a completed temporal CNN run.

Temporal model input:
    t-2, t-1, t  -> target label at t

Paired single-year model input:
    t             -> SAME target label at t

This isolates the value of temporal context.

Expected temporal run directory files:
    train_sequences.csv
    validation_sequences.csv
    test_sequences.csv
    metrics.json

Outputs:
    best_model.pt
    normalization.json
    training_history.csv
    train_paired.csv
    validation_paired.csv
    test_paired.csv
    validation_predictions.csv
    test_predictions.csv
    metrics.json
    paired_comparison.json   (when temporal metrics.json is available)
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
except Exception as exc:
    raise SystemExit("PyTorch is required: pip install torch") from exc

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from tqdm import tqdm

VERSION = "1.0-paired-single-year-ablation"
EXPECTED_BANDS = 9
EXPECTED_SIZE = 64
BAND_NAMES = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12"]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(path, map_location="cpu"):
    # PyTorch 2.6 defaults weights_only=True. This is a local checkpoint created
    # by this script, so loading the metadata dictionary is intentional.
    return torch.load(path, map_location=map_location, weights_only=False)


def infer_final_image_column(frame: pd.DataFrame) -> str:
    cols = []
    for c in frame.columns:
        if c.startswith("image_path_"):
            try:
                cols.append((int(c.rsplit("_", 1)[1]), c))
            except ValueError:
                pass
    if not cols:
        raise ValueError(
            "No temporal image_path_N columns found. Expected columns such as "
            "image_path_0, image_path_1, image_path_2."
        )
    cols.sort()
    return cols[-1][1]


def convert_split(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    final_col = infer_final_image_column(frame)
    required = {"base_id", "target_year", "target", final_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{split_name}: missing required columns: {sorted(missing)}")

    out = pd.DataFrame({
        "image_path": frame[final_col].astype(str),
        "base_id": frame["base_id"].astype(str),
        "year": frame["target_year"].astype(int),
        "target": frame["target"].astype(int),
    })

    if "sequence_id" in frame.columns:
        out["sequence_id"] = frame["sequence_id"].astype(str)
    else:
        out["sequence_id"] = out["base_id"] + "_" + out["year"].astype(str)

    out["source_temporal_image_column"] = final_col
    out["split"] = split_name
    return out.reset_index(drop=True)


def validate_exact_splits(train, val, test):
    a, b, c = set(train.base_id), set(val.base_id), set(test.base_id)
    if a & b or a & c or b & c:
        raise RuntimeError("base_id leakage detected in temporal split files.")
    for name, frame in [("train", train), ("validation", val), ("test", test)]:
        dup = frame.duplicated(["sequence_id"])
        if dup.any():
            raise RuntimeError(f"Duplicate sequence_id values found in {name} split.")
        bad = ~frame.target.isin([0, 1])
        if bad.any():
            raise RuntimeError(f"Non-binary target values found in {name} split.")


def validate_raster(path: str):
    with rasterio.open(path) as src:
        if src.count != EXPECTED_BANDS:
            raise ValueError(f"{path}: expected {EXPECTED_BANDS} bands, got {src.count}")
        if src.width != EXPECTED_SIZE or src.height != EXPECTED_SIZE:
            raise ValueError(
                f"{path}: expected {EXPECTED_SIZE}x{EXPECTED_SIZE}, "
                f"got {src.width}x{src.height}"
            )


def compute_normalization(frame: pd.DataFrame, max_images: int, seed: int):
    if len(frame) > max_images:
        sample = frame.sample(max_images, random_state=seed)
    else:
        sample = frame

    sums = np.zeros(EXPECTED_BANDS, dtype=np.float64)
    sums2 = np.zeros(EXPECTED_BANDS, dtype=np.float64)
    counts = np.zeros(EXPECTED_BANDS, dtype=np.int64)

    for path in tqdm(sample.image_path, desc="Computing normalization"):
        with rasterio.open(path) as src:
            arr = src.read(out_dtype="float32")
            valid = np.isfinite(arr)
            if src.nodata is not None:
                valid &= arr != src.nodata
            for b in range(EXPECTED_BANDS):
                vals = arr[b][valid[b]]
                if vals.size:
                    sums[b] += vals.sum(dtype=np.float64)
                    sums2[b] += np.square(vals, dtype=np.float64).sum(dtype=np.float64)
                    counts[b] += vals.size

    if np.any(counts == 0):
        raise ValueError(f"No valid pixels for bands {np.where(counts == 0)[0].tolist()}")

    means = sums / counts
    var = np.maximum(sums2 / counts - means**2, 1e-12)
    return means.astype(np.float32), np.sqrt(var).astype(np.float32)


class PatchDataset(Dataset):
    def __init__(self, frame, means, stds, augment=False):
        self.frame = frame.reset_index(drop=True)
        self.means = np.asarray(means, dtype=np.float32)[:, None, None]
        self.stds = np.maximum(np.asarray(stds, dtype=np.float32), 1e-6)[:, None, None]
        self.augment = augment

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        with rasterio.open(row.image_path) as src:
            arr = src.read(out_dtype="float32")
            valid_mask = src.read_masks(1) > 0

        arr[:, ~valid_mask] = np.nan
        for b in range(arr.shape[0]):
            bad = ~np.isfinite(arr[b])
            if bad.any():
                arr[b, bad] = float(self.means[b, 0, 0])

        arr = (arr - self.means) / self.stds

        if self.augment:
            if random.random() < 0.5:
                arr = np.flip(arr, axis=2).copy()
            if random.random() < 0.5:
                arr = np.flip(arr, axis=1).copy()
            k = random.randint(0, 3)
            if k:
                arr = np.rot90(arr, k=k, axes=(1, 2)).copy()

        x = torch.from_numpy(arr.astype(np.float32))
        y = torch.tensor(float(row.target), dtype=torch.float32)
        return x, y, idx


class MultispectralCNN(nn.Module):
    """Same spatial CNN family used for the original single-year baseline."""
    def __init__(self, in_channels=9, dropout=0.30):
        super().__init__()
        self.features = nn.Sequential(
            self.block(in_channels, 32),
            nn.MaxPool2d(2),
            self.block(32, 64),
            nn.MaxPool2d(2),
            self.block(64, 128),
            nn.MaxPool2d(2),
            self.block(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    @staticmethod
    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)


def binary_metrics(y_true, probs, threshold):
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    pred = (probs >= threshold).astype(int)

    result = {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, probs))
        result["average_precision"] = float(average_precision_score(y_true, probs))
    else:
        result["roc_auc"] = None
        result["average_precision"] = None
    return result


def choose_threshold(y_true, probs):
    # Optimize F1 on validation only. Test data never selects the threshold.
    thresholds = np.arange(0.05, 0.951, 0.01)
    scores = [
        f1_score(y_true, (probs >= t).astype(int), zero_division=0)
        for t in thresholds
    ]
    best = int(np.argmax(scores))
    return float(thresholds[best])


def run_epoch(model, loader, criterion, optimizer, device, training):
    model.train(training)
    losses, ys, ps, ids = [], [], [], []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for x, y, idx in loader:
            x = x.to(device)
            y = y.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(logits, y)

            if training:
                loss.backward()
                optimizer.step()

            losses.append(float(loss.item()))
            ys.append(y.detach().cpu().numpy())
            ps.append(torch.sigmoid(logits).detach().cpu().numpy())
            ids.append(np.asarray(idx))

    return (
        float(np.mean(losses)),
        np.concatenate(ys),
        np.concatenate(ps),
        np.concatenate(ids),
    )


def make_loader(dataset, batch_size, workers, shuffle=False, sampler=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_temporal_splits(run_dir: Path):
    paths = {
        "train": run_dir / "train_sequences.csv",
        "validation": run_dir / "validation_sequences.csv",
        "test": run_dir / "test_sequences.csv",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing temporal split files:\n  " + "\n  ".join(missing)
        )

    train = convert_split(pd.read_csv(paths["train"]), "train")
    val = convert_split(pd.read_csv(paths["validation"]), "validation")
    test = convert_split(pd.read_csv(paths["test"]), "test")
    validate_exact_splits(train, val, test)
    return train, val, test


def compare_with_temporal(single_metrics, temporal_metrics):
    t = temporal_metrics.get("test_metrics") or temporal_metrics.get("test")
    s = single_metrics["test_metrics"]
    if not t:
        return {"warning": "Could not locate temporal test metrics."}

    keys = [
        "accuracy", "balanced_accuracy", "precision", "recall",
        "f1", "roc_auc", "average_precision"
    ]
    comparison = {}
    for key in keys:
        sv = s.get(key)
        tv = t.get(key)
        comparison[key] = {
            "paired_single_year": sv,
            "temporal_3yr": tv,
            "delta_single_minus_temporal": (
                None if sv is None or tv is None else float(sv - tv)
            ),
        }
    return comparison


def command_train(args):
    seed_everything(args.seed)
    temporal_dir = Path(args.temporal_run_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train, val, test = load_temporal_splits(temporal_dir)

    print(f"Paired split loaded from: {temporal_dir.resolve()}")
    print(f"Train:      {len(train)} samples / {train.base_id.nunique()} base_ids")
    print(f"Validation: {len(val)} samples / {val.base_id.nunique()} base_ids")
    print(f"Test:       {len(test)} samples / {test.base_id.nunique()} base_ids")
    print("Target-year counts (test):")
    print(test.year.value_counts().sort_index().to_string())

    if args.validate_rasters:
        all_paths = pd.concat([train, val, test]).image_path.unique()
        for p in tqdm(all_paths, desc="Validating final-year rasters"):
            validate_raster(p)

    train.to_csv(out / "train_paired.csv", index=False)
    val.to_csv(out / "validation_paired.csv", index=False)
    test.to_csv(out / "test_paired.csv", index=False)

    means, stds = compute_normalization(train, args.normalization_images, args.seed)
    (out / "normalization.json").write_text(
        json.dumps({
            "bands": BAND_NAMES,
            "means": means.tolist(),
            "stds": stds.tolist(),
            "normalization_source": "paired training final-year images only",
        }, indent=2),
        encoding="utf-8",
    )

    ds_train = PatchDataset(train, means, stds, augment=True)
    ds_val = PatchDataset(val, means, stds, augment=False)
    ds_test = PatchDataset(test, means, stds, augment=False)

    sampler = None
    if args.balanced_sampler:
        counts = train.target.value_counts().to_dict()
        weights = train.target.map(lambda z: 1.0 / counts[z]).values
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    ltrain = make_loader(ds_train, args.batch_size, args.workers, shuffle=True, sampler=sampler)
    lval = make_loader(ds_val, args.batch_size, args.workers)
    ltest = make_loader(ds_test, args.batch_size, args.workers)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    model = MultispectralCNN(dropout=args.dropout).to(device)

    positives = int(train.target.sum())
    negatives = len(train) - positives
    pos_weight = torch.tensor(
        [negatives / max(positives, 1)], dtype=torch.float32, device=device
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=None if args.balanced_sampler else pos_weight
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    history = []
    best_score = -math.inf
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _, _ = run_epoch(
            model, ltrain, criterion, optimizer, device, True
        )
        val_loss, vy, vp, _ = run_epoch(
            model, lval, criterion, optimizer, device, False
        )
        vm = binary_metrics(vy, vp, 0.5)
        score = (
            vm["average_precision"]
            if vm["average_precision"] is not None
            else vm["f1"]
        )
        scheduler.step(score)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_f1_at_0.5": vm["f1"],
            "val_roc_auc": vm["roc_auc"],
            "val_average_precision": vm["average_precision"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_f1={vm['f1']:.4f} "
            f"val_AP={vm['average_precision']}"
        )

        if score > best_score + 1e-6:
            best_score = score
            bad_epochs = 0
            torch.save({
                "state_dict": model.state_dict(),
                "means": means.tolist(),
                "stds": stds.tolist(),
                "dropout": args.dropout,
                "epoch": epoch,
                "best_validation_score": float(score),
                "script_version": VERSION,
            }, out / "best_model.pt")
        else:
            bad_epochs += 1

        # patience=0 means intentionally run every requested epoch.
        if args.patience > 0 and bad_epochs >= args.patience:
            print("Early stopping")
            break

    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)

    checkpoint = load_checkpoint(out / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    _, vy, vp, vidx = run_epoch(model, lval, criterion, optimizer, device, False)
    threshold = choose_threshold(vy, vp)
    val_pred = val.iloc[vidx].copy()
    val_pred["probability_agave"] = vp
    val_pred["prediction"] = (vp >= threshold).astype(int)
    val_pred.to_csv(out / "validation_predictions.csv", index=False)

    _, ty, tp, tidx = run_epoch(model, ltest, criterion, optimizer, device, False)
    test_pred = test.iloc[tidx].copy()
    test_pred["probability_agave"] = tp
    test_pred["prediction"] = (tp >= threshold).astype(int)
    test_pred.to_csv(out / "test_predictions.csv", index=False)

    per_year = {
        str(year): binary_metrics(
            group.target.values,
            group.probability_agave.values,
            threshold,
        )
        for year, group in test_pred.groupby("year")
    }

    report = {
        "script_version": VERSION,
        "experiment": "paired_single_year_vs_temporal_ablation",
        "temporal_run_dir": str(temporal_dir.resolve()),
        "device": str(device),
        "best_model_epoch": int(checkpoint.get("epoch", -1)),
        "epochs_requested": int(args.epochs),
        "train_n": int(len(train)),
        "validation_n": int(len(val)),
        "test_n": int(len(test)),
        "train_base_ids": int(train.base_id.nunique()),
        "validation_base_ids": int(val.base_id.nunique()),
        "test_base_ids": int(test.base_id.nunique()),
        "selected_threshold": float(threshold),
        "test_metrics": binary_metrics(ty, tp, threshold),
        "per_target_year_metrics": per_year,
    }

    (out / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    temporal_metrics_path = temporal_dir / "metrics.json"
    if temporal_metrics_path.exists():
        temporal_metrics = json.loads(temporal_metrics_path.read_text(encoding="utf-8"))
        comparison = {
            "experiment": "same samples and same geographic splits",
            "paired_single_year_metrics_path": str((out / "metrics.json").resolve()),
            "temporal_metrics_path": str(temporal_metrics_path.resolve()),
            "sample_counts_match": {
                "single_year_test_n": int(len(test)),
                "temporal_test_n": int(
                    temporal_metrics.get("test_metrics", temporal_metrics.get("test", {})).get("n", -1)
                ),
            },
            "overall_comparison": compare_with_temporal(report, temporal_metrics),
        }

        temporal_per_year = temporal_metrics.get("per_target_year_metrics", {})
        year_comparison = {}
        for year, sm in per_year.items():
            tm = temporal_per_year.get(year)
            if tm:
                year_comparison[year] = {}
                for key in ["f1", "precision", "recall", "roc_auc", "average_precision"]:
                    sv, tv = sm.get(key), tm.get(key)
                    year_comparison[year][key] = {
                        "paired_single_year": sv,
                        "temporal_3yr": tv,
                        "delta_single_minus_temporal": (
                            None if sv is None or tv is None else float(sv - tv)
                        ),
                    }
        comparison["per_year_comparison"] = year_comparison
        (out / "paired_comparison.json").write_text(
            json.dumps(comparison, indent=2), encoding="utf-8"
        )

    print(json.dumps(report, indent=2))
    print(f"Outputs written to: {out.resolve()}")


def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Paired single-year CNN ablation using the exact samples and "
            "base_id splits from a temporal CNN run."
        )
    )
    p.add_argument("--version", action="version", version=VERSION)
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser(
        "train",
        help="Train final-year-only CNN on the exact temporal train/val/test split.",
    )
    t.add_argument("--temporal-run-dir", required=True)
    t.add_argument("--output-dir", required=True)
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--workers", type=int, default=0)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--learning-rate", type=float, default=1e-3)
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--dropout", type=float, default=0.30)
    t.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early-stopping patience. 0 = run every requested epoch.",
    )
    t.add_argument("--normalization-images", type=int, default=2000)
    t.add_argument("--balanced-sampler", action="store_true")
    t.add_argument("--validate-rasters", action="store_true")
    t.add_argument("--device", default=None)
    t.set_defaults(func=command_train)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    print(f"Agave paired CNN script version: {VERSION}")
    args.func(args)