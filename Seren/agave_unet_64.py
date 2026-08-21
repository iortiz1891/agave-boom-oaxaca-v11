#!/usr/bin/env python
"""Train and apply a baseline multispectral U-Net to 64x64 Agave patches.

Designed for files such as:
    INT_0002_2025_sentinel2_dry_median_64_snapped.tif

Expected imagery:
    - 9 float32 bands
    - 64 x 64 pixels
    - band order: B2, B3, B4, B5, B6, B7, B8, B11, B12

A true U-Net requires a pixel mask for every training image. Masks should be
single-band rasters aligned to the image, with 1 = cultivated agave and
0 = non-agave. Use the ``inventory`` command before training.

Examples (PowerShell):
    python agave_unet_64.py inventory --image-dir "C:\\data\\agave_sentinel2_full_64_dry_median"

    python agave_unet_64.py build-manifest `
        --image-dir "C:\\data\\agave_sentinel2_full_64_dry_median" `
        --mask-dir "C:\\data\\agave_masks_64" `
        --output-csv "C:\\data\\agave_unet_manifest.csv"

    python agave_unet_64.py train `
        --manifest "C:\\data\\agave_unet_manifest.csv" `
        --output-dir "C:\\data\\agave_unet_run_01" `
        --epochs 100 --batch-size 16

    python agave_unet_64.py predict `
        --model "C:\\data\\agave_unet_run_01\\best_model.pt" `
        --image-dir "C:\\data\\agave_sentinel2_full_64_dry_median" `
        --output-dir "C:\\data\\agave_unet_predictions"
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.features import rasterize

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required. Install dependencies with:\n"
        "  pip install torch rasterio numpy pandas scikit-learn tqdm"
    ) from exc

try:
    import pandas as pd
    from sklearn.model_selection import GroupShuffleSplit
    from tqdm.auto import tqdm
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install with:\n"
        "  pip install pandas scikit-learn tqdm"
    ) from exc


SCRIPT_VERSION = "5.0-mask-builder"
EXPECTED_BANDS = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12")
FILENAME_RE = re.compile(
    r"^(?P<location_id>INT_.+?)_(?P<year>\d{4})(?:_(?P<year_repeat>\d{4}))?_sentinel2_dry_median_64_snapped$",
    flags=re.IGNORECASE,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_identity(path: Path) -> tuple[str, int]:
    """Extract location and imagery year from either supported export name.

    Supported forms include:
      INT_0002_2025_sentinel2_dry_median_64_snapped.tif
      INT_1778232235_2025_2025_sentinel2_dry_median_64_snapped.tif
      INT_1778262218_93dd2c_2025_2025_sentinel2_dry_median_64_snapped.tif

    Everything between ``INT_`` and the final year/year-pair is retained as
    the spatial grouping ID. This supports numeric IDs and IDs with suffixes
    while keeping every year from the same location in one data split.
    """
    match = FILENAME_RE.match(path.stem)
    if not match:
        raise ValueError(
            f"Unexpected filename: {path.name}. Expected either "
            "INT_<location-key>_YYYY_sentinel2_dry_median_64_snapped.tif or "
            "INT_<location-key>_YYYY_YYYY_sentinel2_dry_median_64_snapped.tif. "
            "The location key may contain digits, letters, and underscores."
        )

    year = int(match.group("year"))
    repeated = match.group("year_repeat")
    if repeated is not None and int(repeated) != year:
        raise ValueError(
            f"Conflicting years in filename {path.name}: {year} and {repeated}"
        )
    return match.group("location_id").upper(), year


def list_tifs(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    return sorted([*folder.rglob("*.tif"), *folder.rglob("*.tiff")])


def inspect_raster(path: Path) -> dict:
    with rasterio.open(path) as src:
        descriptions = tuple(d or "" for d in src.descriptions)
        return {
            "path": str(path.resolve()),
            "bands": src.count,
            "height": src.height,
            "width": src.width,
            "dtype": src.dtypes[0],
            "crs": str(src.crs),
            "nodata": src.nodata,
            "band_descriptions": descriptions,
            "transform": tuple(src.transform),
        }


def validate_image(path: Path) -> list[str]:
    problems: list[str] = []
    try:
        info = inspect_raster(path)
    except RasterioIOError as exc:
        return [f"cannot open raster: {exc}"]

    if info["bands"] != 9:
        problems.append(f"expected 9 bands, found {info['bands']}")
    if (info["height"], info["width"]) != (64, 64):
        problems.append(
            f"expected 64x64 pixels, found {info['height']}x{info['width']}"
        )
    descriptions = tuple(info["band_descriptions"])
    if any(descriptions) and descriptions != EXPECTED_BANDS:
        problems.append(
            f"unexpected band order {descriptions}; expected {EXPECTED_BANDS}"
        )
    try:
        parse_identity(path)
    except ValueError as exc:
        problems.append(str(exc))
    return problems


def validate_pair(image_path: Path, mask_path: Path) -> list[str]:
    problems = validate_image(image_path)
    try:
        with rasterio.open(image_path) as image, rasterio.open(mask_path) as mask:
            if mask.count != 1:
                problems.append(f"mask must have 1 band, found {mask.count}")
            if (mask.height, mask.width) != (image.height, image.width):
                problems.append("mask dimensions do not match image")
            if mask.crs != image.crs:
                problems.append("mask CRS does not match image")
            if not mask.transform.almost_equals(image.transform):
                problems.append("mask transform/grid does not match image")
            values = np.unique(mask.read(1, masked=True).compressed())
            invalid = values[~np.isin(values, [0, 1, 255])]
            if invalid.size:
                problems.append(
                    f"mask contains values other than 0, 1, or optional 255 ignore: "
                    f"{invalid[:10].tolist()}"
                )
    except RasterioIOError as exc:
        problems.append(f"cannot open image/mask pair: {exc}")
    return problems


def candidate_mask_paths(image_path: Path, mask_dir: Path) -> Iterable[Path]:
    location_id, year = parse_identity(image_path)
    names = [
        image_path.name,
        f"{image_path.stem}_mask.tif",
        f"{location_id}_{year}_mask.tif",
        f"{location_id}_{year}_{year}_mask.tif",
        f"{location_id}_{year}_agave_mask.tif",
        f"{location_id}_{year}_{year}_agave_mask.tif",
        f"{location_id}_{year}_mask_64.tif",
        f"{location_id}_{year}_{year}_mask_64.tif",
    ]
    for name in names:
        yield mask_dir / name



def _load_vector_labels(label_file: Path, layer: Optional[str] = None):
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise SystemExit(
            "Mask creation requires GeoPandas. Install it with:\n"
            "  pip install geopandas pyogrio shapely"
        ) from exc
    kwargs = {} if layer is None else {"layer": layer}
    gdf = gpd.read_file(label_file, **kwargs)
    if gdf.empty:
        raise ValueError(f"No features found in label file: {label_file}")
    if gdf.crs is None:
        raise ValueError(f"Label file has no CRS: {label_file}")
    return gdf


def _infer_label_field(columns: Iterable[str]) -> Optional[str]:
    lookup = {str(c).lower(): str(c) for c in columns}
    candidates = (
        "label", "class", "target", "is_agave", "agave", "presence",
        "crop", "classification", "class_id", "value", "y"
    )
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def _is_positive_label(value, positive_values: set[str]) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    normalized = str(value).strip().lower()
    return normalized in positive_values


def command_inspect_labels(args: argparse.Namespace) -> None:
    label_file = Path(args.label_file)
    gdf = _load_vector_labels(label_file, args.layer)
    print(f"Label file: {label_file.resolve()}")
    print(f"Features: {len(gdf):,}")
    print(f"CRS: {gdf.crs}")
    print(f"Geometry types: {gdf.geometry.geom_type.value_counts().to_dict()}")
    print("Columns:")
    for column in gdf.columns:
        if column == gdf.geometry.name:
            continue
        unique = gdf[column].dropna().astype(str).value_counts().head(12).to_dict()
        print(f"  {column}: dtype={gdf[column].dtype}; top values={unique}")
    inferred = _infer_label_field(gdf.columns)
    print(f"Inferred label field: {inferred or 'NONE'}")


def command_build_masks(args: argparse.Namespace) -> None:
    image_dir = Path(args.image_dir)
    label_file = Path(args.label_file)
    mask_dir = Path(args.mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    images = list_tifs(image_dir)
    if not images:
        raise SystemExit(f"No GeoTIFFs found under {image_dir}")

    gdf = _load_vector_labels(label_file, args.layer)
    label_field = args.label_field or _infer_label_field(gdf.columns)
    if not label_field or label_field not in gdf.columns:
        available = [c for c in gdf.columns if c != gdf.geometry.name]
        raise ValueError(
            "Could not identify the agave/non-agave field. Run inspect-labels, "
            f"then pass --label-field. Available columns: {available}"
        )

    positive_values = {v.strip().lower() for v in args.positive_values.split(",") if v.strip()}
    if not positive_values:
        raise ValueError("--positive-values cannot be empty")
    positive = gdf[gdf[label_field].map(lambda v: _is_positive_label(v, positive_values))].copy()
    if positive.empty:
        sample = gdf[label_field].dropna().astype(str).value_counts().head(20).to_dict()
        raise ValueError(
            f"No positive polygons found in {label_field!r} using {sorted(positive_values)}. "
            f"Observed values: {sample}"
        )

    year_field = args.year_field
    if year_field and year_field not in positive.columns:
        raise ValueError(f"Year field {year_field!r} not found. Columns: {list(positive.columns)}")

    print(f"Using label field: {label_field}")
    print(f"Positive values: {sorted(positive_values)}")
    print(f"Positive polygons: {len(positive):,} / {len(gdf):,}")
    if year_field:
        print(f"Filtering polygons by year field: {year_field}")

    by_crs = {}
    report_rows = []
    written = 0
    full_positive = 0
    empty_masks = 0
    failures = []

    for image_path in tqdm(images, desc="Rasterizing masks"):
        try:
            location_id, year = parse_identity(image_path)
            with rasterio.open(image_path) as src:
                if src.crs is None:
                    raise ValueError("image has no CRS")
                crs_key = src.crs.to_string()
                if crs_key not in by_crs:
                    projected = positive.to_crs(src.crs)
                    by_crs[crs_key] = projected
                projected = by_crs[crs_key]
                subset = projected
                if year_field:
                    numeric_year = pd.to_numeric(subset[year_field], errors="coerce")
                    subset = subset[numeric_year == year]
                bounds = src.bounds
                try:
                    hits = list(subset.sindex.intersection(bounds))
                    candidates = subset.iloc[hits]
                except Exception:
                    candidates = subset.cx[bounds.left:bounds.right, bounds.bottom:bounds.top]
                if not candidates.empty:
                    candidates = candidates[candidates.intersects(
                        __import__('shapely.geometry', fromlist=['box']).box(*bounds)
                    )]
                shapes = [(geom, 1) for geom in candidates.geometry if geom is not None and not geom.is_empty]
                mask = rasterize(
                    shapes=shapes,
                    out_shape=(src.height, src.width),
                    transform=src.transform,
                    fill=0,
                    default_value=1,
                    dtype="uint8",
                    all_touched=args.all_touched,
                )
                profile = src.profile.copy()
                profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")
                out_path = mask_dir / f"{image_path.stem}_mask.tif"
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(mask, 1)
                    dst.set_band_description(1, "agave_binary_mask")
            fraction = float(mask.mean())
            empty_masks += int(fraction == 0.0)
            full_positive += int(fraction == 1.0)
            written += 1
            report_rows.append({
                "image": image_path.name, "mask": out_path.name,
                "location_id": location_id, "year": year,
                "positive_fraction": fraction,
                "intersecting_positive_polygons": len(shapes),
            })
        except Exception as exc:
            failures.append({"image": image_path.name, "error": str(exc)})

    report_csv = mask_dir / "mask_creation_report.csv"
    pd.DataFrame(report_rows).to_csv(report_csv, index=False)
    report = {
        "images_found": len(images),
        "masks_written": written,
        "failed": len(failures),
        "empty_masks": empty_masks,
        "fully_positive_masks": full_positive,
        "partially_positive_masks": written - empty_masks - full_positive,
        "label_file": str(label_file.resolve()),
        "label_field": label_field,
        "positive_values": sorted(positive_values),
        "year_field": year_field,
        "all_touched": bool(args.all_touched),
        "failures": failures[:100],
    }
    (mask_dir / "mask_creation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Per-mask report: {report_csv.resolve()}")
    if full_positive > 0:
        print(
            "WARNING: Some masks are 100% agave. If the source polygons are training "
            "rectangles rather than true field boundaries, these are weak patch masks, "
            "not precise segmentation labels."
        )


def command_inventory(args: argparse.Namespace) -> None:
    image_dir = Path(args.image_dir)
    files = list_tifs(image_dir)
    if not files:
        raise SystemExit(f"No GeoTIFFs found under {image_dir}")

    rows = []
    invalid_count = 0
    for path in tqdm(files, desc="Inspecting imagery"):
        problems = validate_image(path)
        if problems:
            invalid_count += 1
        try:
            location_id, year = parse_identity(path)
        except ValueError:
            location_id, year = "", -1
        info = inspect_raster(path)
        rows.append(
            {
                "filename": path.name,
                "location_id": location_id,
                "year": year,
                "bands": info["bands"],
                "height": info["height"],
                "width": info["width"],
                "dtype": info["dtype"],
                "crs": info["crs"],
                "valid": not problems,
                "problems": " | ".join(problems),
            }
        )

    df = pd.DataFrame(rows)
    output_csv = Path(args.output_csv) if args.output_csv else image_dir / "unet_image_inventory.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"\nGeoTIFFs: {len(df):,}")
    print(f"Unique locations: {df.loc[df.location_id != '', 'location_id'].nunique():,}")
    print(f"Years: {sorted(df.loc[df.year >= 0, 'year'].unique().tolist())}")
    print(f"Invalid files: {invalid_count:,}")
    print(f"Inventory written to: {output_csv.resolve()}")
    if invalid_count:
        print("\nFirst validation problems:")
        print(df.loc[~df.valid, ["filename", "problems"]].head(20).to_string(index=False))


def command_build_manifest(args: argparse.Namespace) -> None:
    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    output_csv = Path(args.output_csv)
    images = list_tifs(image_dir)
    if not images:
        raise SystemExit(f"No GeoTIFFs found under {image_dir}")

    records = []
    missing = []
    invalid = []
    for image_path in tqdm(images, desc="Matching image/mask pairs"):
        try:
            location_id, year = parse_identity(image_path)
        except ValueError as exc:
            invalid.append((image_path.name, str(exc)))
            continue

        mask_path = next((p for p in candidate_mask_paths(image_path, mask_dir) if p.exists()), None)
        if mask_path is None:
            missing.append(image_path.name)
            continue

        problems = validate_pair(image_path, mask_path)
        if problems:
            invalid.append((image_path.name, " | ".join(problems)))
            continue

        records.append(
            {
                "image_path": str(image_path.resolve()),
                "mask_path": str(mask_path.resolve()),
                "location_id": location_id,
                "year": year,
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_csv, index=False)

    report = {
        "images_found": len(images),
        "valid_pairs": len(records),
        "missing_masks": len(missing),
        "invalid_pairs": len(invalid),
        "missing_mask_examples": missing[:25],
        "invalid_pair_examples": invalid[:25],
    }
    report_path = output_csv.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nManifest written to: {output_csv.resolve()}")
    print(f"Validation report: {report_path.resolve()}")
    if not records:
        raise SystemExit(
            "No valid image/mask pairs were found. U-Net training cannot begin until "
            "aligned pixel masks exist."
        )


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"image_path", "mask_path", "location_id", "year"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Manifest contains no rows")
    return df


def make_group_splits(
    df: pd.DataFrame, seed: int, val_fraction: float, test_fraction: float
) -> pd.DataFrame:
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be > 0 and sum to < 1")
    groups = df["location_id"].astype(str).to_numpy()
    if len(np.unique(groups)) < 3:
        raise ValueError("At least 3 unique location_id groups are required")

    first = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_val_idx, test_idx = next(first.split(df, groups=groups))
    train_val = df.iloc[train_val_idx].copy()
    test = df.iloc[test_idx].copy()

    adjusted_val = val_fraction / (1.0 - test_fraction)
    second = GroupShuffleSplit(n_splits=1, test_size=adjusted_val, random_state=seed + 1)
    tv_groups = train_val["location_id"].astype(str).to_numpy()
    train_idx, val_idx = next(second.split(train_val, groups=tv_groups))

    out = df.copy()
    out["split"] = ""
    out.loc[train_val.iloc[train_idx].index, "split"] = "train"
    out.loc[train_val.iloc[val_idx].index, "split"] = "val"
    out.loc[test.index, "split"] = "test"
    return out


def compute_band_stats(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros(9, dtype=np.float64)
    squared_sums = np.zeros(9, dtype=np.float64)
    counts = np.zeros(9, dtype=np.int64)

    for path_str in tqdm(df.image_path, desc="Computing train-band statistics"):
        with rasterio.open(path_str) as src:
            x = src.read(out_dtype="float32")
        valid = np.isfinite(x)
        sums += np.where(valid, x, 0).sum(axis=(1, 2), dtype=np.float64)
        squared_sums += np.where(valid, x * x, 0).sum(axis=(1, 2), dtype=np.float64)
        counts += valid.sum(axis=(1, 2))

    mean = sums / np.maximum(counts, 1)
    variance = squared_sums / np.maximum(counts, 1) - mean**2
    std = np.sqrt(np.maximum(variance, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


def augment_pair(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Safe geometric augmentation for north-independent field patterns.
    if random.random() < 0.5:
        x, y = torch.flip(x, dims=[2]), torch.flip(y, dims=[2])
    if random.random() < 0.5:
        x, y = torch.flip(x, dims=[1]), torch.flip(y, dims=[1])
    k = random.randint(0, 3)
    if k:
        x, y = torch.rot90(x, k, dims=[1, 2]), torch.rot90(y, k, dims=[1, 2])
    return x, y


class AgaveDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        mean: np.ndarray,
        std: np.ndarray,
        augment: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.mean = torch.from_numpy(mean).view(9, 1, 1)
        self.std = torch.from_numpy(std).view(9, 1, 1)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with rasterio.open(row.image_path) as src:
            x_np = src.read(out_dtype="float32")
        with rasterio.open(row.mask_path) as src:
            y_np = src.read(1, out_dtype="float32")

        x_np = np.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0)
        ignore = y_np == 255
        y_np = (y_np > 0).astype(np.float32)

        x = (torch.from_numpy(x_np) - self.mean) / self.std
        y = torch.from_numpy(y_np).unsqueeze(0)
        valid = torch.from_numpy((~ignore).astype(np.float32)).unsqueeze(0)

        if self.augment:
            stacked = torch.cat([y, valid], dim=0)
            x, stacked = augment_pair(x, stacked)
            y, valid = stacked[:1], stacked[1:]

        return x, y, valid, str(row.image_path)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 9, base: int = 32) -> None:
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.bottleneck = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.output = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.output(d1)


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0, smooth: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight, reduction="none"
        )
        bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)

        probability = torch.sigmoid(logits) * valid
        target_valid = target * valid
        dims = (1, 2, 3)
        intersection = (probability * target_valid).sum(dims)
        denominator = probability.sum(dims) + target_valid.sum(dims)
        dice_loss = 1.0 - ((2.0 * intersection + self.smooth) / (denominator + self.smooth)).mean()
        return 0.5 * bce + 0.5 * dice_loss


def estimate_pos_weight(df: pd.DataFrame) -> float:
    positive = 0
    negative = 0
    for mask_path in tqdm(df.mask_path, desc="Estimating class balance"):
        with rasterio.open(mask_path) as src:
            y = src.read(1)
        valid = y != 255
        positive += int(((y > 0) & valid).sum())
        negative += int(((y == 0) & valid).sum())
    if positive == 0:
        raise ValueError("Training masks contain no positive agave pixels")
    return float(np.clip(negative / positive, 1.0, 25.0))


@dataclass
class BinaryCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> None:
        p = pred.bool()
        t = target.bool()
        v = valid.bool()
        self.tp += int((p & t & v).sum().item())
        self.fp += int((p & ~t & v).sum().item())
        self.fn += int((~p & t & v).sum().item())
        self.tn += int((~p & ~t & v).sum().item())

    def metrics(self) -> dict[str, float]:
        eps = 1e-9
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        dice = 2 * self.tp / (2 * self.tp + self.fp + self.fn + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn + eps)
        return {
            "precision": precision,
            "recall": recall,
            "dice": dice,
            "iou": iou,
            "accuracy": accuracy,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
        }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses = []
    counts = BinaryCounts()

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for x, y, valid, _ in tqdm(loader, leave=False, desc="train" if training else "evaluate"):
            x, y, valid = x.to(device), y.to(device), valid.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y, valid)
            if training:
                loss.backward()
                optimizer.step()
            losses.append(float(loss.item()))
            pred = torch.sigmoid(logits) >= threshold
            counts.update(pred, y >= 0.5, valid >= 0.5)

    result = counts.metrics()
    result["loss"] = float(np.mean(losses)) if losses else math.nan
    return result


def save_checkpoint(
    path: Path,
    model: nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    args: argparse.Namespace,
    epoch: int,
    val_metrics: dict,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "in_channels": 9,
            "base_channels": args.base_channels,
            "mean": mean,
            "std": std,
            "band_names": EXPECTED_BANDS,
            "threshold": args.threshold,
            "epoch": epoch,
            "val_metrics": val_metrics,
        },
        path,
    )


def command_train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_manifest(Path(args.manifest))
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Validating pairs"):
        problems = validate_pair(Path(row.image_path), Path(row.mask_path))
        if problems:
            raise ValueError(f"Invalid pair {row.image_path}: {' | '.join(problems)}")

    if "split" not in df.columns or not set(df.split.unique()).issuperset({"train", "val", "test"}):
        df = make_group_splits(df, args.seed, args.val_fraction, args.test_fraction)
    split_path = output_dir / "manifest_with_splits.csv"
    df.to_csv(split_path, index=False)

    # Verify no location appears in more than one split.
    leakage = df.groupby("location_id")["split"].nunique()
    if (leakage > 1).any():
        raise ValueError("Spatial leakage detected: a location_id occurs in multiple splits")

    train_df = df[df.split == "train"].copy()
    val_df = df[df.split == "val"].copy()
    test_df = df[df.split == "test"].copy()
    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise ValueError("Train, validation, and test splits must all contain samples")

    mean, std = compute_band_stats(train_df)
    pos_weight = estimate_pos_weight(train_df)
    normalization = {
        "band_names": EXPECTED_BANDS,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "positive_class_weight": pos_weight,
    }
    (output_dir / "normalization.json").write_text(json.dumps(normalization, indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")
    print(f"Rows: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(
        "Unique locations: "
        f"train={train_df.location_id.nunique()}, "
        f"val={val_df.location_id.nunique()}, test={test_df.location_id.nunique()}"
    )
    print(f"Positive class weight: {pos_weight:.3f}")

    train_ds = AgaveDataset(train_df, mean, std, augment=True)
    val_ds = AgaveDataset(val_df, mean, std, augment=False)
    test_ds = AgaveDataset(test_df, mean, std, augment=False)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_kwargs)

    model = UNet(in_channels=9, base=args.base_channels).to(device)
    criterion = BCEDiceLoss(pos_weight=pos_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )

    history = []
    best_iou = -1.0
    epochs_without_improvement = 0
    best_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, args.threshold, optimizer=optimizer
        )
        val_metrics = run_epoch(model, val_loader, criterion, device, args.threshold)
        scheduler.step(val_metrics["iou"])

        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

        print(
            f"Epoch {epoch:03d} | train loss={train_metrics['loss']:.4f} "
            f"IoU={train_metrics['iou']:.4f} | val loss={val_metrics['loss']:.4f} "
            f"IoU={val_metrics['iou']:.4f} Dice={val_metrics['dice']:.4f}"
        )

        if val_metrics["iou"] > best_iou + args.min_delta:
            best_iou = val_metrics["iou"]
            epochs_without_improvement = 0
            save_checkpoint(best_path, model, mean, std, args, epoch, val_metrics)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = run_epoch(model, test_loader, criterion, device, args.threshold)
    summary = {
        "best_epoch": checkpoint["epoch"],
        "best_validation": checkpoint["val_metrics"],
        "test": test_metrics,
        "device": str(device),
        "threshold": args.threshold,
        "splits": {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "train_locations": int(train_df.location_id.nunique()),
            "val_locations": int(val_df.location_id.nunique()),
            "test_locations": int(test_df.location_id.nunique()),
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nFinal test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print(f"\nBest model: {best_path.resolve()}")


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = UNet(
        in_channels=int(checkpoint.get("in_channels", 9)),
        base=int(checkpoint.get("base_channels", 32)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    threshold = float(checkpoint.get("threshold", 0.5))
    return model, mean, std, threshold


def predict_one(
    model: nn.Module,
    image_path: Path,
    output_dir: Path,
    mean: np.ndarray,
    std: np.ndarray,
    threshold: float,
    device: torch.device,
) -> None:
    problems = validate_image(image_path)
    if problems:
        raise ValueError(f"Invalid image {image_path.name}: {' | '.join(problems)}")

    with rasterio.open(image_path) as src:
        x = src.read(out_dtype="float32")
        profile = src.profile.copy()
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = (x - mean[:, None, None]) / std[:, None, None]
    tensor = torch.from_numpy(x).unsqueeze(0).to(device)
    with torch.no_grad():
        probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy().astype("float32")
    mask = (probability >= threshold).astype("uint8")

    profile.update(count=1, dtype="float32", nodata=None, compress="deflate")
    probability_path = output_dir / f"{image_path.stem}_agave_probability.tif"
    with rasterio.open(probability_path, "w", **profile) as dst:
        dst.write(probability, 1)
        dst.set_band_description(1, "agave_probability")

    profile.update(dtype="uint8", nodata=255)
    mask_path = output_dir / f"{image_path.stem}_agave_mask.tif"
    with rasterio.open(mask_path, "w", **profile) as dst:
        dst.write(mask, 1)
        dst.set_band_description(1, "agave_binary_mask")


def command_predict(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images = list_tifs(image_dir)
    if not images:
        raise SystemExit(f"No GeoTIFFs found under {image_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, mean, std, saved_threshold = load_model(Path(args.model), device)
    threshold = saved_threshold if args.threshold is None else args.threshold
    print(f"Device: {device}; threshold: {threshold}")

    failures = []
    for image_path in tqdm(images, desc="Predicting"):
        try:
            predict_one(model, image_path, output_dir, mean, std, threshold, device)
        except Exception as exc:  # continue processing a large folder, but report failures
            failures.append({"file": str(image_path), "error": str(exc)})

    report = {
        "images_found": len(images),
        "successful": len(images) - len(failures),
        "failed": len(failures),
        "failures": failures[:100],
    }
    (output_dir / "prediction_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Baseline 9-band, 64x64 U-Net workflow for Agave Sentinel-2 patches."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_labels = subparsers.add_parser("inspect-labels", help="Inspect polygon label fields and values")
    inspect_labels.add_argument("--label-file", required=True)
    inspect_labels.add_argument("--layer")
    inspect_labels.set_defaults(func=command_inspect_labels)

    masks = subparsers.add_parser("build-masks", help="Rasterize agave polygons onto each image grid")
    masks.add_argument("--image-dir", required=True)
    masks.add_argument("--label-file", required=True)
    masks.add_argument("--mask-dir", required=True)
    masks.add_argument("--layer")
    masks.add_argument("--label-field")
    masks.add_argument("--positive-values", default="1,agave,true,yes")
    masks.add_argument("--year-field")
    masks.add_argument("--all-touched", action="store_true")
    masks.set_defaults(func=command_build_masks)

    inv = subparsers.add_parser("inventory", help="Inspect imagery before mask creation/training")
    inv.add_argument("--image-dir", required=True)
    inv.add_argument("--output-csv")
    inv.set_defaults(func=command_inventory)

    manifest = subparsers.add_parser("build-manifest", help="Match and validate images and masks")
    manifest.add_argument("--image-dir", required=True)
    manifest.add_argument("--mask-dir", required=True)
    manifest.add_argument("--output-csv", required=True)
    manifest.set_defaults(func=command_build_manifest)

    train = subparsers.add_parser("train", help="Train U-Net from a validated CSV manifest")
    train.add_argument("--manifest", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--base-channels", type=int, default=32)
    train.add_argument("--threshold", type=float, default=0.5)
    train.add_argument("--val-fraction", type=float, default=0.15)
    train.add_argument("--test-fraction", type=float, default=0.15)
    train.add_argument("--patience", type=int, default=15)
    train.add_argument("--min-delta", type=float, default=1e-4)
    train.add_argument("--workers", type=int, default=0, help="Use 0 first on Windows")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--cpu", action="store_true")
    train.set_defaults(func=command_train)

    predict = subparsers.add_parser("predict", help="Create probability and binary GeoTIFFs")
    predict.add_argument("--model", required=True)
    predict.add_argument("--image-dir", required=True)
    predict.add_argument("--output-dir", required=True)
    predict.add_argument("--threshold", type=float)
    predict.add_argument("--seed", type=int, default=42)
    predict.add_argument("--cpu", action="store_true")
    predict.set_defaults(func=command_predict)

    return parser

def main() -> None:
    print(f"Agave U-Net script version: {SCRIPT_VERSION}")
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        raise SystemExit("Interrupted by user")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()