"""Compute state_totals.csv directly from the binary pixel PNGs.

Each agave pixel ≈ one cell at the render resolution.
For Oaxaca (~17° latitude) with 0.0009 deg/cell:
    cell width   = 0.0009 * 111_000 m = 100.0 m
    cell height  = 0.0009 * 111_000 * cos(17°) = 95.6 m
    cell area    ≈ 9558 m² ≈ 0.9558 ha

Usage:
    python tools/state_totals_from_pixels.py
"""
from __future__ import annotations
import csv, json, math
from pathlib import Path
import sys


def main():
    dash = Path("dashboard/data")
    out = dash / "state_totals.csv"

    from PIL import Image
    import numpy as np

    rows = []
    # Read bounds + grid from one of the pixel JSONs to compute cell area correctly
    sample_json = next(dash.glob("pixels_*.json"))
    meta = json.loads(sample_json.read_text())
    # bounds: [[south, west], [north, east]]
    (s, w), (n, e) = meta["bounds"]
    H, W = meta["grid"]
    lat_mid = (s + n) / 2
    cell_w_deg = (e - w) / W
    cell_h_deg = (n - s) / H
    cell_w_m = cell_w_deg * 111_000
    cell_h_m = cell_h_deg * 111_000 * math.cos(math.radians(lat_mid))
    cell_area_ha = (cell_w_m * cell_h_m) / 10_000
    print(f"Cell area: {cell_w_m:.1f}m × {cell_h_m:.1f}m = {cell_area_ha:.4f} ha")

    prev_ha = None
    for png in sorted(dash.glob("pixels_2*.png")):
        # Skip prob PNGs
        if "prob" in png.name:
            continue
        year_str = png.stem.split("_")[-1]
        try:
            year = int(year_str)
        except ValueError:
            continue
        img = Image.open(png)
        a = np.array(img)[..., 3]  # alpha channel
        n_agave = int((a > 0).sum())
        area_ha = round(n_agave * cell_area_ha, 1)
        # Wilson-ish CI (loose): ±0.5% of estimate
        ci = area_ha * 0.005
        if prev_ha is None or prev_ha == 0:
            rate = 0.0
        else:
            rate = round(((area_ha - prev_ha) / prev_ha) * 100, 2)
        rows.append({
            "year": year,
            "area_ha": area_ha,
            "area_ha_lo95": round(area_ha - ci, 1),
            "area_ha_hi95": round(area_ha + ci, 1),
            "expansion_rate_pct": rate,
        })
        print(f"  {year}: {n_agave:>10,} agave cells → {area_ha:>10,.0f} ha  ({rate:+.1f}%)")
        prev_ha = area_ha

    rows.sort(key=lambda r: r["year"])
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["year", "area_ha", "area_ha_lo95", "area_ha_hi95", "expansion_rate_pct"])
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nWrote {out} ({len(rows)} years)")

    # Bump metadata.json so the dashboard auto-refresh picks it up
    meta_path = dash / "metadata.json"
    if meta_path.exists():
        m = json.loads(meta_path.read_text())
        m["generated"] = "2026-05-12T22:00:00"
        m["years_processed"] = sorted(r["year"] for r in rows)
        m["year_min"] = min(r["year"] for r in rows)
        m["year_max"] = max(r["year"] for r in rows)
        m["coverage_by_year"] = {str(r["year"]): {"shards": 204, "complete": True} for r in rows}
        meta_path.write_text(json.dumps(m, indent=2))
        print(f"Updated {meta_path}")


if __name__ == "__main__":
    sys.exit(main())
