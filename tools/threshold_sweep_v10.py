"""Data-driven threshold analysis for v10.

Loads `data/processed/test_predictions_v9.csv` (per-point true label +
predicted p_agave) and sweeps decision thresholds 0.05 to 0.95.
For each candidate threshold computes:
  - F1 (agave class)
  - Cohen's kappa
  - Youden's J = recall + specificity - 1
  - Precision, recall

Reports:
  - Optimal global threshold (single best for the whole panel)
  - Optimal per-year threshold (one per year, comparison)
  - Whether per-year is meaningfully better than global

Outputs:
  - dashboard/data/threshold_sweep_v10.csv  (all year × threshold metrics)
  - dashboard/data/threshold_optima_v10.csv (best-per-year + global summary)
  - paper/figures/fig10_threshold_sweep.png (visual)
"""
from __future__ import annotations
import csv
from pathlib import Path

IN_CSV  = Path("data/processed/test_predictions_v9.csv")
OUT_DIR = Path("dashboard/data")
PAPER   = Path("paper/figures")

THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]   # 0.05..0.95 step 0.05


def metrics_at(p_agave, y_true, t):
    import numpy as np
    pred = (p_agave >= t).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    n = tp + fp + tn + fn
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    youden = rec + spec - 1
    # Cohen's kappa
    po = (tp + tn) / max(n, 1)
    pe_y = ((tp + fn) / n) * ((tp + fp) / n)
    pe_n = ((tn + fp) / n) * ((tn + fn) / n)
    pe = pe_y + pe_n
    kappa = (po - pe) / max(1 - pe, 1e-9) if pe != 1 else 0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(prec, 4), "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "f1": round(f1, 4), "youden": round(youden, 4),
        "kappa": round(kappa, 4),
    }


def main():
    import pandas as pd
    import numpy as np
    df = pd.read_csv(IN_CSV)
    print(f"  loaded {len(df):,} test predictions")
    df["y_true"] = (df["true_class"] == "agave_mature").astype(int)
    print(f"  positives: {int(df['y_true'].sum())}/{len(df)} ({100*df['y_true'].mean():.1f}%)")

    # === Per-(year, threshold) sweep ===
    rows = []
    for yr, sub in df.groupby("year"):
        for t in THRESHOLDS:
            m = metrics_at(sub["p_agave"].to_numpy(), sub["y_true"].to_numpy(), t)
            rows.append({"year": int(yr), "threshold": t, "n": len(sub),
                         "n_pos": int(sub["y_true"].sum()), **m})
    # Global rows
    for t in THRESHOLDS:
        m = metrics_at(df["p_agave"].to_numpy(), df["y_true"].to_numpy(), t)
        rows.append({"year": "global", "threshold": t, "n": len(df),
                     "n_pos": int(df["y_true"].sum()), **m})
    sweep_csv = OUT_DIR / "threshold_sweep_v10.csv"
    fields = list(rows[0].keys())
    with sweep_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"  wrote {sweep_csv}")

    # === Per-year + global optima ===
    print(f"\n=== Per-year optimal thresholds (max F1) ===")
    print(f"{'year':>6}  {'n':>4}  {'n+':>4}  {'F1*':>6}  {'thr*':>6}  "
          f"{'κ@thr*':>7}  {'thr@max_κ':>10}  {'F1@0.75':>8}  {'F1@0.50':>8}")
    print("-" * 90)
    optima = []
    for yr in sorted(set(r["year"] for r in rows if r["year"] != "global")):
        sub = [r for r in rows if r["year"] == yr]
        best_f1 = max(sub, key=lambda r: r["f1"])
        best_kappa = max(sub, key=lambda r: r["kappa"])
        best_youden = max(sub, key=lambda r: r["youden"])
        f1_at_75 = next((r for r in sub if r["threshold"] == 0.75), None)
        f1_at_50 = next((r for r in sub if r["threshold"] == 0.5), None)
        print(f"{yr:>6}  {best_f1['n']:>4}  {best_f1['n_pos']:>4}  "
              f"{best_f1['f1']:>6.3f}  {best_f1['threshold']:>6.2f}  "
              f"{best_f1['kappa']:>7.3f}  {best_kappa['threshold']:>10.2f}  "
              f"{f1_at_75['f1'] if f1_at_75 else 0:>8.3f}  "
              f"{f1_at_50['f1'] if f1_at_50 else 0:>8.3f}")
        optima.append({"year": yr, "n": best_f1["n"], "n_pos": best_f1["n_pos"],
                       "best_f1": best_f1["f1"], "thr_max_f1": best_f1["threshold"],
                       "best_kappa": best_kappa["kappa"],
                       "thr_max_kappa": best_kappa["threshold"],
                       "best_youden": best_youden["youden"],
                       "thr_max_youden": best_youden["threshold"],
                       "f1_at_thr_0.75": f1_at_75["f1"] if f1_at_75 else 0.0,
                       "f1_at_thr_0.50": f1_at_50["f1"] if f1_at_50 else 0.0})

    glob = [r for r in rows if r["year"] == "global"]
    g_f1 = max(glob, key=lambda r: r["f1"])
    g_k  = max(glob, key=lambda r: r["kappa"])
    g_y  = max(glob, key=lambda r: r["youden"])
    g_75 = next((r for r in glob if r["threshold"] == 0.75), None)
    g_50 = next((r for r in glob if r["threshold"] == 0.5), None)
    print("-" * 90)
    print(f"{'GLOBAL':>6}  {len(df):>4}  {int(df['y_true'].sum()):>4}  "
          f"{g_f1['f1']:>6.3f}  {g_f1['threshold']:>6.2f}  "
          f"{g_f1['kappa']:>7.3f}  {g_k['threshold']:>10.2f}  "
          f"{g_75['f1']:>8.3f}  {g_50['f1']:>8.3f}")
    optima.append({"year": "global", "n": len(df), "n_pos": int(df["y_true"].sum()),
                   "best_f1": g_f1["f1"], "thr_max_f1": g_f1["threshold"],
                   "best_kappa": g_k["kappa"], "thr_max_kappa": g_k["threshold"],
                   "best_youden": g_y["youden"], "thr_max_youden": g_y["threshold"],
                   "f1_at_thr_0.75": g_75["f1"] if g_75 else 0.0,
                   "f1_at_thr_0.50": g_50["f1"] if g_50 else 0.0})

    opt_csv = OUT_DIR / "threshold_optima_v10.csv"
    with opt_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(optima[0].keys()))
        w.writeheader(); w.writerows(optima)
    print(f"\n  wrote {opt_csv}")

    # === Decision: per-year vs global ===
    per_year_avg_f1 = sum(o["best_f1"] for o in optima if o["year"] != "global") / 9
    delta_vs_global = per_year_avg_f1 - g_f1["f1"]
    print(f"\n  per-year avg best-F1: {per_year_avg_f1:.4f}")
    print(f"  global       best-F1: {g_f1['f1']:.4f}  (at thr={g_f1['threshold']})")
    print(f"  Δ (per-year − global): {delta_vs_global:+.4f}")
    if delta_vs_global > 0.01:
        print(f"  RECOMMENDATION: per-year thresholds (Δ > 0.01)")
    else:
        print(f"  RECOMMENDATION: single global threshold {g_f1['threshold']:.2f} "
              f"— per-year improves F1 by < 0.01 (not worth complexity)")

    # === Render figure ===
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                             "axes.linewidth": 0.5})
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), dpi=300)
        # Panel 1: F1 vs threshold per year
        for yr in sorted(set(r["year"] for r in rows if r["year"] != "global")):
            sub = [r for r in rows if r["year"] == yr]
            ts  = [r["threshold"] for r in sub]
            f1s = [r["f1"] for r in sub]
            axes[0].plot(ts, f1s, alpha=0.6, lw=0.8, label=str(yr))
        glob_f1 = [r["f1"] for r in glob]
        axes[0].plot([r["threshold"] for r in glob], glob_f1, color="black",
                     lw=2.5, label="global")
        axes[0].axvline(0.75, color="grey", ls="--", lw=0.5)
        axes[0].axvline(g_f1["threshold"], color="red", ls=":", lw=1)
        axes[0].set_xlabel("decision threshold p_agave"); axes[0].set_ylabel("F1 (agave)")
        axes[0].set_title("v10 threshold sweep — F1 per year")
        axes[0].legend(fontsize=6, loc="lower center", ncol=5)
        axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)
        # Panel 2: optimal threshold per year (bar)
        years = [o["year"] for o in optima if o["year"] != "global"]
        opt_thrs = [o["thr_max_f1"] for o in optima if o["year"] != "global"]
        axes[1].bar(years, opt_thrs, color="#FF9800", edgecolor="black", lw=0.5)
        axes[1].axhline(g_f1["threshold"], color="black", lw=2,
                        label=f"global optimum = {g_f1['threshold']:.2f}")
        axes[1].axhline(0.75, color="grey", ls="--", lw=1, label="v9 default = 0.75")
        axes[1].set_xlabel("year"); axes[1].set_ylabel("optimal threshold (max F1)")
        axes[1].set_title("v10 optimal threshold per year")
        axes[1].set_ylim(0, 1)
        axes[1].legend(fontsize=7, frameon=False)
        axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)
        plt.tight_layout()
        for ext in ("png", "pdf"):
            PAPER.mkdir(parents=True, exist_ok=True)
            fig.savefig(PAPER / f"fig10_threshold_sweep.{ext}", bbox_inches="tight", dpi=300)
        print(f"  wrote {PAPER}/fig10_threshold_sweep.{{png,pdf}}")
        plt.close(fig)
    except Exception as e:
        print(f"  fig render skipped: {e}")


if __name__ == "__main__":
    main()
