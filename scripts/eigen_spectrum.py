#!/usr/bin/env python3
"""
Marchenko–Pastur + parallel-analysis eigenspectrum of a strategy correlation matrix.

Two modes:

  * synthetic (default):
      Builds a Gaussian-noise correlation matrix of N=512 strategies × T=60
      windows, then plots the empirical eigenvalue spectrum together with the
      Marchenko–Pastur bulk edge λ₊ and the parallel-analysis 95% null cutoff.
      Reproducible (RNG seed=2026), no external data required.

  * from-tables:
      Reads precomputed per-asset eigenvalue tables from a thesis data root
      and reproduces the corresponding figure. The data root is supplied
      either via --from-data-root or the STRATEGY_DATA_ROOT environment
      variable; the script never assumes a hardcoded path.

      Expected layout:
        $STRATEGY_DATA_ROOT/
          03_Random_Matrix_Theory/tables/table1_rmt_summary.csv
          03_Random_Matrix_Theory/tables/table2_signal_eigenvalues.csv

Emits:
  figures/eigen_spectrum.png  — bar plot of the spectrum, signal eigenvalues
                                shaded, λ₊ drawn as a horizontal reference.
  rmt_eigen.json              — compact summary consumed by the portfolio site.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend so the script runs over SSH / CI
import matplotlib.pyplot as plt


def mp_bounds(sigma2: float, gamma: float) -> tuple[float, float]:
    """Marchenko–Pastur bulk edges for noise variance σ² and shape γ = T/N.

    Eigenvalues outside [λ₋, λ₊] are candidates for genuine factor signal
    rather than sample-noise artefacts.
    """
    return sigma2 * (1 - np.sqrt(gamma)) ** 2, sigma2 * (1 + np.sqrt(gamma)) ** 2


def parallel_analysis(X: np.ndarray, n_perm: int = 200, seed: int = 0) -> np.ndarray:
    """Horn's parallel analysis: permutation-based 95% null cutoff per rank.

    Each column of X is independently row-permuted, breaking any
    cross-strategy structure while preserving the marginal distributions.
    The 95th percentile of the rank-k permuted eigenvalue across n_perm
    repeats is the operational cutoff that a real eigenvalue must exceed
    to be called a signal.
    """
    rng = np.random.default_rng(seed)
    T, N = X.shape
    max_eigs = np.zeros((n_perm, N))
    for b in range(n_perm):
        Xp = np.stack([rng.permutation(col) for col in X.T], axis=1)
        Xp = (Xp - Xp.mean(0)) / Xp.std(0, ddof=0).clip(1e-12)
        w = np.linalg.eigvalsh(np.corrcoef(Xp, rowvar=False))
        max_eigs[b] = np.sort(w)[::-1]
    return np.quantile(max_eigs, 0.95, axis=0)


def synthetic_demo(N: int = 512, T: int = 60, seed: int = 2026) -> tuple[np.ndarray, np.ndarray]:
    """Reproducible 4-factor + noise universe used when no real data is present.

    Returns the sorted eigenvalues of the sample correlation matrix and the
    parallel-analysis 95% cutoff per rank — exactly the two arrays the
    plotting code below consumes.
    """
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((T, 4))             # 4 latent factors over T windows
    L = rng.standard_normal((N, 4)) * 0.35      # loadings of N strategies on the factors
    X = F @ L.T + rng.standard_normal((T, N))   # additive idiosyncratic noise
    X = (X - X.mean(0)) / X.std(0, ddof=0).clip(1e-12)
    R = np.corrcoef(X, rowvar=False)
    eigs = np.sort(np.linalg.eigvalsh(R))[::-1]
    pa_95 = parallel_analysis(X)
    return eigs, pa_95


def from_tables(data_root: Path, asset: str) -> dict:
    """Load per-asset eigenvalue tables produced by the thesis pipeline."""
    import pandas as pd
    t1 = pd.read_csv(data_root / "03_Random_Matrix_Theory" / "tables" / "table1_rmt_summary.csv")
    t2 = pd.read_csv(data_root / "03_Random_Matrix_Theory" / "tables" / "table2_signal_eigenvalues.csv")
    row = t1[t1["Asset"] == asset].iloc[0]
    eigs = t2[t2["Asset"] == asset].sort_values("Factor")["Eigenvalue"].to_numpy()
    signal_count = int((t2[t2["Asset"] == asset]["Exceeds PA 95%"] == "Yes").sum())
    return {
        "asset": asset,
        "eigs": [float(v) for v in eigs],
        "lambda_plus": float(row["lambda_+"]),
        "signal_count": signal_count,
        "n_strategies": int(row["N (strategies)"]),
    }


def resolve_data_root(arg_value: Path | None) -> Path | None:
    """Resolve --from-data-root precedence: CLI arg > $STRATEGY_DATA_ROOT > None."""
    if arg_value is not None:
        return arg_value
    env = os.environ.get("STRATEGY_DATA_ROOT")
    return Path(env) if env else None


def main():
    ap = argparse.ArgumentParser(
        description="Marchenko–Pastur + parallel-analysis eigenspectrum (synthetic by default)."
    )
    ap.add_argument("--asset", default=None,
                    help="e.g. BTC, DOGE, SOL. If omitted, runs the reproducible synthetic demo.")
    ap.add_argument("--from-data-root", type=Path, default=None,
                    help="Path to the thesis data root. Falls back to $STRATEGY_DATA_ROOT. "
                         "Only required when --asset is set.")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent.parent / "figures",
                    help="Output directory for figures (default: ../figures relative to this script).")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.asset:
        data_root = resolve_data_root(args.from_data_root)
        if data_root is None:
            ap.error("--asset requires --from-data-root or $STRATEGY_DATA_ROOT to be set.")
        payload = from_tables(data_root, args.asset)
        eigs = np.array(payload["eigs"])
        lambda_plus = payload["lambda_plus"]
        signal_count = payload["signal_count"]
        title = f"{args.asset} · N = {payload['n_strategies']:,}"
    else:
        eigs, pa_95 = synthetic_demo()
        lambda_plus = mp_bounds(1.0, 60 / 512)[1]
        signal_count = int((eigs[:15] > pa_95[:15]).sum())
        title = "synthetic · 4 latent factors + noise"

    # plot
    fig, ax = plt.subplots(figsize=(7, 3.6))
    xs = np.arange(len(eigs))
    cols = ["#b6ff4a" if i < signal_count else "#6f7680" for i in xs]
    ax.bar(xs, eigs, color=cols, width=0.8)
    ax.axhline(lambda_plus, color="#4ec9e0", linestyle="--", linewidth=1,
               label=f"Marchenko–Pastur λ₊ = {lambda_plus:.3f}")
    ax.set_xlabel("factor rank")
    ax.set_ylabel("eigenvalue λ")
    ax.set_title(f"Strategy correlation spectrum · {title}")
    ax.legend(loc="upper right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out / "eigen_spectrum.png", dpi=160)
    print("wrote", args.out / "eigen_spectrum.png")

    # JSON for the portfolio site
    json_out = {
        "asset": args.asset or "synthetic",
        "eigs": [round(float(v), 4) for v in eigs[:40]],
        "lambda_plus": round(float(lambda_plus), 4),
        "signal_count": int(signal_count),
        "n_strategies": int(payload["n_strategies"]) if args.asset else 512,
    }
    (args.out.parent / "rmt_eigen.json").write_text(json.dumps(json_out, indent=2))
    print("wrote", args.out.parent / "rmt_eigen.json")


if __name__ == "__main__":
    main()
