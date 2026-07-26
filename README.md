# strategy-rmt

**Random-matrix analysis of strategy correlation matrices.**

> Code behind [Eigenspectrum of the strategy correlation matrix](https://daru.finance/projects/strategy-rmt), one of the M-series models Daniel Gatto publishes on [daru.finance](https://daru.finance).

Noise-vs-signal decomposition of sample correlation matrices over large populations
of algorithmic trading strategies, using Marchenko–Pastur asymptotics and
parallel-analysis permutation testing.

## Reproduce

```bash
git clone https://github.com/DaruFinance/strategy-rmt
cd strategy-rmt
pip install -e .
python scripts/eigen_spectrum.py
```

Runs the reproducible synthetic demo (no external data, deterministic RNG seed)
and writes `figures/eigen_spectrum.png` plus `rmt_eigen.json`.

## Problem statement

Given `N` strategies with `T` window-level return observations each, the
sample correlation matrix `R̂ ∈ ℝ^{N×N}` is highly noise-dominated when
`γ = T/N ≪ 1`. The Marchenko–Pastur density characterises the bulk of the
spectrum under the null (independent Gaussian returns):

```
λ₋ = σ² · (1 − √γ)²
λ₊ = σ² · (1 + √γ)²
```

Eigenvalues beyond `λ₊` are candidates for genuine factors. Parallel analysis
(Horn 1965) refines the threshold by permutation: for each rank `k`, the
95th-percentile null eigenvalue from row-permuted data is the operational
cutoff. This repository implements both cuts and produces a denoised
covariance for downstream portfolio construction.

## Usage

The default mode runs a synthetic demo (4 latent factors + Gaussian noise,
RNG seed = 2026):

```bash
python scripts/eigen_spectrum.py
```

To reproduce the per-asset thesis figures, point the script at the data
root either via the `--from-data-root` flag or the `STRATEGY_DATA_ROOT`
environment variable:

```bash
export STRATEGY_DATA_ROOT="$HOME/PhD_Research"   # adjust for your machine
python scripts/eigen_spectrum.py --asset BTC
```

The expected layout under `$STRATEGY_DATA_ROOT` is:

```
03_Random_Matrix_Theory/tables/table1_rmt_summary.csv
03_Random_Matrix_Theory/tables/table2_signal_eigenvalues.csv
```

Output: `figures/eigen_spectrum.png` and `rmt_eigen.json` (consumed by the
portfolio site at <https://github.com/DaruFinance>).

## Key result (BTC, N = 30,801, T = 27)

- Marchenko–Pastur bulk bound `λ₊ ≈ 1.109`
- 4 eigenvalues exceed the parallel-analysis 95% threshold (interpretable factors)
- The remaining bulk is statistically indistinguishable from Gaussian noise.

## References

- Marchenko, V. A. & Pastur, L. A. (1967). *Distribution of eigenvalues for some sets of random matrices.*
- Tracy, C. A. & Widom, H. (1994). *Level-spacing distributions and the Airy kernel.*
- Horn, J. L. (1965). *A rationale and test for the number of factors in factor analysis.*
- Laloux, P. *et al.* (1999). *Noise dressing of financial correlation matrices.* PRL 83.

## License

MIT © Daniel Vieira Gatto.
