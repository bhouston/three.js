# Spatial phase ablations

CPU-only exploratory reference; no production sampler is changed. Run `python utils/noise-comparison/explorations/spatial.py` with NumPy and Pillow. All 61 attempted phase configurations and their parameters remain in `spatial-results.json` (screen seed 1, frames 0–1023). Eight initial candidates plus two ternary candidates are checked at seeds 1 and 2 and independent accumulation starts 0, 17 and 113, including RMSE for `u < 0.37` and `u²`. This is a small investigation, not an exhaustive optimizer or a state-of-the-art claim.

## Findings

The original three-level construction (`L3-hash`) assigns different orientations to adjacent 2×2 strata and adds a full-range independent phase per 8×8 tile. Thus it does not enforce approximately uniform coverage of **every sliding 3×3 window**. More levels alone do not fix this: the first few rank bits and the independent phase shifts remain the important local behavior. Animated IGN's sliding neighborhood coverage is a more direct match to the sigma=1 reconstruction filter. The Weyl sequence contributes excellent long-prefix temporal convergence but cannot by itself repair those early spatial correlations.

The most useful simple alternative in this spatial experiment is to use an IGN spatial phase and advance it with the same Weyl step. This is a conventional spatial low-discrepancy phase plus temporal irrational rotation, not a new claim to blue-noise spectra. It retains first-frame IGN quality and inherits the temporal convergence, but still loses to Animated IGN at several short prefixes. In the original seed-1/start-0 uniform benchmark:

| Phase | 1 | 8 | 128 | 256 | 512 | 1024 |
|---|---:|---:|---:|---:|---:|---:|
| IGN phase + Weyl | .029332 | .0080171 | .0007093 | .0003483 | .0001728 | .0001013 |
| Animated IGN reference | .0293557 | .00603109 | .000939296 | .00060265 | .00027788 | .00020536 |

Two seeds × three starts, mean error over six runs:

| Phase | Metric | 1 | 8 | 128 | 1024 |
|---|---|---:|---:|---:|---:|
| Original 3LHW | filtered standard deviation | .0498055 | .0107474 | .0010152 | .0001117 |
| IGN phase + Weyl | filtered standard deviation | .0293339 | .0080223 | .0007096 | .0001013 |
| Original 3LHW | threshold RMSE, p=.37 | .0809279 | .0085342 | .0014689 | .0001565 |
| IGN phase + Weyl | threshold RMSE, p=.37 | .0451341 | .0104233 | .0009139 | .0001183 |

The threshold test is particularly useful: the phase substitution improves early spatial error and late convergence, but worsens the 8-frame threshold error. It does not dominate all uses. The two-seed phase change for IGN phase + Weyl is a global random Cranley–Patterson shift of the same IGN phase field, rather than a newly randomized IGN field; therefore this confirms phase/start sensitivity, not a broad family of independent spatial realizations.

Other attempts:

- One through six binary levels, reduced or absent tile jitter, and fixed orientations improve selected frame counts but show no general dominance.
- Affine coordinate mixing disrupts the useful local strata and makes early errors worse.
- A ternary 3×3 hierarchy still has independently oriented tiles, so it does not enforce the sliding-window property; the tested versions do not solve the problem.
- Linear lattice phases, including the plastic-constant R2 pair and coefficients near IGN's effective slopes, perform well on selected metrics but leave directional spectral lines and aliasing risks. Coefficient tuning toward this benchmark is not evidence of better general rendering.
- Fixed-orientation, zero-jitter rank patterns can make filtered **standard deviation** nearly zero while leaving large mean bias. Example `L1-fixed-zero`: at frame 1, mean filtered standard deviation .0024932 but uniform RMSE .0591334, and threshold RMSE .1283826. Such spectacular-looking ordered-pattern results are rejected as replacements. All results remain in the JSON to make this failure visible.

No downloaded texture or per-frame generation is required by these phase formulas. The script does not measure GPU cost or validate rendering artifacts. Do not promote a winner solely from this Gaussian uniform-field score.
