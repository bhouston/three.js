# FAST / STBN scalar noise textures

Scalar (uniform [0, 1]) spatiotemporal noise from https://github.com/electronicarts/fastnoise (`noise.zip`, BSD-3-Clause, see LICENSE.txt), used by `webgpu_postprocessing_sampling_noise.html`.

Each atlas packs 32 slices of 128×128 into an 8×4 grid (slice `i` at column `i % 8`, row `floor( i / 8 )`), 8 bit grayscale.

| File | Source (`noise/real/...`) | Optimized for |
|---|---|---|
| `fast_binomial3x3_exp.png` | `temporal/exp/real_uniform_binomial3x3_exp0101_product_*.png` | binomial 3×3 spatial × EMA (α = 0.1) temporal |
| `fast_box3x3_exp.png` | `temporal/exp/real_uniform_box3x3_exp0101_product_*.png` | box 3×3 spatial × EMA (α = 0.1) temporal |
| `fast_gauss1_exp_separate.png` | `temporal/exp/real_uniform_gauss1_0_exp0101_separate05_*.png` | Gaussian (σ = 1) spatial, EMA temporal, separate (no spatial filter assumed) |
| `stbn_scalar.png` | `temporal/gauss/real_uniform_gauss1_0_Gauss10_separate05_*.png` | Gaussian spatial + Gaussian temporal, separate (equivalent to STBN) |
| `bluenoise_2d.png` | `real_uniform_gauss1_0.png` | 2D blue noise (single 128×128 slice) |

References: Donnelly et al., "FAST: Filter-Adapted Spatio-Temporal Sampling for Real-Time Rendering", I3D 2024. Wolfe et al., "Spatiotemporal Blue Noise Masks", EGSR 2022.
