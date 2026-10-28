# Temporal schedule ablation

`temporal.py` preserves the original 3LHW spatial phase and changes only its animation. It evaluates two hierarchy seeds, starts 0, 13, and 97, powers of two and additional 3, 7, 31, and 100 sample endpoints. `temporal-results.json` contains raw and Gaussian-filtered uniform-field standard deviation plus filtered threshold RMSE at probabilities 0.1, 0.5 and 0.9. IGN retains the original seed-1 implementation in these runs. These are CPU reference quality measurements, not runtime benchmarks or independent evidence of rendering quality.

Changing the temporal schedule alone cannot improve frame 1. Every candidate retains the same spatial phase. In this experiment, no temporal replacement consistently fixes the early-frame IGN gap while retaining all later advantages.

| Schedule, seed 1/start 0 | 1 | 4 | 16 | 64 | 128 | 1024 |
|---|---:|---:|---:|---:|---:|---:|
| Animated IGN | 0.0294 | 0.00991 | 0.00336 | 0.00137 | 0.000939 | 0.000205 |
| Original golden Weyl | 0.0497 | 0.0187 | 0.00459 | 0.00184 | 0.000999 | 0.000109 |
| Silver Weyl | 0.0497 | 0.0172 | 0.00618 | 0.00128 | 0.000746 | 0.000102 |
| Additive radical inverse | 0.0497 | 0.0296 | 0.0124 | 0.00384 | 0.00198 | 0.000245 |

Values are sigma-1 filtered standard deviations. Silver is the fixed-point step approximating sqrt(2)-1 (`1779033703`); it improves some endpoints and worsens others, including 16 frames. This is not a Pareto improvement. The sqrt(3)-1 step, fixed dyadic strata interleaved with Weyl rotations, and spatially varying signs of the golden step also fail to improve the early range consistently. Spatial sign changes are hashed at pixel, 2x2, and 8x8 scales; they preserve the frame-zero phase.

Radical-inverse sequences demonstrate the distinction between per-pixel discrepancy and spatially filtered error. At 16 samples, additive radical inverse has lower **raw** standard deviation than golden Weyl (0.0180 vs 0.0243), yet nearly three times worse **filtered** deviation (0.0124 vs 0.00459). Dyadic temporal integration cancels the spatially well-stratified high digits, leaving the lower tile-hash digits as a coherent spatial residual. Temporal stratification alone does not make that residual blue.

`complement-Weyl` is a diagnostic, not a recommended replacement. Its odd sample is the exact 24-bit complement of the preceding even sample. This makes uniform-field linear-mean standard deviation zero at every even endpoint, and balances the 0.5 threshold. It remains an ordinary uniform-marginal antithetic sampler, but this benchmark fits it exceptionally well. At 16 samples its 0.1-threshold filtered RMSE is 0.00960, worse than golden Weyl's 0.00714 and IGN's 0.00498. At 1024 it is 0.000218, worse than golden Weyl's 0.000182. A spectacular linear-mean curve would not establish a general sampling improvement. Shifted starts and odd endpoints expose its dependence on complementary pair completion.

Recommendation: retain the inexpensive golden temporal progression while improving the spatial phase or its local phase correlations. Do not select a different Weyl constant merely because it lowers a few powers-of-two endpoints. Before production, evaluate nonlinear integrands, threshold probabilities, moving/reprojected histories, and several starts rather than only the mean of scalar samples.
