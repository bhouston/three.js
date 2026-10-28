# Zero-download runtime comparison

All entries are RMSE ratios to the shipped FAST scalar B reference; lower is better. Aggregates retain bias and pool both seeds. Thresholds: nine values, including .07/.37/.83/.94; smooth functions: u²/u⁴/cos(2πu)/sin(14πu). Ramp/sphere are spatially varying analytic threshold images. All-phase windows, actual continuous frame indices, wrapped sigma1 spatial filtering. See [RUNTIME.md](RUNTIME.md) for limitations.

| Method | Identity 32 | Threshold 1 | Threshold 32 | Threshold 128 | Threshold EMA | Smooth 32 | Ramp/sphere 32 | Worst / 160 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| white | 10.248 | 2.167 | 3.575 | 3.793 | 3.167 | 2.274 | 3.803 | 16.773 |
| animated-ign | 1.579 | 0.880 | 0.622 | 0.571 | 0.660 | 0.657 | 0.693 | 2.202 |
| r2-weyl | 2.183 | 0.808 | 0.856 | 0.489 | 1.012 | 0.296 | 0.745 | 2.183 |
| hashed-digital | 1.281 | 2.171 | 0.543 | 0.295 | 1.215 | 0.168 | 0.655 | 3.217 |
| hierarchical-digital | 3.464 | 1.182 | 1.328 | 0.860 | 1.466 | 0.300 | 1.267 | 3.836 |
| houston-stratified-noise | 2.046 | 1.328 | 0.734 | 0.557 | 0.971 | 0.241 | 0.810 | 2.334 |
| phase-16-p8 | 2.245 | 1.302 | 0.896 | 0.592 | 1.153 | 0.361 | 0.837 | 2.470 |
| phase-32-p0 | 2.638 | 2.113 | 1.015 | 0.608 | 1.342 | 0.414 | 0.987 | 3.214 |
| phase-32-p2 | 2.355 | 1.492 | 0.932 | 0.581 | 1.174 | 0.376 | 0.882 | 2.550 |
| phase-32-p8 | 2.291 | 1.328 | 0.899 | 0.584 | 1.139 | 0.354 | 0.851 | 2.454 |
| phase-64-p8 | 2.272 | 1.327 | 0.901 | 0.580 | 1.145 | 0.361 | 0.853 | 2.428 |
| phase-128-p8 | 2.256 | 1.343 | 0.898 | 0.584 | 1.152 | 0.364 | 0.853 | 2.458 |
| FAST-scalar | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
