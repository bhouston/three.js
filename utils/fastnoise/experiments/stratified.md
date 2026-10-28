# Cyclic Latin-hypercube threshold optimization

`generate({ size=128, depth=32, seed=1, passes=64, temporalWeight=.5, temporalAlpha=.25, temporalRadius=4, thresholdCount=31, cycleShift=false })` returns z/y/x `Float32Array` values in [0,1). Browser and Node compatible, no dependencies. `passes:0` returns initialization. `thresholdCount:0` selects a raw-value quadratic surrogate. Optional `cycleShift:true` incorporates the comparison harness's +71/+37 spatial translation when the temporal kernel crosses a cycle boundary; use it only with the matching consumer.

Every pixel begins with one independently jittered sample per temporal stratum and a random permutation. Candidate swaps exchange temporal positions at the same pixel. Consequently each pixel retains its complete original Latin hypercube, even after optimization. This gives a distributional constraint rather than merely encouraging one in a weighted objective. All cyclic temporal phases participate in the stationary quadratic energy, rather than optimizing only a phase-zero prefix.

The objective sums filtered Heaviside-mask energies over `(c+.5)/thresholdCount`, using the autocorrelation of a 3x3 binomial spatial filter plus the autocorrelation of a truncated temporal EMA. Cached potentials permit exact greedy acceptance for this finite objective. The objective is inspired by FAST's filter-aware sample optimization and PR117's cached threshold-energy implementation; its additional hard constraint preserves per-pixel temporal stratification. It remains an experiment, not a new general state-of-the-art claim.

A screen at 128x128x32, seed1, 128 passes, 31 thresholds, temporalWeight=.5, periodic seam took 38.2 seconds in Node on this runner (concurrent tasks may influence time). The default halves the passes. Using the established 8-bit bin-center quantization and existing compare protocol:

| Diagnostic | FAST scalar baseline | This 128-pass screen |
|---|---:|---:|
| spatial low-frequency ratio | .003042 | .007425 |
| temporal low-frequency ratio | .154208 | .125010 |
| blurred scalar error, 1 frame | .025209 | .017223 |
| blurred scalar error, 4 frames | .007762 | .006189 |
| blurred scalar error, 32 frames | .000583 | .000450 |
| unblurred scalar error, 32 frames | .003976 | .001586 |
| scalar EMA | .023825 | .024310 |
| blurred scalar EMA | .002283 | .001967 |
| blurred threshold .1, 32 frames/all phases | .005345 | .005800 |
| blurred threshold .25, 32 frames/all phases | .005210 | .004398 |
| blurred threshold .5, 32 frames/all phases | .005238 | .004203 |
| blurred threshold .1, EMA/all phases | .007733 | .008661 |
| blurred threshold .25, EMA/all phases | .007764 | .007558 |
| blurred threshold .5, EMA/all phases | .007836 | .007241 |

This improves a broad subset of diagnostics but does **not** dominate FAST: spatial spectral suppression, unblurred EMA, and tail threshold cases remain worse. These are one-seed tuning screens, not held-out proof. No rendered-scene validation has been performed. Increasing passes from 12 to 128 materially improved spatial quality while retaining the exact same per-pixel full-cycle mean.

Limitations: finite threshold grids; a spatial kernel differing from the evaluation Gaussian; independent spatial and temporal objectives rather than every composed filter; Latin-hypercube constraints limit optimization freedom; the consumer's spatial cycle shifts break same-pixel stratification for windows crossing the cycle seam; finite-cycle structure and potential correlations across dimensions. The optional helical kernel produced similar scalar results but did not resolve the tail-threshold weakness in a 64-pass screen. Sample coordinates and final estimators require independent validation, especially for discontinuous or multidimensional integrands.

Validation: deterministic repeatability, finite range, exact preservation of every pixel's original sample multiset, and exactly one sample per temporal stratum checked for size 1/3/8, depth 1/3/8, both seam modes (54 cases, three passes). No rendered GPU validation is implied.
