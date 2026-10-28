# IGN phase and Weyl exploration

This is a CPU reference screen, not a graphics throughput measurement. Run `ign-phase.py` with the same NumPy environment as `compare.py`. Full results retain every attempted method, two independent seed/window holdouts, off-power-of-two frame counts, four off-grid/dyadic thresholds and the nonlinear integrand `u²`. No method uses offline assets, generation, value averaging, or frame-dependent switching between estimators.

The leading simple candidate is `fract(IGN(x,y,0) + n*goldenRatio)`: preserve IGN's strong spatial arrangement while replacing its scrolling time sequence with a low-discrepancy per-pixel sequence. An integer approximation of the inner spatial IGN lattice produces almost identical results. It does **not** solve the early-prefix problem completely.

| Frames | Animated IGN | 3LHW | IGN phase + Weyl | 16-frame IGN block + Weyl |
|---|---:|---:|---:|---:|
| 1 | .0294 | .0497 | .0293 | .0294 |
| 4 | .00991 | .0187 | .0139 | .00991 |
| 8 | .00603 | .0107 | .00800 | .00603 |
| 16 | .00336 | .00459 | .00490 | .00336 |
| 32 | .00222 | .00288 | .00233 | .00231 |
| 64 | .00137 | .00184 | .00115 | .00154 |
| 128 | .000939 | .000999 | .000710 | .000954 |
| 256 | .000603 | .000478 | .000349 | .000495 |
| 512 | .000278 | .000293 | .000173 | .000271 |
| 1024 | .000205 | .000109 | .000103 | .000131 |

These are Gaussian-filtered arithmetic-mean standard deviations, seed 1, start frame 0. Phase+Weyl's improvements persist for seed 2/frame 17 and seed 7/frame 113: 1024-frame error is 49% and 48% of corresponding Animated IGN, and squared-integrand RMSE is about 34%. Its 16-frame error is 45–46% worse, so it trades short-prefix quality for long convergence instead of dominating.

The block candidate uses `fract(IGN(x,y,n%16) + floor(n/16)*goldenRatio)`. It exactly retains the original first 16 samples when the window starts at frame 0, but that is partly window alignment: starting at frame 17 or 113 makes 16-frame error 27–28% worse than Animated IGN. 1024-frame performance remains about 64% of IGN. Do not market exact early equality as a general property.

Other retained failures: inner-IGN Weyl, block sizes 4/8/32, radical inverse instead of Weyl, and scrolling Animated IGN plus outer increments 1/2, 1/4, golden ratio, silver ratio and 1/3. None delivers broad domination. Radical inverse has excellent late power-of-two results but poor 4–16 frame results; this is progressive stratification with strong sample-count alignment, not evidence of improved spatiotemporal blue-noise spectra.

All arithmetic transformations preserve a scalar sample rather than averaging sample values (which would bias sampling). As with existing IGN, its noninteger outer multiplier makes the underlying marginal approximately, rather than perfectly, uniform. A genuinely randomized outer Cranley–Patterson shift can enforce a uniform ensemble marginal, but this screen does not claim that its fixed seed is such an unbiased ensemble.

The causal distinction is spatial phase quality versus temporal discrepancy: 3LHW gives a strongly stratified hierarchy with random coarse-cell phase and residual within-cell jitter; it does not optimize the Gaussian-filtered short prefixes. IGN has a well-spread small-neighborhood pattern, so its spatial filter sees less early error. A Weyl progression guarantees neither a particular first 4/8/16 sample pattern nor spatially blue error after thresholding. At larger frame counts its temporal coverage can overcome that weaker spatial start. There is no universal 128-frame crossover: even the existing 512-frame result reverses it, and other functions/filter widths/window starts change it.
