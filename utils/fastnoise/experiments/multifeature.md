# Multifeature projected gradient experiment

This generator minimizes local low-frequency errors for several transformations
of each scalar sample, rather than only its identity value. The features are
`cos(k*pi*u)` for `k=1..3`, with weights proportional to `1/k²`. These are the
Neumann cosine eigenfunctions associated with integrated threshold discrepancy:
retaining several modes approximates an integrand-aware threshold objective.
The finite truncation remains a strong approximation and does not protect all
thresholds or high-frequency functions.

Each iteration computes feature-space spatial and temporal low-pass correlations,
takes a gradient step in scalar sample space, and restores a globally uniform
discrete marginal using stable 65,536-bin rank ordering. Spatial and temporal
kernels are separate periodic five-tap `[1,4,6,4,1]/16` filters; the spatial
filter is separable. They approximate local discrepancy energy, not FAST's full
Gaussian/EMA energy. A 4,096-entry feature table avoids trigonometric evaluation
per voxel. Global rank restoration has bucket-order ties; it preserves the
global marginal exactly but does not guarantee an unbiased randomized sampler
or per-pixel/per-slice marginals.

`generate({size=128,depth=32,seed=1,iterations=48,features=3,step=.3,
temporalWeight=1,initial=null})` returns z/y/x Float32 values in `[0,1)`.
`initial` optionally supplies an existing Float32Array, copied before optimization.
The default starts from seeded white noise. With three features, the held-out
`cos(8*pi*u)` is not explicitly optimized. More features are supported for
exploration, but are not a holdout-safe default.

## Preliminary result: does not beat FAST

For 128×128×32, seed 1, default construction took about 1.2 seconds in the initial
CPU run on MediaPC. A single timing is diagnostic, not a stabilized benchmark.
Increasing to 192 iterations took about 7 seconds and did not fix held-out error.
Small 8×8×4 checks passed determinism, finite range, and exact sorted marginal
`(rank+.5)/count`.

The following quick diagnostic uses the existing all-32-start-phase scorer's
sigma-1 spatial blur and 32-frame means, including the cycle shifts and 8-bit
midpoint quantization. Lower values are better RMSE, including bias:

| Construction | Identity | Threshold .37 | cos(8*pi*u) |
| --- | ---: | ---: | ---: |
| Default 3 features / 48 iterations | .0017954 | .0103593 | .0403403 |
| 3 features / 192 iterations | .0015610 | .0104180 | .0421118 |
| 8 features / 48 iterations, cosine now trained | .0020170 | .0085043 | .0196556 |
| Shipped FAST separate scalar | .0014134 | .0053524 | .0162269 |

This is a negative experiment, not a new state-of-the-art claim. Multi-integrand
optimization avoids the raw-only objective conceptually, but a low-order smooth
basis and short temporal kernel do not reproduce FAST's broad threshold quality.
Increasing feature count improves nonlinear functions while worsening identity,
illustrating the tradeoff rather than establishing a winner. The default three
features overfit low-order smooth functions sufficiently to make held-out cosine
error substantially worse. Gradient/rank projection is nonconvex, and more
iterations alone do not establish general improvement. No rendered-scene
superiority or hardware rendering validation is claimed.
