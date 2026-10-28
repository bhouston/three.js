# Comparison with PR #117 and interpretation of the chart

The exceptional chart result is a real reduction in filtered **sample-value**
error. It is not evidence of a generally superior graphics sampler, or a new
sampling state of the art. This distinction is demonstrated by both PRs and by
the common-protocol tests below, not merely a theoretical concern.

## What the two experiments found

This branch, [PR #116](https://github.com/bhouston/three.js/pull/116), is compared
with [PR #117](https://github.com/bhouston/three.js/pull/117), source pinned to
`502a026ac59540f305d6f05fa2e8a53c1301815a`. Its source and report were read,
and all three default seed-1 volumes were regenerated for this comparison.

| Family | PR #116 | PR #117 | Interpretation |
| --- | --- | --- | --- |
| Filter values, restore histogram | Relaxation: eight rounds, five-tap spatial/temporal filters, approximate 8,192-bin CDF remap | Filter/rank: twelve rounds, three-tap filters and blended spatial/temporal high-pass product, 65,536-bin ordering and exact discrete global marginal | Same broad spectral-shaping family. Both specialize in cancelling raw values; #117 pushes that advantage further, while worsening important transformed-value tests. |
| Hierarchical ordering | Locally scrambled hierarchical spatial digits plus temporal bit reversal | Scrambled Bayer-style spatial ranks plus temporal bit reversal; exact per-slice histogram, power-of-two restrictions | Closely related inexpensive stratification, with different spatial structure. Both exhibit favorable dyadic thresholds and unfavorable other cases. |
| More balanced sampling attempt | Optimize a 2D circular-phase shift tile, then use rank-1 temporal progression | Greedy cached energy updates for fifteen binary threshold masks, 32 attempts per voxel | Distinct ideas. #117 targets integration errors more directly; our lattice is much cheaper to construct. Neither dominates the common quality tests. |

#117's cached method uses a short spatial autocorrelation plus a truncated
temporal EMA autocorrelation, alpha .25/radius 4, and thresholds `(c+.5)/15`.
It approximates FAST's integrand-aware optimization, rather than duplicating its
sigma-1/alpha-.1 full loss. It spends more CPU time and cache memory to obtain
more balanced threshold errors than its filter/rank method.

The original timing reports are **different machines/backends**: our physical
AMD GCN-5 versus #117's physical NVIDIA GTX 1050 for FAST. This follow-up does
not rebenchmark generation or compare their reported speedups directly. Our
original lattice times are about .09–.11 seconds, while #117 reports about
5.4 seconds for cached threshold energy on its machine. Construction cost is
also distinct from runtime sampling cost: both ultimately produce textures.

## A common numerical comparison

The PRs' original threshold EMA numbers cannot be combined directly: #117's
scorer uses unshifted periodic EMA, while our extended scorer shifts frozen
volumes each cycle and evaluates every starting phase. I regenerated all six
default seed-1 volumes and ran them through **one** scorer with one reference.

- 128×128×32; quantize to 8-bit midpoint values; wrap spatial coordinates.
- Known-integral RMSE, including bias, after sigma=1/radius=3 spatial blur.
- Average squared errors over all pixels and all 32 temporal starting phases.
- Frozen-volume access shifts by (-71,-37) per completed 32-frame cycle.
- Box means at 1/4/8/16/32/64 frames; EMA alpha .1 over four cycles, averaging
  errors in the last cycle. Initialization is the volume's temporal mean.
- Threshold aggregate: RMS over .1/.25/.5/.75/.9; off-grid aggregate:
  .17/.37/.63/.83. These extra thresholds are not on #117's fifteen-point
  training grid, but are near grid points and are only a limited holdout.
- Also test identity, x², x⁴, cos(2πx), and cos(8πx). Quantization is applied
  before each function; known continuous integrals retain quantization bias.
- FAST reference: shipped separate atlas **B scalar channel**. This is the
  scalar reference from our integration tests, not the legacy chart's R
  vector-component curve. White is a fixed seeded volume with the same access.

All values below are lower-is-better RMSE, not generation times. The identity
column uses the same all-phase RMSE protocol as the transformed-function columns,
so the comparison does not depend on selecting frame zero or hiding mean bias.

| Method | Identity, 32 frames | Five thresholds, 32 frames | Five thresholds, EMA | Off-grid thresholds, EMA | x², 32 frames | cos(2πx), 32 frames |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 116-hierarchy | 0.0085769 | 0.0087203 | 0.0120710 | 0.0157376 | 0.0085830 | 0.0020788 |
| 116-lattice | 0.0041350 | 0.0062762 | 0.0100247 | 0.0092245 | 0.0041433 | 0.0023779 |
| 116-relaxation | 0.0004725 | 0.0163051 | 0.0209639 | 0.0214194 | 0.0058411 | 0.0529796 |
| 117-filter-rank | 0.0001966 | 0.0221089 | 0.0276298 | 0.0287080 | 0.0090555 | 0.0802016 |
| 117-hierarchy | 0.0076313 | 0.0078622 | 0.0108239 | 0.0146325 | 0.0076439 | 0.0036608 |
| 117-local-energy | 0.0022595 | 0.0076761 | 0.0102271 | 0.0100404 | 0.0025975 | 0.0101369 |
| FAST-scalar | 0.0014134 | 0.0052813 | 0.0077785 | 0.0079121 | 0.0015641 | 0.0054185 |
| White | 0.0142474 | 0.0199075 | 0.0258965 | 0.0279378 | 0.0146904 | 0.0352031 |

Complete values, including bias and unblurred errors, are in
[research-comparison.json](results/research-comparison.json). This is one seed,
one filter family and synthetic spatially constant integrands, not rendered
scene validation or a statistically comprehensive ranking.

Observations under this shared protocol:

- Our relaxation beats FAST's identity error at 32 frames by about 3x, but
  loses by about 3.1x on the five-threshold aggregate and by about 9.8x on
  cos(2πx). The loss on that cosine is worse than white noise. The original
  phase-zero scalar standard-deviation metric showed about 6x improvement;
  random starting phases and volume seams reduce that particular advantage.
- #117's filter/rank beats FAST identity error by about 7.2x, yet its threshold
  aggregate at 32 frames is about 4.2x worse than FAST and 11% worse than white.
  Its cos(2πx) error is about 14.8x worse than FAST. Stronger raw suppression
  therefore does not resolve the sampling problem.
- Our lattice and #117's cached-energy method are much more balanced. Their
  five-threshold EMA errors are respectively about 29% and 31% above FAST;
  their off-grid EMA errors are about 17% and 27% above FAST. Lattice wins
  those tests and the tested cosine functions; cached energy wins identity
  and x² error. The full JSON also contains x⁴ and other frame counts.
- Hierarchies remain threshold-dependent, with visible ordered structure.
  Their very cheap generation does not imply general blue-noise error quality.

## Why the apparently spectacular improvement happens

The chart computes a filtered average of `u`, where `u` is the stored sample.
This is essentially testing the single integrand `f(u)=u`, whose uniform
integral is 1/2. High-pass/remap algorithms explicitly encourage neighboring
values and nearby times to cancel around that mean. Low-pass filtering then
removes the residual high-frequency fluctuations very effectively.

Graphics sampling normally computes `f(u)`: visibility, a lighting contribution,
transmittance, a warped direction or a threshold/quantization decision. The
covariance and spectrum of `f(u)` need not resemble those of `u`. Nonlinear
operations can destroy the anticorrelations that made the values cancel.

There is also a direct relationship: `u = integral_0^1 1[u > t] dt`. A small
identity-integration error can result from positive and negative threshold
errors cancelling as t varies. It does not imply that the squared errors for
individual thresholds are small. Optimizing the raw average and optimizing
error across a class of thresholds are therefore different objectives.

A simple example illustrates this without any implementation details. Draw a
uniform `u`, then pair it with `1-u`. Their mean integrates `f(u)=u` exactly.
For `g(u)=(u-1/2)^2`, however, the two evaluations are identical: averaging
them gives no variance reduction relative to one evaluation. Both samples
individually remain uniform. A perfect result for one function does not
establish uniformly better integration for other functions.

Thus the chart is numerically valid, but it is a **specialized benchmark**.
The constructors were not fitted to its plotted measurements; their operators
nevertheless align closely with its raw-value low-pass objective. This is
objective specialization, and presenting the winner as a general-purpose
sampling advance would overgeneralize the benchmark. The common transformed
tests supply concrete counterexamples.

An additional reason for differing outcomes is that FAST spends its optimization
budget on a broader class of integrands and chosen filters. It is not trying to
win every conceivable statistic of the stored values. FAST and scalar STBN
are also distinct research methods; the chart's named reference is FAST
separate, not an exhaustive comparison with every modern sampling technique.

## Is this useful in the graphics industry?

Potentially, where the actual operation is additive scalar noise followed by
similar **linear** averaging/filtering. The chart measures a meaningful property
for that narrow signal-processing use, and cheap procedural construction may
be useful. A production claim would still need tests for anisotropy, spectra,
filter width, tiling, seed variation and visual appearance in the intended use.

It does not currently justify replacing FAST/STBN in AO, shadow visibility,
stochastic transparency, ray marching or general path tracing. Thresholding,
nonlinear warping, multiple sample dimensions, changes across pixels, history
rejection and camera/object motion can invalidate the cancellation. Even
"dithering" must be qualified: additive noise averaged directly is different
from using that noise to choose a quantization or threshold outcome.

Correlated samples are not automatically biased. Uniform marginals can give
an unbiased Monte Carlo estimator despite correlation. But an exact global
histogram across one finite volume does not prove uniform randomized marginals
at every pixel/frame, nor good joint multidimensional distributions. Neither
branch establishes general unbiased rendered-image behavior from its tests.
Low finite-sample error and asymptotic convergence are also separate questions.

## Is this discussed in the literature?

Yes; neither the spectral-shaping family nor the distinction between sample
spectra and useful integration error is new.

- [Wronski, 2020](https://bartwronski.com/2020/04/26/optimizing-blue-noise-dithering-backpropagation-through-fourier-transform-and-sorting/)
  discusses high-pass-and-renormalize methods and their limitations, then
  explores Fourier-spectrum optimization with histogram constraints. This is
  directly related to the family explored by relaxation and filter/rank;
  neither PR reproduces his Fourier-gradient method.
- [Belcour and Heitz, 2021](https://belcour.github.io/blog/supp/2021-sampling-bluenoise/)
  evaluate blurred integration errors using Heaviside functions, validate
  rendered scenes, and discuss eigenintegrands: changing the sampler changes
  which functions account for its errors. This is why a single favorable
  integrand is insufficient evidence of general sampler quality.
- [FAST, Donnelly et al., 2024](https://arxiv.org/abs/2310.15364)
  derives filtered error over multidimensional Heaviside integrands, adapting
  samples to reconstruction filters and sample distributions. Its validation
  includes AO, volumetric ray marching and image dithering, substantially
  broader evidence than either PR's raw-value chart.
- [Scalar STBN, Wolfe et al., 2022](https://arxiv.org/abs/2112.09629)
  addresses spatial and temporal sampling behavior together and evaluates
  rendering applications; ordinary 3D spectral suppression alone does not
  establish the desired blue spatial slices and temporal integration behavior.

Optimizing for a renderer's filters is legitimate and central to FAST itself.
The missing step here is showing generalization from our simplified objective
to the relevant integrands, scene variation and reconstruction pipeline.
Neither PR demonstrates superiority over the state of the art. The defensible
result is faster texture construction with application-dependent tradeoffs.

## Reproduction and validation

The six inputs are default `generate({seed:1})` Float32 outputs from this
branch's `alternatives/{hierarchy,lattice,relaxation}.js` and the pinned #117
commit's `methods/{filter-rank,hierarchy,local-energy}.js`. Save them in one
directory as `116-hierarchy.f32`, `116-lattice.f32`, `116-relaxation.f32`,
`117-filter-rank.f32`, `117-hierarchy.f32`, `117-local-energy.f32`.
All inputs are 128×128×32 little-endian z/y/x floats; their SHA-256 digests
are recorded in the JSON. The scorer loads shipped FAST and seeded white itself.

```sh
python utils/fastnoise/compare-research.py comparison-volumes research-comparison.json
```

Requires NumPy/Pillow. Source from #117 can be read at the pinned commit with
`git-dedup show` or checked out with `git-dedup worktree add` (PowerShell:
`git-dedup.cmd`). No implementation on #117 was modified or merged here.

Validation: all six volumes have the expected shape and finite [0,1) values;
our seed-1 hashes and all ten original quality metrics match the previously
committed results. All eight methods have identity, five standard thresholds,
four off-grid thresholds, x², x⁴ and both cosine tests at every reported frame
count and under EMA. Repository core lint and generator invariant tests pass.
No typecheck script exists. This research follow-up does not modify renderer
code or claim GPU-rendered validation; the earlier full core-suite backend
failures remain documented in ALTERNATIVES.md.
