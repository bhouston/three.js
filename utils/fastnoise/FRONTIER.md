# Research round: trying to improve FAST separate broadly

## Frozen evaluation target

Before examining the new candidates, freeze suite `frontier-v1`: the original
ten `compare.py` metrics, plus 60 filtered RMSE metrics from 15 analytic
integrands at 1/4/32 frames and EMA alpha .1, and four aggregate threshold tests
at held-out spatial filter widths .5 and 2. All 74 are lower-is-better.
Integrands: identity, squared, fourth power, cosine; thresholds
.1/.25/.5/.75/.9; held-out thresholds .07/.19/.37/.63/.83/.94. The optimizers
use midpoint threshold grids, not a list of these held-out test thresholds.

Use 128×128×32 scalar volumes, quantize to 8-bit midpoints, two seeds, all 32
starting phases for integrand RMSE, original cycle offsets and wrapped filters.
Record bias diagnostics separately; primary RMSE already includes bias.
Primary reference is the unmodified 10,000-step FAST separate constructor for
the same seed, not the vector R atlas component. Bias of finite quadrature is
not removed. Dataset metadata and every regression must be retained.

Success means strictly smaller values in **all 74 metrics on both seeds**;
averages, weighted scores or improved subsets are not called dominance. This
is an empirical finite-suite target, not a claim about every rendering problem.
Fixed-distribution quantities such as the variance of an unfiltered single
uniform sample cannot be strictly improved while retaining that distribution;
they are not meaningful optimization targets and are not among these 74.

Generation cost is measured separately. Scratch constructors and FAST-plus-
refinement constructors must be distinguished. A refinement's total cost
includes its input FAST volume, not only the polishing step. Preserve the
original generator and previous experiments, and do not replace shipped assets.

## Hypotheses

1. Multi-scale cached CDF energy may improve several nonlinear integrands
   without the raw-value-only failure. Exact cyclic EMA and Gaussian kernels
   replace the earlier short binomial approximation. Try both scratch and FAST
   warm starts; histogram-preserving swaps keep the distribution fixed.
2. Temporal-stratum-preserving refinement may retain full-cycle quadrature
   quality while optimizing spatial and short-window error. The hard constraint
   may also prevent good spatial arrangements; check all short-window metrics.
3. Playback-aware GPU loss accounts for the spatial shift at the depth seam,
   which the original toroidal objective ignores. A joint spatial-temporal
   kernel mixture may target the reconstructed error more directly. Both can
   hurt the original unshifted metrics, so the strict comparison is essential.

The attempts build on the
[FAST objective](https://www.ea.com/news/spatio-temporal-sampling) and
[STBN's space/time distinction](https://arxiv.org/abs/2112.09629), with the
[earlier integrand-aware comparison](PR116-COMPARISON.md) as motivation. They
are research hypotheses, not claims of novel or superior samplers.
