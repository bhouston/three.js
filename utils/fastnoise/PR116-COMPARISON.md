# What the chart winner means, and comparison with PR #116

**Filter/rank is better at the chart's particular task, not established as a
better general graphics sampler.** The dramatic gap is a real numerical result
for directly filtered sample values. It is not evidence of a state-of-the-art
rendering advance. Fresh common-protocol tests below reverse that ranking for
nonlinear functions, including functions smoother than visibility thresholds.

## Comparison of the two investigations

Reviewed [PR #116](https://github.com/bhouston/three.js/pull/116), its report,
implementations and scorer at commit `54a6dfe336cd33fc1fe129e4e1e6b10900395832`.
Our methods are from PR #117 at `502a026ac59540f305d6f05fa2e8a53c1301815a`.

| PR #116 | PR #117 | Interpretation |
| --- | --- | --- |
| Hierarchical strata + digital temporal progression | Scrambled Bayer + temporal XOR | Related inexpensive dyadic constructions; both leave structured correlations and threshold-dependent results |
| Alternating spatial/temporal high-pass + approximate histogram CDF restoration | Product high-pass + exact global rank projection | Closely related value-spectrum shaping; both look exceptional for raw averages and fail to generalize to arbitrary functions |
| Optimized circular spatial shifts + rank-1 temporal lattice | Cached local quadratic energies for 15 threshold masks | Different useful directions: cheap progression/phase construction versus explicitly optimizing a finite set of binary-integrand energies |

PR #116 has a stronger original evaluation than our initial one: it tests x²,
x⁴ and all 32 start phases, rather than only raw values and fixed-phase
thresholds. Its lattice is its best threshold compromise. Our local-energy
method is our most balanced original compromise. The two winners depend on
the integrand; neither dominates the following common tests.

Published generation timings are not directly comparable across these PRs:
#116 used an AMD Ryzen/GCN system; #117 used a GTX 1050 Windows system. Both
compare CPU constructors with a GPU FAST reference. No new timing ranking is
claimed here, and neither report establishes equal-quality acceleration of FAST.

## Fresh common-protocol check

Regenerated all six methods from seed 1, quantized to the same 8-bit midpoints.
Added the original shipped FAST scalar **B** channel, vector-component **R**
channel, and a common frozen white volume. Evaluate analytic integrands over
all pixels and all 32 starting phases, with the same 32-frame volume shifts
and wrapped sigma-1 Gaussian. Errors below are **32-frame RMSE**, including
bias, against the known integral. This is not the phase-zero standard deviation
shown in the earlier chart. These tests are numerical, not rendered scenes.

| Method | f(u)=u | Five-threshold aggregate | u² | u⁴ | cos(2πu) |
| --- | ---: | ---: | ---: | ---: | ---: |
| FAST scalar B | .001413 | .005281 | .001564 | .001812 | .005419 |
| #116 hierarchy | .008577 | .008720 | .008583 | .008518 | .002079 |
| #116 lattice | .004135 | .006276 | .004143 | .004101 | .002378 |
| #116 relaxation | .000472 | .016305 | .005841 | .010077 | .052980 |
| #117 filter/rank | **.000197** | .022109 | .009055 | .015619 | .080202 |
| #117 hierarchy | .007631 | .007862 | .007644 | .007643 | .003661 |
| #117 cached threshold energy | .002260 | .007676 | .002598 | .003232 | .010137 |
| White | .014242 | .019981 | .014768 | .013295 | .035251 |

Thresholds are .1/.25/.5/.75/.9; aggregate is the square root of their mean
squared RMSE. Integrals are respectively 1/2, the threshold, 1/3, 1/5 and 0.
The JSON also retains raw results, 1/4-frame errors and per-phase bias summaries.

Compared with scalar FAST, filter/rank is about **7.2× better** for the identity
integrand, but **4.2× worse** on aggregate thresholds, **5.8× worse** on u²,
**8.6× worse** on u⁴ and **14.8× worse** on cosine. It even loses to white on
aggregate thresholds, u⁴ and cosine. Thus this is not merely an issue with
discontinuous visibility: smooth nonlinear transforms also expose the weakness.

The lattice has 19% higher threshold RMSE than FAST; our cached threshold
method has 45% higher threshold RMSE, but lower u²/u⁴ errors than the lattice.
These relationships are specific to this small test family, not overall rankings.
The new #116 values reproduce that PR's published threshold/u²/u⁴ table to its
printed precision, providing a cross-check of the common protocol.

### A channel distinction worth making explicit

`make_atlas.py` packs an optimized vector2 into RG and a separately optimized
scalar into B. The original chart, and both extended charts, use **R** to match
the original example. The generators here produce scalars. The common study
therefore uses B as the primary scalar reference and reports R separately in
the JSON. This distinction does not invalidate the displayed curves, but
they should not be read as an exact scalar-versus-scalar algorithm comparison.
R's threshold aggregate in this test is .008244 versus B's .005281. Channel,
phase, integrand and metric differences must be controlled before ranking methods.

## Why averaging the numbers gives a spectacular result

The chart effectively integrates just **f(u)=u**, whose answer is 1/2. Our
filter/rank operator explicitly removes low spatial and temporal frequencies
from the values u, then restores a global uniform histogram. Averaging and
Gaussian blur remove high frequencies, so this construction is closely aligned
with that test. Negative correlations cancel the first moment very effectively.

A renderer generally uses u to choose a ray, light, lens point, distance or
visibility decision. Its contribution is f(u), or f(u)/p(u) under importance
sampling, not u itself. Small filtered error in u does not imply small filtered
error in f(u). Nonlinear transforms change the spectrum and its correlations.
In addition, neighboring pixels usually have different integrands, whereas the
chart uses the same uniform-field problem everywhere.

An exact counterexample makes the distinction clear. Draw uniform U, then use
the pair U and 1-U. The average is exactly 1/2: it beats every nonzero-error
result in our raw-value chart after only two samples. Yet for f(u)=cos(2πu),
the pair produces identical contributions. Their estimator has variance 1/2,
while two independent samples have variance 1/4. Perfect raw-value cancellation
can coexist with twice the variance for another smooth integrand. This is an
illustration of integrand-dependent correlations, not a claim that our actual
volume consists of explicit antithetic pairs.

In general the variance of a correlated average contains covariance terms:

```
Var(mean(f(U_i))) = (sum_i Var(f(U_i)) + 2 sum_{i<j} Cov(f(U_i),f(U_j))) / N²
```

Making the covariance negative for f(u)=u says little about its sign or size
for another f. Even proving uniform marginal samples would establish neither
independence nor low variance for arbitrary integrands.

## Is it artificial, useful, or novel?

It is **benchmark specialization**, not fabricated data. We did not optimize
the ten plotted frame counts individually, but the method shapes the same
kind of value field and frequency band that the chart measures. A different
integrand is a more meaningful generalization test than another frame count.
Standard deviation also omits spatially constant bias; the new tests use RMSE
against analytic truth. Changing blur, temporal weighting, start phase, sampling
dimensions or scene can change the ordering.

Potentially useful applications are additive scalar perturbations where those
values themselves are filtered, or a specific sampling problem whose actual
transformed contribution has been validated. Even additive dither can pass
through nonlinear quantization, so the word "dither" alone is not sufficient.
Using this volume as random input to AO, ray tracing or stochastic transparency
is possible, but these measurements do not justify recommending it over FAST.
It cannot simply replace a multidimensional sampler: copying a scalar into
several channels does not create a correct joint sampling distribution.

An exact global discrete histogram is not by itself a proof of per-pixel
uniformity over random seeds or unbiased continuous Monte Carlo estimation.
It guarantees a uniform histogram when choosing a voxel uniformly. A rendering
use needs the correct marginal and joint distributions or appropriate weights;
correlations must then be assessed for that integrand and reconstruction filter.

No general superiority over state of the art, universal convergence theorem,
or novelty claim is supported. FAST itself is filter/integrand-family adapted;
it does not claim to minimize error for every possible function. Its broader
objective explains why it can lose the identity benchmark and still win many
more representative rendering proxies.

## Relationship to literature

- [Wronski (2020)](https://bartwronski.com/2020/04/26/optimizing-blue-noise-dithering-backpropagation-through-fourier-transform-and-sorting/)
  explicitly discusses high-pass-and-renormalize constructions, their limitations,
  and spectrum/histogram optimization. Our local filter/rank heuristic belongs
  to this broad family; the general idea is not new.
- [Ren, Zhao and Ermon (2019)](https://proceedings.mlr.press/v97/ren19b.html)
  study antithetic sampling and explain that effective correlation design depends
  on the application. It supplies a useful statistical explanation, not a
  derivation or endorsement of our specific generator.
- [Georgiev and Fajardo (2016)](https://www.iliyan.com/publications/DitheredSampling/)
  shape screen-space **estimation error**, rather than equating raw random-value
  spectra with all integration errors.
- [Belcour and Heitz (2021)](https://belcour.github.io/blog/research/publication/2021/06/24/sampling_bluenoise_sig21.html)
  optimize rank-1 lattice shifts against filtered test-integrand error and discuss
  multidimensional renderer use. This is the closer established direction for
  improving the cheap lattice attempt in PR #116.
- [FAST, Donnelly et al. (2024)](https://www.ea.com/news/spatio-temporal-sampling)
  derives filtered error for a family of Heaviside integrands. This targets
  transformed sampling contributions more broadly than matching the mean of u.
  [STBN, Wolfe et al. (2022)](https://arxiv.org/abs/2112.09629) likewise evaluates
  spatiotemporal masks in graphics applications rather than just a uniform field.

The literature supports integrand-aware error shaping and carefully designed
correlations. It does not turn our low raw-value curve into a general sampling
result or establish our exact implementation as state of the art.

## Practical next step

Keep FAST as the general reference. Explore PR #116's lattice for a very low
construction budget and our cached threshold method for an explicitly optimized
scalar objective. Before adopting either, test held-out integrands and scenes,
multiple seeds and random start phases, correct vector/multidimensional sampling,
different filters and sample counts, motion/disocclusion, and actual image RMSE,
bias and artifacts. A scalar numerical advantage is a candidate for an experiment,
not a production quality claim.

## Reproduction and validation

The committed [JSON](results/pr116-comparison.json) contains all new scores.
No modifications were made to PR #116. Export its three source files as `.mjs`
from the pinned commit, then run:

```powershell
New-Item -ItemType Directory -Force pr116-source | Out-Null
git-dedup.cmd fetch origin alternative-blue-noise
foreach ($method in @('hierarchy', 'lattice', 'relaxation')) {
  git-dedup.cmd show "54a6dfe336cd33fc1fe129e4e1e6b10900395832:utils/fastnoise/alternatives/$method.js" | Set-Content -Encoding utf8 "pr116-source/$method.mjs"
}
node utils/fastnoise/compare-pr116.js pr116-source comparison-volumes
python utils/fastnoise/compare-pr116.py comparison-volumes utils/fastnoise/results/pr116-comparison.json
```

Requires existing Node dependencies and NumPy. The run used the machine's
existing Blender Python/NumPy; no API keys, model providers or software GPU
fallbacks were used. All computations here are CPU numerical diagnostics.
Source review also received an independent sub-agent check. The common
protocol reproduces #116's published values, and original generator invariants
and scoped JS lint pass. All three regenerated #116 volume SHA-256 hashes
match that PR's recorded hashes exactly. Full `npm test` passed: 1,982 core
passes (one TODO), 141 addon passes, zero failures. The existing Windows
hardware-launcher adaptation was temporary and restored; no typecheck script
is defined. No new generation performance or rendered-scene quality claims
are made.
