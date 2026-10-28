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
4. Pareto-guarded refinement keeps separate spatial and temporal CDF energies
   and feature energies for identity, squared, fourth-power and cosine. A swap
   must not increase any of these ten aggregate objectives. This is stronger
   than a weighted sum, but does not guarantee improvements in all held-out
   thresholds, phases, box windows or spectra; it may stall.

The attempts build on the
[FAST objective](https://www.ea.com/news/spatio-temporal-sampling) and
[STBN's space/time distinction](https://arxiv.org/abs/2112.09629), with the
[earlier integrand-aware comparison](PR116-COMPARISON.md) as motivation. They
are research hypotheses, not claims of novel or superior samplers.

## Outcome: promising seed-1 sweep, unsuccessful independent confirmation

**No tested new constructor met the all-74-metrics, both-seeds target.** The
most promising schedule improved every metric on exploratory seed 1, but six
regressed on confirmation seed 2. Generation was also slower, not faster.
This round therefore does not establish a state-of-the-art improvement.

The light seam schedule is:

1. Generate ordinary FAST separate for 10,000 steps.
2. Run 2,000 additional original-loss steps from that volume.
3. Run 1,000 steps with a 3% playback-aware / 97% original temporal-topology mix.

The last two stages restart keys and suppression as specified by the standalone
research generator; this is a reproducible restarted polishing schedule, not
exactly the uninterrupted 13,000-step baseline trajectory. Swaps stay within
slices, so the original per-frame value histograms are preserved.

| Profile | Seed 1 improved / 74 | Seed 2 improved / 74 | Worst ratio, seed 1 / seed 2 |
| --- | ---: | ---: | ---: |
| Original-loss restart polish, +2,000 steps | 70 | 61 | 1.001218 / 1.006605 |
| Original-loss restart polish, +3,000 steps | 71 | 64 | 1.001063 / 1.007001 |
| Original-loss approximate time control, +4,000 steps | 72 | 66 | 1.002830 / 1.003328 |
| Light seam schedule, +3,000 steps | **74** | **68** | **.999636 / 1.008221** |
| Ten-objective Pareto polish | 32 | 32 | 1.004355 / 1.002137 |

The seed-1 light-seam improvements range from **0.0364% to 1.6543%**, median
**0.1982%**. On seed 2 the median ratio improves by **0.1057%**, but the worst
error rises by **0.8221%**. These are small deterministic point differences,
not confidence intervals or evidence of statistical significance.

The six seed-2 regressions are retained explicitly:

| Metric | Relative increase |
| --- | ---: |
| Original blurred 32-frame error | .3874% |
| Original blurred 256-frame error | .8221% |
| All-phase threshold .75, 32 frames | .0409% |
| All-phase threshold .07, 4 frames | .0079% |
| All-phase threshold .07, EMA | .0181% |
| Sigma-2 threshold aggregate, 32 frames | .2488% |

Compared directly with the same-iteration original-loss control, the new light
schedule improves 54/74 metrics on seed 1 and 52/74 on seed 2. Against the
approximately equal-runtime original-loss control, it improves only 39/74 and
38/74, respectively. It does **not** dominate either stronger control. This is
why comparing only against the 10,000-step reference would overstate progress.

The chosen .03 seam weight was selected after the exploratory seed-1 screen;
the parameters were then fixed for seed 2. The thresholds/widths called held-out
are absent from the local optimization loss, but were visible during exploratory
hyperparameter selection. They are not a fully blind function-level test set.
Two seeds and constant-integrand diagnostics remain far short of scene-level
or universal sampling validation.

## Updated common-protocol comparison with PR #116

The latest [PR #116](https://github.com/bhouston/three.js/pull/116), pinned at
`d8b1acffaada09223b5133d965f3a7cf9028fe5b`, contains stronger methods than its
three initial cheap constructors. We regenerated its three promising frontier
profiles on this machine, at seeds 1 and 2, and evaluated them with the exact
unchanged **74-metric** protocol above. Every previously scored row is unchanged.
The five outputs published by #116 reproduce byte for byte, and 58 compatible
measurements per published output agree with its independent scorer within
1e-12. Autocorrelation seed 2 is an additional confirmation run here.

![Focused comparison of both investigations](results/cross-frontier-summary.png)

| Method | Seed 1 wins / 74 | Seed 2 wins / 74 | Worst ratio, seed 1 / seed 2 | Total seconds, seed 1 / seed 2 |
| --- | ---: | ---: | ---: | ---: |
| #116 annealed stratified hybrid | 65 | 64 | 1.054430 / 1.057645 | 37.49 / 37.71 |
| #116 greedy stratified hybrid | 53 | 53 | 1.082664 / 1.084890 | 37.56 / 37.60 |
| #116 valid mixed autocorrelation | 52 | 53 | 1.092379 / 1.106372 | 37.80 / 37.80 |
| #117 light seam 3% | 74 | 68 | .999636 / 1.008221 | 117.11 / 117.66 |
| #117 extra-FAST time control | 72 | 66 | 1.002830 / 1.003328 | 116.06 / 116.67 |

All ratios in this table divide by ordinary FAST for the same seed. The #116
constructors start from scratch; #117 polishing includes its FAST dependency.
The new runs used the same physical NVIDIA Pascal adapter, warmup, sequential
completed readbacks and no concurrent scoring. They take about half the earlier
74.69/75.33-second FAST reference times on this machine. This is a practical
construction-cost comparison at different quality, not an equal-quality speedup
or a rendering throughput benchmark. The historical runs were separate sessions.

The **annealed method is the strongest #116 candidate here**. It improves all
ten original metrics on both seeds: raw per-pixel 32-frame error is about 60%
lower, single-frame blurred error 20–21% lower, and scalar EMA+blur 23% lower.
Directly against #117's light-seam method, it has lower error in 65/74 and 64/74
measurements, but higher error in the remaining 9 and 10. Its worst relative
regression is about 5.85%/5.82%. Neither method dominates the other. The annealed
method makes larger gains; light-seam stays closer to FAST across this suite.

The largest annealed regression against FAST is the sigma-2 five-threshold
aggregate at four frames, up 5.44%/5.76%. Threshold .07, .83 and .94 also expose
losses; seed 2 additionally loses on threshold .1 EMA. These results retain
bias, seam crossings and all start phases instead of selecting a favorable
frame or integrand. The full heatmap still includes every earlier rejected
profile and control; the [focused table](results/cross-frontier-summary.md)
also compares #116 directly with light-seam, extra FAST and Pareto refinement.

The annealed construction initializes one jittered sample in each of 32 strata
per pixel, then swaps temporal positions within that pixel. It preserves the
full-cycle sample multiset while optimizing a mixed threshold/raw-value loss.
Its hard stratification explains much of the strong full-cycle quadrature
result. The compact mixed kernel in the annealed/greedy profiles is a heuristic,
whereas the third profile uses a valid binomial autocorrelation. Neither this
constraint nor annealing guarantees better error for every transformed
integrand, window, reconstruction filter or scene.

**The 74- and 318-metric suites differ; their win counts are not interchangeable.**
Our suite adds some tail thresholds and sigma-.5/2 aggregates; #116's broader
suite adds unblurred errors, more temporal windows and high-frequency functions.
It reports 262 wins, 53 losses and 3 ties for the annealed method on each seed,
with roughly 7% worst regression on a high-frequency sine. That function is
absent from our frozen 74, so this chart cannot replace that audit. #116 uses
an absolute 1e-12 tie tolerance; our original strict comparisons remain intact.
Both are correlated development diagnostics, not independent trials, fully
blind tests, rendered scenes or state-of-the-art evidence.

The example keeps just two complementary leading experiments: **#116 annealed
stratified** for its larger improvements and cheaper construction, and **#117
light seam 3%** for its conservative FAST refinement. Both load precomputed
PNGs. This is a practical selection for visual comparison, not a universal
ranking. Other methods remain available through the research tools.

Pinned sources, execution adaptations and exact schedules are documented in
[pr116-frontier/README.md](pr116-frontier/README.md). Reproduce the added data:

```sh
node utils/fastnoise/pr116-frontier/run.js pr116-data
python utils/fastnoise/research-score.py pr116-data utils/fastnoise/results/frontier.json
node utils/fastnoise/merge-pr116-frontier.js pr116-data/pr116-generation.json
node utils/fastnoise/check-cross-frontier.js
node utils/fastnoise/chart-frontier.js
node utils/fastnoise/chart-cross-frontier.js
```

The default physical backend replaces #116's forced D3D11 selection. GPU work
is drained every ten iterations without changing its logical 100-step RNG and
suppression schedule; published output hashes confirm the result. The pinned
[published measurements](results/pr116-frontier-published.json) and
[reproduction records](results/pr116-frontier-generation.json) retain provenance.

## Other attempts and rejected profiles

Seed-1 screening results (rejected here because they already fail the target;
these profiles were not all repeated on seed 2):

| Profile | Metrics improved / 74 | Worst ratio | Generation time |
| --- | ---: | ---: | ---: |
| Multi-scale CDF, scratch, 16 passes | 0 | 6.8560 | 15.01 s |
| Multi-scale CDF, 4-pass FAST polish | 8 | 1.0979 | 3.61 s polish + 74.69 s input |
| Temporal strata, scratch, jittered | 23 | 4.8152 | 7.60 s |
| Temporal strata, scratch, midpoint | 21 | 4.5161 | 8.54 s |
| 25% seam-aware FAST polish | 69 | 1.0389 | 43.93 s polish + 74.69 s input |
| 15% joint-kernel FAST polish | 64 | 1.1451 | 45.29 s polish + 74.69 s input |
| 25% seam + 15% joint polish | 65 | 1.1964 | 118.00 s polish + 74.69 s input |
| Lighter 1.5% seam schedule | 70 | 1.001246 | 117.14 s total |

The multi-scale method lowers its stated CDF energy, but a weighted sum of
different scales and periodic EMA weights does not imply lower individual
suite metrics. Hard temporal strata retain exact per-pixel sets while limiting
spatial freedom; midpoint quadrature does not solve seam-crossing windows or
short-window spatial error. Stronger seam/joint weights help many reconstructed
errors while sacrificing some original projections. Each failure is a useful
constraint on future algorithms, not proof that the entire approach cannot work.

The Pareto method uses deployed 8-bit midpoint features and accepts a swap only
if neither spatial nor temporal CDF/identity/squared/fourth/cosine aggregate
energy increases (numerical tolerance 1e-12). That protects **ten aggregate
objectives**, not the 74 individual measurements. It improves the four
continuous-integrand single-frame all-phase errors on both seeds, but can
worsen other thresholds, temporal box windows and spectra. There are 32 wins,
4 exact ties and 38 losses on seed 1; 32 wins, 8 ties and 34 losses on seed 2.
The full-size potential cache is about 294 MB. Independent direct convolution
checks agree with the predicted cached energy changes on small volumes.

## Construction costs and hardware

Windows, AMD Ryzen 9 5950X, Node 26.11.1, Chrome 156, physical NVIDIA GTX 1050.
Adapter reports NVIDIA/Pascal, fallback=false; software rasterization is
disabled. GPU runs disable vsync and frame limits, warm up first, run serially
and wait for mapped GPU output before switching. These are compute volumes,
not display-frame timings. CPU runs warm up and count at least two completed
volumes over at least five seconds, including initialization/allocation.
No tests or numerical scoring ran concurrently with measured generation.

Reference FAST took **74.69 / 75.33 s** for seeds 1/2. The light-seam schedule
took **117.11 / 117.66 s total**: about **1.56–1.57× the cost** of the baseline,
not a speedup. The approximate time controls took **116.06 / 116.67 s**. The
Pareto refinement took **8.34 / 8.04 s**, plus its FAST input, totaling
**83.03 / 83.37 s**. All dependency costs are included recursively in the JSON.
CPU and GPU constructors use different backends, so scratch timings measure
practical construction cost, not equal-backend acceleration.

An initial joint-kernel run with 100 iterations per GPU submission hit the
Windows watchdog/device-loss limit. The research shader now drains every ten
iterations while retaining the logical 100-step keys, counter and suppression
boundaries. Both ordinary and drained zero-option 100-step controls matched
the original baseline SHA-256 exactly. The original generator is unchanged.
The earlier completed 25% seam run used the larger submissions; its timing is
retained rather than silently replaced. No timing is claimed for the failed run.

## Reviewable results and reproduction

The [full heatmap and table](results/frontier-summary.md) include every scored
profile, including regressions. [Scores](results/frontier.json) contain all
primary values, input hashes, exact ratios and bias diagnostics;
[generation records](results/frontier-generation.json) contain parameters,
completed-work times, logs and recursively accounted input costs. Raw volumes
remain task artifacts and can be regenerated from these sources.

![All research error ratios](results/frontier-summary.png)

From repository root, with npm dependencies and NumPy available:

```sh
node utils/fastnoise/check-frontier.js
node utils/fastnoise/check-pareto.js
node utils/fastnoise/research-run.js utils/fastnoise/research-baseline.json research-results
node utils/fastnoise/research-run.js utils/fastnoise/research-screen.json research-results
node utils/fastnoise/research-run.js utils/fastnoise/research-followup.json research-results
node utils/fastnoise/research-run.js utils/fastnoise/research-confirm.json research-results
python utils/fastnoise/research-score.py research-results utils/fastnoise/results/frontier.json
```

Use an installed Chrome through `CHROME_PATH` on another platform. The runner
starts and closes its own server/browser. `--resume` skips completed profile
names; only use it with unchanged implementations/options and intact input
volumes. Copy generation.json to results/frontier-generation.json before
running `node utils/fastnoise/chart-frontier.js`. Score caching is keyed by
volume SHA-256 and the frozen protocol version.

Validation: hardware controls and reference hashes match, all 74 metrics are
finite, generator/quantized Pareto invariant checks and scoped JS/HTML lint
pass. `npm test` passes: 1,982 core tests (one TODO), 141 addon tests, zero
failures. The Windows unit-launcher hardware adaptation was temporary and
restored; no typecheck script exists. The heatmap is a 2D numerical chart,
not rendered-scene validation. Original FAST assets and root build outputs are
unchanged; the separate experimental example catalog is described in
[EXAMPLE.md](EXAMPLE.md).

## What would justify the next advance?

The next optimizer should protect the **actual reconstructed error operators**
for both toroidal and shifted playback, rather than only separate spatial and
temporal energies. A useful direction is constrained multi-objective descent
on those operators, followed by a fresh-seed and scene evaluation that has not
influenced parameter selection. This round implements and tests weaker feature
guards, and documents where they fail; it has not solved that harder problem.
FAST remains the stronger general reference and production defaults are unchanged.
