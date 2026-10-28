# Multi-integrand threshold refinement experiment

`generate(options)` returns an independent `Float32Array` in z/y/x order in
`[0,1)`. Defaults: `size=128`, `depth=32`, `seed=1`, `passes=16`,
`thresholdCount=15`, `scalarWeight=4`, `thresholdWeight=1`,
`smoothWeight=.25`, `temporalWeight=1`, `jointWeight=.5`,
`preserveColumns=true`. Every weight is nonnegative. For sampling-oriented
screens use `scalarWeight:1, thresholdWeight:4, smoothWeight:.125` too.

The experimental objective sums filtered quadratic energies of 15 Heaviside
features, the identity, and the first sine/cosine pair. Kernels are the sum
of a 3x3 binomial spatial reconstruction autocorrelation, a finite 8-tap
EMA (.25) autocorrelation, and a small joint space/time box reconstruction
autocorrelation. Periodic kernel duplicates are aggregated exactly, including
on tiny volumes. Cached potentials give exact swap-energy differences for
this objective, not an approximation of the delta. The objective itself is
a finite, imperfect surrogate for rendering integration error.

Scratch initialization independently shuffles jittered temporal strata at
every pixel. By default swaps remain within that pixel's temporal column:
this preserves per-pixel histograms and complete-cycle sample means. This
constraint can help long-box integration while limiting spatial optimization.
`preserveColumns:false` enables arbitrary voxel swaps and trades those
invariants for a wider feasible set. Neither version guarantees Pareto
improvement across arbitrary metrics.

`initial` can be a matching Float32Array. It is validated and copied; the
caller's input is never mutated. This supports refining FAST separately from
generating from scratch. A FAST warm-start must include FAST's generation
cost and must not be reported as a scratch-generator speedup.

`verify:true` is limited to at most128 voxels. On every accepted swap it
independently recomputes the full quadratic energy and checks its difference
against the cached delta. A 3x3x3 test with four passes passed this check,
input immutability, and exact sample-multiset preservation. Reproduce:

```js
const initial = Float32Array.from({ length:27 }, (_, i) => (i + .5) /27);
const original = initial.slice();
const output = generate({size:3, depth:3, initial, passes:4, verify:true});
console.assert(initial.every((v, i) => v === original[i]));
console.assert([...initial].sort().join() === [...output].sort().join());
```

Initial CPU screens on MediaPC: unconstrained swaps, default feature weights,
128x128x32 and16 passes took29.24 seconds. Versus EA's shipped FAST scalar
atlas, this improved raw-value low temporal frequency, blurred1/4-frame error,
and EMA/blur; it regressed complete-cycle errors and threshold integrands.
Constrained columns with8 passes and thresholdWeight4/scalarWeight1 took
17.83 seconds; it substantially improved32/256-frame raw-value error but
regressed spatial spectrum and shorter-frame error. These are partial,
one-seed numerical screens, not evidence of exceeding FAST on all metrics.
The benchmark harness should evaluate complete held-out integrands, thresholds,
start phases, seeds and frame counts before recommending any configuration.
