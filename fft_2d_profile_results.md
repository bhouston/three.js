# FFT2D performance profile

Profiler ported from `cmh-improve-prefix` (commit `0911a1a5`): `test/unit/addons/gpgpu/perf-utils.js`
(`benchmark`/`report`/seeded-data helpers) plus a new `FFT2D.perf.tests.js`, wired into a new
`test-unit-addons-perf` npm script (`UnitTestsAddonsPerf.html`). Run with:

```
npm run test-unit-addons-perf
```

Each size runs `FFT2D.computeForward()` on a deterministically-seeded complex source texture
(mulberry32 PRNG, fixed seed, so every run/branch transforms identical input data), 5 untimed
warmup iterations followed by 50 timed iterations, each synced to the GPU via a read-back probe so
timings include actual device execution time, not just command submission.

Environment: Apple M3 (macOS, arm64), Chrome headless via Puppeteer (`--enable-unsafe-webgpu`).

## Baseline (before adaptive workgroup/shared-memory sizing)

Commit: `fde9a88178`

| Size | Elements | Mean (ms) | Median (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|---|
| 256x256 | 65,536 | 0.852 | 0.800 | 0.700 | 2.200 |
| 512x512 | 262,144 | 1.224 | 1.200 | 1.100 | 2.200 |
| 1024x1024 | 1,048,576 | 5.114 | 5.100 | 4.900 | 6.300 |
| 2048x2048 | 4,194,304 | 20.782 | 20.500 | 20.200 | 25.100 |

## After adaptive workgroup/shared-memory sizing

`examples/jsm/gpgpu/FFT2D.js` already derived its fused-line-length and transpose-tile-size
choices from the real `GPUDevice.limits` (`getComputeLimits`), but fell back to a hardcoded
`MINIMUM_COMPUTE_LIMITS` object when a WebGPU device wasn't available, and its elementwise/
fallback kernels (conjugate, load, store, and the per-stage fallback butterfly dispatch) always
used a hardcoded `DEFAULT_WORKGROUP_SIZE = 256`, regardless of the device.

Following the pattern from `examples/jsm/gpgpu/GPGPUUtils.js` on the `5eb1c7a8` commit
(`requireLimit`, `pickWorkgroupSize`),
`getComputeLimits` now throws instead of falling back if a real device isn't available, and a new
`pickWorkgroupSize(limits, preferred)` picks the largest power-of-two workgroup size (up to a
preferred upper bound) that actually fits the device's invocation limits, replacing every use of
`DEFAULT_WORKGROUP_SIZE`. This also required moving the elementwise conjugate kernel's
construction out of the constructor (which has no renderer/limits available) into
`_ensureButterfliesBuilt`, alongside the row/column/transpose kernels.

| Size | Elements | Mean (ms) | Median (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|---|
| 256x256 | 65,536 | 0.772 | 0.800 | 0.600 | 1.200 |
| 512x512 | 262,144 | 1.174 | 1.200 | 1.000 | 1.500 |
| 1024x1024 | 1,048,576 | 5.326 | 5.200 | 5.000 | 6.400 |
| 2048x2048 | 4,194,304 | 21.832 | 21.500 | 20.300 | 29.200 |

No meaningful difference from baseline - all within run-to-run noise. Expected: on this M3
(`maxComputeInvocationsPerWorkgroup` well above 256), `pickWorkgroupSize(limits, 256)` still
resolves to 256, the same value the old hardcoded constant used. The change is a robustness/
correctness one rather than a speedup here - it removes a silent fallback that could pick an
invalid workgroup size on a device with smaller real limits (e.g. some mobile GPUs), and would
have masked that with wrong output instead of a clear error.

## Studying `Token-Gremlin/natural-disasters`' `OceanFFT.js`

That implementation is a WebGL2 fragment-shader Cooley-Tukey FFT (no compute shaders, so its
workgroup-sizing concerns don't transfer at all) built for real-time ocean simulation, with a
different problem shape than a generic complex 2D FFT: it always runs a small, fixed set of
transform sizes repeatedly every frame, and its input is always real-valued (a height/slope
field), never arbitrary complex data.

Two ideas stood out:

- **Precomputed twiddle-factor + butterfly-index texture.** It precomputes a `(stages x N) x 4`
  texture of twiddle factors and shuffled indices on the CPU once, then every butterfly stage is
  just a texture fetch, no per-invocation `cos`/`sin`. This one is directly applicable regardless
  of the WebGL/WebGPU or fragment/compute-shader difference - implemented below as a twiddle
  lookup table.

- **Real-signal conjugate symmetry, exploited for a single real image.** A real-valued signal's
  spectrum is conjugate-symmetric, which lets *any* two real signals packed as `f + i*g` share one
  complex FFT - initially implemented (and profiled) here as `computeForwardReal2`/
  `computeInverseReal2`, packing two *different* real images (e.g. an RGB image's R and G
  channels) together. That wasn't actually what was wanted, though: reverted in favor of
  `computeForwardReal`/`computeInverseReal` below, which get the same kind of win for *one* real
  image at a time (row-pair packing, not two-image packing) - see that section.

Not applicable: bit-reversal permutation and butterfly-index precomputation. `FFT2D` already uses
the Stockham autosort formulation (ping-ponging between two buffers each stage), which produces
correctly-ordered output without a separate bit-reversal pass - the exact problem that technique
solves for the classic Cooley-Tukey layout `OceanFFT.js` uses.

## After the twiddle-factor lookup table

Implemented the first idea (see `examples/jsm/gpgpu/FFT2D.js`'s `buildTwiddleTable`): every
stage's `cos`/`sin` pair is now a lookup into a `max(width,height)/2`-entry table built once per
`FFT2D` instance, shared by every stage, row and column alike, instead of two transcendental calls
per invocation per stage.

| Size | Elements | Mean (ms) | Median (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|---|
| 256x256 | 65,536 | 0.855 | 0.800 | 0.600 | 2.700 |
| 512x512 | 262,144 | 1.173 | 1.100 | 1.000 | 1.900 |
| 1024x1024 | 1,048,576 | 5.083 | 5.075 | 4.950 | 5.650 |
| 2048x2048 | 4,194,304 | 20.999 | 20.750 | 20.250 | 22.900 |

(Mean of 3 runs at each size.) No measurable difference from the adaptive-sizing numbers above -
these GPU-bound dispatches were never ALU-bound on `cos`/`sin` in the first place; modern GPUs
have fast native transcendental units, so trading that for an extra memory indirection nets out
close to even here. Verified correct via the forward+inverse round-trip sanity check
(`FFT2D.sanity.tests.js`) throughout.

## `computeForwardReal`/`computeInverseReal`: a real FFT for a single real image

Real-valued images are `FFT2D`'s main use case (`webgpu_fft_2d.html`, and image/filter FFTs
generally), so a genuine speedup for *one* real image, not just for pairs of them, is worth having
in `FFT2D` itself. The technique: pack pairs of rows of the one real image into one complex signal
half the height (`z[p] = row[2p] + i*row[2p+1]`), run one *ordinary* full 2D complex FFT on that
half-height array (a nested half-height `FFT2D` instance, reusing every row/column/transpose
kernel completely unchanged), then recombine that smaller spectrum into the true full spectrum
with one cheap elementwise pass - real-signal conjugate-symmetry unmixing (the same math the
reverted `computeForwardReal2` used, just applied to one image's own row parities instead of two
images) plus a single radix-2 twiddle-recombine stage. `computeInverseReal` mirrors this exactly
in reverse. See `computeForwardReal`'s docstring in `examples/jsm/gpgpu/FFT2D.js` for the full
derivation. Verified against `computeForward` (matching spectra) and via its own forward+inverse
round trip (`FFT2D.sanity.tests.js`).

| Size | Elements | `computeForward` mean (ms) | `computeForwardReal` mean (ms) | Forward speedup | `computeInverseReal` mean (ms) |
|---|---|---|---|---|---|
| 256x256 | 65,536 | 0.798 | 0.818 | 0.98x | 0.770 |
| 512x512 | 262,144 | 0.844 | 0.834 | 1.01x | 0.722 |
| 1024x1024 | 1,048,576 | 5.226 | 2.426 | 2.15x | 2.126 |
| 2048x2048 | 4,194,304 | 21.470 | 11.994 | 1.79x | 11.836 |

(`computeInverse` isn't included as a reference column since there's no complex-input equivalent
call to compare `computeInverseReal` against here - it's benchmarked against a real spectrum
produced by `computeForwardReal`, which `computeInverse` can also consume, at the same cost as
`computeForward`.)

Unlike the twiddle table, this one shows a real, substantial win at the two larger sizes - up to
~2.15x at 1024x1024, and holding at ~1.8x at 2048x2048 (both forward and inverse, as expected since
they're structurally near-mirror-images of each other). At 256x256 and 512x512 there's no
measurable difference: these sizes are small enough that fixed per-dispatch overhead (pipeline
setup, command submission) dominates over actual GPU compute time, and the row-pair-packing
approach doesn't reduce dispatch *count* (it still runs the same six passes: pack, row, transpose,
column, transpose-back, recombine - just on half the row count) - so there's nothing for it to save
until the workload is large enough to be genuinely compute-bound rather than overhead-bound. This
matches expectations: FFT cost is `O(n log n)`, so halving the element count run through the
row/transpose/column/transpose-back passes approaches (but, per the log factor, never quite
reaches) a full 2x, and that only shows up once dispatch overhead stops dominating.

## Discovery: every number above was measured against WebGPU's spec-*minimum* limits, not this M3's real ones

Digging into *why* `computeForwardReal` only won at 1024/2048, not 256/512, surfaced something
that changes every number above: `renderer.backend.device.limits` - what `FFT2D`'s adaptive
sizing (`getComputeLimits`/`computeMaxFusedLineLength`) reads - was reporting
`maxComputeInvocationsPerWorkgroup: 256`, `maxComputeWorkgroupStorageSize: 16384`. Those are
WebGPU's spec-guaranteed *minimum* limits, not this M3's real ones - `WebGPURenderer`'s
`requiredLimits` defaults to `{}`, and per spec, `requestDevice({requiredLimits: {}})` gives the
device the floor for every limit not explicitly requested, not the adapter's actual capabilities.
Querying `navigator.gpu.requestAdapter()` directly shows this M3's real limits are
`maxComputeInvocationsPerWorkgroup: 1024`, `maxComputeWorkgroupStorageSize: 32768`.

With the floor limits, `computeMaxFusedLineLength` (the largest line length the fast, single-
dispatch, shared-memory-fused butterfly kernel can handle) works out to 512 - so 256/512 used the
fused kernel on both axes, while 1024/2048 fell back to the slower per-stage, global-memory
multi-dispatch kernel (`log2(N)` dispatches per axis instead of 1) on both axes. That's the real
reason `computeForwardReal` only won big at 1024/2048: at those sizes its internal half-height
instance's *column* axis (`height/2`) could cross back under the 512 fusion threshold even though
the *row* axis (`width`, unchanged) couldn't - at 1024, halving 1024 rows to 512 flips the column
pass from 10 fallback dispatches to 1 fused one, on top of the general halved-data-volume win. At
2048, halving 2048 to 1024 doesn't cross the threshold either way, so that case was measuring pure
`O(n log n)` scaling with no dispatch-count change at all. Dispatch counts, worked out from the
pass structure (`load`/`loadRowPairs` -> row-pass -> `transpose` -> col-pass -> `transpose-back`
-> `recombine`/`store`, each pass counting as 1 dispatch if fused or `log2(N)` if not):

| Size | `computeForward` dispatches | `computeForwardReal` dispatches |
|---|---|---|
| 256x256 | 1+**1**+1+**1**+1+1 = 6 | 1+**1**+1+**1**+1+1+1 = 7 |
| 512x512 | 1+**1**+1+**1**+1+1 = 6 | 1+**1**+1+**1**+1+1+1 = 7 |
| 1024x1024 | 1+**10**+1+**10**+1+1 = 24 | 1+**10**+1+**1**+1+1+1 = 16 |
| 2048x2048 | 1+**11**+1+**11**+1+1 = 26 | 1+**11**+1+**10**+1+1+1 = 26 |

At 256/512, `computeForwardReal` issues *one more* dispatch than `computeForward` (the extra
`recombine` pass) for the same all-fused structure on both, explaining the flat/negative result
there.

Fixed by having `perf-utils.js`'s `createRenderer()` and `webgpu_fft_2d.html`'s renderer setup
both request a throwaway adapter first and pass its real limits as `requiredLimits`, so the actual
device gets them (see that commit). Confirmed the fix works (`device.limits` now reports
1024/32768) and re-ran every benchmark above under it - visual output is unaffected (0.0% e2e
screenshot diff), only performance:

| Size | `computeForward` mean (ms) | `computeForwardReal` mean (ms) | `computeInverseReal` mean (ms) |
|---|---|---|---|
| 256x256 | 0.902 | 0.793 | 0.694 |
| 512x512 | 0.882 | 0.837 | 0.728 |
| 1024x1024 | 1.776 | 1.447 | 1.359 |
| 2048x2048 | 5.839 | 4.706 | 4.287 |

(Mean of 2 runs at each size.) Two things jump out:

- **`computeForward` itself got dramatically faster** at 1024/2048 - 5.226ms -> 1.776ms (~2.9x) at
  1024x1024, 21.470ms -> 5.839ms (~3.7x) at 2048x2048 - since both now use the fast fused kernel
  instead of the 10-11-dispatch fallback. 256/512 are unchanged (already fused either way).
- **`computeForwardReal`'s advantage over `computeForward` shrank** at 1024/2048 (now ~1.23x and
  ~1.24x, down from ~2.15x/~1.79x) since the fusion-threshold-crossing bonus is gone - with real
  limits, `computeMaxFusedLineLength` is large enough (2048) that *every* size here fuses on both
  axes regardless, so the remaining gap is purely the `O(n log n)` benefit of halving the row
  count, the same effect the pre-fix 2048x2048 case saw in isolation. `computeForwardReal` is
  still faster than `computeForward` everywhere except being roughly break-even at 256/512, where
  the extra `recombine` dispatch's fixed cost is the whole story again.

The headline finding: getting the renderer's device to actually report real hardware limits
mattered far more here than either of `FFT2D`'s own optimizations (the twiddle table, or even
`computeForwardReal`'s halved workload) - it's what let the *existing* adaptive fused/fallback
logic (already in `FFT2D.js` before any of this work) do its job in the first place.
