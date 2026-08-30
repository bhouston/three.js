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
