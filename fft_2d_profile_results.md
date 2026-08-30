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
