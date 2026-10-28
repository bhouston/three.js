# Sampling throughput

Run from the repository root after `npm ci`:

```sh
node utils/noise-comparison/benchmark-sampling.js
```

Requires installed Chrome (`CHROME_PATH` overrides its executable) and a physical
WebGPU adapter. The runner rejects software/fallback adapters, disables software
rasterization, vsync and frame caps, serves source modules without rebuilding
`build/`, and closes Chrome and its HTTP server on exit. Results are written to
[sampling-throughput.json](sampling-throughput.json).

The benchmark uses the same six TSL sampler expressions and FAST atlas loader as
the main blue-noise example. Each fullscreen draw generates **1,048,576 scalar
samples** (1024×1024), storing them in the R channel of an RGBA32F render target.
The sample index cycles through 0–1023, with fixed seed 1 for procedural methods.
FAST samples use the actual atlas lookup and cycle offsets; Golden Ratio uses
Product's first slice and the example's 24-bit Weyl fraction. White uses the
example's PCG hash rather than a CPU random-number generator.

Methods run sequentially, each with a two-second warmup followed by at least
five seconds of sustained work. Eight complete draws are submitted per batch;
the device queue is awaited before counting those draws as completed. GPU work
is drained before switching methods. Three trials rotate the method order.
There is no individual-frame timer or GPU timestamp instrumentation. Readback
after measurement checks finite unit-range outputs and verifies the procedural
methods against the independent CPU reference at four pixels per trial.

The reported milliseconds per million samples are **amortized from sustained
completed throughput**, not the latency of a single million-sample request.
They include CPU/driver submission, target clearing and output writes, and queue
completion waits. They exclude the example's accumulation, thresholding, blur,
display, image fetching/decoding, and warmed shader compilation/atlas upload.
Both procedural methods have no precomputation; FAST uses offline-optimized
assets whose download costs are documented separately in the PR.

These measurements describe this workload and hardware, not isolated shader
instruction costs or guaranteed speedups inside a larger renderer. RGBA32F
output traffic and submission overhead can hide differences between samplers.

## Measured results

Physical NVIDIA GeForce GTX 1050 (WebGPU adapter: NVIDIA / Pascal), Windows, AMD Ryzen 9 5950X, Chrome 156, Node v26.11.1. Three trials per method; medians and min–max ranges below. No test suites or other benchmark jobs ran alongside these trials.

| Method | Median ms / 1M samples | Trial range (ms / 1M) | Median million samples/s |
| --- | ---: | ---: | ---: |
| White | 0.447 | 0.397–0.448 | 2238 |
| FAST Separate | 0.443 | 0.415–0.452 | 2258 |
| FAST Product | 0.441 | 0.410–0.443 | 2266 |
| Golden Ratio | 0.434 | 0.421–0.436 | 2302 |
| 3LHW | 0.442 | 0.439–0.448 | 2263 |
| Animated IGN | 0.416 | 0.407–0.453 | 2402 |

Each trial completed billions of samples; exact counts, elapsed seconds, readbacks and adapter/browser metadata are in the JSON report. The trial ranges overlap substantially: these results do not establish a reliable speed ranking among methods.
