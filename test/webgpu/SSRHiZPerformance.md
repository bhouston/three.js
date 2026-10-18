# SSR Hi-Z performance investigation

The original Hi-Z path is conservative and preserves the linear marcher's results, but its construction cost overwhelms its modest tracing savings. The changes here reduce construction work and retain that output contract. They do not replace SSR with AMD's traversal algorithm.

The old builder independently reduces the original depth texture for every mip. Every level covers all source pixels, so it performs approximately N × L source reads: 22.8 million at 1920×1080 with 11 levels. Calling this one compute dispatch does not make it an SPD downsampler. The replacement reduces each level from its predecessor, with approximately 7N/3 total (for large two-dimensional textures) input/intermediate reads including the level-zero copy. Separate intermediate textures and ordered dispatches avoid cross-workgroup publication and read/write aliasing. Odd trailing footprints, raw depth, reversed depth, and both nearest/farthest channels are preserved. Intermediate textures add approximately 8N/3 bytes of storage, about 5.5 MB at 1080p; additional dispatches can matter at small sizes.

The SSR optimization reduces four raw nearest bounds before linearizing once. Conventional depth uses min and reversed depth uses max. This removes three depth conversions per batch check without changing the selected bound.

The current SSR code does not traverse hierarchy cells. It checks aligned groups of up to 16 original samples, bounds their entire padded screen rectangle with four hierarchy reads, and skips the group only when it is wholly in front of every surface. Failed checks still pay for the original samples. This is intentionally stronger output equivalence than a new ray sampling algorithm provides, but has several performance limits:

- Nearby geometry anywhere inside the rectangle prevents rejection, including geometry away from the actual ray. The origin surface is especially problematic.
- Each check adds four atlas reads, coordinate/LOD calculations, and a branch. Saving a handful of leaf samples may not repay that work.
- Stochastic SSR uses a fixed budget of 64 × quality steps, independent of screen resolution. Castle's default quality 0.25 permits only 15 loop samples and one batch opportunity. Quality 0.5 permits 31 samples and two opportunities; quality 1 permits 63 and four. Mirror SSR instead scales its budget with projected ray length, so higher resolution gives skipping more work to save.
- Divergent stochastic rays can leave neighboring GPU lanes working after one ray skips. This is a plausible additional limitation; wave occupancy was not profiled here.
- Denoising, blur, scene rendering, environment shading, and compositing remain outside the accelerated traversal.

AMD's [SSSR shader](https://raw.githubusercontent.com/GPUOpen-Effects/FidelityFX-SSSR/master/ffx-sssr/ffx_sssr.h) advances to cell boundaries, coarsens after crossing empty cells, and descends when a depth plane might intersect the ray. It hoists direction reciprocals and updates mip resolution incrementally. Its non-mirror path can terminate low-occupancy waves. This gives the hierarchy control over traversal rather than inserting tests around an existing sample schedule.

AMD's [SSSR documentation](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/stochastic-screen-space-reflections/) also describes roughness classification, variable ray density, temporal variance selection, and denoising. Its complete pipeline's performance cannot be attributed to Hi-Z alone. [McGuire and Mara's perspective-correct DDA](https://jcgt.org/published/0003/04/04/) is an efficient screen-space tracing baseline; the current mirror marcher already traces in screen space, so comparisons against naive view-space marching would exaggerate the gain available here.

[AMD SPD](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/single-pass-downsampler/) reuses intermediate reductions within workgroups and relies on globally coherent publication for its final cross-workgroup reduction. The portable replacement here uses dispatch boundaries instead. AMD also documents odd-dimension handling limitations; preserving this implementation's trailing-edge coverage is necessary.

There is no universal expected speedup. Net improvement requires the traversal savings to exceed construction cost. If traversal were 20% of a frame and became twice as fast, the maximum frame-rate gain would be about 11% before building the hierarchy. Long rays through large empty regions are favorable; short rays, early hits, dense depth bounds, and fixed small stochastic step budgets are unfavorable.

The next substantial algorithm experiment should implement cell-boundary traversal as a separately selectable mode, with explicit hit-distance, thickness, refinement, and quality comparisons against the current marcher. It should not silently replace the exact-output batch optimization. For stochastic SSR, roughness-based ray classification and fewer rays per quad with denoising are also promising but change the quality contract. A tiled local multi-level reduction followed by a second dispatch could further reduce builder dispatch overhead while preserving portable synchronization.

Validation: 200 hardware pyramid cases, 1,936,440 exact bounds checks, 2,400 CPU-reference trace cases, and 48 actual SSR comparisons covering perspective/orthographic, conventional/reversed/log depth, mirror/stochastic, refinement, and quality. All passed; SSR hit masks and output channels were identical. Core unit tests: 1,957 passed, one TODO, zero failures. Addon unit tests: 102 passed. Core/addon/test/changed-file lint passed. This JavaScript repository has no configured typecheck. Browser test cleanup printed harmless process-already-exited messages; addon tests printed a resource 404 while all assertions passed.

Benchmark numbers below are GPU-completed throughput measurements including CPU submission and synchronization, not GPU timestamp measurements of individual passes. Each mode warms up for one second, runs sequentially for at least five seconds, counts batches of eight completed frames, and drains the queue before switching. Frame nodes advance explicitly, and SSR execution counts and compute-call counts are checked. Vsync/frame caps are disabled. No GPU tests run alongside the retained measurements. Frozen hierarchy is only a diagnostic for these static views. Small differences near a few percent should be treated as noise; the results represent one GPU, one fixed view per demo, and the stated settings.

Hardware: Windows001, NVIDIA GeForce RTX 3060 Ti, driver 32.0.16.1714, Chrome 155.0.8059.79, Node 26.11.1. WebGPU reported NVIDIA Ampere and fallback/software adapters were rejected. Baseline: hi-z-pyramid commit 5317042363f6c90b07ce12036ab94ad08a863f99. Optimized shaders: commits c90d773c8d2 and 73b1ebf06b0; subsequent source changes only remove an unused field and restore an explanatory comment.

Measurements at 801×603 (quality 0.25 is the castle demo default; 0.5 is the mirror demo default):

| Scene | Quality | Old linear ms | Old Hi-Z ms | New linear ms | New Hi-Z ms | New frozen Hi-Z ms | New Hi-Z FPS vs new linear |
|---|---:|---:|---:|---:|---:|---:|---:|
| Steampunk (mirror) | 0.25 | 1.251 | 1.863 | 1.268 | 1.241 | — | 2.2% |
| Castle (stochastic) | 0.25 | 1.695 | 2.397 | 1.668 | 1.685 | — | -1.0% |
| Steampunk (mirror) | 0.5 | 1.257 | 2.082 | 1.269 | 1.310 | 1.285 | -3.1% |
| Steampunk (mirror) | 1 | 1.300 | 2.332 | 1.313 | 1.479 | 1.359 | -11.2% |
| Castle (stochastic) | 0.5 | 1.678 | 2.486 | 1.726 | 1.707 | 1.686 | 1.1% |
| Castle (stochastic) | 1 | 1.661 | 2.738 | 1.678 | 1.933 | 1.805 | -13.2% |

Measurements at 1920×1080, quality 1:

| Scene | Quality | Old linear ms | Old Hi-Z ms | New linear ms | New Hi-Z ms | New frozen Hi-Z ms | New Hi-Z FPS vs new linear |
|---|---:|---:|---:|---:|---:|---:|---:|
| Steampunk (mirror) | 1 | 5.644 | 9.518 | 5.604 | 5.083 | 4.925 | 10.3% |
| Castle (stochastic) | 1 | 4.999 | 9.915 | 5.016 | 5.535 | 5.392 | -9.4% |

Reversing mode order (Hi-Z before linear) at 1080p reproduced the conclusion:

| Scene | Quality | Old linear ms | Old Hi-Z ms | New linear ms | New Hi-Z ms | New frozen Hi-Z ms | New Hi-Z FPS vs new linear |
|---|---:|---:|---:|---:|---:|---:|---:|
| Steampunk (mirror) | 1 | 5.568 | 9.498 | 5.568 | 5.113 | — | 8.9% |
| Castle (stochastic) | 1 | 4.988 | 9.836 | 5.018 | 5.543 | — | -9.5% |

Independent pyramid construction throughput (includes CPU submission, excludes scene rendering):

| Resolution | Old ms/build | New ms/build | Throughput ratio |
|---|---:|---:|---:|
| 801×603 | 1.301 | 0.256 | 5.1× |
| 1920×1080 | 7.459 | 0.254 | 29.4× |
| 3840×2160 | 35.361 | 0.906 | 39.0× |

The large builder improvement is clear. It cuts the complete 1080p Hi-Z frame from 9.50–9.52 to 5.08–5.11 ms in steampunk and from 9.84–9.92 to 5.53–5.54 ms in castle. Against linear SSR, however, the optimized mirror view gains only 8.9–10.3% FPS; stochastic castle loses about 9.4–9.5%. At 801×603 there is no consistent net win. In particular, castle at its default quality is essentially tied (within a few percent), rather than showing the previous large construction penalty.

The frozen-pyramid result isolates the remaining algorithm problem: at 1080p the mirror path is ~4.93 ms versus ~5.60 ms linear, whereas castle is ~5.39 ms versus ~5.02 ms linear. These are full-frame diagnostic comparisons, not isolated GPU-pass timings. They show why a cheaper builder can expose a mirror win but cannot fix stochastic traversal overhead. Small low-resolution deltas should not be interpreted as reliable speedups.

Raw runs and environment metadata are committed in [hi-z-performance-results.json](./hi-z-performance-results.json). Screenshots are view captures only; directly edited benchmark uniforms do not update inspector DOM values.


Reproduce from the repository root after npm ci and npm run build. Set CHROME_PATH for a different Chrome installation. The benchmark scripts use headful Chrome and shut down their browsers and servers on completion. They leave the example source untouched and expose local variables through intercepted HTML.

```powershell
# Export both baseline shaders beside each other, outside the repository.
git-dedup.cmd show 5317042363f6c90b07ce12036ab94ad08a863f99:examples/jsm/tsl/display/HiZNode.js | Set-Content -Encoding utf8 ../HiZNode-baseline.js
git-dedup.cmd show 5317042363f6c90b07ce12036ab94ad08a863f99:examples/jsm/tsl/display/SSRNode.js | Set-Content -Encoding utf8 ../SSRNode-baseline.js
node test/webgpu/benchmarkSSRHiZ.js ../optimized.json
node test/webgpu/benchmarkSSRHiZ.js ../baseline.json ../HiZNode-baseline.js
$env:BENCH_WIDTH='1920'; $env:BENCH_HEIGHT='1080'; $env:BENCH_QUALITY='1'
node test/webgpu/benchmarkSSRHiZ.js ../optimized-1080p.json
node test/webgpu/benchmarkSSRHiZ.js ../baseline-1080p.json ../HiZNode-baseline.js
node test/webgpu/benchmarkHiZBuild.js ../build-optimized.json
node test/webgpu/benchmarkHiZBuild.js ../build-baseline.json ../HiZNode-baseline.js
```

BENCH_MODES can select comma-separated modes: linear, build-only, hi-z, frozen-hi-z. The independent builder benchmark uses deterministic R32F input and 16-build batches, excludes the initial upload/compilation warmup, checks device errors, and includes CPU dispatch submission. Its source texture differs from the examples' depth attachment, so use it to compare construction strategies, not as an exact subtraction from full-frame timing.
