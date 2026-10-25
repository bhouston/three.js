# Interior rendering performance

The Space Ship Hallway and Dungeon Warkarma views are primarily limited by GPU pixel work on Apple M3. Half-resolution SSR improves sustained throughput by about 21–24%, while lowering the overall render resolution has a much larger effect. Dungeon batching substantially reduces CPU work but improves the complete scene by only about 4% in the tested view.

Measurements were taken on October 24, 2026, against checkout `b0875904ca91`. The benchmark substitutes the example HTML in a dedicated browser; production rendering code and example defaults are unchanged.

## Measurement method and cameras

Hardware Chrome 155 uses WebGPU on Apple M3. Adapter information reports Apple, Metal 3, and no fallback adapter; ANGLE independently identifies Apple M3. Software adapters are rejected. Chrome starts with vsync and frame-rate limits disabled.

Each implementation runs sequentially after baking all 16 interleaved iterations with the original 4096 light-map atlas setting. Main runs warm up for at least 2.5 seconds, then measure three five-second intervals. The higher-resolution runs use two intervals. Frames include the complete Beauty pipeline, controls update, and all lighting/depth history copies. After each batch of four frames, the benchmark waits for `GPUQueue.onSubmittedWorkDone()` before counting those frames. The interval includes its final GPU drain. Inspector collection and timestamp queries are disabled during throughput measurement; a separate 24-frame diagnostic collects pass timings.

The CSS viewport is 1280 by 720. Pixel ratio 1 renders 1280 by 720; pixel ratio 2 renders 2560 by 1440. These are sustained completed-frame rates for stationary cameras, without animation-loop pacing.

The hallway uses its existing interior camera. The dungeon uses the final camera setup in [webgpu_postprocessing_ssr_denoise.html](../../examples/webgpu_postprocessing_ssr_denoise.html), with the same 0.1 model scale:

- Position: `(1.259878548682251, 0.5391287340899181, -0.27217301481427114)`.
- Orbit target: `(1.0258536154689288, 0.2746440590977971, -1.0815876858987743)`.
- Vertical field of view: 35 degrees; near/far: 0.1 and 8.

This frames the throne room from inside the dungeon. The hallway has 8 meshes and 53,400 triangles; the dungeon has 798 meshes and 72,137 triangles. Frustum culling reduces the visible dungeon workload to 356 total draw calls per full frame, versus 28 for the hallway.

## Throughput at 1280 by 720

Values are median completed frames per second. Each row changes only the named setting from the baseline, except the explicitly combined SSR setting. Disabling effects is a cost experiment and changes appearance.

| Setting | Hallway fps | Dungeon fps |
| --- | ---: | ---: |
| Baseline | 82.5 | 71.2 |
| SSR at half width and height | 99.8 | 86.3 |
| SSR at quarter width and height | 104.3 | 91.7 |
| SSR quality 0.5 | 91.2 | — |
| SSR half resolution and quality 0.5 | 106.2 | 88.8 |
| SSR disabled | 162.3 | 126.0 |
| SSGI disabled | 121.2 | 101.7 |
| AO disabled | 87.1 | 81.9 |
| SSR, SSGI and AO disabled | 652.9 | 243.9 |
| Half the overall pixel count | 147.9 | 135.3 |
| SSR denoising disabled | 99.5 | 85.2 |
| SSGI denoising disabled | 88.7 | 76.2 |
| Pre-pass lighting disabled | — | 72.0 |
| Static meshes merged by material | — | 74.0 |
| Baseline repeated at the end | 82.4 | 71.4 |

Hallway half/quarter-resolution SSR and disabled-SSR results come from an initial sweep with a baseline of 82.0 fps and 64 warmup frames per implementation. Its baseline differs from the main sweep by about 0.6%. All measured intervals in both sweeps last at least five seconds, with three repeats.

“Half the overall pixel count” sets pixel ratio to `Math.SQRT1_2`, reducing each dimension to about 71% while retaining all effects. It is a different experiment from halving only SSR dimensions, which reduces tracing pixels to one quarter.

## Throughput at 2560 by 1440

| Setting | Hallway fps | Dungeon fps |
| --- | ---: | ---: |
| Baseline | 21.0 | 15.9 |
| SSR at half width and height | 26.1 | 19.6 |
| Baseline repeated at the end | 21.0 | 15.9 |

The example uses `window.devicePixelRatio`, so this larger workload is relevant to Retina displays. At the tested window size, returning to pixel ratio 1 raises throughput approximately fourfold. Half-resolution SSR alone does not bring either scene close to 60 fps at pixel ratio 2 on this GPU.

## Optimization priorities

1. **Expose a render-resolution budget and half-resolution SSR as performance options.** These have the strongest demonstrated gains. Half-resolution SSR is the first setting to assess for acceptable reflection detail: quarter resolution adds only another 4.5–5.5 fps at pixel ratio 1. A render-scale control or adaptive resolution can address the larger Retina workload while retaining native-resolution UI.
2. **Optimize reflection reconstruction alongside tracing.** Disabling SSR denoising raises throughput by about 20% in both scenes, whereas halving the tracing quality gives the hallway only about 10%. The default chain resolves radiance to a texture, performs temporal reprojection, and runs recurrent denoising. [TemporalReprojectNode](../../examples/jsm/tsl/display/TemporalReprojectNode.js) and [RecurrentDenoiseNode](../../examples/jsm/tsl/display/RecurrentDenoiseNode.js) size their targets from the full drawing buffer, even when tracing is reduced. Reduced-resolution reconstruction with an edge-aware final upsample, or fusing compatible passes, are candidates requiring quality and motion tests.
3. **Reduce SSGI reconstruction and source-validation work.** SSGI already traces at half resolution, but [CoverageDenoiseNode](../../examples/jsm/tsl/display/CoverageDenoiseNode.js) runs its spatial and temporal targets at full resolution. Its spatial filter evaluates nine guided taps; temporal reconstruction also gathers and validates history. [ReprojectedRadianceNode](../../examples/jsm/tsl/display/ReprojectedRadianceNode.js) validates four historical texels each time a contributing SSGI sample requests source radiance. Caching this validated source once per current raster texel is another candidate. Preserve per-texel validation, coverage, and foreground/background boundaries when testing either approach; their speedups have not been measured here.
4. **Treat dungeon batching as a CPU optimization.** The static-merge prototype reduces draw calls from 356 to 50 and diagnostic render-pipeline CPU time from approximately 4.3 ms to 1.5 ms. Complete throughput rises only from 71.2 to 74.0 fps; the three merged samples range from 71.4 to 74.1 fps. A production implementation should preserve useful culling through spatial groups or batched meshes, and preserve the baked atlas UVs. Disabling pre-pass lighting similarly reduces CPU work but changes sustained throughput by only about 1%.

AO is a secondary target, particularly in the dungeon, where disabling it improves throughput by about 15%. The current baked shadows already disable automatic updates; this study does not identify repeated shadow-map rendering as a steady-state bottleneck.

## Image checks and timing limitations

Stationary screenshots were inspected for baseline, reduced-resolution SSR and merged dungeon geometry. Their overall appearance remains similar in this view. This does not establish reflection accuracy, motion stability, disocclusion quality, or acceptable quality on other cameras and GPUs. Keep the denoisers enabled in a quality-preserving performance preset until replacement reconstruction has been validated.

Inspector GPU durations did not reconcile with sustained throughput. For example, the hallway SSGI temporal denoiser reported about 9.9 ms and SSR tracing about 4.0 ms. These disjoint passes alone exceed the approximately 12.1 ms per completed frame implied by the complete pipeline's 82.5 fps. Raw timestamps are retained for investigation; optimization rankings above use completed-frame throughput. Diagnostic CPU parent durations include nested calls and should not be added to their children.

All retained scene runs completed without recorded browser or GPU errors. The benchmark passes ESLint. No production renderer changes were made.

## Data and reproduction

- [Main hallway measurements and pass diagnostics](interior-performance-data/hallway.json).
- [Initial hallway SSR sweep](interior-performance-data/hallway-initial.json).
- [Dungeon measurements and pass diagnostics](interior-performance-data/dungeon.json).
- [Both scenes at pixel ratio 2](interior-performance-data/retina.json).
- Dungeon screenshots: [baseline](interior-performance-data/dungeon-baseline.png), [half-resolution SSR](interior-performance-data/dungeon-ssr-half.png), [static merge](interior-performance-data/dungeon-merge-static.png).
- Hallway screenshots at pixel ratio 2: [baseline](interior-performance-data/hallway-baseline.png), [half-resolution SSR](interior-performance-data/hallway-ssr-half.png).

Run from the repository root with dependencies installed and hardware Chrome available:

```sh
node test/benchmarks/interior-performance.mjs
```

The runner starts and stops its own local server and browser. It defaults to both cameras, three five-second intervals, and output under `/tmp/three-interior-profile-20261024`. For the extended dungeon study:

```sh
INTERIOR_PROFILE_MODELS='Dungeon Warkarma' \
INTERIOR_PROFILE_VARIANTS='baseline,ssr-half,ssr-quarter,no-ssr,no-ssgi,no-ao,no-screen-effects,half-pixels,ssr-half-quality-half,ssr-no-denoise,ssgi-no-denoise,prepass-unlit,merge-static,baseline-repeat' \
node test/benchmarks/interior-performance.mjs
```

To repeat the higher-resolution comparison:

```sh
INTERIOR_PROFILE_PIXEL_RATIO=2 INTERIOR_PROFILE_REPEATS=2 \
INTERIOR_PROFILE_VARIANTS='baseline,ssr-half,baseline-repeat' \
INTERIOR_PROFILE_OUTPUT=/tmp/three-interior-profile-retina \
node test/benchmarks/interior-performance.mjs
```

`INTERIOR_PROFILE_OUTPUT`, `INTERIOR_PROFILE_MODELS`, `INTERIOR_PROFILE_VARIANTS`, `INTERIOR_PROFILE_SECONDS`, `INTERIOR_PROFILE_REPEATS`, `INTERIOR_PROFILE_PIXEL_RATIO`, and `CHROME_PATH` customize the run. Static merging is restricted to the opaque static meshes supported by this experiment; the runner fails on unsupported meshes rather than silently dropping them.
