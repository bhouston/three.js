# SSR eligibility classification and reconstruction distance

October 24, 2026. Follow-up to the [Hi-Z investigation](interior-ssr-hi-z.md), starting at `502588f2f7a0`. This change moves the existing roughness exclusion policy before ray generation, rather than introducing a new roughness cutoff or reducing samples on contributing surfaces. It is enabled by default in Interior at full tracing resolution and can be disabled with **SSR Settings → Skip Unused Rays**. Linear traversal remains the default.

## Reference research and design

[AMD's classifier](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/blob/v1.1.4/sdk/include/FidelityFX/gpu/sssr/ffx_sssr_classify_tiles.h) decides whether a surface requires a reflection ray before building the ray list. Very rough surfaces use fallback instead. It separately handles mirror rays, lower ray counts on glossy quads, variance-driven extra rays, and denoiser tiles. Those are distinct policies: deterministic exclusion of a surface whose screen reflection is unused does not require the reconstruction needed for stochastic/subsampled missing rays.

[NRD's hit-distance guidance](https://github.com/NVIDIA-RTX/NRD#input-signals) emphasizes that hit distance controls reflection reprojection and must describe a relevant specular-lobe sample. NRD has its own conventions for misses and skipped samples; those numeric encodings cannot be copied into our custom denoisers. Our pipeline already uses an environment sentinel for unknown fallback hit distance. A finite hit from a ray whose effective coverage has become zero is not a valid guide for the fallback radiance replacing that ray.

[AMD SSSR's overall pipeline](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/stochastic-screen-space-reflections/) also uses compact compute dispatches and denoiser classification. This implementation is an early fragment eligibility mask, not that full architecture. The [previous workgroup builder improvement](interior-ssr-hi-z.md#quality-preserving-improvement-retained) already brought construction closer to [SPD's local shared-memory reductions](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/single-pass-downsampler/). Classification addresses the larger measured cost of tracing unused pixels.

## Two separately committed changes

1. `c4b95c80f94e` fixes reconstruction semantics. `resolveSSRHitDistance(rawDistance, effectiveCoverage)` returns the finite screen distance only for positive effective coverage and a valid positive screen-hit distance. Otherwise it returns `ENV_RAY_LENGTH`. Partial coverage retains its screen distance, and RGB never determines validity: a black reflection is still a hit. Interior uses this resolved distance for both before-filtering reconstruction and coverage-aware virtual-hit reprojection.
2. `f4e57ed5aba0` adds `SSRNodeOptions.traceMaskNode`, a boolean node evaluated before analytic/custom noise generation, GGX sampling, ray construction and traversal. It requires `outputCoverage: true`, with fallback handled externally. Masked MRT texels clear to zero radiance, coverage and distance, including when they contained a hit in the preceding frame. `useTraceMask` permits runtime comparisons without rebuilding the tracer shader.

Interior's mask is `roughness < maxRoughness`, using the same roughness guide and cutoff as the existing downstream fade. It retains the transition band below the cutoff. It does not discard low-radiance or black reflections and does not turn up the cutoff to get a speedup. Both classified and unclassified comparisons use the corrected distance policy. The distance fix itself can change reconstruction relative to the old, incorrect fallback guide; this report does not claim bitwise equivalence to that older pipeline.

At reduced tracing resolution, a texel centered on a rough surface can still contribute through bilinear upsampling to a neighboring glossy surface. A center-only mask is insufficient to prove that such a texel is unused. Interior therefore bypasses classification below full tracing resolution until eligibility covers the entire reconstruction footprint. The half-resolution tests verify that bypass preserves the existing signal exactly. Custom callers of `traceMaskNode` are responsible for their sampling/reconstruction footprints.

## Hardware throughput

Apple M3, physical non-fallback Metal adapter, hardware Chrome, 1920×1080, pixel ratio 1, quality 1, otherwise Interior defaults with a 1024 bake atlas. Each traversal runs classification off, on, then off. Every mode warms up for one second, runs for at least five seconds, and counts frames only after batches of four complete on the GPU. Vsync/frame caps are disabled, modes run sequentially, and queues drain before switching. Measurements include all Beauty work: scene rendering, SSGI, AO, reconstruction, fallback, TRAA and Hi-Z construction where selected. They are sustained throughput including CPU submission, not individual-frame timings or isolated GPU timestamps.

| Scene | Traversal | Off before FPS | On FPS | Off after FPS | Gain versus geometric mean of off controls |
| --- | --- | ---: | ---: | ---: | ---: |
| Cornell mirror wall | Linear | 43.82 | 65.99 | 44.09 | 50.1% |
| Cornell mirror wall | Conservative | 43.50 | 65.31 | 43.56 | 50.0% |
| Cornell mirror wall | Adaptive | 40.71 | 64.90 | 40.62 | 59.6% |
| Mirror floor | Linear | 64.97 | 66.45 | 65.02 | 2.2% |
| Mirror floor | Conservative | 62.15 | 62.73 | 62.21 | 0.9% |
| Mirror floor | Adaptive | 65.53 | 67.43 | 65.52 | 2.9% |

The earlier untimed diagnostics found approximately 80% of Cornell's traced surface pixels but only 5% of the floor's pixels beyond the downstream roughness cutoff. The gains follow that workload difference. Cornell's large gain reproduces across the off controls and all traversals. The floor differences are small and need more repetitions/hardware before treating them as robust universal improvements. Classification does not establish an advantage for Hi-Z over linear, nor does it fix adaptive traversal's independent quality limitations.

[Raw Cornell throughput](ssr-classification-data/cornell-throughput.json), [raw floor throughput](ssr-classification-data/floor-throughput.json). Both runs have zero recorded browser/GPU errors. Results are from these fixed views on this GPU; they do not predict arbitrary scene or hardware performance.

## Correctness and quality checks

The comparison rig freezes HDR source lighting and replays the same stochastic sequence and camera trajectory. The camera stays fixed for sixteen frames, then translates for sixteen frames. It reads linear HDR resolved and filtered RGBA at the end of each phase, before display tone mapping. Classification off/on/off uses the same corrected distance contract and resets histories. The separate resolved RTT is explicitly evaluated even when the coverage-aware output does not depend on it, preventing a stale-target comparison. Nonzero lighting and one SSR execution per frame are asserted.

Both Cornell and the floor test all three traversals, both fallback/filter orders, and resolution scales 1 and 0.5: **48 stationary/moving comparison cases total**. Every classified-versus-unclassified and repeated-unclassified comparison has **zero maximum channel error** in both resolved and filtered output. This establishes preservation of the corrected pipeline for the tested views and trajectories, not physical correctness of the underlying SSR estimator, adaptive marcher, or probe fallback.

[Raw Cornell quality](ssr-classification-data/cornell-quality.json), [raw floor quality](ssr-classification-data/floor-quality.json).

Focused GPU units additionally verify:

- Positive and partial coverage retain screen-hit distance; zero coverage, cleared/negative distances and environment sentinels cannot guide fallback as geometry hits.
- A mask is rejected without the external-fallback/coverage contract.
- Rejecting a previously lit texel clears stale radiance, distance and coverage; disabling the mask restores the original hit exactly.
- Eligible rays preserve hit radiance/distance; valid black hits keep their geometry distance and partial coverage.
- Existing endpoint/refinement and GGX pole regressions still pass.

The six focused unit tests execute without skips. Actual test backends identify Apple Metal and hardware WebGL ANGLE Metal on Apple M3. The existing 200-case pyramid validation and 192 traversal/coverage cases also pass. [Unit hardware and results](ssr-classification-data/units.json), [traversal validation](ssr-classification-data/traversal-validation.json). Changed-file lint and whitespace checks pass; generated root build artifacts are excluded.

The existing Damaged Helmet fallback/HDR regression also passes on hardware, with zero browser/GPU errors. The forced-unfiltered-miss visor comparison has relative RMSE 0.0759 against SSR disabled (the existing tolerance is 0.1). Filtered comparisons remain inside their existing 0.25 tolerance. Bright HDR reflections remain present: the disabled reference's top-100 mean luminance is 119.4 and default SSR's is 83.25, above the existing 50% preservation bound. This is a regression bound, not exact equivalence or a physical-reference accuracy claim. [Raw fallback/HDR results](ssr-classification-data/fallback-hdr.json).

## API and reproduction

```js
import { resolveSSRHitDistance } from 'three/addons/tsl/utils/SpecularHelpers.js';

const reflections = ssr( source, depth, normal, {
    camera,
    stochastic: true,
    outputCoverage: true,
    roughnessNode: roughness,
    traceMaskNode: roughness.lessThan( cutoff )
} );
reflections.useTraceMask.value = true;
// Resolve RGB/coverage with external fallback. Before using distance for history:
const distance = resolveSSRHitDistance( raw.a, effectiveCoverage );
```

From the repository root:

```sh
HIZ_OUTPUT=/tmp/classification-validation node test/benchmarks/ssr-hi-z-validation.mjs
SSR_FALLBACK_OUTPUT=/tmp/classification-fallback node test/benchmarks/interior-ssr-fallback.mjs
CLASSIFY_MODEL='Cornell Box (Mirror Wall)' CLASSIFY_SKIP_BENCHMARK=1 CLASSIFY_OUTPUT=/tmp/classification-cornell-quality node test/benchmarks/interior-ssr-classification.mjs
CLASSIFY_SKIP_BENCHMARK=1 CLASSIFY_OUTPUT=/tmp/classification-floor-quality node test/benchmarks/interior-ssr-classification.mjs
CLASSIFY_MODEL='Cornell Box (Mirror Wall)' CLASSIFY_SKIP_QUALITY=1 CLASSIFY_WIDTH=1920 CLASSIFY_HEIGHT=1080 CLASSIFY_OUTPUT=/tmp/classification-cornell-performance node test/benchmarks/interior-ssr-classification.mjs
CLASSIFY_SKIP_QUALITY=1 CLASSIFY_WIDTH=1920 CLASSIFY_HEIGHT=1080 CLASSIFY_OUTPUT=/tmp/classification-floor-performance node test/benchmarks/interior-ssr-classification.mjs
```

`CHROME_PATH` selects hardware Chrome. Keep other renders stopped and run GPU workloads sequentially. Quality and throughput are separate phases; timed loops contain no readbacks or instrumentation. Source imports avoid generating root build outputs.

## Remaining architecture work

The next step toward AMD's architecture is conservative eligibility over reconstruction footprints, followed by compact compute ray queues and indirect dispatch. Full-rate mirror rays should remain, while lower density and variance-driven extra rays on rough surfaces require an explicit missing-sample reconstruction policy. Denoiser tile classification can save more work than tracing alone, but fallback radiance and history must remain valid outside traced tiles. Subgroups should be an optional optimization over a portable baseline.

Keep these separate from tracer correctness: adaptive needs a nearest-hit/DDA reference, proper finite hit-distance validation, self-hit/backface rejection and matched-quality reference images. The current speedup avoids provably unused work under the existing cutoff; it does not justify promoting adaptive as a quality-preserving default or weakening hit validation for more FPS.
