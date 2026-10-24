# Validation of the combined interior audits

Compared the SSR-only branch at `876361339e40` with the combined SSR, SSGI and temporal-AA audits at `fc1d256152c9` on October 24, 2026. The comparison uses Apple hardware WebGPU and ANGLE Metal on Apple M3 for the WebGL test cases. Software adapters are rejected.

The merges improve several correctness properties, but the evidence does not establish uniformly better image quality. Stationary mirror views remain broadly similar. Controlled AA accuracy is essentially unchanged, a synthetic double-filtering stress test becomes blurrier, and recovery after camera motion has mixed results. Keep the fixes for their demonstrated correctness benefits; temporal filtering still needs quality tuning.

## Exact correctness comparisons

The same test definitions run against both implementations through HTTP module substitution. The original SSR fixes remain enabled in both. One new per-pixel age attachment test is skipped on the baseline because it requires an attachment the old implementation did not have. A separate output-only disocclusion test runs on both and compares observable reconstruction without inspecting internal storage. The new pure jitter helper export remains available to the test imports; the original runtime pipeline does not call it.

| Suite | SSR only | Combined audits |
| --- | --- | --- |
| SSGI | 4 passed, 6 failed | 10 passed |
| Coverage reconstruction | 20 passed, 3 failed, 1 intentionally skipped | 24 passed |
| Temporal pipeline | 0 passed, 2 failed | 2 passed |

All 36 comparison tests pass on the combined implementation. Some test cases run the same checks on both WebGPU and WebGL, so these counts do not represent distinct defects.

The meaningful improvements include:

- Projected solid angle: the central half of a frontal slice changes from `0.707107` to the numerical reference `0.5`; tilted-sector quadrature and matched-source/fallback normalization also pass.
- Sparse reconstruction: invalid bright texels no longer enter valid lighting through interpolation. The old fixture produced red-channel values from approximately 23.6 to 64 despite a valid source value of 4.
- Disocclusion: a newly revealed surface gives its second observation half weight, rather than inheriting the mature history weight of the rest of the screen.
- Jitter-only motion: lighting history uses the previous raster grid while physical motion vectors remain unjittered.
- Camera state: custom projections and existing tiled view offsets survive AA jitter and cleanup.

The second baseline temporal test fails only its expected jitter-index count. Its ownership, changing-projection and stationary-motion assertions already pass. It is not evidence that AA was disabled in the baseline. Full failure details are retained in the [pipeline comparison data](interior-merge-validation-data/pipeline-comparison.json).

## Antialiasing against an independent raster reference

A synthetic scene uses unlit diagonal silhouettes and subpixel lines at 256 by 192. Its reference is ordinary rasterization at four times the resolution in each axis, box-downsampled in linear RGB. Both implementations produce identical reference images. Measurements use frame 96 after temporal accumulation, no tone mapping, and linear float output.

| Fixture | SSR only RMSE | Combined RMSE | Interpretation |
| --- | ---: | ---: | --- |
| Sharp Beauty with an upstream temporal dependency | 0.012825 | 0.012864 | No meaningful demonstrated accuracy improvement; difference about +0.3% |
| Entire Beauty passed through temporal reprojection and then TRAA | 0.019145 | 0.030380 | More blur and approximately 59% higher reference error |

The first fixture preserves sharp input and uses a runtime-zero dependency to exercise the temporal graph without filtering geometry colors twice. The second intentionally filters the entire sharp image twice. The interior example denoises indirect lighting, so the stress result does not directly predict its Beauty output. It does show that fixing coordinate contracts does not by itself guarantee a sharper temporal filter.

Both implementations retain TRAA ownership, produce 32 distinct jittered projections, and have stationary motion below `6e-8` in these fixtures. The ownership defect is not reproduced here. Gradient-energy ratios also show the extra smoothing in the stress case, but that diagnostic alone is not a quality score. [Complete AA measurements](interior-merge-validation-data/aa-reference.json).

Reference and frame 96 images:

| Fixture | Reference | SSR only | Combined |
| --- | --- | --- | --- |
| Sharp input | [4x raster](interior-merge-validation-data/dependency-reference-4x.png) | [Before](interior-merge-validation-data/dependency-baseline-96.png) | [After](interior-merge-validation-data/dependency-current-96.png) |
| Double-filtering stress | [4x raster](interior-merge-validation-data/stress-reference-4x.png) | [Before](interior-merge-validation-data/stress-baseline-96.png) | [After](interior-merge-validation-data/stress-current-96.png) |

## Cornell mirror wall and Primitive mirror floor

Both methods render the requested scenes at 800 by 600, pixel ratio 1, with the same 1024 bake-atlas setting, materials, exposure, sampling selections, camera poses and frame schedules. Each method bakes independently; byte-identical baked inputs are not established. Comparisons therefore do not attribute arbitrary absolute RGB differences solely to the merges.

After warmup, both implementations retain TRAA ownership and produce 16 distinct geometry projections over 16 frames. The static Beauty images look broadly similar; there is no demonstrated major visual AA gain in these scenes. All four scene/method runs complete with no recorded browser or GPU errors.

| Scene | SSR only | Combined |
| --- | --- | --- |
| Cornell mirror wall | [Stationary](interior-merge-validation-data/cornell-before-stationary.png) | [Stationary](interior-merge-validation-data/cornell-after-stationary.png) |
| Primitive mirror floor | [Stationary](interior-merge-validation-data/floor-before-stationary.png) | [Stationary](interior-merge-validation-data/floor-after-stationary.png) |

The camera then pans horizontally over 32 frames by 10% of its distance from the orbit target. Recovery samples after stopping are compared against each implementation's own later 32-frame average, following another 128 warmup frames. The lighting feedback remains active. This measures whole-pipeline settlement, including ongoing lighting changes and stochastic noise; the target is not physical ground truth or a common cross-method image.

| Scene and stopping frame | SSR only linear HDR RMSE | Combined linear HDR RMSE |
| --- | ---: | ---: |
| Cornell, frame 8 | 5.8237 | 6.0881 |
| Cornell, frame 32 | 1.0883 | 1.5831 |
| Cornell, frame 64 | 0.5560 | 1.3817 |
| Primitive floor, frame 8 | 0.01129 | 0.00919 |
| Primitive floor, frame 32 | 0.00390 | 0.00403 |
| Primitive floor, frame 64 | 0.00301 | 0.00358 |

The early floor recovery improves, while later residuals and Cornell results worsen. Raw HDR error is especially sensitive to bright reflected content. These results do not support a universal temporal-quality improvement. They motivate separating noisy indirect-lighting recovery from final AA sharpness and tuning them against appropriate references.

## Reproduction

With dependencies installed, generate local development entry points and run:

```sh
node utils/build/dev.js
node test/benchmarks/interior-merge-validation.mjs
node test/benchmarks/temporal-aa-reference.mjs
```

The first runner stores all scene captures and full comparison data in `/tmp/three-interior-merge-validation`. `MERGE_VALIDATION_UNIT_ONLY=1` runs just the differential tests; `MERGE_VALIDATION_SKIP_UNITS=1` runs just scenes. The second runner writes its raster references and AA captures to `/tmp/three-merge-aa-reference`. Both use visible hardware Chrome and make no throughput claims. Changed test/benchmark files pass ESLint. Root `build/` artifacts are excluded from commits.
