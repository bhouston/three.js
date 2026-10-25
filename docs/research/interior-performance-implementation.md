# Interior performance implementation results

The interior example now defaults to half-resolution SSR tracing, caches validated SSGI source radiance, and batches compatible static meshes after baking. Render Scale controls the native pixel ratio independently of the interface. Recurrent denoising retains its eight-sample default and offers a four-sample spatial budget as an explicit quality/performance option. Temporal reconstruction and geometry guides retain their full rendering resolution.

On Apple M3 at 1280 by 720, the final defaults measure approximately 109 fps in the hallway and 89 fps in the interior dungeon view, compared with approximately 83 and 71 fps in the original study. These cumulative figures come from separate warmed runs; the individual comparisons below isolate each change. The cameras and sustained GPU-completed-frame measurement method are described in the [original study](interior-performance.md).

## Changes and individual validation

| Change | Commit | Validation and measured result |
| --- | --- | --- |
| Render Scale control | `7bdb4bb92a38` | Both scenes at device pixel ratio 2; scale changes 1 → 0.5 → 0.8 → 1, resize events, temporal-target dimensions, finite output, and camera motion/recovery pass. |
| Half-resolution SSR default | `b55c72947d82` | Alternating full/half/full tracing: hallway about 83 → 102 fps; dungeon about 70 → 85 fps. Full-resolution tracing remains selectable. |
| Recurrent denoising sample budget | `5ca4e7ab2e48` | Four samples give about 102 → 105 fps in the hallway and 84 → 88 fps in the dungeon. Eight GPU tests cover constant HDR energy, finite bounded budgets, normal edges, and existing demodulation behavior; temporal tests and scene motion/recovery pass. Four samples remain opt-in because they reduce spatial noise suppression. |
| SSGI source cache | `284a4c69d49e` | Hallway about 101 → 110 fps; dungeon about 85 → 87 fps. Cached and direct sampling agree within `1e-5` in GPU fixtures covering rejection, camera motion, raster boundaries and resized guides. History invalidation, scene resizing and motion/recovery pass. |
| Generic static batching addon | This change | Dungeon about 87 → 89 fps; 357 → 91 draw calls, with 75 spatial/material batches. Diagnostic render-pipeline CPU time falls from about 4 ms to about 2 ms. The hallway has no compatible material-sharing pairs, and shows no established batching gain. |

The SSGI cache uses a nearest-sampled RGBA32F texture on the current depth-texel grid, preserving the validated RGB and coverage values. This adds approximately 14 MiB at 1280 by 720. It is opt-in on the reusable radiance helper and enabled for the example's diffuse source. Outside the current raster grid, evaluation retains the original direct path.

The batching comparison alternates disabled/enabled/disabled/enabled, with three five-second measurements per configuration. Small throughput changes should be interpreted alongside the more substantial reduction in CPU work. Geometry is merged within spatial cells, retaining separate culling bounds. Original geometries and shared materials are retained for rebaking and cleanup.

## Reusable batching API

[batchStaticMeshes.js](../../examples/jsm/utils/batchStaticMeshes.js) is a renderer-independent addon, also exported through `three/addons`. The example calls the same utility for every selected model. Compatible static meshes can be batched in any scene:

```js
import { batchStaticMeshes } from 'three/addons/utils/batchStaticMeshes.js';

const batches = batchStaticMeshes( gltf.scene, {
    animations: gltf.animations
} );

// Before changing descendant transforms, geometry, or starting a new bake:
batches.dispose();
```

Options include `cellSize` in root-local units, animation clips, and an additional mesh filter. The result exposes `group`, `sourceMeshes`, `batchCount`, and idempotent `dispose()`. The root can still move as a whole. Animated descendants identified by supplied or attached clips are excluded; externally driven meshes must be excluded with the filter. Skinned, morphed, instanced, transparent and unsupported custom-rendering meshes remain untouched.

The utility preserves atlas UVs, material references, shadow flags, layers, render order and ancestor group ordering. Quantized transformed attributes and half-float attributes are promoted to decoded float values before merging, avoiding truncation or loss of half-float interpretation. Cleanup frees only the new batch geometry and restores the originals.

In the dungeon's unlit atlas comparison, 45 of 921,600 pixels exceeded a linear RGB difference of 0.005 between original and batched rendering, consistent with small raster differences from baking transforms into vertices. Vertex atlas UV values are checked directly in the utility tests. Batching toggles, complete rebaking and switching to the hallway all pass; the rebake restores all 798 original dungeon meshes before recreating batches.

## Retained results

- [Render Scale validation](interior-performance-data/render-scale-validation.json).
- [SSR default comparison](interior-performance-data/ssr-default-validation.json).
- [Denoising budget scene comparison](interior-performance-data/denoise-budget-validation.json) and [GPU tests](interior-performance-data/denoise-budget-units.json).
- SSGI source caching: [disabled](interior-performance-data/source-cache-before.json), [enabled](interior-performance-data/source-cache-after.json), [GPU tests](interior-performance-data/source-cache-units.json).
- [Generic batching comparison and lifecycle validation](interior-performance-data/batching-validation.json).
- [Final selected unit tests](interior-performance-data/implementation-units.json): 63 passed, none failed or skipped. Changed code passes ESLint.

Hardware adapters are verified and software rasterizers are disabled. Implementations run sequentially with vsync and frame-rate limits disabled; each timing interval counts completed GPU work over at least five seconds. Inspector CPU/GPU collection runs separately from throughput measurement. The original study's limitations on Inspector GPU durations still apply.

Run the final selected suites with:

```sh
INTERIOR_PROFILE_UNIT_FILTERS='batchStaticMeshes,RecurrentDenoiseNode,ReprojectedRadianceNode,SSGINode,CoverageDenoiseNode,TemporalPipeline' \
INTERIOR_PROFILE_OUTPUT=/tmp/three-interior-units \
node test/benchmarks/interior-performance.mjs
```

Run batching throughput, resize/motion checks, atlas comparison, rebaking and model-switch validation with:

```sh
INTERIOR_PROFILE_VARIANTS='no-batching,baseline,no-batching,baseline' \
INTERIOR_PROFILE_VALIDATE=1 INTERIOR_PROFILE_VALIDATE_BATCHING=1 \
INTERIOR_PROFILE_OUTPUT=/tmp/three-interior-batching \
node test/benchmarks/interior-performance.mjs
```

`INTERIOR_PROFILE_SOURCE_CACHE=0` disables the example's source cache through HTML substitution for comparison. `ssr-full` selects original tracing resolution, and `ssr-filter-4` selects the reduced denoising budget. The initial study's timing data describes its earlier defaults; the benchmark's `baseline` always uses the checkout's current defaults.
