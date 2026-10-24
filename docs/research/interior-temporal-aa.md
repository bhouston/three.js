# Interior temporal AA and camera jitter audit

Audited `interior-blue-noise` at `eedbf3e9833e` on 24 October 2026.

The interior pipeline already uses several sound foundations: centered Halton(2,3)
jitter, nearest-depth motion-vector dilation for final TRAA, neighborhood variance
clipping, HDR-aware blending, and a final temporal resolve before output tone
mapping. MSAA is disabled. However, the jitter and motion-coordinate contracts were
inconsistent across the lighting denoisers and final resolve.

## Findings and fixes

1. **The reflection denoiser could disable final AA jitter.**
   `TemporalReprojectNode`, although explicitly a non-jittering denoiser, registered
   pipeline callbacks and claimed `viewOffsetOwner`. TRAA built its upstream nodes
   before trying to claim ownership, so a lighting dependency could win. Its
   first-update synchronization could also rebuild the camera projection after
   geometry had rendered. Removed the denoiser's projection mutation and callbacks;
   TRAA claims ownership before building dependencies. Only AA controls camera
   jitter. Debug outputs without AA consequently render without AA jitter.

2. **Coverage-denoiser motion used the wrong units and Y direction.**
   `VelocityNode` produces current-minus-previous NDC motion. The denoiser subtracted
   it directly from top-left-origin UV coordinates. It now multiplies by
   `(0.5, -0.5)` first. Previously horizontal history moved twice as far as it
   should, and vertical history moved twice as far in the opposite direction.
   Both coverage-aware SSR and SSGI use this denoiser.

3. **Raw lighting histories omitted the jitter delta.**
   Motion vectors intentionally exclude camera jitter. But the lighting denoisers
   store images on each frame's jittered raster grid, unlike final resolved AA
   color history. Their surface reprojection now uses:

   ```text
   previousUV = currentUV - motionNDC * (0.5, -0.5)
                + previousRasterJitterUV - currentRasterJitterUV
   ```

   A shared helper measures raster displacement relative to the projection used
   by velocity, including perspective, orthographic, and tiled projections.
   Updated both `CoverageDenoiseNode` and `TemporalReprojectNode`. Final TRAA color
   history continues using jitter-free motion; its raw previous-depth lookup now
   receives the jitter delta. SSR virtual-hit reprojection already projects through
   the previous jittered matrix, so it needs no second correction.

4. **TRAA overwrote camera projection/view state.**
   Calling `updateProjectionMatrix`, replacing `camera.view`, and clearing it
   discarded custom projections and tiled view offsets. TRAA now translates clip
   space on the existing projection, updates its inverse, and restores the original
   projection afterward. Cleanup is guarded because the first compilation frame
   can register an after callback after the before-callback phase has already run.

## Pipeline contracts checked

| Stage | Projection / history convention |
| --- | --- |
| Normal/depth pre-pass and lit scene pass | Same current jittered camera |
| Motion vectors | Current and previous unjittered projection; current-minus-previous NDC |
| GTAO, SSGI, SSR ray generation | Current projection and inverse, including jitter |
| Previous scene lighting source | Previous jittered projection/inverse captured during scene rendering; depth validation |
| Lighting denoisers | Raw raster history; NDC-to-UV conversion plus jitter delta |
| SSR virtual-hit history | Direct projection using previous jittered matrix |
| Final TRAA color | Accumulated output grid; jitter-free motion reprojection |
| Final TRAA depth validation | Previous raw depth grid; jitter delta included |
| Output transform | After temporal resolve, preserving linear HDR histories |

The previous-lighting copy in `webgpu_interior.html` already snapshots the matrices
inside `scene.onAfterRender`, before AA restores the camera. This correctly keeps
its color, depth, and projection snapshots together. GTAO's temporal option varies
sampling over time; it does not maintain another reprojected history buffer.

## Validation

Ran ESLint on all changed JavaScript files. Ran the existing hardware-only test
runner, extended to include `TemporalPipeline`:

```sh
node utils/build/dev.js
SAMPLING_UNIT_ONLY=1 SAMPLING_OUTPUT=/tmp/three-temporal-aa-unit-final \
  node test/benchmarks/interior-sampling.mjs
```

All 23 tests passed without skips: FASTNoise (2), SSGINode (4),
CoverageDenoiseNode (15), and TemporalPipeline (2). Chrome 155 used an Apple
Metal-3 WebGPU adapter, with `isFallbackAdapter: false`; software rasterization
was disabled. New GPU tests verify horizontal/vertical motion, jitter-only
reprojection, AA ownership through an upstream denoiser, changing geometry jitter,
zero stationary motion, and projection restoration. Perspective/orthographic
camera tests cover existing tile settings and custom projection preservation.

Also rendered Dual Rooms at 960×720 with a reduced 1024 bake atlas. Instrumented
both SSR resolve orders: each retained TRAA ownership, produced eight distinct
scene projection samples over eight steps, and restored the camera projection.
Captured and inspected beauty and lighting debug images; browser error capture was
empty. Local captures are in `/tmp/three-temporal-aa-final`.

This is a coordinate/lifecycle correctness audit, not a new throughput or image
quality benchmark. No FPS or quantified quality improvement is claimed. The
existing blue-noise sampling measurements predate these fixes and should be rerun
before using them to rank sampling methods with the corrected Beauty pipeline.
Animated transparency, camera-cut detection, WebGL fallback, and TAAU-specific
reconstruction are not validated by these tests. Existing neighborhood clipping
and geometry checks remain heuristics and cannot eliminate every ghosting case.

## Primary references

- [AMD FSR2 integration documentation](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/super-resolution-temporal/): jitter, jitter cancellation for motion vectors, depth conventions, and temporal history inputs. The coordinate contracts apply here without implying that this implementation is FSR2.
- [Playdead's temporal AA source and INSIDE presentation](https://github.com/playdeadgames/temporal): projection jitter, temporal reprojection, neighborhood history clipping, and accumulation.
- [NVIDIA: Adaptive Temporal Antialiasing](https://research.nvidia.com/sites/default/files/pubs/2018-08_Adaptive-Temporal-Antialiasing/adaptive-temporal-antialiasing-preprint.pdf): reprojection failures at disocclusions, subpixel features, transparency, shadows, and reflections; limits of neighborhood heuristics.
