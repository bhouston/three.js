# Interior SSGI algorithm and parameter audit

The `interior-blue-noise` implementation has a sensible lightweight SSGI architecture, but it is not a complete implementation of current reconstruction and GI best practices. Several correctness defects can be repaired locally. Larger limitations remain in directional fallback, source radiance, temporal reconstruction, and the geometry available to screen tracing. Replacing the noise pattern does not address those limitations.

This audit starts at `eedbf3e9833e`, covers `webgpu_interior.html`, and focuses on Cornell Box and Dual Rooms. The fixes are on `interior-ssgi-audit`. The previous [FAST sampling study](interior-sampling.md) supports retaining Original sampling as the starting configuration. Its reference uses the same screen-space integrator, so it measures sampling behavior rather than physical GI correctness. The changed integration and filter would require a new comparison to establish the sampling ranking after these fixes.

Implementation commit: `97a890a5970a` (`Correct SSGI projected coverage and sparse lighting reconstruction`).

## Existing sound choices

The example uses visibility-bitmask horizon gathering, rather than tracing independent hemisphere rays to a single hit. That is an established way to collect several contributions per screen direction with relatively low noise. Its fixed-thickness geometry remains an approximation. [SSILVB paper](https://arxiv.org/html/2301.11376v1)

The coverage output used by this example also makes several good choices:

- RGB represents incoming diffuse irradiance, before receiver albedo and material response. The final lighting model applies that response once.
- Coverage is independent of brightness. Valid black samples consume coverage; rejected source history leaves coverage to fallback.
- The coverage path applies source-facing rejection without multiplying by another emitter cosine. Once the input is outgoing radiance and integration is over solid angle, an extra source cosine would change the transport equation. [PBRT transport equation](https://pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/The_Light_Transport_Equation)
- Lighting history is linear HDR before final tone mapping and output conversion. Tracing current geometry against previous lighting avoids a cyclic dependency between passes.
- Diffuse reconstruction uses surface motion and constant roughness, rather than specular virtual-hit reprojection or roughness-dependent smoothing.
- SSGI replaces a fraction of the fallback estimate instead of simply adding all screen-space lighting to all baked lighting.

Feeding resolved previous lighting back into GI can approximate additional bounces. This is an established practice, not intrinsically an error; AMD distinguishes direct-only input for one bounce from composited input for multiple bounces. [Brixelizer GI documentation](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/brixelizer-gi/)

## Corrections implemented

### Projected solid angle

The old `ssgiSectorCoverage()` integrated `0.5 cos(beta)` across normal-relative sector angles. However, the slices rotate around the **view direction**, so their spherical area measure also depends on the polar angle relative to that direction. This coordinate system is explicit in the [authors' implementation explanation](https://cdrinmatane.github.io/posts/ssaovb-code/).

For projected-normal length `p`, signed projected-normal angle `n`, and sector angle `beta` relative to that normal, the corrected slice measure is:

```text
p * integral cos(beta) * abs(sin(beta + n)) d beta
```

An antiderivative before taking the sine's absolute value is:

```text
F(beta) = -cos(2 beta + n) / 4 + beta sin(n) / 2
```

The implementation splits intervals at the sine's zero. For a frontal normal, the middle 16 of 32 sectors previously received weight `0.707107`; the correct projected measure is `0.5`. Uniform-light normalization alone would not reveal this directional error.

Individual tilted slices have unequal full measures. Both lighting and coverage therefore normalize by the summed **full projected measure of the same sampled slices**, rather than just their count. This preserves a matched white screen source plus white fallback with only two slices. It is a normalized finite-sample estimator: it preserves the coverage partition, but does not make the whole screen-space method unbiased.

GPU tests compare sector weights with independent numerical quadrature, test tilted uniform illumination, and test two-slice complementary masks at different rotations.

### Sector classification and invalid geometry

Coverage mode now uses accurate `acos` and classifies sectors by their centers. The previous floor-start/ceil-width rule promoted arbitrarily small horizon errors into an entire occupied sector. With quantized normals this caused false coplanar coverage. The SSILVB paper uses a half-covered-sector rounding criterion; the new midpoint rule follows that intent. Thin geometry narrower than a sector can consequently be missed, which is the resolution tradeoff of a binary mask. [SSILVB sector criterion](https://arxiv.org/html/2301.11376v1#S3.SS1)

Background depth is skipped before reserving visibility bits. Degenerate projected normals use a guarded normalization. The legacy non-coverage path retains its original sector integration, quantization, and fast angle approximation.

### Spatial reconstruction

The old spatial stage interpolated unweighted RGB and coverage and only then multiplied them. These operations do not commute. Halfway between `(RGB=8, coverage=1)` and `(RGB=0, coverage=0)`, it produced premultiplied contribution `2` instead of `4`. Invalid bright texels could also contaminate valid lighting.

The nine spatial taps now read exact input texel centers, multiply each texel by its coverage before accumulation, and evaluate geometry at the corresponding depth texel center. History-clipping taps also use exact spatial-output texels. The filter retains nine source taps; fractional radii select discrete source texels rather than requesting hardware interpolation.

Regressions include a linearly filtered HDR input with bright invalid texels and a tilted plane reconstructed from a lower-resolution lighting buffer.

### Temporal validity and age

The old history gather validated individual bilinear taps, but the final blend independently validated interpolated geometry. That interpolation could match the current surface even when every underlying tap failed. The blend now uses the gather's actual accepted weight.

History age is now stored per pixel, independently of lighting coverage. Previously a newly revealed pixel inherited the global mature `15/16` history weight on its second accepted frame. It now receives the appropriate `1/2` weight. Zero lighting coverage does not itself invalidate geometry or reset age.

The age storage adds two R16F textures, approximately **7.9 MiB at 1920 by 1080**, plus associated bandwidth. `CoverageDenoiseNode` is shared with the optional coverage-aware SSR path, so these corrections also affect that path. The example's default SSR reconstruction uses its separate recurrent denoiser.

## Remaining integration limitations

### Ambient occlusion attenuates SSGI again

`updateAO()` installs GTAO on the entire resolved indirect-lighting chain. `PhysicalLightingModel.ambientOcclusion()` then multiplies all indirect diffuse lighting, including SSGI that has already integrated visible sectors. This is additional artistic attenuation, and can suppress bounce light. Baked GI also receives that multiplication.

An appropriate next comparison is AO disabled versus AO applied only to fallback contributions that need unresolved contact occlusion. Simply disabling AO globally would also change probe and specular behavior. This patch does not make that broader integration change.

### The source contains view-dependent specular

`lightingHistory` stores full beauty, including specular reflection. A camera-facing reflection on an object can therefore be used as though it were outgoing diffuse radiance toward another object. This is particularly relevant to Cornell metallic and mirror variants.

A separate SSGI source containing emission plus diffuse outgoing radiance would remove that approximation while retaining full beauty for SSR. This would deliberately exclude some specular-to-diffuse transport, which requires directional source information to reproduce correctly. It is a useful controlled experiment rather than a complete caustic solution.

### Scalar fallback loses missing-direction information

The current chain resolves approximately:

```text
covered screen irradiance + (1 - coverage) * whole fallback irradiance
```

The missing directions may have very different lighting from the hemisphere average. Colored Cornell walls and the Dual Rooms doorway are strong tests of this assumption. A directional probe/environment representation integrated over unoccupied sectors is a better match; the SSILVB authors demonstrate sampling ambient illumination over those sectors. A scalar light-map irradiance value cannot reconstruct that information. [Authors' slides, ambient lighting](https://cdrinmatane.github.io/cgspotlight-slides/ssilvb_slides.pdf)

### Feedback rejection is coupled to thickness

`createHistorySource()` uses the tracing thickness as its previous-depth tolerance. Geometric slab thickness and reprojection uncertainty are different quantities. They should have independent controls; a future rejection rule can use pixel footprint, surface-plane distance, motion, and depth precision.

The source reprojection assumes static world geometry. Supporting moving objects also requires accounting for hit-surface motion, rather than only transforming the current hit through previous camera matrices.

## Remaining tracing and denoising limitations

- A single depth layer cannot recover hidden or offscreen surfaces, true backfaces, or actual object thickness. More samples do not restore missing scene information.
- The world-space `radius` controls a projected search footprint, not a strict sphere around the receiver. There is no world-distance rejection, and `_halfProjScale` is half the usual pixel focal scale. Treat existing radius values as calibrated search parameters rather than guaranteed metric cutoffs.
- Screen-angle rotation is not uniform view-space azimuth away from the image center. The corrected per-slice integral does not fix that sampling-density approximation.
- Visibility is assigned in screen-search order, not by exact sorted intersections along every represented direction. Thirty-two binary sectors introduce angular quantization and can miss small bright emitters.
- The denoiser still uses a fixed spatial kernel and neighborhood extrema clipping. It has no luminance moments, variance-guided multiscale filtering, explicit lighting-change confidence, or adaptive work for disoccluded regions.
- A very small surviving history footprint can still be normalized to full history confidence. Per-pixel age repairs one failure mode, not every disocclusion artifact.

SVGF is a strong next denoising reference: it combines temporal statistics with variance-guided hierarchical filtering. NRD additionally documents directional SG/SH reconstruction against full-resolution guides for reduced-resolution output. These are larger quality improvements than changing a noise texture alone. [SVGF authors](https://research.nvidia.com/labs/rtr/publication/schied2017spatiotemporal/), [NRD documentation](https://github.com/NVIDIA-RTX/NRD/blob/master/README.md#iq-vs-performance-tracing-and-denoising-resolution)

## Parameters shared across scenes

There is no universally correct numeric SSGI or SSR preset. Separate the **quality budget**, **geometric search scale**, and **lighting response**. The following are starting points, not a measured optimum for either scene.

| Parameter | Current interior value | Guidance |
| --- | --- | --- |
| SSGI sampling | Original | Retain the measured choice. |
| Resolution scale | 0.5 | Reasonable medium setting; 1 is the quality baseline for edges and small objects. Half width and height traces one quarter of the pixels. |
| Slices and steps | 2 and 8 | 32 depth candidates per traced pixel. Test 3 by 8 against 2 by 12 at the same 48-candidate budget: more slices address angular instability; more steps address radial misses. |
| Step exponent | 2 | Reasonable near-field concentration; retain until distance sweeps expose a specific problem. |
| GI intensity | 1 | Retain. Increasing gain does not fix missing coverage or incorrect integration. |
| Backface lighting | 0 | Retain for ordinary opaque one-sided transport. |
| Linear thickness | Disabled | Retain; tying thickness to camera far distance is not a geometric model. |
| History | 16 | Start at 8–16 for interaction. Longer history reduces noise while increasing lag; it is not a substitute for lighting-change rejection. |
| Spatial radius | 1.5 source texels | Start at 1–1.5. Larger values can blur geometry and lighting detail. |
| History depth threshold | 0.01 of view depth | Existing denoiser heuristic, not a universal physical tolerance. Keep source-history tolerance separate. |
| Normal history threshold | Dot product greater than 0.9 | About 25.8 degrees; a fixed compromise rather than curvature-aware rejection. |
| Maximum luminance | 35 | Scene-lighting-dependent clamp on conditional irradiance. It introduces bias; test against the unclamped/high-threshold result before using it to hide fireflies. |

At saturation, 16-frame accumulation is an exponential average with history weight `15/16`, not a sliding window of 16 observations. Replacing 95 percent of old lighting takes about 46 frames, or 0.77 seconds at 60 Hz. At 8 frames it takes about 22 frames, or 0.37 seconds. Feedback and final antialiasing can add response delay. These times are analytical filter behavior, not measurements of the complete example.

## Cornell Box and Dual Rooms parameters

The example currently multiplies radius and thickness by the whole model's bounding-box diagonal:

| Scene | Bounds in world units | Diagonal | Current radius at 0.25 | Current thickness at 0.01 |
| --- | --- | --- | --- | --- |
| Dual Rooms | 16 by 5 by 8 | 18.5742 | 4.6435 | 0.18574 |
| Cornell Box | 20 by 15.5 by 20 | 32.2529 | 8.0632 | 0.32253 |

Dual Rooms includes a table top approximately 0.12 thick, legs 0.10 thick, and a divider 0.30 thick. A single proxy thickness of 0.186 cannot faithfully represent them all. Extending a level with another room should not change an unchanged table's effective thickness.

Keep the shared quality settings above while testing these geometric candidates:

| Scene | Radius candidates at relative 0.10 / 0.15 / 0.25 | Thickness candidates at relative 0.0025 / 0.005 / 0.01 |
| --- | --- | --- |
| Dual Rooms | 1.857 / 2.786 / 4.644 | 0.0464 / 0.0929 / 0.1857 |
| Cornell Box | 3.225 / 4.838 / 8.063 | 0.0806 / 0.1613 / 0.3225 |

For Dual Rooms, begin thickness comparisons near the thinner candidates and inspect the table, doorway, and divider together: reducing thickness can restore gaps while worsening wall leakage. For Cornell, inspect box contacts, wall color bleeding, and grazing surfaces. The larger scene justifies testing a larger search footprint; it does not prove that all surfaces need a thicker proxy. The current GTAO radius is an absolute 0.4, so it does not scale with these SSGI settings.

### Automatic scene loading

Encode shared low/medium/high quality presets according to hardware and resolution. Read geometric overrides from scene metadata, ideally per room or surface category. A practical interface is independent `ssgiRadius`, `ssgiThickness`, `ssrMaxDistance`, `ssrThickness`, and `historyDepthTolerance` values, plus a declared world-unit scale or local room bounds.

For scenes without metadata, the current bounding-diagonal policy can remain a fallback. It provides scale consistency under uniform model scaling, not an automatic quality optimum. Prefer a local room diagonal over the bounds of an entire level. Do not infer physical thickness from a camera depth buffer or arbitrary mesh bounding boxes: sheets, rotated walls, and compound furniture make those guesses unreliable.

SSR maximum distance is also a reach parameter that can follow scene scale. Its thickness is an intersection acceptance tolerance, while SSGI thickness constructs an angular occluder slab; use separate values even when they happen to start numerically equal. SSR quality, roughness cutoff, refinement, and denoising settings can mostly follow shared quality/material policy. A luminance clamp belongs to lighting configuration, not bounds-based scene loading.

This patch deliberately preserves the example's parameter defaults. Numerical tuning against the old biased integration/reconstruction would confound correctness with artistic compensation.

## Techniques to investigate after FAST

1. **Variance-guided reconstruction and anti-lag.** Add temporal moments, short-history handling, and adaptive spatial filtering. Test camera motion and changing lights, not only stationary convergence.
2. **Directional fallback and source separation.** These directly address missing-direction bias and reflected highlights becoming diffuse sources.
3. **Depth hierarchy and better geometry proxies.** Explore carefully filtered depth MIPs for bandwidth and thin-geometry behavior, and authored or additional-layer thickness where available. XeGTAO is a useful implementation reference, but its AO heuristics cannot be copied indiscriminately into a radiance gather. [XeGTAO](https://github.com/GameTechDev/XeGTAO)
4. **Screen probes, radiance caches, and lighting-guided importance sampling.** These change where work is spent and provide information beyond a single screen buffer. Lumen describes previous-lighting-guided sampling and screen/world radiance caches. This is an architectural project, not a noise-pattern swap. [Lumen engineers' explanation](https://www.unrealengine.com/tech-blog/unreal-engine-5-goes-all-in-on-dynamic-global-illumination-with-lumen)
5. **Hilbert-indexed R2 sampling**, if another low-cost sampling comparison is desired after the filter is corrected. XeGTAO documents this alternative and its temporal behavior; it is not a guaranteed improvement here. [XeGTAO sampling](https://github.com/GameTechDev/XeGTAO/blob/master/README.md#sampling-noise)

## Validation and reproduction

All **28 targeted GPU tests passed**, with zero skips and zero rendering errors: 2 FAST atlas tests, 10 SSGI cases including both hardware backends for the mathematical tests, and 16 coverage-denoiser tests. ESLint passes on both changed nodes, their tests, and the new benchmark script. The WebGPU adapter reports `vendor: apple`, `architecture: metal-3`, and `isFallbackAdapter: false`; WebGL reports `ANGLE Metal Renderer: Apple M3`. The host has an Apple M3 with 10 GPU cores.

The scene harness preserves the production lighting pipeline and defaults, with a 960 by 720 viewport, pixel ratio 1, and a 1024 light-map atlas for practical comparison. It captures both requested scenes and measures the full Beauty pipeline. Medians of three five-second runs:

| Scene | Baseline completed FPS | Corrected completed FPS | Change |
| --- | --- | --- | --- |
| Cornell Box | 107.72 | 108.19 | +0.43 percent |
| Dual Rooms | 181.87 | 181.81 | -0.03 percent |

These small differences do not establish a speedup or slowdown. The full pipeline performed 28 draws per Cornell frame and 64 per Dual Rooms frame, and the harness verified that both SSGI-denoiser and SSR update counts matched completed frames. The age buffers add storage even though total throughput stayed similar in this setup.

Both scenes render successfully. Filtered SSGI and coverage visibly change after the weighting corrections; the final Beauty change is modest with the baked fallback enabled. Residual structured noise remains visible in the isolated SSGI output, so these fixes should not be described as a complete visual denoising solution. The captures contain no moving-camera or changing-light sequence; temporal rejection and disocclusion changes are covered by the synthetic regressions.

Raw counts and adapter metadata are retained in [benchmark data](interior-ssgi-audit-data/results.json) and [test data](interior-ssgi-audit-data/unit-tests.json). Sixteen comparison PNGs are written to `/tmp/three-ssgi-audit` by the harness; their names are listed in the benchmark data.

```sh
node utils/build/dev.js
SSGI_AUDIT_UNIT_ONLY=1 SSGI_AUDIT_FILTERS=FASTNoise,SSGINode,CoverageDenoiseNode node test/benchmarks/interior-ssgi-audit.mjs
node test/benchmarks/interior-ssgi-audit.mjs
```

The benchmark disables vsync, frame limits, and software rasterizers, verifies the adapter, warms each implementation, drains queued GPU work, and runs implementations sequentially. Each of three repetitions counts GPU-completed batches of complete frames over at least five seconds. It performs no per-frame timing or readbacks in the timed loop. `SSGI_AUDIT_BASE` selects the baseline commit; `SSGI_AUDIT_METHODS`, `SSGI_AUDIT_MODELS`, and `SSGI_AUDIT_OUTPUT` select runs and output location. Discard generated root `build/` changes after validation.

The analytic and regression tests establish the repaired properties. Scene captures establish rendering viability and reveal visual changes. A path-traced reference with matching materials, lights, exposure, and camera is still needed to rank parameter presets by physical accuracy; the existing screen-space sampling reference cannot establish that ranking.
