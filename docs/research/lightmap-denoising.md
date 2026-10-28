# Denoising UV-space light bakes

Research date: 2026-10-28. Scope: [zalo/three.js PR #2](https://github.com/zalo/three.js/pull/2), inspected at head `76f49e8facba9ded04eecb0ee35fd6b4b38568c5`. This is an analysis and experiment proposal, not a new denoiser implementation or a measured quality/performance claim.

## Recommendation

Yes, ray-traced lightmaps can be denoised, and this is established production practice. Start with the variance-guided lightmap denoiser **already present in this PR branch**, retain final-pass denoising as the default, and evaluate cross-chart surface neighborhoods as an incremental improvement. Compare against native Open Image Denoise's dedicated `RTLightmap` model before investing in a browser ML integration.

Denoising improves finite-budget reconstruction and can reach acceptable visual quality with fewer rays. It does not make the original Monte Carlo sample mean converge faster. Filtering exchanges variance for bias: a permanently blurred solution can remain wrong even after infinitely many rays. A useful quality target must measure lost shadow detail and light leakage as well as residual grain. Sampling/path-guiding improvements are complementary.

## What this branch already does

These observations come from the pinned source, not from assuming that a released Three.js package has these addons:

- [LightMapDenoiser.js](https://github.com/bhouston/three.js/blob/76f49e8facba9ded04eecb0ee35fd6b4b38568c5/examples/jsm/lighting/LightMapDenoiser.js) implements four default à-trous passes with a 5×5 B3 kernel and increasing tap spacing. It rasterizes world position, normal and chart identity into UV-space guide textures. It rejects other charts and unsampled texels, compares normals and tangent-plane displacement, and makes luminance rejection depend on estimated noise. It fits local linear gradients rather than just averaging; that reduces one-sided border bias, but does not establish that the complete nonlinear denoiser is unbiased. Moments provide a per-texel noise estimate; a relative-noise fallback also exists. It has a correlated-noise option for stochastic radiosity.
- [LightBaker.js](https://github.com/bhouston/three.js/blob/76f49e8facba9ded04eecb0ee35fd6b4b38568c5/examples/jsm/lighting/LightBaker.js) enables `denoise` and `stitchSeams` by default. `process()` denoises each atlas page and then stitches it. The bake normally presents raw samples until completion and processes afterwards. `LightMapper` is constructed with moments enabled. This already implements the proposed position/normal-guide approach.
- [LightMapSeams.js](https://github.com/bhouston/three.js/blob/76f49e8facba9ded04eecb0ee35fd6b4b38568c5/examples/jsm/lighting/LightMapSeams.js) matches world-space edges with compatible normals, solves least-squares constraints on bilinear seam samples using GPU Jacobi iterations, and refills padding. It operates per page; matching is across the objects supplied on that page. Stitching boundary values is different from supplying complete filter neighborhoods across charts.
- The filter reads accumulated irradiance into separate targets. Its hard chart rejection prevents atlas bleeding but cannot borrow samples from a physically adjacent chart. Its spatial stencil is still measured in UV texels, so the world-space guide does not make the support radius invariant to texel density. The dominant direction texture is assigned separately in `LightBaker`; the RGB denoiser does not also denoise that directional representation. Residual normal-map/directional artifacts deserve separate inspection.

Three.js also has camera-space denoisers. [DenoiseNode](https://threejs.org/docs/pages/DenoiseNode.html) takes depth, normals and a camera and targets signals such as raw AO. The checked-out branch also contains `BilateralBlurNode`, `RecurrentDenoiseNode` and `PoissonDenoiseShader`. Their screen-space assumptions make them references rather than drop-in atlas filters. Do not confuse upstream screen-space support with the dedicated lightmap addons in this fork.

## Published and production precedents

| Source | What has been tried | Relevance and limitation |
| --- | --- | --- |
| [Dammertz et al., 2010, Edge-Avoiding À-Trous Wavelet Transform](https://diglib.cgv.tugraz.at/items/89ef8071-636b-407c-b56c-2e2ab391bf0a) | Multiscale edge-aware reconstruction of noisy GI. | Foundation for a compact GPU filter; original image neighborhoods do not solve atlas adjacency. |
| [Schied et al., 2017, SVGF](https://research.nvidia.com/publication/2017-07_spatiotemporal-variance-guided-filtering-real-time-reconstruction-path-traced) | Temporal accumulation and luminance variance drive hierarchical image-space filtering; publication links implementation in Falcor. | Borrow variance-aware spatial filtering. A static lightmap already has stable texel identities and does not need camera reprojection. |
| [Frostbite, 2018, Battling Noise](https://developer.nvidia.com/blog/?p=12209) | Tracks variance per lightmap texel, filters in lightmap space using hierarchical à-trous before presentation, and fades filtering as convergence improves. | Direct evidence for progressive preview denoising, not only final processing. This is a production description, not an open-source baker. |
| [Saber/AMD, GDC 2019, Enabling Real-Time Light Baking Workflows](https://gpuopen.com/gdc-presentations/2019/gdc-2019-s1-enabling-light-baking-workflows.pdf) | Bilateral filtering in lightmap/spatial space guided by normals, positions and radiance. | Direct precedent for the proposed joint bilateral filter; the presentation is not a reusable implementation. |
| [Munkberg et al., 2016, Texture Space Caching and Reconstruction for Ray Tracing](https://www.intel.com/content/dam/develop/external/us/en/documents/author-preprint-texture-space-caching-and-reconstruction-for-ray-tracing-676729.pdf) | Caches and reconstructs surface shading on local texture grids, including secondary-ray shading. | Surface reconstruction is well motivated beyond the screen; this is broader than post-bake denoising. Fixed discretization/reconstruction can retain bias. |
| [Binder et al., 2019, Massively Parallel Path Space Filtering](https://research.nvidia.com/publication/2019-02_massively-parallel-path-space-filtering) | Shares path-vertex contributions through spatial hashing. | An architectural precedent for world-space reuse, not a ready-made UV lightmap postprocess. |
| [Burley and Lacewell, 2008, Ptex](https://media.disneyanimation.com/technology/opensource/ptex/ptex.pdf), [open-source project](https://www.disneyanimation.com/open-source/ptex/) | Per-face textures filtered with mesh adjacency. | Proof that disconnected texture storage can support continuous surface filtering. Ptex itself is not a Monte Carlo denoiser. |
| [Silvennoinen and Sloan, 2019, Ray Guiding for Production Lightmap Baking](https://www.activision.com/cdn/research/Ray_Guiding_for_production_Lightmap_Baking_author_final.pdf) | Improves the sampling distribution for lightmap integration. | Addresses variance at its source, especially difficult indirect illumination; combine with denoising rather than interpreting blur as improved sampling. |

## Open-source options

**Godot is the most directly useful external implementation to study.** Its [lightmap documentation](https://docs.godotengine.org/en/latest/tutorials/3d/global_illumination/using_lightmap_gi.html) describes GPU joint non-local means (JNLM) and an optional OIDN workflow. The former is included; the latter uses a separately installed executable. Inspect the MIT-licensed [compute shader](https://github.com/godotengine/godot/blob/master/modules/lightmapper_rd/lm_compute.glsl) (`MODE_DENOISE`) and [bake orchestration](https://github.com/godotengine/godot/blob/master/modules/lightmapper_rd/lightmapper_rd.cpp). This supplies both a traditional GPU baseline and a production integration example. It does not imply that an arbitrary atlas can safely be blurred without chart treatment.

**Native OIDN provides an actual lightmap model.** The [documented `RTLightmap` filter](https://www.openimagedenoise.org/documentation.html#rtlightmap) accepts HDR color, or normalized directional lightmap coefficients with its directional option. Unlike the generic `RT` example, its documented inputs do **not** include position, normal or albedo guides. It also does not expose a chart mask or mesh adjacency. Plan chart-aware extraction/extension and compare seams rather than assuming the network understands atlas topology. Manual tiling needs the specified overlap/alignment and consistent HDR input scaling. The [Apache-2.0 source](https://github.com/RenderKit/oidn) is reusable, but native CPU/CUDA/HIP/SYCL integration is not a browser WebGPU dependency.

**Browser ML is possible, but verify the model first.** The MIT [pmndrs/denoiser](https://github.com/pmndrs/denoiser) project runs OIDN networks through browser runtimes and integrates with Three.js on a shared GPU device. Its README describes path-traced frames, alpha API changes, and Apple-only validation so far. That is a candidate for experimentation, not evidence that the dedicated `RTLightmap` weights and atlas handling are already supported. Verify those requirements and Windows GPU behavior before choosing it.

[Unity's official artifact guide](https://docs.unity.com/en-us/engine/6000.3/manual/lighting-overview/direct-and-indirect-lighting/lightmapping/baking-before-runtime/troubleshooting/troubleshooting-lightmapping-artifacts) is a useful counterexample to treating denoising as automatic: it documents blur/leaks and low-sample à-trous speckles. Production adoption does not remove the need for enough initial samples.

## What guides and neighborhoods should we use?

The following are proposed design choices, not measurements from this investigation.

Filter linear HDR **irradiance**, before applying the receiving material's albedo or tone mapping. Preserve raw sums, counts and moments. A useful guide record contains position, normal, validity/coverage, chart ID and texel footprint. Geometric normal and surface/primitive identity help distinguish folds and nearby disconnected layers. Position can instead be reconstructed from triangle identity and barycentrics; textures are convenient, not mandatory.

A joint bilateral weight can combine a spatial kernel, normal agreement, tangent-plane/world distance, a validity/connectivity gate, and a variance-scaled luminance difference. Geometry alone cannot protect a shadow edge on a flat wall. Conversely, aggressive luminance rejection preserves isolated bright noise as false edges. Use enough independent samples, stabilize local variance, and test HDR outliers. À-trous increases support cheaply with sparse passes; NLM compares patches and costs more. Neither recovers illumination paths that were never sampled.

UV seams introduce three separate problems:

1. Unrelated charts lie next to each other in the atlas: masks/chart rejection prevent bleeding.
2. Adjacent surface points lie far apart in the atlas: chart-local rejection leaves asymmetric neighborhoods and potentially different estimates on each side.
3. Bilinear/mipmap sampling crosses chart borders: gutters and chart-safe mip generation are still required after filtering. Padding is not a substitute for cross-chart denoising.

For this baker, prefer **surface adjacency as the next experiment**. Reuse seam-edge discovery to map taps through neighboring triangles/charts, with orientation and texel-density changes handled explicitly. Extend correspondence across atlas pages. Retain normal/visibility-aware rejection: a UV cut is artificial, but a crease can have a real irradiance discontinuity. Existing seam stitching can remain as a final sampling-consistency step.

A second option is a **world-space surfel/texel hash**: insert valid texels using their positions, query a radius based on world-space footprint, reject incompatible normals/layers/connectivity, and write results back into the same atlas. This reaches across UV seams and mesh boundaries. Euclidean proximity alone is unsafe: close parallel surfaces can exchange light through a wall even with similar normals. Tangent-plane tests help but do not prove connectivity or visibility. Stronger gates cost preprocessing, memory or rays. Weight for surface area when densities vary so oversampled charts do not dominate.

Mesh/geodesic neighborhoods most directly follow the surface and avoid many proximity errors, at the cost of irregular GPU reads and topology work. A full Ptex conversion is unnecessary: retain UV storage and change only how filter neighbors are found.

## When to denoise, and what does convergence mean?

Keep final processing as the first baseline. Optional progressive preview processing can run every several batches from raw accumulations, as Frostbite demonstrates. Do not repeatedly filter yesterday's filtered preview and call it a new sample. Blend toward raw results or reduce filtering strength with confidence to avoid a persistent smoothing floor.

Do not use post-filter smoothness as the bake stopping statistic. Raw moments assess sampling variability, not reconstruction bias. This baker also uses an iteratively solved bounce cache in `RayTracedRadiosity`: final texel noise cannot establish that the cache is unbiased or sufficiently resolved. Feeding a filtered map back into that cache would change the transport solve and could spread leaks; treat that as a separate, explicitly biased experiment.

Where direct lighting is baked, consider filtering direct and indirect components independently to preserve sharp direct shadows. Default live analytic direct lighting reduces that concern but does not remove indirect occlusion edges. Directional maps need coherent treatment of intensity and directional coefficients; blindly blurring unit directions or signed coefficients as positive RGB is invalid.

## Concrete evaluation plan

1. Establish raw / current denoiser / current denoiser plus stitching baselines at fixed ray budgets. Include Cornell Box, Courtyard and Space Ship Hallway, plus a smooth plane split across charts/pages, unequal texel densities, thin parallel walls and a sharp coplanar shadow.
2. Build a substantially higher-sample reference with a sufficiently converged bounce cache. Compare linear irradiance error on covered texels with surface-area weighting, bilinear seam mismatch in surface space, and rendered shadow/contact-detail crops. Use multiple independent seeds. Do not score atlas padding or only a tone-mapped screenshot.
3. Compare native `RTLightmap` as an offline quality baseline with chart-boundary treatment documented. Then prototype adjacency-aware taps using the existing seam metadata. Add world hashing only if adjacency fails to cover important modular surfaces or preprocessing becomes impractical.
4. Measure time to a specified reference error/quality, including preprocessing, denoising, stitching, transfers and memory. Report raw-noise and filtered-error targets separately; no numeric speedup is justified yet.
5. Verify screenshots on a confirmed physical WebGPU adapter. If measuring rendering throughput, disable vsync/frame caps, warm up, run each variant sequentially for several seconds, count GPU-completed frames and drain GPU queues between variants. Do not time individual frames or use software rendering.

No rendering or denoiser benchmarks were performed for this research report. The immediate opportunity is to evaluate and extend the existing filter, especially cross-chart neighborhoods and directional output, rather than introduce another basic bilateral implementation.
