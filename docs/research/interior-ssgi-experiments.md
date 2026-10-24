# Interior SSGI technique experiments

Research record, 24 October 2026. Isolated trials and selected integration changes build on `8de0e0315c5c`.

The useful merges are focused correctness and integration changes: uniform view-space azimuth, accepted-support history confidence, per-texel source-history validation with independent tolerances, diffuse-plus-emission input, fallback-only diffuse AO, and authored scene settings. They are combined on `interior-ssgi-audit`; other prototypes remain in their isolated worktrees and branches.

These changes improve the implementation, but do not make screen-space GI complete physical GI. Noise replacement alone did not help: the tested Hilbert R2 variant increased temporal variation. Variance-guided filtering reduced variation at a substantial cost, while directional fallback and the rolling world cache exposed unresolved quality or performance problems. Geometry defaults retain the baseline values; the comparative sweeps below evaluate alternatives. The measured source and AO changes deliberately preserve existing behavior on unsupported layered, transmissive, and custom materials.

## Scope and measurement

Experiments branch independently from `8de0e0315c5c`, following the [algorithm audit](interior-ssgi-audit.md). The requested scenes are Cornell Box and Dual Rooms. Unless overridden, the common budget is Original sampling, half width/height tracing, two slices, eight steps per side, relative radius 0.25, relative thickness 0.01, 16 history frames and spatial radius 1.5. Emission and diffuse response remain linear HDR through reconstruction. The example includes baked diffuse fallback, SSR, AO and final antialiasing, so Beauty FPS measures the complete example rather than SSGI alone.

The common harness uses 960×720, pixel ratio one and a 1024 light-map atlas on the Apple M3 host described in the audit. The adapter records report Apple / Metal 3 and no fallback adapter. It verifies hardware rendering, disables vsync and frame limiting, warms implementations, and runs them sequentially for at least five seconds. Each batch submits four whole frames and waits for GPU completion before counting them. It verifies denoiser frame counts. Readbacks, CPU reference work and screenshots occur outside throughput timing. These are individual sustained runs, without confidence intervals; small percentage differences are not established speedups. Earlier audit medians used a separate run series and should not be mixed with this baseline.

## Completed throughput measurements

Percentages below compare each scene with this series' baseline, 109.30 completed FPS for Cornell and 186.08 for Dual Rooms. The pilot run is excluded from comparisons.

| Method | Cornell completed FPS (change) | Dual Rooms completed FPS (change) |
| --- | ---: | ---: |
| Corrected audit baseline | 109.30 (+0.00%) | 186.08 (+0.00%) |
| Adaptive history / anti-lag | 101.86 (-6.81%) | 165.59 (-11.01%) |
| Variance-guided multiscale filter | 88.26 (-19.25%) | 135.11 (-27.40%) |
| Explicit bilinear reconstruction | 85.62 (-21.66%) | 150.77 (-18.98%) |
| Hilbert R2 | 108.46 (-0.77%) | 184.83 (-0.68%) |
| Uniform view azimuth | 109.19 (-0.10%) | 185.96 (-0.06%) |
| Metric world radius | 108.89 (-0.38%) | 185.21 (-0.47%) |
| Source-associated depth hierarchy | 102.31 (-6.40%) | 177.30 (-4.72%) |
| Independent source-history tolerance | 103.33 (-5.46%) | 174.34 (-6.31%) |
| Diffuse-only source | 107.84 (-1.33%) | 181.56 (-2.43%) |
| Fallback-only diffuse AO | 108.72 (-0.53%) | 183.97 (-1.13%) |
| Directional missing-sector fallback | 94.36 (-13.66%) | 152.19 (-18.21%) |
| Stratified radial control | 109.12 (-0.16%) | 185.17 (-0.49%) |
| Lighting-guided radial selection | 84.19 (-22.97%) | 149.68 (-19.56%) |
| Rolling world-space SH cache | 62.06 (-43.22%) | 43.32 (-76.72%) |
| Gather-confidence correction | 109.12 (-0.16%) | 185.59 (-0.26%) |
| Optimized source-history validation | 104.09 (-4.76%) | 176.17 (-5.33%) |
| Guarded diffuse source | 107.65 (-1.51%) | 181.88 (-2.26%) |
| Guarded fallback-only AO | 109.14 (-0.14%) | 185.66 (-0.23%) |

## Reconstruction and sampling diagnostics

`RMSE / temporal deviation` below concerns filtered **screen contribution**, RGB multiplied by coverage, in linear irradiance units. The reference averages 64 frames of an independently hashed, eight-slice, 32-step full-resolution version of the branch's screen-space estimator with frozen lighting and a high luminance threshold; measured variation uses 32 frames. It is useful for convergence and temporal reconstruction diagnostics. It does not replace a path-traced reference. References are branch-specific: modified geometry sampling, source lighting and fallback change the target. Even when the estimator is unchanged, source brightness can affect an absolute noise statistic.

| Method | Cornell RMSE / temporal deviation | Dual Rooms RMSE / temporal deviation |
| --- | ---: | ---: |
| Corrected audit baseline | 0.47906 / 0.003014 | 0.09431 / 0.001019 |
| Adaptive history / anti-lag | 0.47902 / 0.003043 | 0.09408 / 0.001298 |
| Variance-guided multiscale filter | 0.48217 / 0.002013 | 0.09643 / 0.000458 |
| Explicit bilinear reconstruction | 0.47737 / 0.003736 | 0.09382 / 0.001632 |
| Hilbert R2 | 0.47654 / 0.005514 | 0.09413 / 0.001895 |
| Uniform view azimuth | 0.47141 / 0.003026 | 0.09337 / 0.001015 |
| Metric world radius | 0.54668 / 0.003689 | 0.10992 / 0.001243 |
| Source-associated depth hierarchy | 0.46910 / 0.003050 | 0.09401 / 0.001013 |
| Independent source-history tolerance | 0.47914 / 0.003014 | 0.09435 / 0.001015 |
| Diffuse-only source | 0.46885 / 0.002954 | 0.08789 / 0.000950 |
| Fallback-only diffuse AO | 0.47998 / 0.003020 | 0.09429 / 0.001011 |
| Directional missing-sector fallback | different output quantity | different output quantity |
| Stratified radial control | 0.47714 / 0.002865 | 0.09348 / 0.000942 |
| Lighting-guided radial selection | 0.45172 / 0.003208 | 0.09071 / 0.001041 |
| Rolling world-space SH cache | 0.15340 / 0.001144 | 0.05138 / 0.000659 |
| Gather-confidence correction | 0.47926 / 0.003012 | 0.09440 / 0.001023 |

The variance-guided prototype reduces temporal deviation by approximately 33% in Cornell and 55% in Dual Rooms, while completed throughput falls by 19% and 27%. It adds roughly 63.3 MiB at 1920×1080 and three full-resolution filter passes. Its RMSE is slightly higher, and the filter may smooth lighting detail. This is an optional quality/cost direction, not a free improvement or a complete implementation of SVGF/NRD.

The Hilbert R2 prototype costs under 1% in these runs but raises temporal deviation by approximately 83% and 86%. It uses R2 for the slice angle and one radial dimension plus independent integer hashes for remaining dimensions; this result does not reject every low-discrepancy sampler. The stratified radial control reduces deviation by approximately 5% and 7.5%, with nearly unchanged throughput. Those small changes need repeated measurements and visual inspection before a default change.

Adaptive history does not establish an overall gain in these static diagnostics: it costs roughly 7–11%, with near-baseline Cornell variation and higher Dual variation. Explicit bilinear reconstruction costs roughly 19–22%, with slightly lower RMSE but higher temporal variation. Their synthetic edge/disocclusion properties may still be valuable. The gain-step and camera-motion records are retained in JSON; their error includes estimator bias and source differences, so they should not be described as pure lag or ghosting measurements.

## Independent geometry reference

The CPU reference traces actual triangles: a 32×24 primary grid, 128 deterministic cosine-weighted hemisphere rays per valid receiver, exact first world-geometry hit within the supplied radius, and a separate camera ray to confirm that the corresponding screen pixel sees that same source surface. Primary and screen rays honor rendered material sidedness; secondary rays use double-sided geometry for blocking. Source-facing tests retain coverage but suppress backface lighting. Normals are interpolated and transformed, including instancing.

The first reference accidentally forced camera rays to see culled room walls, producing zero available coverage in Dual Rooms. That reference is superseded. Version `sidedness-and-source-facing-v2` fixes camera sidedness and source-facing lighting. Five CPU regressions and an actual-GLB diagnostic cover the fix. At baseline radius, corrected mean available coverage is **0.24831 for Cornell** and **0.21433 for Dual Rooms**, with 768 and 320 valid receivers. Corrected geometry records are distinguished in the saved results; earlier throughput and temporal measurements are unaffected by this CPU-only correction.

The target is screen-available geometry within a strict world radius. Lighting still reuses captured screen radiance and assumes direction-independent diffuse outgoing radiance. It supplies neither hidden/offscreen lighting nor full path-traced global illumination, and omits normal maps, transparency and dynamic deformation. The full-image mean coverage and sparse valid-receiver mean cover different sample domains and should not be directly equated. Finite angular sampling, the coarse receiver grid, and nearest-pixel surface matching limit precision, particularly at grazing angles. The original radius is only a projected search footprint, so a strict world-radius reference also exposes a difference in support semantics. Small error differences do not establish a physical GI winner. Directional fallback includes additional illumination and is excluded from this screen-only comparison.

## Interpretation of the technique trials

- **Uniform view azimuth:** constructs an orthonormal tangent basis around the receiver-to-camera direction and projects slice planes back into screen directions. This repairs off-axis angular sampling geometry, with effectively unchanged full-pipeline throughput here. It changes the screen-space estimator, so its branch-specific RMSE is not direct proof of closer physical GI. Finite-slice normalization and incomplete geometry remain.
- **Metric radius:** uses the full camera focal scale, a consistent radial search extent and a strict world-distance bound. Performance is essentially unchanged, but numerical radius semantics change, and temporal deviation rises roughly 22%. Existing radius values must be retuned rather than copied as though they denoted the same old search footprint.
- **Depth hierarchy:** carries linear min/max depth and the exact original UV of the nearest representative through two reductions. A distant gather adopts a representative only for depth-continuous footprints; radiance and normals are fetched from that same finest source. It averages no unrelated color MIP. This is conservative source LOD, not ray skipping. The extra construction passes cost about 5–6% without a compelling reconstruction improvement in these runs.
- **Source-history tolerance:** separates reprojection uncertainty from the tracing slab thickness. The initial measured implementation costs about 5–6%; the optimized projection-scale version measures 104.09/176.17 FPS, approximately 5% below baseline. Static scenes do not sufficiently exercise moving geometry or all rejection cases.
- **Diffuse-only source:** removes camera-view specular from SSGI input while retaining the full source for SSR. It addresses reflected highlights being re-emitted as diffuse radiance, at approximately 1–2.5% throughput cost in these runs. The extra full-resolution HDR attachment/history alone uses about 31.6 MiB at 1920×1080. It deliberately excludes directional specular-to-diffuse transport. The guarded follow-up retains legacy behavior for unsupported layered, transmissive and custom material cases; the guarded follow-up passed its hardware rendering regressions and scene checks.
- **Fallback-only AO:** avoids attenuating already integrated diffuse SSGI again, while retaining appropriate fallback/specular behavior. It costs around 0.5–1% here. This is an integration-scope change; a screen-only SSGI RMSE is not a sufficient measure of its final Beauty benefit. A guarded variant restricts the changed path to supported ordinary diffuse materials.
- **Directional fallback:** integrates a directional radiance provider over unfilled sector masks, which can represent missing-direction lighting better than multiplying a whole-hemisphere scalar fallback by `1 - coverage`. It costs approximately 14–18% here. Its RGB and alpha now include fallback, so the screen-only error/noise columns are not comparable. The current prototype also turns source validity into a binary mask using `valid > 0.5`, while screen energy uses fractional validity: fractional values can lose a fraction or double-count fallback. Current boolean history validity does not test that contract. Per-sector confidence or an explicitly binary-only interface is required before treating it as general coverage-correct integration.
- **Lighting-guided radial selection:** selects among candidate gather positions using previous lighting. It costs roughly 23% Cornell and 20% Dual, and does not reduce temporal variation in this run. Its lower self-reference RMSE does not establish less bias: the selection changes horizon traversal and binary-mask ownership without a corresponding unbiased probability estimator. This is biased adaptive quadrature, not a validated importance-sampled path tracer. The simpler stratified control is retained as a separate comparison.
- **Rolling world-space cache:** demonstrates a separate 4×2×2 world SH probe grid refreshed incrementally, extending available lighting beyond a single screen buffer. Its captures, updates, storage and temporal lag belong in the measured cost. SH is low-frequency, visibility/light leakage needs evaluation, and the prototype is not a complete Lumen-style screen-probe/radiance-cache system. The completed prototype costs approximately 43% Cornell and 77% Dual throughput. It also changes fallback and feedback brightness, so its lower absolute screen-error/noise values do not isolate improved denoising. This implementation is an architectural demonstration, not a performance-ready replacement.
- **Authored scene settings:** separates quality budgets from physical distances; accepts authored local scale or local/world radius and thickness. This avoids changing an object's declared thickness merely because distant level bounds grow. The `.005` thickness / `.25` radius profiles for the two requested scenes are candidates, not established defaults. Uniform-scaling and bounds-growth CPU tests pass. The initial profile comparison was invalid: Inspector's implicit slider step changed `.005` to `.0001` and `.01` to `.0101` through its listening/writeback path. This caused the apparent near-zero coverage. Explicit radius/thickness steps and a real DOM slider regression fix the problem. The near-zero result is not evidence against thickness `.005`; controlled corrected sweeps below determine the recommendation. The metadata/quality infrastructure is merged with baseline geometry defaults.
- **Gather-confidence follow-up:** discounts temporal history age by accepted bilinear gather support, rather than promoting a tiny surviving history footprint to mature confidence. This has a focused fractional-support regression; hardware scene timing is essentially unchanged, and static noise remains near baseline. Its improvement is the bounded confidence contract at partial history footprints, rather than a large aggregate screenshot change.

## Parameter trial matrix

The dedicated sweep runs fifteen cases per scene without rebaking each setting: nine combinations of relative radius `0.10 / 0.15 / 0.25` and thickness `0.0025 / 0.005 / 0.01`, then independent `3×8`, `2×12`, full-resolution, 8-frame history, 32-frame history and spatial-radius-1 alternatives. Every case resets to the common `2×8`, half-resolution, 16-frame, radius-1.5 filter baseline before applying its override. It measures completed-frame throughput before readbacks and CPU geometry work, then restores settings even on error.

The measured world values should accompany every recommendation. Relative radius/thickness are scene-diagonal policies in the baseline, whereas the metric-radius branch uses stricter semantics. At the retained `.25 / .01` settings, Cornell uses radius **8.063** and slab thickness **0.3225**; Dual Rooms uses **4.644 / 0.1857**. These thicknesses are effective occluder proxies, not inferred material thickness.

A scene loader should apply a shared work budget, then authored scale/distance metadata. The merged resolver reads `gltf.scene.userData.ssgi` (glTF scene `extras.ssgi`). It supports `localScale`, `localRadius`, `localThickness`, `worldRadius`, and `worldThickness`. Local distances scale with the uniformly scaled model; world distances stay fixed. Authored values take precedence over relative GUI controls. For example:

```json
{
  "ssgi": {
    "localRadius": 4.644,
    "localThickness": 0.1857
  }
}
```

This example preserves the present Dual Rooms proxy distances under model scaling. It is not a universal room prescription. Use bounds-based values only as fallback metadata, prefer a meaningful local room scale, and keep the lighting clamp scene dependent. The source-history floor remains a separate relative scene-scale control, with pixel-footprint tolerance for reprojection; it does not reuse the tracing thickness. Independently moving hit surfaces remain outside this camera-only history model.

SSR keeps its own reach/thickness/roughness policy. Its history tolerance now has independent pixel/world controls and uses the validated helper. These experiments do not establish universally optimal SSR marching settings or authorize copying SSGI slab values into SSR. The [initial audit](interior-ssgi-audit.md) explains their different meanings.


- **Retain the medium budget:** half resolution, `2 × 8`, 16 history frames, denoise radius 1.5. These tests do not establish a compelling replacement for the shared default.
- **Offer `2 × 12` as the next quality step.** At the same radius/thickness, it lowers independent geometry coverage RMSE by about 8.4% Cornell and 11.5% Dual, for about 6.4% and 6.7% lower completed FPS. This is the clearest measured coverage/cost improvement among the tested work-budget changes. It spends extra work radially rather than adding a third slice.
- **Keep relative thickness 0.01 for these two scenes among the tested candidates.** At every fixed radius, 0.01 gives lower coverage error than 0.005 or 0.0025 in both implementations. At radius 0.25, merged coverage RMSE for thickness `0.0025 / 0.005 / 0.01` is Cornell `0.27724 / 0.25909 / 0.23234`, Dual `0.21885 / 0.20165 / 0.17244`. Smaller values recover less of the geometry-reference coverage; a lower absolute temporal deviation can simply mean less SSGI signal. This does not prove that 0.01 represents actual surface thickness or handles every thin object best. Authored local overrides remain useful.
- **Retain relative radius 0.25 provisionally; do not choose 0.10 solely because its absolute error is lower.** Each radius changes the CPU target's world support. Mean true available coverage at `0.10 / 0.15 / 0.25` is Cornell `0.10618 / 0.15630 / 0.24831`, Dual `0.09155 / 0.13601 / 0.21433`. A shorter reach is a different near-field effect, with less transport to recover. Compare room coverage and fallback behavior before changing the shared reach.
- **Keep 16 history frames as the compromise.** Eight slightly increases captured temporal variation but can improve responsiveness; 32 slightly reduces variation here but increases lag. This short, static capture does not justify a universal history-length change.
- **Keep spatial radius 1.5 for the shared default.** Radius 1 modestly increases temporal variation with little matched coverage-error benefit. Use 1 when preserving sharper lighting detail is worth the tradeoff. Full-resolution tracing costs approximately 29% Cornell and 27% Dual completed FPS for only about 1.6–2.0% improvement in this coarse coverage metric; its edge/small-object benefits require visual inspection beyond the coarse grid.

### Work-budget comparison on the corrected merged implementation

All six alternatives reset to relative radius 0.25, thickness 0.01 and the medium budget before applying one override. Baseline Cornell is **103.75 FPS / 0.23234 coverage RMSE / 0.017920 temporal deviation**; Dual is **174.01 / 0.17244 / 0.005508**.

| Alternative | Cornell: FPS / coverage RMSE / temporal deviation | Dual Rooms: FPS / coverage RMSE / temporal deviation |
| --- | ---: | ---: |
| 3 × 8 | 97.23 / 0.23224 / 0.016643 | 162.04 / 0.17316 / 0.004901 |
| 2 × 12 | 97.07 / 0.21273 / 0.017482 | 162.40 / 0.15263 / 0.006166 |
| Full resolution | 73.32 / 0.22857 / 0.019656 | 127.18 / 0.16907 / 0.007349 |
| History 8 | 103.37 / 0.23231 / 0.018574 | 174.51 / 0.17272 / 0.005771 |
| History 32 | 103.68 / 0.23263 / 0.017591 | 174.50 / 0.17299 / 0.005413 |
| Denoise radius 1 | 103.79 / 0.23215 / 0.019017 | 174.61 / 0.17399 / 0.005715 |

Three slices reduce captured temporal deviation about 7% Cornell and 11% Dual, with nearly unchanged coverage error and roughly the same cost as 2 × 12. It is a reasonable alternative when angular flicker is the main concern; the geometry metric favors extra radial steps for these views. The earlier baseline sweep agrees with the thickness ordering and the advantage of 2 × 12 for coverage.


Sweep FPS is a separate single-run series; compare alternatives within this table rather than combining it with the repeated native medians below. The temporal statistic includes post-freeze history settling and brightness differences, so it is not a guaranteed stationary-noise measure. Source-aligned geometry was recaptured separately without retiming the implementations; saved records preserve original timing/temporal statistics and mark the refreshed geometric reference. Irradiance rankings are not used to select defaults.

## Selected merges and combined validation

The following experiment branches were merged into `interior-ssgi-audit`:

| Change | Merge commit | Why selected |
| --- | --- | --- |
| View-space azimuth | `291434ae9e9e` | Correct off-axis slice geometry; effectively unchanged measured throughput. |
| Scene metadata and quality budgets | `d3b6c5a07fee` | Explicit scale/distance policy and reusable quality settings; baseline geometric defaults retained. |
| Gather confidence | `cfc2890b14fd` | Partial accepted history cannot retain full mature confidence; no additional texture reads. |
| Guarded diffuse/emission source | `8fefbe9cf7c6` | Avoid re-emitting camera-dependent specular into diffuse GI. |
| Guarded fallback-only AO | `040bbc755937` | Avoid applying extra screen AO to SSGI's integrated visibility; preserve material/specular AO. |
| Independent source reprojection | `dd98ac3d2c8c` | Validate each historical source texel before interpolation, using pixel/world uncertainty instead of slab thickness. |

Integration commit `613c9261db02` selects one history source texture before gathering; it avoids fetching both Beauty and diffuse at every validated tap. `0725d7c0a269` includes the isolated parameter benchmark. The source and AO material guard rejects active clearcoat, sheen, transmission, transparency, corresponding node overrides, and custom lighting/output/MRT paths. Such materials retain baseline behavior, rather than receiving a partially implemented layered transport model. The source helper currently supports camera motion over static world geometry; independently moving hit surfaces still require hit-surface motion information.

The final combined targeted hardware suite passed **43 tests**, with zero failures or skips: SSGI/material integration, coverage reconstruction, source history, scene settings, and FAST sampling. The CPU reference has five additional regressions. Both requested scenes and the Cornell metallic-sphere/mirror-wall variants render without shader errors; screenshots were inspected. The initial combined high-sample diagnostic and timings were affected by GUI rounding and are superseded. The corrected combined diagnostic gives RMSE/temporal deviation **0.46171/0.002950 Cornell** and **0.08751/0.000954 Dual Rooms**. The source has changed, so these branch-specific reductions are not physical-ground-truth accuracy claims. Three sequential five-second native runs per scene measured these completed-frame medians after the fixes:

| Scene | Audit baseline | Combined branch | Throughput change |
| --- | ---: | ---: | ---: |
| Cornell Box | 109.34 FPS | 101.98 FPS | -6.7% |
| Dual Rooms | 185.51 FPS | 166.46 FPS | -10.3% |

These are correctness/integration improvements with a measurable cost, principally validated source gathers and the extra diffuse attachment/history. They are not a performance optimization. Small absolute differences remain hardware/run dependent. Source selection, geometric defaults, intensity, and backface policy are explicit controls. The expensive reconstruction/cache prototypes were not merged.

Inspector slider precision is fixed in `0ea99815dc7b`, with a real DOM regression merged in `1a3992de4985`. Previous-frustum near/far rejection is on `interior-ssgi-history-clip-bounds` (`ba5a469a9477`) and passed the final source-history tests. Parameter benchmarks now reject any requested settings changed by GUI writeback and read the actual SSGI radiance source for their geometry reference.

## Branch and commit inventory

Branches are isolated experiments from the corrected audit base. The table records experiment branch heads; benchmark JSON contains the actual measured commit, which can precede a later follow-up.

| Branch | Head | Role / status |
| --- | --- | --- |
| `interior-ssgi-audit` | see merge record | Selected changes combined |
| `interior-ssgi-adaptive-history` | `2ffc3689e571` | Adaptive history prototype |
| `interior-ssgi-variance-atrous` | `75cdbafa45f5` | Temporal moments and multiscale filter |
| `interior-ssgi-bilinear-reconstruction` | `870580ae9c92` | Explicit geometric bilinear taps |
| `interior-ssgi-r2` | `b4abc7ba5506` | Hilbert R2 plus remaining hash dimensions |
| `interior-ssgi-view-azimuth` | `d6fd204add7a` | Uniform view-space slice azimuth |
| `interior-ssgi-world-radius` | `e3bc96ddd4b8` | Metric radius semantics |
| `interior-ssgi-depth-hierarchy` | `2b9383bd8e4e` | Source-associated depth reductions |
| `interior-ssgi-history-tolerance` | `ce7d0a59964d` | Independent source reprojection; head is later optimization |
| `interior-ssgi-diffuse-source` | `07b726e0ea23` | Diffuse source experiment |
| `interior-ssgi-diffuse-source-guarded` | `e4d8a30a9ada` | Material-scope guarded follow-up |
| `interior-ssgi-fallback-ao` | `599544103ab8` | Fallback-only diffuse AO |
| `interior-ssgi-fallback-ao-guarded` | `edd7b5fea6d3` | Material-scope guarded follow-up |
| `interior-ssgi-directional-fallback` | `9e8767b2f277` | Missing-sector directional lighting; changed quantity |
| `interior-ssgi-light-guided` | `93f162275d73` | Both guided and stratified modes |
| `interior-ssgi-world-cache` | `22db4d30dee7` | Rolling world-space SH fallback |
| `interior-ssgi-scene-settings` | `74e8902bf2bd` | Authored metadata and candidate profiles |
| `interior-ssgi-history-clip-bounds` | `ba5a469a9477` | Previous near/far history rejection |
| `interior-ssgi-gather-confidence` | `2574cf5074ae` | Fractional history-support correction |
| `interior-ssgi-parameter-sweep` | `f63ce810929b` | Fifteen-case numeric benchmark module |

Measured source-history results use `2992159e64d4`, preceding the branch-head projection-scale simplification. Guarded source/AO variants, optimized source reprojection, and gather confidence have their own completed follow-up measurements above. World-cache and scene-profile measurements include their required runtime work.

## Primary references and reproduction

The design references remain the [SSILVB paper](https://arxiv.org/html/2301.11376v1), [authors' implementation explanation](https://cdrinmatane.github.io/posts/ssaovb-code/), and [directional ambient-lighting slides](https://cdrinmatane.github.io/cgspotlight-slides/ssilvb_slides.pdf). They explain view-aligned slices, finite thickness and missing-direction lighting. [XeGTAO](https://github.com/GameTechDev/XeGTAO) motivates the R2/Hilbert and depth hierarchy experiments; its AO heuristics are not automatically correct for a GI radiance gather. [SVGF](https://research.nvidia.com/labs/rtr/publication/schied2017spatiotemporal/) and [NRD](https://github.com/NVIDIA-RTX/NRD) motivate temporal statistics, confidence and multiscale reconstruction. [Lumen's engineering description](https://www.unrealengine.com/tech-blog/unreal-engine-5-goes-all-in-on-dynamic-global-illumination-with-lumen) motivates lighting guidance and screen/world caches, whose complete architecture is much larger than these prototypes. The [transport equation](https://pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/The_Light_Transport_Equation) distinguishes outgoing source radiance from receiver BRDF weighting.

A compact, durable [results record](interior-ssgi-experiment-results.json) includes measurements, corrected geometry comparisons and test results. Raw results are under `/tmp/three-ssgi-experiments/<label>/results.json`, with Beauty, filtered lighting and coverage screenshots beside them. The common script is `test/benchmarks/interior-ssgi-experiments.mjs`; manifests select isolated worktrees and optional geometry/quality/parameter-sweep work. The parameter module is `test/benchmarks/ssgi-parameter-sweep.js`, with a CPU-only harness test and README in its branch. The independent CPU triangle reference is `test/benchmarks/ssgi-geometry-reference.js`.

```sh
node test/benchmarks/interior-ssgi-experiments.mjs test/benchmarks/ssgi-experiments.json
node test/benchmarks/interior-ssgi-experiments.mjs test/benchmarks/ssgi-geometry.json
node test/benchmarks/interior-ssgi-experiments.mjs test/benchmarks/ssgi-parameters.json
node test/benchmarks/ssgi-parameter-sweep.test.mjs
```

The final command runs from the parameter-sweep worktree. The central harness serves that worktree's sweep module and its own geometry reference. Keep implementations sequential, count GPU-completed whole frames, include required cache/filter construction work, verify hardware adapters, and discard generated repository-root `build/` changes before branch changes. The raw self-reference and geometric comparisons are diagnostics; the selected merges also rely on mathematical and material integration regressions.
