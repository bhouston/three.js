# Interior SSR correctness and parameter review

Reviewed on October 24, 2026, against `interior-blue-noise` at `eedbf3e9833e`. Implementation and regression fixes are on `interior-ssr-review`. The focus is `webgpu_interior.html`, Cornell Box (Mirror Wall), and Primitives Floor (Metallic Mirror).

The implementation has good modern foundations, but it is not fully best practice or a physically consistent reflection integrator. Several definite correctness bugs are fixed on this branch. The largest remaining gaps are incomplete traversal, permissive hit validation, inconsistent sampling and material BRDF models, and reconstruction that can blur or misclassify valid dark reflections. Changing blue-noise patterns cannot resolve these gaps.

## What already follows good practice

The tracer samples a bounded GGX visible-normal distribution, squares perceptual roughness into alpha, evaluates reflection weights without explicitly dividing two very large GGX distribution values, interpolates perspective ray depth through reciprocal Z, and clips perspective rays at the near plane. These are sound foundations. [Bounded VNDF paper](https://gpuopen.com/download/publications/Bounded_VNDF_Sampling_for_Smith-GGX_Reflections.pdf), [McGuire and Mara](https://jcgt.org/published/0003/04/04/paper.pdf).

The interior integration keeps linear HDR lighting separate from final tone mapping, carries coverage independently of RGB, demodulates material response before denoising, uses geometric history validation, and supplies probe/environment fallback. It avoids adding SSR and fallback at full strength to the same lobe. Specular history includes virtual-hit reprojection and curvature/geometry confidence. Those choices align with the intent of modern reflection denoisers, although this custom implementation is not NRD. [NRD integration guidance](https://github.com/NVIDIA-RTX/NRD#material-demodulation).

The existing FAST comparison is useful evidence for the noise pattern under this integrator. Its reference runs the same tracing and weighting code, so it cannot detect their common bias. Lower error against that reference establishes better reconstruction of the current estimator, not physical correctness.

## Fixes made on this branch

| Change | Why it matters | Evidence |
| --- | --- | --- |
| Stop stochastic marching when its screen-space ray parameter exceeds 1 | The minimum one-texel step could extend a short projected ray beyond its endpoint and near-plane clipping | A synthetic target beyond the endpoint reflects with the original code and correctly contributes zero with the fix; a longer ray still reaches it |
| Refine only a genuine front-to-back depth bracket, starting the first bracket at the ray origin | A thickness-rejected behind-depth sample is not a valid front endpoint; the previous jitter expression could also produce an invalid first bound | Original refinement loses the synthetic long-ray hit after a rejected depth gap; fixed refinement preserves it |
| Recheck thickness/background after refinement and reject background while marching | Binary search can land at a depth discontinuity, so coarse acceptance does not validate the final UV | Hardware compilation, tracing regression, and both example scene captures pass; no dedicated isolated post-refinement discontinuity test was added |
| Guard the grazing ray-length denominator | Avoids division by zero in the inherited distance extension | Shader compilation and scene captures pass |
| Convert coverage-denoiser NDC motion to UV motion with `(0.5, -0.5)` | Previously horizontal reprojection was doubled and vertical reprojection had the wrong sign | All four new directional movement tests fail with the old expression and pass with the fix |
| Test the GGX tangent cross product before normalizing | Normalizing zero at view-axis normals is undefined and the old fallback checked too late | Exact and near-pole GPU checks pass on WebGPU and hardware WebGL; the original also passes on Apple, so this is portability hardening |
| Default mirror bias to zero in SSRNode and the interior example; correct the unbiased comment | The warp changes the sampling distribution without changing its PDF | Reproducible numerical estimator comparison below |

The motion fix affects default SSGI as well as the optional coverage-aware SSR pipeline. It does not change the default SSR legacy temporal pass, which already converted velocity correctly.

## Remaining tracing gaps

**The current quality control is a fixed sample budget, not complete traversal.** At quality 1 it uses a nominal 64-step budget; the loop starts at 1 with an exclusive end, so at most 63 coarse samples are evaluated. Exponent 3 concentrates those samples near the origin and leaves larger gaps farther away. A binary search only refines a crossing already found; it cannot recover thin objects skipped by the coarse march. Contiguous perspective-correct pixel DDA is a useful correctness baseline, and hierarchical depth traversal is the stronger performance architecture for long mirror-floor rays. Neither is implemented by these fixes. [DDA explanation and code](https://casual-effects.blogspot.com/2014/08/screen-space-ray-tracing.html), [McGuire and Mara](https://jcgt.org/published/0003/04/04/paper.pdf).

**Thickness still measures distance to an infinite ray line.** A reconstructed surface point can be close to that line while far from the actual sampled ray point, especially along the view direction. The hard minimum of three trace pixels also makes acceptance resolution-dependent. Reducing the exposed thickness below that floor has no effect there. Replace this with a finite depth-segment/surface-slab test or a reconstructed ray-point separation test, then derive smooth confidence from the separation. The stochastic path also still lacks explicit hit-surface backface and immediate self-hit rejection. AMD's validator provides concrete examples of those checks and confidence fading. [Pinned AMD validator](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/blob/v1.1.4/sdk/include/FidelityFX/gpu/sssr/ffx_sssr_intersect.h#L220-L265).

**Max distance is not a strict world-space ray length.** The implementation divides it by the incident-view/normal cosine, inherited from the mirror path. Grazing rays can therefore be much longer than the GUI value. The new endpoint check respects that constructed ray, but does not redefine the public distance parameter. A future tracer should expose an actual maximum ray length separately from any plane-distance limit.

Single-layer depth cannot represent hidden surfaces, backs of objects, or offscreen objects. Also, the color buffer stores outgoing radiance toward the main camera. At a glossy secondary hit, that is generally different from radiance toward the mirror. Reusing previous beauty approximates multiple bounces; it does not make mirror-in-mirror shading physically correct. The original Frostbite author identifies the reconstruction as a ratio estimator. [Author's publication page](https://h3.gd/stochastic-ssr/).

## Remaining sampling and integration gaps

The bounded sampler and its unwarped PDF follow Eto and Tokuyoshi. The extra transformations around it do not all preserve that PDF:

- Mirror bias is now off by default. Turning it on intentionally changes the lobe and remains biased.
- The one-time retry of a below-horizon ray changes the distribution without correcting its weight. A mathematically consistent implementation should retain a rejected sample's zero contribution, or use a sampler/PDF that accounts for the conditioning. Coverage/fallback must handle those zero contributions consistently.
- `GeometryTerm` uses separable Smith masking, while the material and DFG LUT use height-correlated Smith. Dividing by the correlated LUT and remodulating later does not repair an inconsistent tracer BRDF.
- The optional `ImportanceSampledEnvironment` BRDF/MIS path evaluates ordinary VNDF weights/PDFs for bounded-VNDF rays. Interior's `outputCoverage` mode uses external probes and does not execute this environment path. It still prevents a claim of general SSR estimator correctness.

The numerical harness independently reproduces the relevant equations, using 1,000,000 samples per configuration, seed 20261024, white incident radiance, and dielectric F0 0.04. Each change below is isolated rather than combined:

| Configuration | Numerical effect |
| --- | ---: |
| Roughness 0.8, N dot V 1, mirror bias 0.5 versus zero, no retry | +17.15% reflected energy |
| Roughness 0.6, N dot V 0.5, one retry versus no retry, bias zero | +4.80% reflected energy |
| Roughness 0.8, N dot V 0.1, separable versus correlated Smith, bias zero and no retry | -22.03% reflected energy |

These are CPU estimator measurements, not image error or GPU throughput measurements. They quantify distinct mathematical issues; they are not predictions of the change in either mirror screenshot. Dielectric reflected energy is not expected to equal one.

The default **Before filtering** integration has two additional limitations. Legacy temporal/recurrent code infers missing samples from near-black RGB, although a valid reflection can be black. Its provider also combines filtered RGB with current unfiltered coverage; when fallback coverage is incomplete, that can discard or distort valid accumulated energy. **After filtering (coverage-aware)** jointly filters premultiplied radiance and coverage and avoids those particular ambiguities. Resolving fallback before filtering is itself legitimate; the problem is inconsistent validity semantics. A universal pipeline switch needs motion, black-surface, and partial-fallback comparisons.

The debug views are also different pipelines: **SSR Filtered** always shows the coverage denoiser, while default Beauty uses the legacy recurrent resolved signal. Use **Indirect Specular Filtered** to inspect the selected resolved pipeline. Do not assume SSR Filtered directly explains default Beauty.

Material handling is correct for these standard isotropic metalness materials: the MRT contains base color before diffuse metalness attenuation. General physical-material support would additionally require consistent IOR/specular intensity, clearcoat, anisotropy, transmission and other lobe data. The query matcher guards the base lobe, but its current small G-buffer cannot describe every physical-material response.

## Parameters for mirrors and general use

These are engineering starting points, not universally optimal settings. Only mirror bias was changed as a production default in this review. The other suggested values need a scene/camera sweep, particularly after replacing the thickness test.

| Parameter | Original interior default | Suggested use |
| --- | --- | --- |
| Resolution scale | 1 | Keep 1 for mirrors; try 0.5 for rough reflections with reconstruction |
| Quality | 1 | Keep 1 for mirror review; 0.5 to 1 for general use after checking thin geometry |
| Relative max distance | 0.5 | Start at 0.5; increase only for required reflection reach, understanding the grazing extension |
| Relative thickness | 0.01 | Sweep 0.001 to 0.003 initially; inspect contacts/silhouettes. The three-pixel floor can dominate |
| Step exponent | 3 | Compare 1 to 2 for distant mirror details; 2 to 3 prioritizes nearby hits. Neither guarantees complete traversal |
| Binary refinement | true | Keep true after bracket and validation fixes |
| Mirror bias | 0.5 | Use 0 for fidelity; nonzero is an explicit biased appearance control |
| Screen edge fade | 0.1 | Try 0.03 to 0.05; retain 0.1 if the broader probe transition is visibly preferable |
| Intensity | 1 | Keep 1 while validating lighting energy |
| Maximum roughness | 0.8 | Compare a perceptual cutoff around 0.5 to 0.7 for general use; keep 0.8 if broad SSR is useful and affordable |
| History frames | 16 | 8 to 16 for motion; 16 to 32 for stationary inspection, with adequate rejection |
| Spatial radius | 1.5 | Start at 0 for mirror sharpness diagnostics; increase only enough to control actual noise |
| Maximum luminance | 35 | Keep as an appearance/firefly compromise; raise substantially during energy diagnostics so clipping does not hide errors |
| Noise | FAST Separate | Keep the measured SSR choice; it improves stability without correcting estimator bias |
| Fallback resolve | Before filtering | Compare the coverage-aware alternative for black surfaces and partial fallback; preserve both until motion/detail tests choose a policy |

The radius numbers are not portable between the two filters: coverage denoising measures stride in input texels, while recurrent denoising derives a projected world radius using hit distance, view depth, roughness and adaptation. A single shared GUI radius does not mean identical blur strength.

AMD treats very smooth mirrors differently from rough reflections and uses hierarchical depth tracing. Its roughness inputs normally use alpha, while this example exposes perceptual roughness; numerical defaults cannot be copied without conversion. For example, alpha 0.2 corresponds to perceptual roughness about 0.447. [AMD SSSR integration and algorithm](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/stochastic-screen-space-reflections/).

## Automatic parameter policy

Most SSR parameters should be shared across scenes. Prefer three layers: global correctness settings, scene scale and authored geometry information, then per-pixel material classification. Scene-name presets cannot account for camera-dependent ray length or mixed roughness within one scene.

| Scope | Policy | When to update |
| --- | --- | --- |
| Global correctness | Mirror bias 0, intensity 1, binary refinement on | Shared across all scenes |
| Scene scale | Use opaque renderable model bounds, excluding sky, helpers and unrelated distant objects, to seed maximum reach and thickness | On scene load and substantial geometry changes |
| Material policy | Smooth mirrors get full-rate tracing and little or no spatial blur; rough lobes get stochastic sampling and broader reconstruction | Per pixel, using effective material roughness |
| Performance | Select ray budget and resolution for target hardware, projected ray length and measured variance | Runtime or quality tier, rather than world size alone |
| Fallback | Use authored room volumes for enclosed interiors and suitable capture placement for open scenes | Scene load, with authored probe layout metadata where possible |

A practical load-time fallback is `maxDistance = 0.5 * D` and `thickness = 0.002 * D`, where D is the relevant model diagonal. These are initial estimates, not guaranteed intersection tolerances. Large floors, long corridors and isolated tiny props can make the global diagonal a poor thickness scale. Allow a world-unit thickness override in scene metadata; do not infer mesh thickness from a flat mesh bounding box. With the present tracer, the three-pixel floor and grazing ray extension still override the intuitive meaning of these values.

For a first material classifier, effective perceptual roughness at or below about 0.08 is a useful candidate mirror band: it includes these materials after their 0.045 floor. Use full resolution, quality 1 and zero/minimal spatial blur there. This threshold needs calibration against desired image detail. Let rougher pixels retain reconstruction; fade very rough pixels to probes using a shared cutoff instead of changing it by scene name. The existing filters already vary radius with roughness, but their nonzero floors and different radius units mean they do not implement an exact mirror bypass.

Both requested scenes therefore start with the same SSR settings. The Primitive floor especially needs reliable long-ray traversal and contact validation, while Cornell especially needs correct room fallback and near-view-direction validation. Those are algorithm/fallback differences, not reasons to increase thickness on one scene until artifacts disappear. An automatic policy should never use extra thickness or blur to compensate for missing coarse intersections.

Do not automatically set the entire scene radius to zero just because it contains one mirror: the Primitive scene also has a roughness-0.3 gold material. A per-pixel policy preserves rough-material denoising. With the current single global radius UI, use zero as a mirror diagnostic and retain a separate general-use setting until a mirror bypass is implemented.

## The two requested scenes

**Cornell mirror wall:** The GLB mirror is metalness 1, roughness 0. The material pipeline clamps effective roughness to at least 0.045 and adds geometric roughness. Near-normal viewing stresses depth-line acceptance; grazing views stress step coverage. The reflected open side of the room requires fallback, so separate missing screen data from incorrect geometry hits. For a true planar mirror, an additional reflected-camera render is a better correctness reference and can include offscreen geometry. Recursive mirrors still need a deliberate recursion policy.

**Primitive mirror floor:** The GLB has zero-roughness metals plus a gold material at roughness 0.3. Long grazing rays, thin primitives, contact edges, and the mirrored sphere stress the algorithm. The example deliberately disables box projection for open-floor probes. Those probes provide approximate fallback parallax, so changing SSR/fallback coverage can reveal a different reflected shape or position. No SSR parameter can make those probes match the reflected camera everywhere.

Both scenes store packed normals and roughness in an unsigned-byte G-buffer. For high-quality mirrors, compare half-float normals/roughness or a higher-precision normal encoding. Also compare exact mirror rays below a deliberate roughness threshold and reduced/bypassed spatial blur. The existing material roughness floor should be considered explicitly rather than assuming GLB roughness 0 produces a perfect mirror.

## Validation and next changes

The focused suite passes 29 tests with no skips: coverage denoising 16, specular helpers 2, SSR 1, recurrent denoising 6, and SSGI 4. WebGPU uses Apple's non-fallback Metal adapter; the helper's WebGL test uses ANGLE Metal on Apple M3. Browser-response overrides of the original source reproduce the four motion failures and the endpoint/refinement failures without changing the worktree. Pole tests also pass on the original Apple shader, as noted above. [GPU results and baseline reproductions](interior-ssr-review-data/gpu-tests.json).

Both requested scenes completed baking and all nine existing debug/Beauty captures at 960 by 720, pixel ratio 1, with a 1024 light-map atlas, verified hardware, and zero recorded browser/GPU errors. These are smoke and visual checks, not path-traced accuracy comparisons or moving-camera quality sweeps. [Cornell metadata](interior-ssr-review-data/cornell-mirror.json), [floor metadata](interior-ssr-review-data/floor-mirror.json).

Reproduce the numerical audit with:

```sh
node test/benchmarks/ssr-sampling-integral.mjs
```

Reproduce the scene captures after installing dependencies and running `node utils/build/dev.js`:

```sh
SAMPLING_CAPTURE_ONLY=1 SAMPLING_MODES=Selected SAMPLING_MODEL='Cornell Box (Mirror Wall)' SAMPLING_OUTPUT=/tmp/ssr-review-cornell node test/benchmarks/interior-sampling.mjs
SAMPLING_CAPTURE_ONLY=1 SAMPLING_MODES=Selected SAMPLING_MODEL='Primitives Floor (Metallic Mirror)' SAMPLING_OUTPUT=/tmp/ssr-review-floor node test/benchmarks/interior-sampling.mjs
```

The harness freezes source lighting for captures and reduces the bake atlas; this isolates inspection from continuing feedback. No new performance claim is made. Generated root `build/` files are excluded from the commits.

The next implementation priorities are: complete DDA or Hi-Z traversal with finite hit validation; consistent bounded-VNDF proposal/weights and correlated Smith across screen/environment paths; then a validity-aware reconstruction policy with mirror-specific spatial filtering. Compare those against a reflected-camera or path-traced reference with stationary and moving cameras, contact edges, offscreen objects, valid black surfaces, and incomplete fallback. Parameter tuning becomes more reliable after those structural fixes.
