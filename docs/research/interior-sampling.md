# Denoising-friendly sampling for the interior SSR and SSGI example

Research and measurements completed on 24 October 2026. Published findings are distinguished from implementation-specific measurements.

**Use FAST Separate for SSR and retain Original sampling for SSGI.** Across four Cornell-box and primitive-floor scenes, the selected SSR pattern reduced stationary filtered temporal variation by 25–46% at the same ray budget, with essentially unchanged throughput. Reference error improved in three scenes; the rough primitive floor had 1.5% higher error. The tested FAST SSGI variants flickered more, so they remain optional experiments.

Filter-adapted spatiotemporal sampling (FAST) arranges sampling error for the reconstruction filter. The implementation preserves SSR’s GGX transform and SSGI’s slice integrator, replacing their random inputs. It includes both product and separate FAST variants and a switch back to the original patterns.

## What the research establishes

**Blue-noise dithered sampling, Georgiev and Fajardo, 2016.** Correlating neighboring pixels' samples can suppress low-frequency estimation error without reducing its total magnitude. This distinction matters: a raw frame can have similar numerical error while looking better and becoming easier to reconstruct. Static spatial masks establish the core idea, but do not by themselves provide a good sequence over time. They are a useful conceptual baseline, rather than the final choice for a renderer that already accumulates history. [Author publication page](https://iliyan.com/publications/DitheredSampling/).

**Spatiotemporal blue noise, Wolfe et al., 2022.** STBN extends masks across two spatial dimensions and one temporal dimension. Individual frames retain spatial blue noise, while a fixed pixel gets a well-distributed temporal sequence. It supports scalar and vector values, including nonuniform distributions. This is substantially more relevant to a temporal denoiser than randomly choosing or translating independent spatial masks every frame. The published work demonstrates volumetric rendering, ambient occlusion, and stochastic convolution; transferring those benefits to this SSR/SSGI implementation still requires testing. [NVIDIA research publication and SDK link](https://research.nvidia.com/publication/2022-07_spatiotemporal-blue-noise-masks).

The authors describe STBN as toroidally progressive: history rejection need not restart the sampling sequence to obtain a useful temporal subsequence. Motion weakens its temporal advantage because a reprojected surface traverses screen pixels; spatial benefits remain. Ordinary 3D blue noise should not be substituted: good frequency behavior in a three-dimensional volume does not guarantee the required two-dimensional spatial slices and one-dimensional time series. Texture access should use integer pixel coordinates and a discrete time slice. These implementation details are central to retaining the designed correlation. [Authors' implementation discussion](https://developer.nvidia.com/blog/rendering-in-real-time-with-spatiotemporal-blue-noise-textures-part-2/).

**FAST, Donnelly et al., 2024.** This is the most directly relevant advance. Blue noise is naturally associated with Gaussian filtering, whereas many real-time pipelines apply a different spatial filter and an exponential moving average in time. FAST optimizes sample arrangements against an analytical post-filtering error objective. The paper supports scalar/vector samples, uniform/nonuniform distributions, and spatial/spatiotemporal filters. Its examples show why choosing the spectrum for the filter can outperform generic STBN: a mask that looks attractive before filtering is not necessarily the mask with the lowest residual error afterward. [Publisher record](https://doi.org/10.1145/3651283).

The paper's conclusion reports better temporal convergence when optimizing for EMA rather than generic temporal blue noise. It also distinguishes filter combinations: a product objective suits spatial-plus-temporal reconstruction, while a separate/additive objective better preserves perceptual quality when spatial filtering is absent. For signals with fine structure, optimization against a smaller spatial filter can work better than matching a large nominal blur. This suggests testing a conservative Gaussian-sigma-1 mask before experimenting with wider or box-shaped alternatives. [Author-hosted FAST paper](https://media.contentapi.ea.com/content/dam/ea/seed/presentations/seed-id3-2024-spatio-temporal-sampling-paper.pdf).

**Importance-sampled FAST, Wolfe, Donnelly, and Halén, 2025.** This extends the optimizer to arbitrary target distributions using representative input point sets. Directly optimizing samples in the desired distribution avoids distortions introduced by warping a uniform square. The demonstrated application is lens sampling with complex bokeh. This is valuable background, but does not automatically give a universal SSR texture: GGX visible-normal distributions change with roughness and viewing angle. A single precomputed distribution cannot represent every pixel's changing lobe. Likewise, this SSGI integrator does not choose a cosine-weighted hemisphere ray; it integrates visibility sectors along screen-space slices. The important lesson is to preserve the desired distribution, not to replace every random pair with hemisphere vectors. [JCGT paper](https://jcgt.org/published/0014/01/08/paper.pdf).

**Newer work checked.** NILE, Ahmed et al., SIGGRAPH 2026, is a modular sampling framework that combines specialized local, low-dimensional samplers with high-dimensional low-discrepancy structure. It is relevant if this renderer grows into a deeper path tracer with many sampling dimensions. Its stated problem is broader than this example's few per-pixel random inputs, and its abstract does not establish a better drop-in solution for the present bilateral/EMA denoising pipeline. FAST therefore remains my first implementation candidate. [NVIDIA primary publication](https://research.nvidia.com/labs/rtr/publication/ahmed2026nile/).

## Progressive samplers and additional recent work

Heitz, Belcour, Ostromoukhov, Coeurjolly, and Iehl's 2019 sampler combines per-pixel Owen-scrambled Sobol sequences with optimized scrambling/ranking keys. It preserves low-discrepancy convergence while arranging screen-space error as blue noise, including progressive power-of-two sample counts. This is a stronger progressive-rendering baseline than simply adding blue-noise offsets to arbitrary sequences. [Author paper](https://perso.liris.cnrs.fr/david.coeurjolly/publication/heitz-19/heitz-19.pdf).

Belcour and Heitz's 2021 improvements introduce a blur-based quality criterion, rank-1 lattice sampling, and parallel GPU optimization. Their experiments connect lower blurred error on test integrands with improved denoising in rendered scenes. These are legitimate alternatives if long-run progressive convergence is a priority. FAST is the first candidate here because this application has short, rejected/clamped EMA histories and very few stochastic inputs; its objective explicitly addresses the temporal and spatial filters. That is an implementation-fit argument, not proof that FAST beats optimized Sobol for this scene. [Author supplemental comparisons](https://belcour.github.io/blog/supp/2021-sampling-bluenoise/), [optimization implementation](https://github.com/unity-grenoble/sampling_bluenoise_sig21).

Z++: Temporal, Non-Uniform, and 3D Z Sampling proposes temporal anticorrelation and support for nonuniform distributions without additional memory or runtime. Its online primary page dates publication to December 2026, after this report date, so it is forthcoming work. It is more directly relevant than NILE to temporal reconstruction, but is not evaluated by these experiments. [NVIDIA project](https://research.nvidia.com/labs/rtr/publication/ahmed2026zpp/).

## What the local implementation needs

These observations come from the repository files, not the publications. `SSRNode.js` obtains a vector from `bindAnalyticNoise`: two components drive stochastic GGX reflection sampling and another jitters ray-march positions, and the fourth modulates mirror bias. An optional environment/MIS path requests another vector. `RNoise.js` generates components from multiples of a common linear scalar and translates a repeating 32-by-32 pattern over time. Those components are not a jointly optimized uniform vector, and the frame-to-frame shifts do not constitute a temporal FAST or STBN sequence.

SSR retains `ggxReflectionSample`, mirror bias, sample weights, and traversal logic. The trials compared a uniform vector3 for the reflection pair and traversal jitter against a more targeted vector2 for GGX plus scalar jitter. The selected vector2/scalar layout uses a spatially offset scalar copy for the fourth, mirror-bias input. This combination preserves uniform marginals but is not jointly optimized in four dimensions. The optional environment/MIS sampler remains unchanged; this interior example uses probe fallback rather than that sampler. Precomputing GGX-distributed directions would require a family of distributions for changing roughness and viewing angle, making a faithful comparison harder.

`SSGINode.js` uses a different algorithm: angular slices, radial steps, and a visibility bitmask. Its current angular randomness combines interleaved gradient noise with temporal rotation. Its radial start is the **sum** of a fractional spatial/temporal offset and another random value. Replacing that sum by a single uniform number changes its range from approximately [0,2] to [0,1], shifting radial samples and confounding the comparison. A fair first implementation uses FAST randomness for the angular offset and preserves the two-component radial sum. Any later radial-distribution redesign should be measured as a separate experiment.

`CoverageDenoiseNode.js` performs a 3-by-3 spatial reconstruction with Gaussian tap coefficients, multiplied by depth, normal, and roughness weights. Tap spacing varies with radius and roughness. Temporal reprojection then blends history with a bounded effective frame count and rejection/clamping. Gaussian-sigma-1 plus EMA-alpha-0.1 FAST is a reasonable approximation, but there is no exact fixed linear filter match: the actual bilateral weights and history acceptance depend on scene content. The default 16-frame limit is also not identical to alpha 0.1. A claimed exact match would overstate the evidence.

## Practical asset and integration choices

The authors provide 128-by-128-by-32 textures. Read `(pixelX % 128, pixelY % 128, frame % 32)` with integer addressing, no mipmaps, and no color-space conversion. A texture array, 3D texture, or atlas with integer addressing can work. Preserve the optimized temporal order. A global spatial tile shift once per complete 32-frame cycle can extend the sequence, but shifting every frame destroys the intended temporal relationships. For extra inputs, the authors suggest spatial offsets that are far apart, including offsets derived from an R2 sequence. These behave approximately independently rather than forming a jointly optimized larger-dimensional sample. [FAST design and usage](https://github.com/electronicarts/fastnoise/blob/main/FastNoiseDesign.md).

The downloadable archive is [noise.zip](https://media.githubusercontent.com/media/electronicarts/fastnoise/main/noise.zip), 198,692,784 bytes. The equivalent raw.githubusercontent.com URL contains a Git LFS pointer. Inside it, start with `noise/vector2/temporal/exp/vector2_uniform_gauss1_0_exp0101_product_0.png` through `_31.png`, plus corresponding `real_uniform_gauss1_0_exp0101_product_*.png` under `noise/real/temporal/exp/`. The `separate05` variants provide the perceptual/raw alternative. Matching `vector3_uniform_gauss1_0_exp0101_{product,separate05}_*.png` files under `noise/vector3/temporal/exp/` support the initial vector3 trial; the selected layout uses the vector2 and scalar sets. PNGs are roughly 58 KB for vector2 and 40 KB for scalar per slice; runtime storage depends on the uploaded format. The repository distributes source and assets under BSD-3-Clause; retain its copyright, conditions, and disclaimer with redistributed material. [Repository](https://github.com/electronicarts/fastnoise), [license](https://raw.githubusercontent.com/electronicarts/fastnoise/main/LICENSE.txt).

## Comparison criteria

Capture identical camera, viewport, resolution scale, material settings, ray budgets, frame indices, exposure, and denoiser settings. Compare the baseline against FAST product and FAST separate. Include SSR Raw, SSR Filtered, SSGI Raw, SSGI Filtered, raw coverage, and filtered coverage. Compare early convergence and settled history, then camera motion and history rejection. A changed mean coverage can signal an accidental distribution or addressing change; less visible grain alone does not establish accuracy.

For quantitative quality, construct a long-run reference from the same integrator, and report image or selected-region error against that reference. A temporally averaged noisy baseline is only an approximate reference and should be identified as such. Record temporal variation in stationary regions separately from spatial residual error. Check both bright glossy reflections and broad diffuse walls; behavior near geometry edges may be dominated by rejection rather than sampling.

Performance should use real GPU rendering, adapter verification, disabled vsync/frame caps, warmup, and sequential several-second runs. Count GPU-completed frames, including all rendering and reconstruction work, and drain queued work before changing methods. A texture fetch may replace arithmetic without changing total throughput; reduced noise at equal throughput is already useful. Claim a speedup only after lowering sample cost while holding measured quality approximately constant. Blue noise cannot recover off-screen geometry, repair insufficient depth representation, or remove all bias from screen-space traversal.

## Initial implementation

The experiment adds optional `noiseNode` inputs to SSR and SSGI. Existing callers retain their original patterns. The interior example offers Original, FAST Product, and FAST Separate independently for each effect; changing a selection resets lighting history. SSR uses xy for GGX, z for traversal, and a separately optimized scalar for mirror bias. SSGI uses x for slice rotation and y+z for the radial offset, preserving the original two-uniform range and approximate triangular distribution.

Both trial layouts pack 32 sampling slices alongside 32 scalar slices into opaque 256×4096 RGB PNGs. The initial vector3 slices were replaced by the vector2/scalar layout described below. Opaque storage avoids browser image alpha conversion corrupting random values. `utils/noise/pack_fast_noise.py` reproduces this layout from the authors' archive. Each texture uses approximately 4 MiB of GPU storage as RGBA8, with no mipmaps. Byte-bin centers map samples strictly inside (0,1). Constant offsets distinguish the two passes, and frame order is retained.

## Measurement method

The focused scenes are Cornell Box (Metallic Sphere), Cornell Box (Mirror Wall), Primitives Floor (Color Rough), and Primitives Floor (Metallic Mirror). Measurements use the hardware Apple M3 Metal adapter, a 960×720 viewport at pixel ratio 1, the existing tracing and denoiser settings, and a 1024-pixel bake atlas. The smaller bake atlas keeps setup practical; it is shared by all methods in each scene.

Quality runs bake once, warm up, then freeze the resolved source-lighting texture. All methods therefore see identical illumination rather than their own evolving multi-bounce feedback. After each sampler change, histories reset and warm up for 64 frames. Each method is measured for 96 frames. An independent 256-frame PCG-uniform run through the same integrators supplies an approximate reference. The FAST volumes repeat every 32 frames; the measurement spans three full periods, not 96 independent FAST samples.

RGB is measured in linear incoming radiance for SSR and linear irradiance for SSGI, multiplied by the corresponding coverage. SSR measurements use the actual denoiser input, including its material demodulation and roughness-dependent coverage fade. They are compared with the corresponding filtered output. SSGI's half-resolution reference is expanded with nearest texel centers for comparison with its full-resolution reconstruction. RMSE includes spatial-filter bias as well as sampling error; temporal standard deviation and adjacent-frame RMS difference separately describe stationary fluctuation. Black background pixels are included equally in every method. The reference retains the screen-space integrators' limitations and finite-sample error; it is not a path-traced ground truth.

The main SSR Raw and SSR Filtered debug views now show the same coverage-weighted incoming radiance. Previously Raw showed a material-weighted tracing contribution while Filtered showed demodulated radiance. The original trace remains available in the Inspector under SSR Trace (BRDF Weighted). SSGI Raw and Filtered already shared units: irradiance before receiver albedo, multiplied by coverage for display. Coverage views show dimensionless coverage/validity values in [0,1].

Throughput runs use full Beauty rendering with source-history copies enabled, disabled vsync and frame-rate limits, warmup, and three sequential five-second measurements per method. The harness renders batches of four complete frames, waits for `GPUQueue.onSubmittedWorkDone()`, and then increments the completed-frame count. It checks that every counted frame also updated the SSGI denoiser, and records draw counts. Timing is over the whole run; there are no individual-frame timers. GPU readbacks and screenshots are outside timed intervals.

## Refinement for the actual sampling dimensions

The first experiment used joint uniform vector3 masks for direction and jitter. The refined layout uses a jointly optimized vector2 for GGX's two inputs and scalar FAST for traversal jitter. A fixed spatial offset (49,83) supplies a decorrelated copy of that scalar for mirror bias. SSGI instead takes scalar noise for slice rotation and sums the vector2's components for radial jitter. The scalar copy is approximately decorrelated, not jointly optimized with the other dimensions. This targets each effect's dominant input more closely while preserving its required uniform marginals and SSGI's radial range.

The packer defaults to this vector2 layout; its optional third argument `vector3` reproduces the first experiment. The atlas loader explicitly disables vertical flipping and color conversion. This keeps slice 0 at the top and preserves the authors' temporal order. The tests check both a synthetic atlas (spatial and temporal wraparound) and source PNG bytes after browser decoding and GPU upload.

## Reproducing the comparisons

Run from the repository root with dependencies installed. `CHROME_PATH` can select a local hardware Chrome executable. The harness refuses software adapters. It opens a visible Chrome window, starts its own local server, and closes both when finished.

```sh
SAMPLING_UNIT_ONLY=1 node test/benchmarks/interior-sampling.mjs
SAMPLING_MODEL='Cornell Box (Metallic Sphere)' SAMPLING_OUTPUT=/tmp/sampling-cornell node test/benchmarks/interior-sampling.mjs
SAMPLING_MODEL='Cornell Box (Mirror Wall)' SAMPLING_OUTPUT=/tmp/sampling-mirror node test/benchmarks/interior-sampling.mjs
SAMPLING_MODEL='Primitives Floor (Color Rough)' SAMPLING_OUTPUT=/tmp/sampling-floor-rough node test/benchmarks/interior-sampling.mjs
SAMPLING_MODEL='Primitives Floor (Metallic Mirror)' SAMPLING_OUTPUT=/tmp/sampling-floor-mirror node test/benchmarks/interior-sampling.mjs
```

`SAMPLING_MODES='Original,FAST Product,FAST Separate,Selected'` also measures the selected production policy: FAST Separate SSR and Original SSGI. `SAMPLING_FRAMES` and `SAMPLING_REFERENCE_FRAMES` default to 96 and 256. Each mode captures all four lighting views, their raw and filtered coverage views, and Beauty. The JSON files retain RGB error, temporal deviation, frame differences, coverage means/errors/deviations, completed-frame counts, draw counts, and adapter metadata. `SAMPLING_CAPTURE_ONLY=1` captures without measuring; `SAMPLING_SKIP_BENCHMARK=1` omits throughput.

To try the alternative vector3 layout, call the packer with an alternate output directory and third argument `vector3`, then set `SAMPLING_ATLAS_DIRECTORY` to that directory and `SAMPLING_SWIZZLE=xyz`. The final vector2/scalar layout uses `wxy` for SSGI.

Validation includes lint on every changed JavaScript/HTML file and 18 targeted hardware GPU tests with no failures or skips: two atlas tests, four SSGI tests, and twelve coverage-denoiser tests. The PNG-upload test verifies independently decoded source bytes at temporal slices 0, 1, and 31. The synthetic-atlas test checks pixel wraparound, temporal wraparound, separate scalar-tile addressing, and byte-bin decoding.

The source archive is pinned by SHA-256 `35fe2ce496a837186931acc5afedcdf53ef22d1aaf6d01ad1dcd416c5ff68cea`; the packer verifies it before producing textures. The authors' BSD-3-Clause license is retained in `examples/textures/noises/fast/LICENSE.txt`.

## Measured results and selected defaults

The following table compares FAST Separate against Original with FAST enabled on both effects. Negative percentages indicate lower measured error or variation. The SSGI columns describe the experiment, not the selected default.

| Scene | SSR filtered RMSE | SSR temporal deviation | SSGI filtered RMSE | SSGI temporal deviation |
| --- | ---: | ---: | ---: | ---: |
| Cornell metallic sphere | -13.5% | -40.8% | +0.5% | +63.8% |
| Cornell mirror wall | -6.0% | -46.4% | +1.0% | +51.2% |
| Primitives color rough | +1.5% | -44.9% | -1.6% | +20.7% |
| Primitives metallic mirror | -0.7% | -24.6% | +1.0% | +20.7% |

SSR's largest improvement is temporal stability. Single-frame raw error stayed broadly similar, while the filtered sequence fluctuated less. This is the expected distinction between changing noise magnitude and changing how well a denoiser removes it. Separate FAST consistently performed well under this implementation's nonlinear reprojection, clipping, and bilateral reconstruction, despite Product being the theoretical choice for a matched fixed spatial-plus-temporal filter. The supplied Gaussian/EMA masks are only an approximate match to these filters.

Coverage confirms that this is not simply reduced SSR participation. For example, Cornell mirror-wall raw mean SSR coverage changed from 0.101050 to 0.101065, while filtered coverage temporal deviation fell 44%. The rough primitive floor's corresponding means were 0.052933 and 0.052941, with 42% lower filtered coverage deviation. FAST SSGI, by comparison, increased filtered coverage fluctuations; its radiance flicker was accompanied by less stable confidence.

The original SSGI sampler has a short repeating angular/radial schedule. FAST's more varied samples sometimes brought mean irradiance or coverage closer to the independent reference, but did not produce a consistently calmer filtered result. The scalar-angle/vector2-radial refinement improved on the first vector3 attempt, but still failed the stability objective. Keeping Original SSGI is the measured choice for the current denoiser and settings.

The selected mixed policy was also measured directly in the metallic-sphere and metallic-mirror-floor scenes. Its frozen-source SSR outputs matched FAST Separate and its SSGI outputs matched Original, confirming that the two controls work independently. Beauty captures for this policy are included in the comparison artifacts.

Median sustained GPU-completed frames per second, from three five-second runs per method:

| Scene | Original | FAST Product | FAST Separate | Selected mixed policy |
| --- | ---: | ---: | ---: | ---: |
| Cornell metallic sphere | 108.22 | 107.81 | 107.84 | 108.34 |
| Cornell mirror wall | 109.68 | 109.18 | 109.38 | — |
| Primitives color rough | 180.76 | 179.51 | 179.95 | — |
| Primitives metallic mirror | 179.61 | 178.38 | 178.21 | 178.94 |

There is no established speedup. The selected policy stayed within about half a percent of Original in the two directly measured mixed-policy runs. No ray-count reduction was adopted; this change improves SSR stability at the existing sample cost.

These are stationary-camera measurements. Camera motion, disocclusion, different history lengths, and different spatial radii may change the relative benefit; the sampling textures are optimized in screen space. The reference retains finite-sample uncertainty and screen-space bias. The 1.5% rough-floor error increase is a measured tradeoff, not evidence of universally improved accuracy.

Complete linear RGB and coverage metrics, per-run throughput, and adapter metadata: [Cornell metallic sphere](interior-sampling-data/cornell-sphere.json), [Cornell mirror wall](interior-sampling-data/cornell-mirror.json), [rough primitive floor](interior-sampling-data/floor-rough.json), [metallic mirror floor](interior-sampling-data/floor-mirror.json), [hardware GPU test results](interior-sampling-data/unit-tests.json).

The final example also completed its normal 4096-pixel bake and Beauty rendering with FAST Separate SSR enabled and Original SSGI enabled, using the Apple Metal hardware adapter, with no browser or GPU validation errors. Repacking the pinned archive reproduced both checked-in noise PNGs byte for byte.
