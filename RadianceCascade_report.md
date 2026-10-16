# Radiance cascades in three.js

Research and experimental implementation, 9 October 2026. Branch: [`bhouston/three.js: radiance_cascades`](https://github.com/bhouston/three.js/tree/radiance_cascades). Based on repository revision `8d486d4cc1`.

## Smoothing experiments — 10 October 2026

Worktree: `~/Coding/.worktrees/three.js-radiance-cascades-smooth`, branch `radiance-cascades-smooth`, based on `c0b581e1021` from `radiance_cascades`. The original preset is retained. The sections below this update describe the previous implementation and its historical results.

### Why the voxel grid becomes visible

Sharing a voxel grid with VXGI does not mean sharing its reconstruction. The original RC implementation uses nearest-cell `load()` calls at mip zero, an occupancy cutoff, and a single opaque first hit. Moving a ray across a cell boundary can abruptly change both visibility and outgoing radiance. VXGI instead uses interpolated volume samples, distance-dependent mip footprints, and front-to-back fractional opacity. It smooths the same scene representation before gathering it.

Two more sources of patchwork are independent of voxel resolution: only 32 base directions represent incident radiance, and each visible pixel originally gathers from four screen-space probes with a hard normal cutoff. A small surface can have no compatible probe at all. The distant cascades also begin at different world-space origins, so simply interpolating their intervals can create gaps and overlap.

The requested [GigiRadianceCascades](https://github.com/Kosmokleaner/GigiRadianceCascades/tree/c3db945bc6796c5116bdec37713c958588a75676) reference was cloned to the sibling directory `~/Coding/OpenSource/GigiRadianceCascades`. It is a **2D** implementation, not a 3D voxel renderer. Its `PropagateCommon.hlsl` uses bilinear scene samples and fractional transmittance; its merge traces each near segment to an upper probe's interval start before composing. Those ideas informed independent TSL experiments; no shader files were copied.

### Controls in both examples

Expand **RC experiments** in either example. The existing lighting selector still compares RC, VXGI, baked probes, direct lighting, and raw RC irradiance.

| Option | Experiment and tradeoff |
| --- | --- |
| Original | Binary DDA and the original four-probe gather, retained for comparison. |
| Filtered voxels | Trilinear mip-zero volume samples, directional opacity, and front-to-back integration. Half-voxel steps sample their midpoint and clip to interval ends; opacity is corrected for actual step length. Changes energy and softens voxel boundaries. |
| Filtered + smooth gather | Adds a positive cubic B-spline over 4×4 probes, continuous normal weighting, and receiver-plane weighting/rejection. Smooths probe-grid derivatives and improves support near edges, but can blur lighting and still miss small surfaces. |
| Angular footprint + smooth | Chooses mip levels from each angular cell's equal-area footprint at its trace distance. The footprint-scale slider trades detail for filtering; larger footprints may leak through thin geometry. This is a cone-like approximation within RC intervals. |
| Connect cascade intervals | Retraces each near segment to each compatible upper probe's interval start before composing radiance/transmittance. Avoids reusing a segment from an unrelated origin, at substantial extra work. Upper angular bins are still averaged, so this is approximate 3D transport. |
| Repair unsupported pixels | Only receivers with effectively zero compatible probe weight trace their own 32-direction hemisphere through the whole cascade range. Diagnoses missing screen-probe support, with extra cost and possible seams where it switches to reconstructed lighting. |
| Base directions: 32 / 128 | Doubles both angular axes while preserving four-child nesting. Uses four times the interval rays and atlas storage; helps separate angular undersampling from voxel artifacts. |
| Volume resolution: 64 / 128 | Revoxelizes the shared RC/VXGI volume. Resolves finer geometry at greater memory and update cost; does not fix probe interpolation by itself. |
| Probe spacing: 4 / 8 / 16 / 32 | Denser screen probes capture more small surfaces; doubling density in both axes uses approximately four times the rays and atlas storage. |
| Normal offset | Moves trace origins in voxel units; too little risks self-intersection, too much loses local occlusion. |

Start with **Filtered + smooth gather**, then try 8-pixel spacing. Compare 32/128 directions independently before raising voxel resolution. Leave interval connection and unsupported-pixel repair off until isolating those specific problems. Presets reset the tracing/reconstruction options while preserving the chosen spatial, angular, and voxel resolutions.

### Validation and practical limits

Both examples were rendered and visually inspected in Chrome on a physical Apple Metal WebGPU adapter (`vendor: apple`, `architecture: metal-3`, fallback false). The smoothing harness rejects software adapters, verifies finite nonnegative HDR irradiance, repeats unchanged frames without temporal history, switches all presets and comparison methods, changes angular and voxel resolutions, exercises interval connection and receiver repair, resizes to 807×613, changes probe spacing, and checks zero irradiance after disabling the sources. The Cornell scene also checks off-screen emissive contribution in every tracing mode. Captured page and WebGPU errors were empty. The legacy Cornell scenario harness also ran, but its 0.1% energy tolerance after revoxelization failed on this adapter. Serving the unmodified `radiance_cascades` node and example reproduced a failure in the same family of restored-scene checks. The tolerance was not relaxed; deterministic voxel representative selection remains a separate issue. The legacy harness now waits for actual example readiness rather than network idleness and reports the measured energy difference on those assertions.

Filtered tracing and wider reconstruction visibly reduce discontinuities around object bases and across the multi-room lighting field. They do **not** remove every artifact. In particular, sparse probe support, coarse voxel geometry, angular quadrature, and approximate displaced-probe transport remain. Filtering changes brightness; a brighter or smoother screenshot is not evidence of improved physical accuracy. No path-traced reference was generated, and no speed advantage over VXGI is claimed. The historical per-frame timings later in this document are not sustained-throughput benchmarks.

The new harness writes unobstructed rendered comparisons, raw irradiance views, and HDR summary data to the ignored `test/e2e/output-screenshots/radiance-smoothing/` directory:

```sh
node utils/build/dev.js
node utils/server.js -p 8088
# In another terminal, from this worktree:
RC_BASE_URL=http://localhost:8088 node test/e2e/radiance-cascades-smoothing.js
```

`CHROME_EXECUTABLE` overrides the default macOS Chrome executable. Generated `build/` files must stay uncommitted and be restored before switching branches. During this session, the source preview at port 8089 supplies development entry points without changing tracked build outputs.

## Recommendation and current result

Radiance cascades are worth evaluating as a **runtime diffuse GI gather**, initially reusing three.js's voxel scene representation. The branch implements that experiment in TSL/WebGPU. It uses **mip-zero voxel traversal, without a software BVH**, and compares radiance cascades, existing VXGI, baked light probes, and direct lighting in two identical scenes:

- [Cornell box](examples/webgpu_radiance_cascades.html), copied from `webgpu_lightprobes.html`.
- [Complex multi-room scene](examples/webgpu_radiance_cascades_complex.html), copied from `webgpu_lightprobes_complex.html`. This is the example with `complex` in its name, rather than Sponza.

Both render and pass the supplied checks on a physical AMD WebGPU adapter. The prototype responds to light changes without baking or temporal accumulation and includes geometry and emitters outside the camera's depth buffer. It remains an approximation: dark patches around silhouettes, missing probe support at corners, coarse voxel geometry, and approximate cascade interpolation are visible. It has not established a fidelity or speed advantage over VXGI. Keep it as an experimental addon until those issues are addressed.

## What radiance cascades are

The method divides incoming light into distance intervals and samples each interval at a different spatial and angular resolution. Nearby occluders need dense spatial samples, whereas distant lighting needs more directional samples. Cascades trade spatial density for angular density as distance increases. Each interval stores radiance and transmittance; farther intervals are merged back into nearer ones. The conceptual basis is described in [Alexander Sannikov's original paper repository](https://github.com/Raikiri/RadianceCascadesPaper). The author explicitly notes that this manuscript was not published in JCGT, despite using its template.

For two consecutive intervals along the same ray, composition is:

```text
L_combined = L_near + T_near * L_far
T_combined = T_near * T_far
```

In this spike, opaque hits have `T = 0`; empty intervals have `T = 1`. The implementation integrates the merged directional radiance against the receiver's cosine hemisphere to produce diffuse irradiance. This composition is the local algorithm used here; actual interpolation between displaced probes is approximate.

The cascade hierarchy is a lighting representation and gather strategy. It still needs a way to intersect the scene and obtain outgoing radiance. It can use depth marching, an SDF, voxels, or other intersection mechanisms; a BVH and hardware ray tracing are not prerequisites.

**It is normally a runtime method, not a lightmap bake.** Geometry may still be preprocessed or voxelized ahead of time. Light injection, tracing, merging, and irradiance gathering can run every frame. One could freeze the resulting field for static content, but that is an application choice. Also, merging distance intervals does not create additional reflection bounces. More bounces require separate lighting feedback or another transport mechanism.

## Use in the game industry

The most prominent developer presentation is [Grinding Gear Games' ExileCon 2023 talk, Rendering Path of Exile 2](https://www.youtube.com/watch?v=TrHHTQqmAaM). It presents a practical GI approach associated with radiance cascades for the game's constrained, approximately top-down view. This is evidence of production-oriented use; it should not be read as proof that every platform or current graphics preset runs an identical algorithm, or that the technique transfers unchanged to unrestricted 3D cameras.

[Lumina's developer page](https://nixon-voxell.itch.io/lumina) explicitly identifies its custom radiance-cascades GI and lists the Windows game as released on 24 April 2025. That gives a smaller released-game example, alongside the many tutorials and engine experiments. The page also links its source repository. These examples establish real use, not broad industry adoption.

The more recent [Split Radiance Cascades paper by Rouli Freeman and Alexander Sannikov](https://arxiv.org/abs/2607.20384), submitted 22 July 2026, tackles general 3D diffuse GI with sparse world-space probes and ray splitting. Its abstract distinguishes this from earlier 2D and screen-space implementations whose volumetric storage costs limit their applicability. It is a promising next research direction, **not the algorithm implemented by this branch**. This branch has dense screen-space probe layouts and separately traced intervals, without a sparse hashmap or ray splitting.

## Public code investigated

License labels below describe the inspected revision. Publicly readable code without a clear license was studied as a reference and was not copied into the branch. Rendering smoke checks do not certify an implementation's correctness or performance.

| Project and inspected revision | License / status | What was useful and what was tested |
| --- | --- | --- |
| [SimonDev's Shaders_RadianceCascades](https://github.com/SimonDevYouTube/Shaders_RadianceCascades/tree/bba7867d1c0f1f0043c0ad618c6967d06d92c11e) | MIT; 2D WebGL tutorial | A readable working example of interval tracing and cascade merging. Rendered on the physical AMD GPU. Its 2D distance-field backend is not a general 3D replacement. |
| [Unity URP RadianceCascades2DGI](https://github.com/Youssef-Afella/UnityURP-RadianceCascades2DGI/tree/7e8d77673004b040e498ce97e303d83fffebaae4) | MIT; Unity 2D | Studied jump-flood distance fields, ping-pong cascade resources, and the outer-to-inner trace/merge organization. Unity execution was not tested here. |
| [Voxel radiance cascades](https://github.com/creikey/voxel-radiance-cascades/tree/1722ebab0e8ca42a3563a6a30db59f8b6619589d) | MIT; voxel experiment | Rendered on the physical GPU. Useful for studying voxel integration, but its indexed Fibonacci directions do not provide geometrically nested four-child angular cells. I would not port its merge mapping unchanged. |
| [chandra](https://github.com/entropylost/chandra/tree/c2902b527a2288b978c0779e23050234cbea3030) | No clear reuse license identified; research reference | Relevant to Split RC. Source was studied; missing sibling dependencies prevented treating it as a ready-to-run integration. |
| [split-radiance-cascades-webgpu](https://github.com/wroughtinator/split-radiance-cascades-webgpu/tree/34255d334b1fc3880664d2580486bbd111f1be52) | No clear reuse license identified; public 3D WebGPU port | Sponza smoke test rendered on the physical GPU without fatal or console errors. Its software BVH conflicts with the requested backend; neither that backend nor its code was adopted. |
| [three-rc](https://github.com/CodyJasonBennett/three-rc) | Explicit all-rights-reserved notice | Relevant three.js experiment, excluded from code reuse. Public availability is insufficient permission to copy it. |

The best approachable licensed learning reference is SimonDev's 2D code. For this repository, the most practical integration base is **the existing MIT-licensed three.js voxel infrastructure**. The new TSL cascade code is an independent implementation; it does not vendor another project's shader files.

## Comparison with existing three.js GI

This comparison reflects the code in this checkout, including the newer `LightProbeGrid` and WebGPU VXGI addons, rather than assuming every three.js release has the same features.

| Method | Scene data and update behavior | Benefits | Main limitations |
| --- | --- | --- | --- |
| `LightProbe` / `LightProbeGrid` | Nine L2 spherical-harmonic coefficients per probe; grid captures rendered cubemaps when baked. Extra bake passes can add indirect bounces. | Cheap runtime interpolation; world-space lighting survives camera changes; grid volumes can be authored independently. | Baked data becomes stale after lighting or geometry changes. Low directional bandwidth; spatial interpolation and volume boundaries can leak light. A basic single probe does not give spatially varying GI. |
| `VXGINode` / `VXGIVolume` | Voxelizes geometry, injects current lighting, builds a radiance mip hierarchy, then gathers with cones. Geometry changes require voxelization updates. | Runtime light response; off-screen geometry; an existing reusable WebGPU backend and optional bounce/temporal machinery. | Coarse voxels and cone mips blur detail and can leak. Geometry updates can be expensive; resolution and memory limit scale. |
| This radiance-cascades spike | Shares the same voxel volume; traces individual cells at mip zero into interval atlases, merges cascades, then gathers screen-space irradiance. | Runtime light response without history; hierarchical angular detail; shared interval work across receivers; off-screen hit data from the voxel volume. | Extra compute and atlas memory; visibility/interpolation errors; screen-space probe support changes with camera; inherits voxel resolution and update costs. No demonstrated win over VXGI yet. |
| `SSGINode` / SSR depth tracing | Camera depth and normal buffers, plus a scene color/radiance buffer. | Avoids voxel preprocessing; fits an existing postprocessing pipeline. | Cannot trace absent, hidden, or behind-camera surfaces; thickness heuristics and missing data remain. Cascades do not restore missing depth-buffer geometry. |

Against baked probes, the clearest benefit is dynamic lighting without rebaking. Against VXGI, RC offers a different allocation of gather work and directional information, rather than a new scene representation. Both share exactly the same off-screen scene data in these comparisons. A timing or accuracy improvement must be measured rather than inferred from the name of the technique.

Relevant local sources: [LightProbeGrid](examples/jsm/lighting/LightProbeGrid.js), [VXGINode](examples/jsm/lighting/vxgi/VXGINode.js), [VXGIVolume](examples/jsm/lighting/vxgi/VXGIVolume.js), [SSGINode](examples/jsm/tsl/display/SSGINode.js), and [SSRNode](examples/jsm/tsl/display/SSRNode.js).

## Prerequisites and integration design

The current implementation requires a hardware-backed WebGPU context, compute shaders, storage textures, RGBA16F intermediate targets, and an ordinary perspective depth buffer. It explicitly rejects the WebGL backend, logarithmic depth, and non-perspective cameras. It also needs scene bounds, voxel occupancy, current linear HDR outgoing radiance, and a material hook accepting diffuse irradiance.

The pipeline is implemented in [RadianceCascadesNode.js](examples/jsm/lighting/RadianceCascadesNode.js):

```text
single-sample depth/normal prepass + current shadows
  -> existing VXGIVolume voxelization / direct-light and emissive injection
  -> trace distance intervals using mip-zero voxel DDA
  -> merge from the farthest cascade to the nearest
  -> cosine-integrate into a small octahedral irradiance tile per probe
  -> gather compatible neighboring probes at each visible receiver
  -> builtinGIContext -> normal material lighting and tone mapping
```

The prepass is explicitly single-sampled, even though the final renderer uses antialiasing. This avoids inconsistent multisampled-depth bindings when targets resize. Renderer state and the GI context are restored after internal passes. Borrowed voxel volumes are not disposed by the RC node.

Four cascades use 32, 128, 512, and 2,048 equal-area angular cells per probe. Doubling azimuth and elevation gives four genuine angular children. Screen-space probe counts shrink by approximately four per cascade. At 800 x 600 and 16-pixel base spacing, rounding produces 259,840 interval rays and approximately **8.15 MiB** of RC textures. That estimate includes interval atlases, merged atlases, prefilter tiles, and full-resolution irradiance; it excludes voxels, prepass targets, shadows, and transient/backend allocations.

Distance intervals grow by a factor of four in length. With a first interval of 0.3 scene units, the four boundaries are 0, 0.3, 1.5, 6.3, and 25.5. The larger multi-room scene uses 0.6 instead, ending at 51 units. Receiver origins move 1.5 voxels along their normals to reduce self-intersection. Cells with occupancy above 0.1 count as opaque; this is a coarse geometric approximation, not alpha-aware surface traversal.

The examples borrow `VXGINode.volume` so RC and VXGI compare against the same voxelization. Both use `bounces = 0`, meaning injection has no additional indirect feedback and the gather supplies one indirect reflection. The probe captures also use `bounces = 0` for a comparable single-indirect-bounce result. VXGI temporal filtering and its extra AO multiplier are disabled. Original geometry, materials, light intensities, camera presets, shadows, and exposure are preserved. The complex scene retains its two 6 x 6 x 6 probe grids and their falloffs. Test controls add scenarios without changing the reset baseline.

## Hardware validation and observations

The tests confirmed a physical AMD adapter (`vendor: amd`, `architecture: gcn-5`, fallback false), with WebGL also identifying AMD Radeon Graphics through ANGLE/D3D11. SwiftShader and software adapters are rejected by the harnesses. The new shaders compiled and both scenes completed their scenario checks with **zero captured page or WebGPU validation errors**.

The [Cornell-box harness](test/e2e/radiance-cascades.js) checks finite, nonnegative HDR irradiance; exact repetition with unchanged inputs and no temporal history; method switching; zero residual light after disabling the source; a genuinely behind-camera emissive object; changed geometry and a thin divider; camera movement; odd-sized viewport resize; and changed probe spacing. The emitter scenario moves the camera inside the room and verifies every emitter vertex lies behind it. Reset restores the original light-probe camera. Revoxelization can select a different representative triangle in cells shared by surfaces, so restored-scene energy uses a 0.1% tolerance rather than bitwise equality.

The [complex-scene harness](test/e2e/radiance-cascades-complex.js) checks the same basic shader/output invariants, compares all four lighting methods, switches each room's light independently, tests both lights off, places the camera inside each room, and exercises resize and probe spacing. In the initial successful run, the right light alone produced approximately 80,333 summed irradiance on the right image half and 2,593 on the left. The left light alone produced approximately 64,995 on the left and 3,023 on the right. These are aggregate image-space HDR samples, not physical flux measurements. The doorway allows real cross-room transport, so these numbers cannot separate leakage from legitimate illumination.

For a fixed image region, mean display RGB (0-255, after tone mapping) was:

| Scene / region | Direct only | Baked probes | VXGI | RC |
| --- | ---: | ---: | ---: | ---: |
| Cornell back wall | 201.27 | 223.01 | 217.59 | 219.19 |
| Complex scene, central room/object region | 54.30 | 94.75 | 87.30 | 81.45 |

These values demonstrate a visible indirect contribution. They are not error metrics: brighter is not necessarily more accurate. No path-traced reference was produced. Screenshots are included in `examples/screenshots/`; fuller scenario PNGs and JSON are written to the ignored `test/e2e/output-screenshots/` directory when tests run.

The complex harness also records median, minimum, and maximum CPU-to-GPU-completion times for twelve warmed frames per method. Each sample advances a real browser frame so nodes marked `FRAME` update, rather than timing repeated rendering of a cached pass. Thirty warmup frames allow switched pipelines to compile. On this AMD adapter at 800 x 600, the observed medians were **7.1 ms direct, 7.2 ms baked probes, 7.9 ms VXGI, and 31.5 ms RC**. This unoptimized spike was substantially slower than VXGI in this scene. Times include CPU submission and synchronization, exclude the initial probe bake/voxelization, and are not isolated GI GPU timestamps. Consult the generated `results.json` on the target machine before making performance decisions.

## Limits and follow-up work

The multi-room scene makes several limits explicit. The longest scene axis is 16 units, so 64 voxels imply roughly 0.25-unit cells before padding. Table legs are only 0.1 units thick and the divider is 0.3. The representation cannot preserve all that detail. A 128-voxel follow-up would help distinguish voxel errors from cascade errors, at increased memory and preprocessing cost.

Screen-space probes sample only selected visible surfaces. Rejecting incompatible normals prevents some blending errors but leaves receivers without support near silhouettes and corners. Distant intervals also originate at displaced probes: the spike does not retrace a visibility connection, apply full parallax correction, or provide interval extension. This can lose or borrow lighting across a boundary. Dense screen-space layout is not the sparse world-space scheme from Split RC.

Other limits: no environment contribution for rays leaving the voxel bounds; binary occupancy; no transparent transport, directional surface radiance, specular GI, or caustics; no cascaded voxel clipmaps for large worlds; no automated ground-truth comparison. Moving mesh geometry must explicitly mark `volume.needsUpdate`, triggering expensive preprocessing. Emissive material changes also need current voxel material data. Light changes update through injection with current shadow maps.

Recommended next steps are better receiver/probe support at discontinuities, a visibility-aware cascade merge, a voxel-resolution comparison, and GPU timestamp profiling against VXGI using the same data. Keep the addon separate from core while assessing those changes.

A depth-marching backend is a reasonable second spike and fits the requested no-BVH constraint. Reuse the repository's SSR/SSGI depth reconstruction and intersection conventions, and supply **direct-only linear HDR scene radiance** to avoid reading GI recursively or counting it twice. It can reduce voxel startup cost, but it loses the verified behind-camera emitter capability unless combined with an additional scene representation. A hybrid depth-first/voxel-fallback backend is a subsequent option rather than something this branch already implements.

## Running the branch

After checking out `radiance_cascades`:

```powershell
npm.cmd ci --ignore-scripts
npm.cmd run build
node utils/server.js -p 8086
```

Open `http://localhost:8086/examples/webgpu_radiance_cascades.html` or `http://localhost:8086/examples/webgpu_radiance_cascades_complex.html` in a WebGPU-capable browser. The lighting selector provides RC, baked probes, VXGI, direct only, and raw RC irradiance. The original light-probe examples remain available for comparison.

With the server running, execute:

```powershell
$env:CHROME_EXECUTABLE = 'C:/Program Files/Google/Chrome/Application/chrome.exe'
node test/e2e/radiance-cascades.js
node test/e2e/radiance-cascades-complex.js
```

An already installed Chrome avoids Puppeteer's skipped browser download. `RC_URL` can override the full example URL; include `?test=1` to expose the validation hook. Both scripts deliberately fail if a physical adapter cannot be confirmed. Build outputs are generated locally and are not part of the experimental commits.
