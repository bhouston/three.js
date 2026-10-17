# G-buffer throughput investigation

Run `node utils/server.js -p 8080` from the repository root and open
`http://localhost:8080/examples/webgpu_gbuffer_performance.html`. The import map
loads this checkout's source modules directly: no generated `build/` files or
unmerged technique branches are needed. WebGPU and a physical GPU are required.

This measures a data-path prototype, not the performance of a finished GI/SSR
renderer. There is no Hi-Z allocation, preparation or traversal, and there is no
SSR/SSGI tracing or signal-denoising algorithm.

## Pipeline and supported shading

Desktop uses the existing renderer's metallic/roughness physical lighting model
on a fullscreen triangle. The geometry pass disables lighting and writes actual
base color (including vertex color), mapped normal, computed roughness,
metalness, emission and unjittered NDC motion. Position is reconstructed from
depth using `getViewPosition` and the jittered inverse projection. The resolve
overrides view/world position, view direction and normal, including the shadow
receiver position. Direct lighting and shadow maps use the same light as the
forward baseline. The ground receives the real model shadow.

The deferred subset is ordinary opaque glTF metallic/roughness, normal/bump maps,
vertex color, double-sided surfaces, alpha tests, baked AO and emission. The
albedo alpha byte contains `(round(AO * 127) * 2 + modelId) / 255`; model ID 1 is
metallic/roughness and clear ID 0 is background. AO therefore has seven bits.
Roughness is captured after geometry-derived specular antialiasing. The resolve
applies its BRDF once. Sky uses the scene background; the prefilter environment
is a cached PMREM of `RoomEnvironment`, shared across modes.

Unlit materials, lightmapped materials, custom depth state and unsupported
physical extensions use an **explicit object-level forward fallback**. Reasons
appear in the UI and snapshots. This includes clearcoat, sheen, anisotropy,
iridescence, nondefault IOR/specular properties, transmission, dispersion and
diffuse roughness. A multi-material object with any unsupported material follows
that fallback as a whole. Its opaque geometry is rendered only once, into the
resolved framebuffer with shared depth, and is counted separately. With TRAA,
that fallback also writes motion into the shared velocity attachment. Lightmaps
retain their actual UV bindings through this fallback; reconstructed position is
never treated as a substitute for an atlas. Arbitrary node-material graphs are
outside this experiment.

The optional depth prepass uses a **zero-color-attachment depth-only target**.
Per-source material clones preserve side, map alpha, alpha map/test,
displacement, skinning, morphing and instancing, with color writes and lighting
disabled. The surface pass uses `EqualDepth` and no depth writes. Shared material
depth state is saved/restored once per material. The same jittered camera is used
for both. Default is off; the benchmark compares off/on at equal resolution.

GTAO is the existing real GTAONode, with its depth-derived normal reconstruction.
It produces scalar visibility; **bent-normal estimation is not implemented**.
Disabled AO uses visibility 1 and the actual surface normal as bent-normal
fallback. The resolve combines screen and baked visibility with `min`, avoiding
multiplication of overlapping cavity occlusion. The physical model applies AO
only to indirect diffuse and indirect specular, never direct light or emission.

Diffuse IBL irradiance and specular prefiltered radiance come from the shared
PMREM. They are multiplied by the material's BRDF/albedo inside the physical
model exactly once. The optional local SSGI signal is *irradiance*, initially 0;
it contains no baked albedo and is added through `getGI`. SSR stores radiance,
coverage and positive hit distance separately. No-hit coverage is 0 and distance
is the finite FP16 sentinel 65504. Environment specular radiance is replaced only
where coverage indicates a hit, so no-hit never removes or doubles the fallback.
Denoiser copies preserve radiance and separately retain coverage/distance;
confidence/history age are not packed into those channels. There is no temporal
signal-denoiser state or scratch texture.

## Neutral pass modes and execution inventory

`bypass` uses constant nodes and aliases normal inputs: it allocates no neutral
signal targets. `materialized` executes the following six fullscreen draws in
order, even though their shader bodies are trivial:

1. AO visibility and surface-normal fallback fill (reads surface normal and AO).
2. Local SSGI irradiance fill (0).
3. SSR radiance fill (0).
4. SSR coverage/distance fill (0, 65504).
5. Specular denoiser copy (reads raw SSR radiance).
6. Diffuse denoiser copy (reads local SSGI irradiance).

The resolve reads those textures. These are explicit draws scheduled by the
pipeline node; constants cannot remove them from the node graph. Their bandwidth,
storage and dispatch overhead are real. There are no sampling loops, spatial
kernels or reprojection algorithms in the neutral passes.

The complete desktop order is shadows → optional depth prepass → surface MRT →
optional GTAO → the six neutral draws when materialized → fullscreen PBR resolve
→ restricted opaque extension fallback, if needed → opaque color/depth/history
publication → copy opaque HDR into the transparent framebuffer → forward
transparents → TRAA → optional sharpen → tonemap/output. Output and sharpening
share one final draw. GTAO adds one real draw. TRAA adds one resolve draw and two
history copies. Source publication adds two GPU texture copies and immutable CPU
geometry snapshots. Opaque fallback needs two additional HDR copies. Glass may
add renderer-owned framebuffer copies/mips; these are included in whole-frame
measurement and tracked allocations. `snapshot().passes` lists stage groups;
one group can contain multiple copies/draws. `renderInfo.draws` counts actual
draws, including geometry and fullscreen work.

Mobile and the matched ordinary forward baseline have the same forward renderer
path here: one opaque material traversal, then forward transparency in the same
HDR/depth framebuffer. They omit surface/signal/source-history targets and the
deferred resolve. Only enabled TRAA requires the motion attachment and temporal
history. This deliberately makes their measurements a repeatability check, not
an artificial preset speedup. Resolution scale is independent, uses DPR 1, and
changes the drawing buffer while preserving CSS display size.

## History and temporal limits

Desktop publishes immutable ping-pong **linear HDR opaque color and depth**,
before transparency, AA, sharpening, tonemapping or output conversion. Each slot
has matching frame, model epoch, dimensions, jittered/unjittered projection and
view matrices, plus immutable CPU snapshots of rigid transforms, bone matrices,
instance transforms and morph weights. Metadata identifies the geometry slot and
frame. These CPU copies are intentionally included in the throughput cost; the
neutral signals do not sample any history. Upstream rendering separately retains
its previous geometry for velocity.

TRAA is the actual existing TRAANode, including its jitter hooks, unjittered
projection for velocity, current depth, HDR history and previous depth. NDC
velocity converts to UV inside TRAA with `(0.5, -0.5)`. Static camera/objects must
have zero surface motion even as projection jitter changes. Drawing-buffer sizing
keeps TRAA's previous-depth copies at the correct resolution.

Model/mode/resolution/lighting changes and explicit camera cuts rebuild/reset
history. Large camera position/orientation jumps also invalidate history.
Rigid and skeletal motion use upstream velocity. Upstream morph velocity lacks
previous weights: while morph animation plays, TRAA history is restarted each
frame, visibly disclosed in the UI. This loses temporal accumulation quality for
morph animation. Forward transparents/glass retain opaque background motion and
depth; no layered transparent velocity or refracted motion is represented, so
moving glass can ghost. This is a temporal/material limitation, not a claim of
production completeness. Material variants are selectable as tracked files;
KHR_materials_variants runtime switching is not added.

## Actual allocation inventory

`snapshot().resources.textures` records **every actual GPUDevice.createTexture**
descriptor, including dimensions, format, mips, layers, samples, label and logical
bytes. Destroyed allocations are removed. It includes renderer-owned shadows,
PMREM, DFG LUT, canvas depth, model textures, morph/bone textures, transmission
copies and Inspector preview resources when enabled. `liveBytes`, preset
`peakBytes` and lifetime `sessionPeakBytes` distinguish steady/preset/session
peaks. The session peak includes initialization and overlapping load/rebuild
resources. GPU buffer descriptors/bytes/peak are reported separately, including
geometry, previous geometry, uniforms and readbacks. Immutable geometry CPU bytes
are reported under `history.geometryCPUBytes`. Driver alignment, heap slack and
browser-owned presentation swap chains are outside these logical counts.

For width W and height H, single-sample allocations are:

| Resource | Format / bytes per pixel | Producer → consumers / lifetime |
| --- | --- | --- |
| albedoId | rgba8unorm / 4 | Surface → resolve and Inspector; desktop |
| normal | rg16float signed oct / 4 | Surface → resolve, optional AO contract fill; desktop |
| roughMetal | rg8unorm / 2 | Surface → resolve; desktop |
| velocity | rg16float NDC / 4 | Surface or forward opaque/fallback → TRAA; desktop always, forward only with TRAA |
| emissive | rg11b10ufloat / 4, or rgba16float / 8 | Surface → resolve; desktop |
| current depth | depth32float / 4 | Geometry/prepass → resolve/GTAO/transparency/TRAA/history; all modes |
| opaque HDR | rgba16float / 8 | Resolve/forward → history/composition; all modes |
| composed HDR | rgba16float / 8 | Seed + forward transparent → TRAA/output; desktop; aliases opaque in forward/mobile |
| source colors ×2 | rgba16float / 16 total | Pre-transparent opaque copy → previous-source publication; desktop |
| source depths ×2 | depth32float / 8 total | Current-depth copy → matching previous-source publication; desktop |
| AO + normal fallback | rgba16float / 8 | Fullscreen contract fill → resolve; desktop materialized |
| SSGI raw + filtered | rgba16float ×2 / 16 | Neutral fill → copy → resolve; desktop materialized |
| SSR raw + filtered | rgba16float ×2 / 16 | Neutral fill → copy → resolve; desktop materialized |
| SSR coverage/distance | rg16float / 4 | Neutral fill → resolve; desktop materialized |
| GTAO visibility | r8unorm / 1 | Real GTAO → resolve/contract fill; AO enabled |
| GTAO noise | small generated texture | GTAO only; see descriptor list for size |
| TRAA history + resolve | rgba16float ×2 / 16 | Temporal resolve/history copy → next frame/output; TRAA enabled |
| TRAA prior depth | depth32float / 4 | Current depth copy → next-frame rejection; TRAA enabled |
| sharpen | no target | Four-neighbor filter in final output draw |
| shadow map color + depth | rgba8unorm + depth24plus / 8 at 1024² | Live key-light shadow pass → all material/resolve lighting; shadows enabled |
| PMREM | rgba16float cube, 256² ×6, six mips | RoomEnvironment prefilter → shared IBL; all modes, cached |
| DFG LUT | rg16float, 16² | Physical BRDF integration lookup; shared |
| model textures | asset-dependent mip chains/formats | GLTFLoader → surface/forward; superseded scenes disposed |

Surface-only storage is **22 bytes/pixel including depth** with packed emission,
or 26 with HDR fallback. WebGPU color attachment validation uses format attachment
cost (22 bytes/sample here), not simply storage-byte sum (18). Five attachments
are required. Both layouts fit the actual device's 32-byte default limit and
were validated at odd dimensions; the adapter advertises eight attachments and
128 bytes/sample. The packed format is used only when the device enabled
`rg11b10ufloat-renderable`.

Neutral materialization adds **44 bytes/pixel** of live targets. Disabled targets
are released on rebuild, including AA/GTAO/shadows and source histories in mobile.
There is no denoise scratch, position target, Hi-Z or future tracing allocation.

## Sustained benchmark and validation

Run `node utils/benchmark-gbuffer.mjs --out /path/to/results`. `CHROME_PATH` can
select an already installed browser. The harness launches headful Chrome with
`--disable-frame-rate-limit --disable-gpu-vsync` and disables background
throttling. It rejects software/fallback adapters, verifies the app backend,
and stops the renderer's internal animation loop. Inspector UI remains available
but its debug rendering is disabled during measurements.

Each whole frame explicitly advances Three.js's node frame once. This matters:
otherwise multiple renders inside one rAF interval can reuse frame-scoped
surface/AA work and produce invalid inflated numbers. Every interval asserts
that published source-frame count equals the fenced completed-frame count.

Configurations run sequentially at 1280×720, DPR 1, scale 1, the same normalized
model/camera/environment/key light/shadows, animation paused, sharpening off,
ACES output and TRAA on unless being isolated. There are two seconds of warmup
and five seconds of measurement, three repetitions per configuration/model.
Eight complete frames are submitted per bounded batch, with one queue-completion
fence per batch, never per frame. A final drain is included in the measured wall
interval. The result counts **GPU-completed frames**, including all rendering,
history and output work. There are no individual-frame timings or rAF FPS claims.
Batch fencing/task-yield/CPU/instrumentation overhead is included. A deeper queue
could change the absolute result. The Inspector button offers the same 2s/5s
batch procedure for one current configuration; use the uncapped browser flags.

The matrix isolates ordinary forward, mobile, deferred bypass, materialized
neutral passes, prepass, real GTAO, and TRAA disabled. Subtracting configurations
does not produce isolated GPU stage timings: shaders, CPU scheduling, pipeline
transitions, cache behavior and history copies also change. This establishes a
throughput floor/ceiling for this prototype and supported shading subset only.

`--smoke` runs mode/model switches, compressed/static/skinned/morph/instanced and
transmissive assets, odd resize, camera and object movement. It also reads actual
GPU buffers to verify finite HDR/normal data, nonflat lighting, zero jitter-only
velocity, nonzero object velocity, and alpha-safe prepass off/on/off agreement.

Targeted tests: `node --test test/benchmarks/gbuffer-*.test.mjs` and
`node utils/generate-gbuffer-model-manifest.mjs --check`. The latter verifies all
63 tracked glTF/GLB paths recursively, case-insensitively, including manual assets.
Paths remain visible even if a required codec/browser extension fails. External
textures/buffers resolve relative to the selected model URL. Loaders configure
DRACO, KTX2 and Meshopt; stale selections are discarded and superseded assets
are disposed. Core/addon unit tests and ESLint are the repository checks; there
is no TypeScript typecheck script in this JavaScript repository.

Build001 evidence and measured results are in [RESULTS.md](RESULTS.md).
