# WebGPU G-buffer prototype

Open `webgpu_gbuffer.html` through a local server. It imports source modules, so
no build is needed. The scene selector loads Cornell, the two-room light-probe
scene, and the GTAO gallery. Scene changes reload the page to release all baked
and temporal resources. Camera orbit and viewport resizing remain interactive.

## Frame graph

The high preset renders an opaque depth pre-pass and an equal-depth forward MRT
into the same depth attachment, with the same TRAA-jittered camera. This source
feeds Hi-Z, GTAO, SSR, and temporal/spatial reflection denoising. A separate forward
resolve draw evaluates direct light and diffuse GI, replaces indirect specular
with the SSR/probe result, and applies AO only to indirect lighting. TRAA runs on
beauty after the lighting resolve. Diagnostic views bypass output color transforms.

| Attachment | Format | Meaning |
| --- | --- | --- |
| Depth | depth32float | Shared by the pre-pass and source MRT |
| Source color | rgba16float | Lit source for reflection hits |
| Normal | rg16float | Signed octahedral view normal; decode before filtering |
| Albedo | rgba8unorm | Linear base color RGB, metalness A |
| Roughness | r8unorm | Perceptual roughness |
| Velocity | rg16float | Motion used by denoising and TRAA |

Hi-Z uses the prototype nearest/farthest depth atlas. Disable it to compare the
linear SSR path. The debug view shows the selected hierarchy level as view-space
distance divided by 20, rather than almost-white nonlinear device depth.

The low preset omits normal/albedo/roughness MRT attachments and the GTAO/SSR/denoise
graph. Velocity is omitted when neither TRAA nor its diagnostic view needs it.
It retains captured specular probes and diffuse probe GI; it is not the deck's
finished lightmap-based low-end preset.

## Lighting and reflection contract

Physical atmosphere sky and direct sun share a fixed scene-referred scale of
0.0001. The IBL excludes the solar disc, avoiding duplication of direct sun.
Existing scene point/spot lights remain live. Visibility-aware diffuse probes
bake once at startup. One grid spans both colored rooms (8 × 4 × 4 probes),
keeping the combined material under WebGPU's portable 16-sampler stage limit.
Single-room scenes use 4 × 4 × 4 probes. Local specular probes are captured after
diffuse GI, filtered, and packed with sky fallback into a cube array.

Probe radiance is weighted with the environment DFG BRDF exactly once. SSR already
returns BRDF-weighted outgoing light. Finite SSR hits replace the probe result;
the SSR environment-distance sentinel selects local probe/sky fallback. This
replaces indirect specular instead of adding it over existing environment light.
The SSR shader's implicit edge fade is disabled because its internal environment
is empty; the material resolve supplies the fallback.

The stock temporal/recurrent denoiser is optional. **SSR demodulation is excluded**
at Ben's request. The alternative reconstruction pipeline from PR #53 is also
left separate; this experiment uses the existing denoiser contract.

## Not ready for general integration

- This is example plumbing, not a reusable `GBufferNode` or a full deferred
  renderer. The extra forward resolve draw is intentional to avoid a dependency
  cycle through SSR; no bandwidth or frame-rate improvement is claimed.
- SSR/probe transitions still use a binary hit/miss decision. Filtering black SSR
  misses can darken boundaries; stochastic coverage, glossy reconstruction, and
  seamless confidence blending remain unfinished. Probe DFG weighting omits the
  full material multiscattering treatment.
- The simple pre-pass supports these opaque static scenes. It does not reproduce
  arbitrary displacement/custom vertex nodes, material arrays, or transparent
  glass/refraction. No claim of general material-aware depth support is made.
- Probe bakes/captures are synchronous startup work, not budgeted background
  configuration updates. Probe/sky settings are fixed for each loaded scene.
- The high graph requests extended sampled-texture limits when supported. Devices
  with fewer than 24 sampled textures per stage use the low graph. Mobile, Safari,
  and other GPUs have not been validated.
- No fidelity-suite score, performance benchmark, or calibrated exposure match to
  a path-traced reference is claimed.

## Validation

From the repository root:

```sh
npm ci
npm test
npm run lint-addons
npx eslint examples/jsm/gbuffer examples/webgpu_gbuffer.html test/webgpu/gbuffer-prototype.js
node test/webgpu/gbuffer-prototype.js
```

The integration test serves the repository on an ephemeral localhost port,
launches Puppeteer, requires an identified hardware adapter, exercises all scenes,
both presets, buffer views, toggles, and odd viewport resizing, and runs the Hi-Z
and SSR/Hi-Z GPU readback suites. It closes the browser and server on completion.
Set `PUPPETEER_EXECUTABLE_PATH` to an installed Chrome if Puppeteer's bundled
browser is unavailable. Set `SWARM_ARTIFACTS` to save screenshots and JSON results.

On Windows001 / RTX 3060 Ti (WebGPU reports `nvidia ampere`):

- Core lint, addon lint, focused example lint, and build passed.
- Unit tests: 1966 core passed, one todo; 111 addon passed.
- Hi-Z: 200 GPU cases, 1,936,440 atlas texels, 2400 ray checks passed.
- SSR/Hi-Z: 48 shader comparisons, zero changed hit masks and zero channel error.
- All three scenes and both presets rendered without WebGPU validation errors.
- Full `lint-examples` has 11 errors and two warnings in unchanged Ammo,
  multiple-elements, HTML-texture, and shadertoy examples. The unit runner also
  emits existing Windows process-cleanup messages and a resource 404. The GTAO
  bust emits a missing-UV warning; rendering completes.

See PR #69 for screenshots and branch-based live demos.
