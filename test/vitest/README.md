# Neural framework test suite (vitest + playwright)

Isolated, incremental tests for the neural-texture / neural-material /
neural-appearance framework (`examples/jsm/neural*`), separate from the
repo's existing QUnit + puppeteer harness (`test/unit/**`, run via `npm run
test-unit-addons`). That harness is kept as-is; this suite exists specifically
because it couldn't do the one thing this framework most needed: actually
*run* a TSL node graph or a WebGPU compute kernel and check the number that
comes back, instead of mocking the renderer or skipping numerical assertions
(see `test/unit/addons/neural-appearance/NeuralAppearanceTSL.tests.js` for an
example of exactly that gap - it tests packing helpers but never evaluates
`applyChannelActivation`, `evaluateNeuralAppearanceOutputs`'s TSL half, or any
compute kernel).

## Running

```
npm run test-neural          # everything (unit + browser)
npm run test-neural-unit     # pure-JS project only, fast, no browser
npm run test-neural-browser  # real WebGPU project only
npm run test-neural-watch    # interactive watch mode, both projects
```

## Two projects, one config (`vitest.config.js`)

- **`unit`** - plain Node. For pure-JS logic: math, MLP forward pass,
  format/manifest encode-decode, byte-level (de)serialization, anything that
  never builds a TSL node or touches a renderer. Fast, deterministic, no
  browser startup cost.
- **`browser`** - real Chromium via the `@vitest/browser-playwright`
  provider, launched with `--enable-unsafe-webgpu` so `navigator.gpu` is a
  real adapter/device, not a mock. For anything that builds or evaluates a
  TSL node graph, drives a `WebGPURenderer`, or runs a compute kernel.
  `test/vitest/browser/helpers/webgpuEval.js` has the reusable
  storage-buffer-compute-readback harness (`evalScalar`/`evalFloats`) - see
  `test/vitest/browser/neural/NeuralOutputActivations.test.js` for the
  pattern: build the node with the real function under test, evaluate it on
  the GPU, compare against a hand-written plain-JS reference formula.

  One important gotcha this config already routes around: WebGPU is only
  exposed to a *secure context*. `about:blank` and `data:` URLs are not
  secure contexts in Chromium, so `navigator.gpu` silently doesn't exist
  there even with the right flags - vitest's browser mode serves tests over
  `http://localhost`, which is trusted, so this isn't something individual
  test files need to think about. If you ever hand-roll a raw Playwright
  script outside vitest, remember this.

## What's done

`examples/jsm/neural/` (the shared primitives every trainer/model builds on)
is fully covered:

| File | Project | Notes |
|---|---|---|
| `NeuralVectorMath.js` | unit | dot/cross/normalize incl. zero-vector and near-zero edge cases |
| `NeuralMLP.js` | unit | activations, He-init scale, RGB-head gray bias, forward pass, zero-weight/zero-input edge cases |
| `NeuralGridModel.js` | unit | grid level geometry incl. degenerate base/target, latent grid init range/determinism |
| `NeuralTrainingUtils.js` | unit | LR annealing incl. clamping, seeded PRNG determinism, `yieldToBrowser`'s RAF/setTimeout fallback |
| `NeuralOutputActivations.js` | browser | forward + backward TSL nodes evaluated on real WebGPU against reference formulas, incl. softplus overflow region and a forward/backward finite-difference cross-check |

## Roadmap for the rest

Every other file, classified by which project it belongs in and what to
prioritize. "Pure JS" here means: no `from 'three/tsl'` or `from
'three/webgpu'` import *and* no renderer/scene/camera use - grep a file
yourself before trusting this table if it's been edited since.

### neural-texture/

| File | Project | What to test |
|---|---|---|
| `NeuralTextureModel.js` (33 lines) | unit | CPU decoder forward pass - smallest, do this first |
| `NeuralTextureSource.js` (113 lines) | **browser** | Renders via `THREE.Scene`/`RenderTarget` even though it doesn't literally import `three/tsl` - `bakeColorNodeToTexture`'s UV-flip-on-bake is exactly the kind of silent-flip bug worth a real round-trip test: bake a node with a known asymmetric pattern, read the render target back, assert the orientation |
| `NeuralTextureTrainer.js` (153 lines) | unit for the orchestration logic, mock the GPU calls it delegates to | Iteration/LR-schedule wiring, not the compute itself (that's `NeuralTextureGPUComputeTSL.js`'s job below) |
| `NeuralTextureGPUModel.js` (268 lines) | browser | Storage buffer layout/allocation - assert byte offsets and sizes match what `NeuralTextureGPUComputeTSL.js` expects to read/write |
| `NeuralTextureGPUComputeTSL.js` (463 lines) | browser | The actual training-step kernels - this is the highest-value target in the whole framework for the `evalScalar`/`evalFloats` pattern: pick one weight, one gradient, one Adam step, hand-compute the expected update in plain JS, run the kernel, compare |
| `NeuralTextureNodeMaterial.js` (186 lines) | browser | Inference-time evaluation - feed a known latent grid + tiny MLP, sample at a known UV, compare against `NeuralTextureModel.js`'s CPU forward pass (cross-check, same pattern as the appearance manifest's `referenceEvaluations`) |

### neural-material/

| File | Project | What to test |
|---|---|---|
| `NeuralMaterialFormat.js` (161 lines) | unit for `layoutChannels`/`buildChannelActivations`/`getChannel` (pure data-layout logic); **browser** for `previewColor` (builds a TSL node) | Channel offset assignment for arbitrary active-channel subsets (this is the encoder/decoder contract other modules index into by flat offset - a layout bug here silently corrupts every channel after the first mismatch), `MAX_TOTAL_CHANNELS` invariant, unknown-key error path |
| `NeuralMaterialSource.js` (378 lines) | mixed - `classifyMaterialChannels` (which channels are "active") is likely pure JS/unit; anything building/resolving TSL nodes (e.g. `resolveAnisotropyNodes`) is browser | Classification given a material with only some node properties set - the "spatially-varying vs constant" decision this drives is a correctness-critical converter step |
| `NeuralMaterialNodeMaterial.js` (292 lines) | browser | `reconstructFinalNormal`'s `sqrt(1 - dx*dx - dy*dy)` z-reconstruction is a natural edge-case target: dx/dy near the unit circle boundary (sqrt of a near-zero or slightly negative value) |

### neural-appearance/

This directory is the biggest and has the clearest "encoder/decoder/
loader/converter" shape of the whole framework:

| File | Project | Role | What to test |
|---|---|---|---|
| `NeuralAppearanceFormat.js` (40 lines) | unit | format constants | Trivial but cheap - lock down `FORMAT`/`VERSION`/size constants so a future edit that changes one without meaning to fails loudly |
| `NeuralAppearanceModel.js` (329 lines) | unit | CPU model (decoder) | No TSL import - this is a pure-JS forward pass, do it early. Cross-check against the browser-side TSL evaluation of the same math once that exists (see `NeuralAppearanceTSL.js` row) |
| `NeuralAppearanceManifest.js` (200 lines) | unit | **encoder** - model → JSON manifest | `serializeLayers` round-trips (weights/biases survive `.slice()`), `createNeuralAppearanceManifest`'s conditional `emission`/`opacity` block inclusion, `layoutChannels`-style offset consistency |
| `NeuralAppearanceRuntime.js` (281 lines) | unit | **decoder** - JSON manifest → evaluated outputs, pure-JS reference implementation | This is the module the manifest's own `referenceEvaluations` are generated from (see `createReferenceEvaluations` in Manifest.js) - test it directly, then use it as the oracle when writing the `NeuralAppearanceTSL.js` GPU tests below, the same way `NeuralOutputActivations.test.js` uses hand-written reference formulas |
| `NeuralAppearanceSampler.js` (477 lines) | unit | teacher-target sampling/normalization | `normalizeDirectLightingTargets`, `assignTeacherTargets` given a fake teacher |
| `NeuralAppearanceTeacherReadback.js` (93 lines) | unit | readback plumbing | Whatever doesn't require a real render target |
| `NeuralAppearanceTrainer.js` (450 lines) | unit for orchestration/schedule, browser for anything that actually dispatches | |
| `NeuralAppearanceValidator.js` (362 lines) | unit | **loader-adjacent** - validates a manifest/model before use | This is exactly where malformed-input edge cases belong: missing fields, wrong array lengths, out-of-range channel counts - the kind of thing that currently fails deep inside a render call with a confusing GPU error instead of a clear validation message |
| `NeuralAppearanceTSL.js` (855 lines) | browser | GPU-side inference node graph | Largest file, highest payoff: cross-check against `NeuralAppearanceRuntime.js`'s CPU reference the same way `createReferenceEvaluations` already does at export time - now made into an actual assertion instead of just JSON that a human has to eyeball |
| `NeuralAppearanceTeacherAtlas.js` (195 lines) | browser | renders a real atlas texture | |
| `NeuralAppearanceTeacherEvaluator.js` (747 lines) | browser | teacher BRDF evaluation kernels | |
| `NeuralAppearanceGPUModel.js` (693 lines) | browser | storage buffer layout (same pattern as `NeuralTextureGPUModel.js`) | |
| `NeuralAppearanceGPUComputeTSL.js` (1211 lines) | browser | training kernels (same pattern as `NeuralTextureGPUComputeTSL.js`) | Biggest single file in the framework - tackle after the smaller GPU kernels above have proven out the pattern |
| `NeuralAppearanceNodeMaterial.js` (308 lines) | browser | inference-time material wiring | |

### Suggested order

1. `NeuralTextureModel.js`, `NeuralAppearanceModel.js`, `NeuralAppearanceFormat.js` -
   quick pure-JS wins, round out `unit`'s coverage of every CPU-side decoder.
2. `NeuralAppearanceRuntime.js` + `NeuralAppearanceManifest.js` - the
   encoder/decoder pair at the JSON-manifest layer; establishes the CPU
   reference oracle everything in step 4 will check GPU output against.
3. `NeuralMaterialFormat.js`'s `layoutChannels`/`buildChannelActivations` -
   the channel-offset contract several other modules depend on.
4. `NeuralAppearanceValidator.js` - malformed-input edge cases, cheapest
   place in the whole framework to catch a bad manifest before it reaches
   the GPU.
5. `NeuralOutputActivations.js`-style GPU node tests, extended to
   `NeuralAppearanceTSL.js`'s per-output evaluation functions, checked
   against the `NeuralAppearanceRuntime.js` oracle from step 2.
6. The two `*GPUModel.js` / `*GPUComputeTSL.js` pairs (texture, then
   appearance - texture's kernel file is a third the size, do it first to
   shake out the storage-buffer-diffing pattern before tackling appearance's).
