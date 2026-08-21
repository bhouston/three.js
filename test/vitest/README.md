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

Also covered, beyond the `neural/` primitives table above:

| File | Project | Notes |
|---|---|---|
| `neural-texture/NeuralTextureModel.js` | unit | CPU decoder construction, decoder input-size arithmetic, deterministic grid/decoder content for a seeded `random` |
| `neural-texture/NeuralTextureSource.js` | browser | `bakeColorNodeToTexture` sign preservation and raw-UV round-trip - this is where the real "brick preset missing normal-map slopes" bug lived, see the file's own comment |
| `neural-texture/NeuralTextureTrainer.js` | browser | end-to-end sign-convergence (a tanh channel alone in its own packed render target, matching the real neural-material `normal` channel's layout) + convergence-rate regression (capacity/learning-rate/divergence ordering checks) |
| `neural-appearance/NeuralAppearanceFormat.js` | unit | format constants, incl. every `*_SIZE` constant's exact arithmetic relationship to `LEVELS`/`CHANNELS_PER_LEVEL` |
| `neural-appearance/NeuralAppearanceModel.js` | unit | `sampleLatents` bilinear/wrap addressing, `forwardDecoderInput`/`buildIBLInput`/`buildIndirectProbeInput`/`unpackIBLOutput` shape and math, `createModel` construction - **plus a documented, currently-`it.fails` known bug** (`NeuralAppearanceModel.levels-bug.test.js`): `LATENT_CHANNELS`/`DECODER_INPUT_SIZE`/`IBL_INPUT_SIZE`/`INDIRECT_INPUT_SIZE` are fixed constants baked from the hardcoded default `LEVELS = 4`, not from a model's actual configured `levels` - so the "grid levels" GUI control in `webgpu_materials_neural_appearance.html` silently produces all-NaN predictions for any non-default value. Not yet fixed (touches every module that imports these as constants); see that test file for the full writeup |
| `neural-appearance/NeuralAppearanceValidator.js` | unit | difference/angular-bin metric accumulation, `evaluateRuntimeValidation` integration incl. the non-finite-value throw path and null-when-absent guards for emission/opacity/ibl losses |
| `neural-appearance/NeuralAppearanceTrainer.js` | browser (perf) | stage-timing breakdown (`NeuralAppearanceTrainer.perf.test.js`, not a pass/fail gate): confirms teacher render+readback (not GPU training compute) dominates wall-clock (~60-80% in a small run), and quantifies IBL sampling's per-iteration evaluateBatch-call multiplier (5 calls/iteration vs. 1 for direct-only) - useful data for anyone chasing the "neural-appearance training is slow" complaint next, see that file's comments |
| `neural-material/NeuralMaterialFormat.js` | unit for `layoutChannels`/`buildChannelActivations`/`getChannel`; browser for `previewColor` | offset-assignment contract, `MAX_TOTAL_CHANNELS` invariant, unknown-key error path, and (browser) the exact display bug fixed in this session - `previewColor`'s `size === 2` branch hardcoding blue to 0 when handed an already-reconstructed 3-component normal |
| `neural-material/NeuralMaterialSource.js` | unit for `classifyMaterialChannels`; browser for `buildPackedColorNodes` | active/constant channel classification and default-value fallbacks (unit); the `flattenChannelComponents` V-flip sign convention - x passthrough, y negated - incl. the exact "channel alone in its own packed render target" layout the real `brick` preset hits (browser) |
| `neural-material/NeuralMaterialNodeMaterial.js` | browser | `reconstructFinalNormal`'s sign fidelity for all four `(dx, dy)` sign combinations and its `sqrt(max(1 - dx² - dy², 0))` near/over-unit-circle clamp |

## Roadmap for the rest

Every other file, classified by which project it belongs in and what to
prioritize. "Pure JS" here means: no `from 'three/tsl'` or `from
'three/webgpu'` import *and* no renderer/scene/camera use - grep a file
yourself before trusting this table if it's been edited since.

### neural-texture/

| File | Project | What to test |
|---|---|---|
| `NeuralTextureTrainer.js` (153 lines) | unit for the orchestration logic, mock the GPU calls it delegates to | Iteration/LR-schedule wiring is now covered end-to-end via the browser convergence/sign-convergence tests above, but a dedicated unit test mocking the GPU calls (to check e.g. abort()/onProgress wiring in isolation) is still open |
| `NeuralTextureGPUModel.js` (268 lines) | browser | Storage buffer layout/allocation - assert byte offsets and sizes match what `NeuralTextureGPUComputeTSL.js` expects to read/write |
| `NeuralTextureGPUComputeTSL.js` (463 lines) | browser | The actual training-step kernels - this is the highest-value target in the whole framework for the `evalScalar`/`evalFloats` pattern: pick one weight, one gradient, one Adam step, hand-compute the expected update in plain JS, run the kernel, compare |
| `NeuralTextureNodeMaterial.js` (186 lines) | browser | Inference-time evaluation - feed a known latent grid + tiny MLP, sample at a known UV, compare against `NeuralTextureModel.js`'s CPU forward pass (cross-check, same pattern as the appearance manifest's `referenceEvaluations`) |

### neural-material/

Fully covered (see the "Also covered" table above) - `NeuralMaterialFormat.js`, `NeuralMaterialSource.js`, `NeuralMaterialNodeMaterial.js`.

### neural-appearance/

This directory is the biggest and has the clearest "encoder/decoder/
loader/converter" shape of the whole framework. `NeuralAppearanceFormat.js`,
`NeuralAppearanceModel.js`, and `NeuralAppearanceValidator.js` are now
covered (see the "Also covered" table above) - remaining:

| File | Project | Role | What to test |
|---|---|---|---|
| `NeuralAppearanceManifest.js` (200 lines) | unit | **encoder** - model → JSON manifest | `serializeLayers` round-trips (weights/biases survive `.slice()`), `createNeuralAppearanceManifest`'s conditional `emission`/`opacity` block inclusion, `layoutChannels`-style offset consistency |
| `NeuralAppearanceRuntime.js` (281 lines) | unit | **decoder** - JSON manifest → evaluated outputs, pure-JS reference implementation | This is the module the manifest's own `referenceEvaluations` are generated from (see `createReferenceEvaluations` in Manifest.js) - test it directly, then use it as the oracle when writing the `NeuralAppearanceTSL.js` GPU tests below, the same way `NeuralOutputActivations.test.js` uses hand-written reference formulas |
| `NeuralAppearanceSampler.js` (477 lines) | unit | teacher-target sampling/normalization | `normalizeDirectLightingTargets`, `assignTeacherTargets` given a fake teacher |
| `NeuralAppearanceTeacherReadback.js` (93 lines) | unit | readback plumbing | Whatever doesn't require a real render target |
| `NeuralAppearanceTrainer.js` (450 lines) | unit for orchestration/schedule | Perf/stage-timing is now covered (see above); still open: `abort()`, `outputFeatures`/`opacityMode` derivation, `validateTrainingSettings`'s error paths |
| `NeuralAppearanceTSL.js` (855 lines) | browser | GPU-side inference node graph | Largest file, highest payoff: cross-check against `NeuralAppearanceRuntime.js`'s CPU reference the same way `createReferenceEvaluations` already does at export time - now made into an actual assertion instead of just JSON that a human has to eyeball |
| `NeuralAppearanceTeacherAtlas.js` (195 lines) | browser | renders a real atlas texture | |
| `NeuralAppearanceTeacherEvaluator.js` (747 lines) | browser | teacher BRDF evaluation kernels | Its `evaluateBatch` MRT grouping/caching behavior is exercised indirectly by the perf test above (call-count assertions would make this explicit) |
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
