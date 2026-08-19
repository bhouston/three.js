# Plan: Eliminate per-iteration teacher-readback stalls in NeuralAppearanceTrainer

## Context for the executing agent

This is a follow-on to a separate, already-in-flight refactor that replaces
neural-appearance's mip-pyramid latent grid with the multiresolution grid
encoding used by `neural-texture`/`neural-material` (see
`.cursor/plans/` for related history if present, and
`examples/jsm/neural/NeuralGridModel.js` for the shared grid code). That
refactor does **not** touch the issue this plan addresses. Do not assume it
has landed or re-do any of its work; if `NeuralGridModel.js` doesn't exist
yet when you start, the grid-encoding refactor may still be in flight —
check current file state before editing anything grid-related, and treat
this plan's file line-number references as approximate (re-locate by
function/symbol name, not by line number, since files may have shifted).

Repo: three.js fork (`three.js-v2.js`), on branch `neural-appearance-ibl`.
Relevant addons live under `examples/jsm/neural-appearance/`,
`examples/jsm/neural-texture/`, `examples/jsm/neural-material/`,
`examples/jsm/neural/` (shared code), and the example page
`examples/webgpu_materials_neural_appearance.html`.

## Problem statement

`NeuralAppearanceTrainer` trains ~10x slower (wall-clock, per iteration)
than `NeuralTextureTrainer`/`NeuralMaterialNodeMaterial`'s shared trainer,
for reasons that are architectural, not just about batch size or network
size. This plan is scoped specifically to closing that gap by removing
CPU↔GPU synchronization stalls from the training loop's hot path.

### Root cause: `neural-texture` never leaves the GPU; `neural-appearance` does, repeatedly, every iteration

**`neural-texture`/`neural-material`** (`examples/jsm/neural-texture/NeuralTextureGPUComputeTSL.js`,
`createTextureTrainBatchComputeNode`): the "teacher" is a static texture
already resident on the GPU. The training compute shader samples it
directly inside the same compute dispatch that does the grid-encode → MLP
forward → loss → backward → gradient-scatter pipeline
(`textureLevel(sourceTexture, uv, 0)`, see line ~98 of that file). One GPU
dispatch chain per iteration, zero CPU readback, zero pipeline stalls.

**`neural-appearance`** (`examples/jsm/neural-appearance/NeuralAppearanceTrainer.js`,
`train()` method, main iteration loop): every iteration calls
`generateTrainingSamples(settings, teacher, this.random, iteration)`
(`examples/jsm/neural-appearance/NeuralAppearanceSampler.js`) **before** the
GPU compute step can even run. That function is a sequence of `await`ed
calls into `NeuralAppearanceTeacherEvaluator.evaluateBatch(...)`
(`examples/jsm/neural-appearance/NeuralAppearanceTeacherEvaluator.js`), and
each `evaluateBatch` call does:

```
this._uploadSamples(batch)              // CPU→GPU: write sample atlas textures
await this._renderAndRead()             // renders the teacher, then:
  → renderer.readRenderTargetPixelsAsync(...)   // GPU→CPU stall, awaited
```

(`renderAndReadTeacher` in `examples/jsm/neural-appearance/NeuralAppearanceTeacherReadback.js`).

Per iteration, `generateTrainingSamples` makes **up to 8 separate,
sequentially-awaited `evaluateBatch` calls**, each with its own render +
readback:

1. `assignTeacherTargets` → 1 call, `targetMode` default `'brdf'`
2. `assignAuxiliaryTeacherTargets` → up to 2 calls: `'emission'`, `'opacity'`
   (only if the material has those inputs — `teacher.supportsEmission`/`supportsOpacity`)
3. `assignIBLTeacherTargets` → up to 5 calls: `'iblQuery'`, `'iblIncoming'`,
   `'iblIrradiance'`, `'iblIndirectRadiance'`, `'iblIndirectIrradiance'`
   (only if `teacher.supportsIBL`, i.e. an environment map is present —
   note this runs **every iteration of the "direct" phase too**, not just
   the dedicated IBL phase at the end of training; see
   `NeuralAppearanceTrainer.js`'s two loops, both of which call code paths
   that end up invoking `assignIBLTeacherTargets`)

**Worse than the readback stall itself**: `NeuralAppearanceTeacherEvaluator`
keeps only *one* scene/material/render-target alive at a time
(`this._targetMode`). Every time `evaluateBatch` is called with a
`targetMode` different from the currently-built one, it does:

```js
} else if ( targetMode !== this._targetMode ) {
    this.dispose();            // tears down scene, material, render target, atlas textures
    this._targetMode = targetMode;
    await this.init();         // → this._createResources(): new Scene, new cloned
                                //   NodeMaterial with a freshly-built TSL shader
                                //   graph, new RenderTarget — implies shader
                                //   (re)compilation
}
```

(`examples/jsm/neural-appearance/NeuralAppearanceTeacherEvaluator.js`,
`evaluateBatch`, `_createResources`, `dispose`). Since the 8 calls above
cycle through up to 8 distinct `targetMode` values in sequence, **every
single training iteration can trigger up to 8 full material-clone +
scene-rebuild + shader-recompile cycles**, each followed by its own
upload+render+readback stall. This is very likely the single largest cost
in the whole training loop — shader/pipeline compilation is typically far
more expensive than the readback itself.

### Secondary contributor: validation runs every iteration when `onProgress` is set

`NeuralAppearanceTrainer.js`, both iteration loops:

```js
const shouldSync = onProgress !== null || ( iteration === settings.iterations - 1 );
```

Whenever the caller passes an `onProgress` callback (which the example page
does, for live UI updates), this triggers a full `gpuModel.syncToCPU(...)`
+ `gpuModel.readLosses(...)` + `evaluateRuntimeValidation(...)` (CPU-side,
over `validationSamples` and `directionalValidationSamples`) **every single
iteration**. Compare to `NeuralTextureTrainer.js`:

```js
const shouldSync = onProgress !== null && ( iteration % 4 === 0 || iteration === iterations - 1 );
```

— which only syncs every 4th iteration. This is a smaller effect than the
teacher-readback problem but compounds it, and is a cheap, low-risk fix.

## Non-goals / do not do

- Do **not** change the actual BRDF/IBL math, sample distributions, or
  training targets/losses — this plan is about removing synchronization
  overhead and redundant resource churn, not changing what gets learned.
  Any change that alters convergence behavior (e.g. skipping IBL target
  computation during the direct-only phase) must be validated empirically
  (loss curves comparable to before) before being treated as safe, and
  should be flagged as an experiment, not shipped silently.
- Do **not** block on or duplicate the concurrent mip-pyramid→multi-res-grid
  refactor's file changes. If both land around the same time, expect merge
  conflicts in `NeuralAppearanceTrainer.js`/`NeuralAppearanceGPUModel.js`;
  resolve by re-reading current file state, don't assume this plan's quoted
  snippets are still verbatim-accurate.
- This plan does not require moving teacher evaluation onto the GPU
  entirely (matching neural-texture's zero-readback architecture) as a
  *requirement* — that is called out below as a stretch goal (Phase 4)
  because it's a much larger redesign. Phases 1–3 should already close most
  of the gap.

## Proposed phases (do in order; each is independently valuable and testable)

### Phase 1 — Persistent per-target-mode teacher resources (highest priority, moderate effort)

Eliminate the dispose()/recreate cycle on every `targetMode` switch.
Restructure `NeuralAppearanceTeacherEvaluator` so that **all** target-mode
scene/material/render-target bundles it will ever need are built **once**,
up front (e.g. during `init()`, or lazily-but-cached on first use per mode
— cached forever after, never disposed until the evaluator itself is
disposed), instead of being torn down and rebuilt every time the active
mode changes.

Concretely:
- Change `this._targetMode`/`this._scene`/`this._material`/etc. (singular)
  into a `Map<targetMode, { scene, camera, geometry, material, mesh, light,
  target }>` cache, keyed by target mode string.
- `_createResources(targetMode)` becomes "build-and-cache-if-absent",
  returning the cached bundle if it already exists for that mode.
- `evaluateBatch(samples, targetMode)` looks up (or lazily builds once) the
  bundle for `targetMode`, uploads atlas samples into *that bundle's*
  atlas textures, renders *that bundle's* scene into *that bundle's* render
  target, and reads back — no dispose, no rebuild.
- `dispose()` (called once when the evaluator itself is torn down, e.g.
  end of training) now needs to dispose every cached bundle, not just the
  single active one.
- Watch for correctness traps: the atlas textures (`_sampleTextures`) are
  currently keyed to `this.batchSize`/`this.tileSize`/atlas dimensions,
  which don't vary by mode, so it's safe/preferable to keep one shared set
  of atlas *input* sample textures (`uv`/`normal`/`tangent`/`bitangent`/
  `wi`/`wo`) reused across all mode bundles, and only per-mode-specific
  state (scene, material, output render target) needs its own cached copy.
  Re-check `_uploadSamples`/`uploadAtlasSamples` to confirm the atlas
  input textures are mode-agnostic before sharing them (they should be,
  since they just encode uv/frame/direction — the *material* built per
  mode is what changes what gets rendered from them).

Expected effect: removes up to 7 of the 8 shader-recompile+resource-churn
cycles per iteration (all but the first time each mode is ever used across
the whole training run). This is likely the single biggest win available.

### Phase 2 — Reduce the number of distinct render+readback round trips via MRT (high priority, higher effort)

Investigate whether three.js's WebGPU renderer supports multiple render
targets (MRT) for a single draw call (check for `RenderTarget` with a
`count`/multiple-color-attachment option, or `MRTNode`/`mrt()` in
`three/tsl` — search the three.js core source under `src/renderers/` and
`src/nodes/` for MRT support before assuming it exists or doesn't).

If MRT is available:
- The `'brdf'`, `'emission'`, `'opacity'` modes render the *same* geometry/
  camera/atlas-sample setup with different `outputNode`s — these are strong
  candidates to merge into a single MRT pass (up to 3 outputs in one draw),
  since they share everything except what gets written out.
- The 5 IBL-related modes (`'iblQuery'`, `'iblIncoming'`, `'iblIrradiance'`,
  `'iblIndirectRadiance'`, `'iblIndirectIrradiance'`) likewise share scene
  setup (same environment, same query directions) and mostly differ only in
  which piece of the lighting-model output gets written — investigate
  merging these into 1–2 MRT passes instead of 5 (may need up to two passes
  if the total attachment count exceeds the GPU's MRT limit, typically 4-8
  color attachments).

This could reduce 8 render+readback round trips down to as few as 2–3,
directly cutting readback-stall count (not just the shader-recompile cost
Phase 1 already removed) by a similar factor. Read back all MRT attachments
in one `readRenderTargetPixelsAsync`-equivalent call per pass if the API
supports multi-attachment readback; otherwise this phase still helps by
cutting render-pass count even if readback calls stay 1:1 with attachments.

If MRT support turns out to be impractical (e.g. not exposed for the
render-target-readback path this evaluator uses), document why and skip to
Phase 3 — don't force it.

### Phase 3 — Pipeline/overlap CPU readback with GPU work (medium priority, higher effort, do after 1–2)

Currently the loop is fully serialized: generate samples (render+await
readback ×N) → upload to GPU training buffers → GPU compute steps → (maybe)
sync back. Restructure to double-buffer so that while the CPU awaits
iteration N's readback, the GPU isn't idle: e.g. kick off iteration N's
teacher render(s) immediately after iteration N-1's samples are uploaded to
the training compute buffers (so the training compute dispatch for N-1 and
the teacher render for N can be in-flight on the GPU at the same time),
only awaiting the readback right before it's needed to build N's training
batch. This is more invasive (changes the trainer's loop structure and
likely needs a small pipeline/prefetch abstraction) — attempt only after
Phases 1–2 land and are measured, since they may already close most of the
gap and reduce the marginal value of this phase. Benchmark before deciding
whether to invest here.

### Phase 4 — (Stretch goal, not required) Move teacher evaluation fully onto the GPU compute pipeline

The architectural end-state that would make neural-appearance match
neural-texture's zero-CPU-involvement training loop: port the relevant
BRDF/IBL shading math (currently expressed as a `NodeMaterial` +
`PhysicalLightingModel` subclass rendered as a full screen-space pass, see
`NeuralTeacherLightingModel`/`NeuralTeacherIBLLightingModel` in
`NeuralAppearanceTeacherEvaluator.js`) into a form callable directly inside
the training compute shader (`NeuralAppearanceGPUComputeTSL.js`), the way
`textureLevel(sourceTexture, uv, 0)` is called inline in
`NeuralTextureGPUComputeTSL.js`. This eliminates the render+readback+upload
cycle entirely, at the cost of re-deriving the shading math as inline
TSL/WGSL compute code rather than reusing three.js's existing
`NodeMaterial`/`LightingModel` machinery. This is a substantial redesign —
scope and estimate it as a separate follow-up plan if Phases 1–3 don't
close the gap sufficiently; do not attempt it as part of this pass unless
explicitly asked.

### Phase 5 — Match neural-texture's validation-sync cadence (low priority, trivial effort, safe to do anytime)

In `NeuralAppearanceTrainer.js`, change both iteration loops' `shouldSync`
condition from:

```js
const shouldSync = onProgress !== null || ( iteration === settings.iterations - 1 );
```

to something like:

```js
const shouldSync = onProgress !== null && ( iteration % 4 === 0 || iteration === settings.iterations - 1 );
```

matching `NeuralTextureTrainer.js`'s cadence (adjust the modulus if a
different interval is more appropriate given neural-appearance's smaller
default batch size/iteration count — use judgment, but default to matching
neural-texture's `% 4` unless there's a clear reason not to). This cuts the
per-iteration CPU-side `syncToCPU`/`readLosses`/`evaluateRuntimeValidation`
cost by ~4x whenever `onProgress` is set (i.e. whenever the example page is
actively driving training with live UI updates, which is the common case
end users will actually experience).

### Phase 6 — Re-tune default batch size (low priority, do last, after measuring)

`NeuralAppearanceTrainer.js`'s `DEFAULT_OPTIONS.batchSize` is `1024` vs.
`NeuralTextureTrainer.js`'s `4096`. Once the fixed per-iteration overhead
from Phases 1–3/5 is reduced, re-benchmark whether a larger default batch
size (matching neural-texture, or some appropriate value given
neural-appearance's larger per-sample compute cost — BRDF+IBL heads vs. a
single small decoder) gives better total-training-time-to-quality. Don't
guess; measure before changing the default, since a larger batch also means
larger teacher atlas textures/render targets and more compute per
iteration, which could shift the bottleneck back.

## Investigate-before-assuming: IBL target computation during the "direct" training phase

`NeuralAppearanceTrainer.js` has two loops: a "direct" phase (calls
`generateTrainingSamples`, which — per above — computes IBL targets too,
every iteration, whenever an environment is present) and a dedicated "IBL"
phase at the end (calls `generateIBLTrainingSamples`, IBL-only).

In `NeuralAppearanceGPUComputeTSL.js`, the training compute kernel
(`createTrainBatchComputeNode`) gates its entire IBL forward/backward
sub-pass on `If (iblWeight.greaterThan(0.0))`, and that IBL sub-pass's
backward pass appears to write gradient contributions back into the shared
latent grid (not just the IBL-specific decoder weights) — meaning IBL
targets computed during the "direct" phase may be intentionally shaping the
shared grid representation throughout training, not merely wasted work.
The direct-phase loop does *not* call the IBL Adam-weights update
(`adamIBLWeightsNode`) — only the IBL-phase loop does — so IBL *decoder*
weights aren't updated until the dedicated phase, but IBL-driven *latent*
gradients may already be flowing in during the direct phase.

Before touching this: read `NeuralAppearanceGPUComputeTSL.js`'s IBL
backward-pass section carefully and confirm whether it does or doesn't
backprop into the shared latent gradient buffer (`gradA0`/equivalent) the
same way the direct-BRDF backward pass does. If it does — skipping IBL
target computation during the direct phase would change what the shared
grid learns, and should be treated as a training-quality experiment (run
both ways, compare loss curves and final validation quality), not a free
performance win to ship without evaluation. If it turns out the IBL
gradient path is inert unless `adamIBLWeightsNode` also runs (i.e. the
latent gradient contribution is negligible/zero during the direct phase),
then skipping `assignIBLTeacherTargets` during the direct-phase loop
becomes a legitimate, larger win on top of Phases 1–3 (removes up to 5 of
the 8 per-iteration evaluateBatch calls for the majority of training
iterations) — but verify first, don't assume.

## Verification / how to know it worked

- Add or reuse simple wall-clock timing around `NeuralAppearanceTrainer.train()`'s
  iteration loop (e.g. `console.time`/`performance.now()` deltas per
  iteration, or reuse whatever timing the example page
  (`examples/webgpu_materials_neural_appearance.html`) already surfaces in
  its training-progress UI) to get a concrete before/after
  milliseconds-per-iteration number. Capture a baseline on the current code
  before making changes.
- After each phase, re-measure milliseconds-per-iteration and record it in
  your final report (per-phase, not just final total), so the user can see
  which phase contributed how much.
- Confirm training still converges to comparable loss/validation quality
  after each phase (compare `validation.loss`/`validation.directional`
  reported by `onProgress` against a pre-change baseline run with the same
  seed/settings) — this work must not silently degrade model quality while
  chasing speed.
- Run the existing unit tests under `test/unit/addons/neural-appearance/`
  (check `package.json` for the test-runner command) and confirm nothing
  regresses; add test coverage for the new persistent-resource-cache
  behavior in `NeuralAppearanceTeacherEvaluator` if none exists (e.g. a
  test asserting that calling `evaluateBatch` with alternating target modes
  does not dispose/recreate resources on the second-and-later use of a
  previously-seen mode).

## Report back

Summarize, per phase attempted: what changed, measured ms/iteration before
vs. after, any convergence/quality comparison performed, what (if anything)
you deferred or couldn't verify (e.g. Phase 2's MRT feasibility, Phase 4
being explicitly out of scope), and the outcome of the "investigate before
assuming" IBL-gradient question above.
