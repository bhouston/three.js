# Plan: Move the neural-appearance teacher fully onto the GPU compute pipeline

## Context for the executing agent

This is Phase 4 of `.cursor/plans/neural_appearance_teacher_readback_speedup.md`
("stretch goal, not required"), split out into its own plan because Phases
1-3/5 of that plan (persistent per-target-mode resource caching, MRT batching
of related render targets, reduced validation-sync cadence) are now largely
landed and measured to *not* close the gap: `NeuralAppearanceTrainer` is
still roughly 5x slower per iteration than `NeuralTextureTrainer` /
`NeuralMaterialNodeMaterial`'s shared trainer, and the example page is still
non-responsive while training. Profiling (Chrome DevTools Performance panel,
plus `performance.mark`/`measure` around each stage of the training loop)
confirms the remaining cost is dominated by the teacher-evaluation
architecture itself, not by fixed per-iteration overhead Phases 1-3
targeted.

Repo: three.js fork (`three.js-v2.js`), on branch `neural-appearance-ibl`.
Relevant addons live under `examples/jsm/neural-appearance/`,
`examples/jsm/neural-texture/`, `examples/jsm/neural-material/`,
`examples/jsm/neural/` (shared code), and the example page
`examples/webgpu_materials_neural_appearance.html`. Re-locate referenced
code by function/symbol name, not by line number, since files may have
shifted since this plan was written.

## Problem statement: the architectural gap Phases 1-3 didn't touch

**`neural-texture`/`neural-material`**'s teacher is trivial by construction:
a static, already-GPU-resident texture (an image loaded once at the start of
training). The training compute shader samples it directly inline --
`textureLevel( sourceTexture, uv, 0 )` in
`examples/jsm/neural-texture/NeuralTextureGPUComputeTSL.js`
(`createTextureTrainBatchComputeNode`) -- inside the *same* compute dispatch
that does grid-encode -> MLP forward -> loss -> backward ->
gradient-scatter. One dispatch chain per iteration. Zero CPU involvement,
zero GPU pipeline stalls, zero JS main-thread work proportional to batch
size.

**`neural-appearance`**'s teacher cannot be a static texture: it's a full
BRDF + lighting evaluation (an arbitrary MaterialX-authored
`MeshPhysicalNodeMaterial` shaded through `PhysicalLightingModel`, plus IBL
via a PMREM-filtered environment map) evaluated at per-sample random
`uv`/`wi`/`wo` directions that change every training iteration -- there is
no fixed image to sample. So even after Phases 1-3/5
(`examples/jsm/neural-appearance/NeuralAppearanceTeacherEvaluator.js`), every
iteration still does, on the CPU/main thread:

1. Render the teacher via the normal rasterization pipeline (a real `Scene`
   + `Camera` + cloned `NodeMaterial`), producing up to 3 merged MRT render
   passes (`direct`, `opacity`, `iblProbe` -- see `GROUP_BY_MODE` and
   `_createResources()` in `NeuralAppearanceTeacherEvaluator.js`).
2. `await renderer.readRenderTargetPixelsAsync(...)` per pass -- a hard
   CPU<->GPU synchronization point
   (`NeuralAppearanceTeacherReadback.js`'s `renderAndReadTeacherAttachments`).
3. Unpack each readback pixel-by-pixel into JS arrays (`readSamplePixel`).
4. Re-serialize all of that into the training compute buffer with a
   per-sample JS loop (`NeuralAppearanceGPUModel.js`'s `uploadSamples`)
   *before* the actual training compute dispatch (`renderer.compute(...)`)
   can even start.

This is a hard architectural floor: no amount of caching, batching, or
sync-cadence tuning removes the GPU pipeline stalls or the CPU-side
pixel-unpack/re-upload work, because the fundamental shape of the loop is
render -> stall -> unpack -> reupload -> compute, once per iteration,
instead of neural-texture's single compute dispatch chain. This also
directly explains the UI non-responsiveness: several sequential awaited
GPU stalls plus synchronous CPU unpack/repack loops happen inside one
iteration, and `yieldEvery` only yields control back to the browser every N
iterations, so the main thread is pinned through several of these
stall+unpack+reupload cycles at a stretch.

## Goal / end state

Match neural-texture's shape: the teacher's BRDF + lighting + IBL math
becomes a set of TSL functions callable *inline*, inside the training
compute shader itself (`examples/jsm/neural-appearance/NeuralAppearanceGPUComputeTSL.js`,
`createTrainBatchComputeNode`), operating directly on the per-sample
`uv`/`wi`/`wo` values already resident in the compute buffer. No render
pass, no `readRenderTargetPixelsAsync`, no pixel unpack, no CPU
re-upload. One dispatch chain per iteration, same as neural-texture.

This is a genuine redesign, not a tuning pass: it replaces "reuse
`NodeMaterial`/`LightingModel`/rasterization to evaluate the teacher" with
"reimplement the specific subset of that math the teacher actually needs,
as inline compute-shader TSL." Budget it as such.

## Non-goals

- Do not change the trained model's format, the grid encoding
  (`examples/jsm/neural/NeuralGridModel.js`), or the MLP architecture.
  This plan is scoped to how *teacher targets* are produced, not what's
  learned from them.
- Do not attempt to support arbitrary `NodeMaterial` graphs generically.
  Scope the compute-shader BRDF reimplementation to what the neural-appearance
  teacher actually needs: the `MeshPhysicalNodeMaterial` feature surface
  used for MaterialX-imported PBR materials (base color, metalness,
  roughness, normal, emission, opacity, clearcoat/sheen/etc. only if
  currently exercised -- check `NeuralAppearanceTeacherEvaluator.js`'s
  `supportsEmission`/`supportsOpacity`/material-feature flags for the
  actual surface to cover, don't over-scope to features it doesn't use).
- Do not silently change what the training targets numerically represent.
  Any place where the compute-shader reimplementation can't be bit-exact
  with the current rasterized version (e.g. due to precision differences,
  or intentionally dropping the `iblIndirectRadiance`/`iblIndirectIrradiance`
  isolate-hack's known multi-scatter cross-term leak -- see the "merge
  remaining IBL passes" follow-up already landed on top of Phase 2) must be
  flagged and validated via before/after loss-curve comparison, not shipped
  silently.
- Do not require this to land in one shot. Phase by material-feature-surface
  (direct BRDF first, then emission/opacity, then IBL) so each phase is
  independently measurable and revertible.

## Proposed phases

### Phase 4.0 -- Baseline measurement (do first, cheap)

Before any redesign work, get real numbers to phase the rest of this plan
against and to prove the eventual win:

- Instrument `NeuralAppearanceTrainer.train()`'s iteration loop with
  `performance.mark`/`performance.measure` around: each `evaluateBatch`
  call (broken out by target-mode group), the pixel-unpack loop,
  `uploadSamples`, and each `renderer.compute(...)` call.
- Run ~50-100 iterations with an environment map present (worst case: all
  3 merged render+readback groups execute) and without (direct-only case),
  record ms/iteration broken down by stage.
- Compare against `NeuralTextureTrainer`'s ms/iteration on a comparable
  batch size, to get a concrete target gap to close (this plan's success
  criterion, not just "faster").

### Phase 4.1 -- Direct-lit BRDF teacher in compute (prototype/validate the approach)

Scope: replace only the `'direct'` MRT group (`brdf` + `emission`) with an
inline compute-shader evaluation. This is the simplest case (a single
directional light, no environment sampling) and validates the approach
before investing in the harder IBL path.

- Port the base-color/metalness/roughness/normal BRDF math currently
  expressed via `NeuralTeacherLightingModel` (extends `PhysicalLightingModel`,
  overrides `direct()`/`indirect()` -- see `NeuralAppearanceTeacherEvaluator.js`)
  into standalone TSL functions callable with plain `vec3`/`float` inputs
  (`uv` -> sampled base color/normal/roughness/metalness via the material's
  existing texture-sampling nodes, `wi`, `wo` -> direct lighting response),
  rather than requiring a full `NodeMaterial`/`Scene`/`Camera` render.
  Investigate whether the *sampling* part (reading base color / normal /
  roughness / metalness / emission textures at a given `uv`, with the
  correct MaterialX-authored node graph) can be reused as-is from the
  source `MeshPhysicalNodeMaterial`'s existing nodes called directly inside
  a compute shader context (TSL nodes are generally context-agnostic;
  confirm texture-sampling nodes work identically inside `Fn(...).compute(...)`
  as they do inside a fragment shader) -- if so, only the *lighting model*
  math (the BRDF response given sampled material properties + `wi`/`wo`)
  needs hand-porting, which is a much smaller surface than re-deriving
  texture sampling too.
- Call this inline, per-sample, inside `createTrainBatchComputeNode`
  (`NeuralAppearanceGPUComputeTSL.js`), replacing the current codepath
  where `assignTeacherTargets` populates `sample.target` on the CPU before
  upload.
- This removes `generateTrainingSamples`'s call to
  `assignTeacherTargets`/`assignAuxiliaryTeacherTargets` (for the modes
  covered) entirely -- the compute shader computes the "teacher" target
  itself, in-line, instead of receiving it pre-computed from the CPU.
  `uv`/`wi`/`wo` sample generation (`NeuralAppearanceSampler.js`'s random
  sampling logic) still happens on the CPU (it's cheap, no readback
  involved) and still gets uploaded, same as before.
- Verify: render+readback count for the direct-lit phase drops from 1 (was
  already merged into one MRT pass by Phase 2) to 0. Compare loss curves
  against a pre-change baseline run with the same seed to confirm the
  reimplemented BRDF math matches (should be exact or near-exact, since
  it's the same formulas, just relocated).

### Phase 4.2 -- Emission + opacity in compute

Once 4.1's approach is validated, port `emission`/`opacity` (currently
their own MRT-merged or single-purpose passes) the same way -- these are
simpler than the BRDF path (emission is a direct texture/node sample,
opacity is alpha-test/blend logic), so should be fast once the "call
material sampling nodes from inside a compute shader" plumbing from 4.1 is
proven out.

### Phase 4.3 -- IBL teacher in compute (hardest, do last)

Scope: replace the `iblProbe` MRT group (`iblQuery`, `iblIncoming`,
`iblIrradiance`, and -- if the "merge remaining IBL passes" follow-up has
landed -- `iblIndirectRadiance`/`iblIndirectIrradiance`) with inline compute
evaluation.

- The core operation is PMREM environment lookups (`TSL.pmremTexture(...)`
  with a direction + roughness-derived mip level, see
  `NeuralTeacherIBLEnvironmentNode`/`createTeacherIBLQueryNodes` in
  `NeuralAppearanceTeacherEvaluator.js`). Confirm `pmremTexture(...)`'s
  underlying texture sampling works when called from a compute shader
  context the same way it does in a fragment shader (this is the main
  technical risk for this phase -- PMREM sampling may have fragment-shader
  assumptions baked in, e.g. implicit derivatives for mip selection; check
  whether an explicit-LOD variant is needed/available for compute).
  Prototype this in isolation before porting the full IBL BRDF-weighted
  indirect math (`indirectDiffuse`/`indirectSpecular`, see
  `src/nodes/functions/PhysicalLightingModel.js`).
- If PMREM sampling turns out to be impractical from compute (e.g. requires
  fragment-only derivatives), consider whether an explicit-LOD PMREM
  variant already exists in three.js core, or whether one needs to be
  added generally (worth checking if this is a broader gap other
  compute-shader use cases would hit too, in which case it's worth fixing
  at the `pmremTexture`/PMREM node level rather than working around it
  locally).

### Phase 4.4 -- Remove the now-dead rasterization path

Once all target modes are compute-resident, delete
`NeuralAppearanceTeacherEvaluator`'s render+readback machinery
(`_renderAndRead`, `NeuralAppearanceTeacherReadback.js`,
`NeuralAppearanceTeacherAtlas.js`'s atlas-texture upload path,
`GROUP_BY_MODE`/`_evaluateGrouped`/`_evaluateUngrouped`) if nothing else
still depends on it. Check `examples/webgpu_materials_neural_appearance.html`
first -- it imports `NeuralTeacherIBLEnvironmentNode`/`NeuralTeacherIBLLightingModel`
directly for a live isolate-mode debug view unrelated to training; that
usage must keep working (or be ported to whatever replaces it) even after
the trainer itself stops using the rasterized path.

## Investigate-before-assuming

- **Can TSL texture-sampling nodes run unmodified inside a `.compute()`
  context?** This determines whether Phase 4.1 is "port the lighting math
  only" (cheap) or "port lighting math + texture sampling + UV derivatives"
  (much more expensive). Spend an hour prototyping this first -- it gates
  the rest of the plan's cost estimate.
- **PMREM sampling from compute** (Phase 4.3's main risk) -- see above.
- **UV gradients for MaterialX node graphs that use `dFdx`/`dFdy`
  (mip selection, anisotropic filtering, etc.)**: the current rasterized
  approach gets these for free from the GPU's fragment quad derivatives
  (see the `duvDx`/`duvDy` atlas-encoding trick in
  `NeuralAppearanceTeacherAtlas.js`, which exists specifically to fake
  derivatives through the atlas layout). A compute shader has no
  quad-derivative equivalent -- confirm whether any MaterialX-authored
  teacher materials in practice depend on derivative-based nodes (mip
  selection is the main one), and if so, decide how to approximate them
  (fixed LOD? explicit uv-gradient parameters passed into the compute
  sampling call, mirroring what the atlas trick already encodes?).

## Verification

- Re-run Phase 4.0's instrumentation after each sub-phase lands, record
  ms/iteration before/after per phase.
- Compare loss curves (`validation.loss`/`validation.directional` from
  `onProgress`) against a pre-Phase-4 baseline run with the same seed for
  each phase, to confirm the reimplemented math doesn't silently change
  what's being learned.
- Confirm example-page UI responsiveness improves subjectively (training
  should no longer visibly stall the page) once Phases 4.1-4.3 land.
- Run `test/unit/addons/neural-appearance/` and `test/e2e/neural-appearance-training.js`;
  update/add coverage for the new compute-resident teacher path.

## Report back

Per phase attempted: what changed, ms/iteration before vs. after, loss-curve
comparison result, and anything deferred (e.g. if PMREM-from-compute proves
impractical and Phase 4.3 needs a different approach than originally
scoped).
