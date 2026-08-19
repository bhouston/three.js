# Plan: Merge the indirectRadiance and indirectIrradiance heads into one MLP

## Context for the executing agent

Follow-on to the runtime de-dup work already landed on `NeuralAppearanceNodeMaterial.js`/`NeuralAppearanceTSL.js` (shared per-fragment grid-fetch/rotation-frame context across BRDF/emission/opacity/IBL). That work was a pure runtime refactor with no training or format impact. This plan is different in kind: it changes the trained model's architecture, the exported file format, and the training compute kernel. It requires retraining and is not backward-compatible with previously-exported `.json` neural-appearance assets without a version-gated fallback (see "Format compatibility" below).

Repo: three.js fork (`three.js-v2.js`), on branch `neural-appearance-ibl`. Relevant files (re-locate by symbol name, not line number, since line numbers will drift):

- `examples/jsm/neural-appearance/NeuralAppearanceModel.js` — CPU model construction (`createModel`), currently builds `indirectRadianceHead`/`indirectIrradianceHead` as two separate `createMLP(...)` calls.
- `examples/jsm/neural-appearance/NeuralAppearanceGPUModel.js` — GPU buffer layout (`computeModelLayout`/`allocateIndirectProbeHead`), `initFromCPUModel`/`syncToCPU` weight copy loops.
- `examples/jsm/neural-appearance/NeuralAppearanceGPUComputeTSL.js` — hand-differentiated forward+backward training kernel (`trainIndirectProbeHead`, called twice per sample today from inside `createTrainBatchComputeNode`).
- `examples/jsm/neural-appearance/NeuralAppearanceTSL.js` — runtime (inference-side) evaluation (`evaluateIndirectProbeHead`, `evaluateNeuralIBLForTexels`, called twice today).
- `examples/jsm/neural-appearance/NeuralAppearanceFormat.js` — format constants (`INDIRECT_INPUT_SIZE`, `INDIRECT_OUTPUT_SIZE`, `VERSION` — currently `7`).
- `examples/jsm/neural-appearance/NeuralAppearanceManifest.js` — JSON export/import (`createNeuralAppearanceManifest`, `outputs.indirectRadiance`/`outputs.indirectIrradiance` keys).
- `examples/jsm/neural-appearance/NeuralAppearanceRuntime.js`, `NeuralAppearanceValidator.js`, `NeuralAppearanceSampler.js`, `NeuralAppearanceTeacherEvaluator.js` — all reference `indirectRadiance`/`indirectIrradiance` by name (8 source files total reference these two names — grep before editing to make sure none are missed).
- Tests: `test/unit/addons/neural-appearance/{NeuralAppearanceTSL,NeuralAppearanceValidator,NeuralAppearanceNodeMaterial,NeuralAppearanceModel,NeuralAppearanceRuntime}.tests.js` all reference these names.

## Problem statement

`indirectRadiance` and `indirectIrradiance` are two separate 2-layer MLPs (`createMLP(INDIRECT_INPUT_SIZE, [iblHiddenSize], INDIRECT_OUTPUT_SIZE, ...)` in `NeuralAppearanceModel.js`), evaluated back-to-back for the same fragment/sample, both conditioned on the same latent code and `wo`, differing only in which environment probe value (`incoming` vs `irradiance`, each a 3-float PMREM sample) is concatenated into the input and which 3-channel target they're trained against (see `NeuralAppearanceGPUComputeTSL.js`'s two `trainIndirectProbeHead(...)` calls — identical shape, different `probeSampleOffset`/`targetSampleOffset`/weight-buffer offsets; and `NeuralAppearanceTSL.js`'s `evaluateNeuralIBLForTexels`, which calls `evaluateIndirectProbeHead` twice the same way at runtime).

Given how structurally similar these two heads are (same input size class, same hidden size, evaluated at the same point in the pipeline, likely correlated outputs per the user's hypothesis — indirect radiance and irradiance from the same environment/material point are physically related), merging them into one MLP with a shared hidden trunk and a 6-wide output layer (3 radiance + 3 irradiance) is architecturally sound and should cut real per-fragment/per-sample MLP work (one hidden-layer matmul instead of two) both at training time (compute cost, part of what's discussed in `.cursor/plans/neural_appearance_teacher_readback_speedup.md` and `neural_appearance_gpu_resident_teacher.md`) and at runtime (part of what made the trained-neural preview material expensive — see conversation history / prior investigation into `NeuralAppearanceNodeMaterial.js`'s per-pixel cost).

## Non-goals

- Do not touch the BRDF decoder, emission/opacity heads, or the IBL-query head — this plan is scoped to the two indirect-probe heads only. (Emission/opacity are already single-linear-layer heads with no hidden layer — not worth merging into anything; see prior investigation. The IBL-query head has a different output shape/purpose — direction+roughness, not a color — and isn't a natural merge candidate here.)
- Do not change the grid encoding, latent channel count, or BRDF rotation-frame mechanism.
- Do not silently break previously-exported `.json` neural-appearance assets — see "Format compatibility."
- Do not skip retraining/quality validation — merging two independently-trained heads into one shared-trunk head is a real architecture change, not a refactor; it can change what's learned (shared trunk means gradients from both targets shape the same hidden features, which could help — if outputs are correlated, as hypothesized — or hurt — if they interfere). Must be validated empirically (loss curves, visual comparison), not assumed safe.

## Design

### Merged head shape

Replace two heads of shape `INDIRECT_INPUT_SIZE(=LATENT_CHANNELS+6) -> iblHiddenSize -> INDIRECT_OUTPUT_SIZE(=3)` with one head:

```
mergedInput = latents(LATENT_CHANNELS) ++ wo(3) ++ incomingProbe(3) ++ irradianceProbe(3)   // = LATENT_CHANNELS + 9
            -> iblHiddenSize (shared hidden layer, ReLU)
            -> 6   // [0:3] = radiance, [3:6] = irradiance
```

This is the natural generalization: today each head's input is `latents ++ wo ++ its-own-probe` (`INDIRECT_INPUT_SIZE = LATENT_CHANNELS + 3 + 3`); the merged head's input is `latents ++ wo ++ both-probes` (`LATENT_CHANNELS + 3 + 3 + 3`), and its output is both heads' outputs concatenated. Define new format constants (`NeuralAppearanceFormat.js`):

```js
const MERGED_INDIRECT_INPUT_SIZE = LATENT_CHANNELS + 3 + 3 + 3;   // was two separate LATENT_CHANNELS+6
const MERGED_INDIRECT_OUTPUT_SIZE = 6;                             // was two separate 3s
```

Keep the old `INDIRECT_INPUT_SIZE`/`INDIRECT_OUTPUT_SIZE` names only if something else still legitimately needs the single-head shape (check before removing — grep shows they're currently only used by the two-head code paths this plan replaces, so they can likely be renamed/removed outright rather than kept alongside new constants; confirm during implementation).

### Training kernel (`NeuralAppearanceGPUComputeTSL.js`)

Replace the two `trainIndirectProbeHead(...)` calls with one call to a new `trainMergedIndirectHead(...)`:

- Forward: one shared hidden layer (`mergedInput -> iblHiddenSize`, ReLU) instead of two; one output layer (`iblHiddenSize -> 6`) instead of two `iblHiddenSize -> 3`s.
- Loss: compute the same per-channel loss (the existing cube-root-compressed L1, see the `predLog`/`refLog`/`diff` math in `trainIndirectProbeHead`) independently for both 3-channel halves of the merged 6-wide output, against their respective targets (`indirectRadiance`/`indirectIrradiance` sample targets, unchanged) — sum into `sampleIblLoss` same as today. The two halves' losses don't need equal weighting changes unless empirical validation shows one target's gradient dominating the shared trunk and hurting the other (watch for this in the "Verification" step below — if radiance consistently degrades relative to a from-scratch two-head baseline, or vice versa, consider a per-target loss-scale knob before assuming the merge is a wash).
- Backward: standard 2-layer MLP backprop, output layer split into two 3-channel delta groups (one per target) that both flow into the *same* shared hidden layer's delta (sum their contributions before back-propagating into the hidden layer and into the shared latent gradient, same accumulation pattern already used elsewhere in this file for shared-trunk gradient flow — e.g. how `actGradLatentsOffset` already accumulates contributions from multiple heads into one shared buffer).
- This removes one entire hidden-layer forward+backward pass (weights, activations, gradient accumulation) worth of GPU compute per training sample versus today's two independent calls.

### GPU buffer layout (`NeuralAppearanceGPUModel.js`)

- Replace the two `allocateIndirectProbeHead(...)` calls (`indirectRadianceHead`, `indirectIrradianceHead` weight offsets) with one `allocateMergedIndirectHead(...)` sized for `MERGED_INDIRECT_INPUT_SIZE -> iblHiddenSize -> MERGED_INDIRECT_OUTPUT_SIZE`.
- Update `initFromCPUModel`/`syncToCPU`'s weight copy loops to copy the single merged head's two layers instead of two heads' two layers each.
- Update activation-buffer offset allocation (`actIndirectA0Offset` etc. in `computeModelLayout`) similarly — one set of activation slots sized for the merged shapes instead of two duplicate sets.

### CPU model (`NeuralAppearanceModel.js`)

Replace:
```js
const indirectRadianceHead = createMLP( INDIRECT_INPUT_SIZE, [ iblHiddenSize ], INDIRECT_OUTPUT_SIZE, random, 'relu', 'linear' );
const indirectIrradianceHead = createMLP( INDIRECT_INPUT_SIZE, [ iblHiddenSize ], INDIRECT_OUTPUT_SIZE, random, 'relu', 'linear' );
```
with:
```js
const indirectHead = createMLP( MERGED_INDIRECT_INPUT_SIZE, [ iblHiddenSize ], MERGED_INDIRECT_OUTPUT_SIZE, random, 'relu', 'linear' );
```

### Runtime evaluation (`NeuralAppearanceTSL.js`)

- Replace the two `evaluateIndirectProbeHead(...)` calls in `evaluateNeuralIBLForTexels` with one call to a new `evaluateMergedIndirectHead(material, head, uniforms, latents, wo, incomingProbe, irradianceProbe)` that builds the concatenated 9-extra-value input, runs one `evaluateMLP(...)`, and returns `{ radiance: toVec3(output.slice(0,3)), irradiance: toVec3(output.slice(3,6)) }` (or equivalent), each still passed through `applyOutputActivation` as today.
- `isolate` handling (`'radiance'`/`'irradiance'`/`'full'`, used by the debug view) still works the same way at the call site — just slice the relevant half of the merged output instead of choosing which of two separate MLP calls to run. Note this technically changes the *isolate debug view's* compute cost profile slightly (it now always runs the full merged MLP even when only one half is displayed, whereas before it could skip evaluating the unneeded head entirely) — acceptable since the debug view is not the hot path, but worth a one-line comment noting the tradeoff.

### Format / manifest (`NeuralAppearanceFormat.js`, `NeuralAppearanceManifest.js`)

- Bump `VERSION` (currently `7`) to `8`.
- Replace the manifest's separate `outputs.indirectRadiance`/`outputs.indirectIrradiance` keys with one `outputs.indirect` key holding the merged head's layers (`{ layers: serializeLayers(model.indirectHead), inputSize, outputSize, outputActivation }`).
- Update every other reader of `outputs.indirectRadiance`/`outputs.indirectIrradiance` (`NeuralAppearanceRuntime.js`, `NeuralAppearanceValidator.js`, `NeuralAppearanceSampler.js`, `NeuralAppearanceTeacherEvaluator.js` — grep for both names before finishing, this list is what today's grep found but re-check after code has shifted) to read `outputs.indirect` instead.

## Format compatibility

Decide and document one of:

1. **Hard version bump, no back-compat loader** (simplest): `NeuralAppearanceLoader`/`NeuralAppearanceRuntime.js` reject `.json` assets with `version < 8` (or any version that predates the merge) with a clear error message telling the user to retrain. Acceptable if there's no meaningful population of previously-exported assets in the wild that need to keep working (check whether the example page or any docs/tests ship a pre-built `.json` asset committed to the repo — if so, it needs to be regenerated as part of this change, not just the loader code).
2. **Version-gated dual code path**: `NeuralAppearanceRuntime.js`'s evaluator checks `outputs.indirect` vs the old `outputs.indirectRadiance`/`outputs.indirectIrradiance` shape and evaluates accordingly (both code paths kept, old one calling the original two-call evaluator). More work, only worth it if there's a real backward-compatibility requirement.

Default to option 1 unless the executing agent finds evidence (committed sample assets, documented external users of the format) that option 2 is actually needed — check `examples/models/` or similar for any committed neural-appearance `.json` files before deciding, and re-export/regenerate any found ones as part of this change either way.

## Investigate-before-assuming

- **Loss weighting between the two halves.** The two heads were independently trained (independent loss terms, independent weights) before; sharing a trunk means their gradients now compete for the same hidden-layer capacity. Confirm after training whether one target's error is measurably worse than a from-scratch two-head baseline trained under the same settings/seed — if so, a per-half loss-scale multiplier (mirroring `highlightLossScale`'s pattern elsewhere in the trainer) may be needed; don't assume equal weighting is correct without checking.
- **`iblHiddenSize` sizing.** The merged head has ~1.5x the input width (LATENT_CHANNELS+9 vs LATENT_CHANNELS+6) and 2x the output width (6 vs 3) of either original head, for the same `iblHiddenSize` (default derived from `hiddenSize`, clamped 16-32). Confirm the default hidden size still has enough capacity for the combined task — if validation loss for either half regresses noticeably versus the two-head baseline, consider whether `iblHiddenSize` needs its own independent default distinct from the IBL-query head's, rather than sharing the same clamp logic.
- **Any other reader of `INDIRECT_INPUT_SIZE`/`INDIRECT_OUTPUT_SIZE`/`allocateIndirectProbeHead`/`indirectRadianceHead`/`indirectIrradianceHead` symbols not caught by this plan's file list** — re-grep at implementation time; the plan's file list reflects state as of this writing and may drift if other neural-appearance work lands first (this addon has an active in-flight refactor history — check current `git log` on this branch before starting).

## Verification

- Add/update unit tests in the 5 test files identified above to reflect the merged head's shape (input/output sizes, one `outputs.indirect` key instead of two, `createModel`/`NeuralAppearanceGPUModel` layout assertions).
- Retrain a representative test material (same one used in `test/e2e/neural-appearance-training.js` if practical) with the merged head, and compare against a pre-merge baseline run with the same seed/settings:
  - Final direct/IBL/validation loss, comparable or better (not silently worse).
  - Visual comparison of the trained-neural preview against the teacher for an IBL-heavy scene (the merge specifically affects indirect lighting quality).
- Re-measure runtime per-pixel cost (reuse the profiling approach from the earlier teacher/runtime-cost investigation) to confirm the expected reduction: one hidden-layer MLP pass removed per fragment in the shaded (non-isolate-debug) view.
- Re-measure training ms/iteration (reuse `performance.mark`/`measure` instrumentation from `neural_appearance_teacher_readback_speedup.md`'s verification section) to confirm the expected reduction in the IBL-phase compute kernel cost.
- Run `test/unit/addons/neural-appearance/` and `test/e2e/neural-appearance-training.js`; all should pass with the updated format.

## Report back

What changed per file, before/after per-pixel MLP-call count and measured runtime cost, before/after training ms/iteration for the IBL phase, loss-curve comparison (merged vs. two-head baseline) for both radiance and irradiance targets, the format-compatibility decision made (and why), and anything deferred (e.g. if loss weighting between the two halves needs follow-up tuning).
