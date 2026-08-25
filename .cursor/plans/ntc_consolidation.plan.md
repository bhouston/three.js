# NTC consolidation plan

Goal: this branch keeps only the **neural material training workflow**
(rebranded to the `NTC` name), and gains the **NTC run-time inference
engine** (loader + decoder) from `ntc_16bit_tsl`, plus a new `NTCExporter`
that replaces `NeuralMaterialLoader`/the ad-hoc export path. Neural
appearance (BRDF/atlas teacher) and standalone neural texture (single-image
demo) are removed. End state: one coherent NTC product — train on this
branch, run inference via the same `.ntc` format/loader/decoder the
`ntc_16bit_tsl` branch already ships.

## Why this ordering

`ntc_16bit_tsl`'s `NTCMLPNode.js`/`NTCFormat.js`/`NTCManifest.js` are
already an inference-only fork of this branch's
`neural-texture/NeuralTextureNodeMaterial.js` +
`neural-material/NeuralMaterialFormat.js` + `NeuralMaterialManifest.js` —
same channel vocabulary, same fp16/mat4-packed decoder (this branch's fp16
TSL work landed on both branches independently, so they're compatible, not
duplicative). So "consolidate" mostly means: rename this branch's trainer
code to match the `ntc_16bit_tsl` naming/module layout, delete the
duplicate inference logic in favor of the already-polished
`ntc_16bit_tsl` version, and reunite training + inference under one
`examples/jsm/ntc/` tree.

## Scope confirmed with user

Remove neural-appearance entirely. Remove the *standalone* neural-texture
demo/loader (`webgpu_materials_neural_texture.html`,
`NeuralTextureLoader.js`), but fold its underlying shared trainer engine
(grid model, MLP, GPU compute, quantization) into the NTC module tree,
since neural-material training is built directly on it.

---

## Phase 1 — Delete neural-appearance, wholesale

Everything BRDF/atlas/teacher-specific has no NTC counterpart and no
consumer once neural-material training doesn't need it.

Delete:
- `examples/jsm/neural-appearance/` (all 13 files)
- `examples/jsm/loaders/NeuralAppearanceLoader.js`
- `examples/webgpu_materials_neural_appearance.html`
- `examples/jsm/neural/NeuralVectorMath.js` (appearance-only consumer)
- `test/unit/addons/loaders/NeuralAppearanceLoader.js`
- `test/unit/addons/neural-appearance/` (5 files)
- `test/vitest/browser/neural-appearance/` (6 files)
- `test/vitest/unit/neural-appearance/` (7 files)
- `test/vitest/unit/neural/NeuralVectorMath.test.js`
- `test/e2e/neural-appearance-training.js`
- `EXT_neural_textures.md` (confirm it's appearance-specific before deleting — verify content)
- `.cursor/plans/neural_appearance_*.md` and `*.plan.md` (11 files) —
  historical planning docs, safe to delete or leave untouched (they're
  dev-only clutter, not shipped code; low priority either way)

Remove references:
- `examples/jsm/Addons.js` — drop the 12 `neural-appearance/*` export lines
- `examples/files.json` — drop `webgpu_materials_neural_appearance`
- `examples/tags.json` — drop its tag entry

## Phase 2 — Delete standalone neural-texture demo, keep the engine

Delete:
- `examples/webgpu_materials_neural_texture.html`
- `examples/jsm/loaders/NeuralTextureLoader.js` (single-texture-only
  loader; NTCLoader supersedes it — but first confirm nothing in the
  material chain calls `NeuralTextureLoader.parseManifestObject`, since
  today's `NeuralMaterialLoader.js` does; that call moves to whatever
  NTCLoader/NTCManifest decode path replaces it in Phase 3)
- `test/vitest/browser/neural-texture/NeuralTextureNodeMaterial.test.js`
  (standalone-material-facing test only, if any — audit before deleting;
  keep tests that exercise the shared trainer engine)
- `examples/files.json` / `tags.json` entries for
  `webgpu_materials_neural_texture`

Keep (rename in Phase 3, don't delete): `NeuralTextureTrainer.js`,
`NeuralTextureModel.js`, `NeuralTextureGPUModel.js`,
`NeuralTextureGPUComputeTSL.js`, `NeuralTextureSource.js`,
`NeuralTextureManifest.js`, `NeuralTextureNodeMaterial.js`'s
`buildLevelTextures`/`evaluateNeuralTextureRaw` helpers (still imported by
`NeuralMaterialNodeMaterial.js`).

## Phase 3 — Rename/restructure into the `ntc/` module tree

Target layout, modeled on `ntc_16bit_tsl`'s `examples/jsm/ntc/` +
mirroring what's genuinely trainer-vs-runtime:

```
examples/jsm/ntc/
  NTCFormat.js            <- merge NeuralMaterialFormat.js (trainer, has
                              resolveNode/resolveConstant) with ntc_16bit_tsl's
                              NTCFormat.js (inference, has applyActive/
                              applyConstant) — this branch's version already
                              needs both, so it becomes the new canonical file
  NTCManifest.js           <- rename NeuralMaterialManifest.js; body already
                              matches ntc_16bit_tsl's encodeNTC() shape —
                              reconcile field-for-field, adopt encodeNTC's
                              cleaner form (drop the NeuralTextureManifest
                              indirection now that there's one flat format)
  NTCOutputTypes.js         <- rename NeuralOutputTypes.js
  NTCOutputActivations.js   <- rename neural/NeuralOutputActivations.js
  NTCBinaryCodec.js         <- rename neural/NeuralBinaryCodec.js
  NTCHalfFloatTexture.js    <- rename neural/NeuralHalfFloatTexture.js
  NTCMLPNode.js             <- new: adopt ntc_16bit_tsl's NTCMLPNode.js
                              verbatim as the inference/decoder half;
                              NeuralMaterialNodeMaterial.js's constructor
                              becomes a thin wrapper that also calls
                              applyNTCToMaterial-equivalent logic, OR
                              NTCMLPNode's applyNTCToMaterial gets extended
                              with the constant/renderFlags handling
                              NeuralMaterialNodeMaterial does today (needs a
                              closer read of both to decide the merge shape
                              — flag as a design decision to make while
                              implementing, not fully resolved by this plan)
  NTCNodeMaterial.js         <- rename NeuralMaterialNodeMaterial.js (trainer-
                              facing parts: classifyMaterialChannels-driven
                              construction), reconciled against NTCMLPNode.js
  NTCSource.js               <- rename NeuralMaterialSource.js
  NTCTrainer.js               <- rename neural-texture/NeuralTextureTrainer.js
  NTCGridModel.js              <- rename neural/NeuralGridModel.js
  NTCMLP.js                    <- rename neural/NeuralMLP.js
  NTCGPUModel.js                <- rename neural-texture/NeuralTextureGPUModel.js
  NTCGPUComputeTSL.js            <- rename neural-texture/NeuralTextureGPUComputeTSL.js
  NTCGPUTrainingConstants.js      <- rename neural/NeuralGPUTrainingConstants.js
  NTCQuantization.js               <- rename neural/NeuralQuantization.js
  NTCTrainingUtils.js                <- rename neural/NeuralTrainingUtils.js
  NTCLossGraph.js                     <- rename neural/NeuralLossGraph.js
  NTCMaterialXSamples.js                <- rename neural/NeuralMaterialXSamples.js
  NTCModelSize.js                        <- rename neural/NeuralModelSize.js
  NTCMLPTSL.js                             <- rename neural/NeuralMLPTSL.js

examples/jsm/loaders/
  NTCLoader.js    <- adopt ntc_16bit_tsl's version, updated to import from
                     the merged NTCManifest.js/NTCFormat.js above instead
                     of its slightly different inference-only versions
  (delete NeuralMaterialLoader.js, NeuralTextureLoader.js,
   NeuralAppearanceLoader.js)

examples/jsm/exporters/
  NTCExporter.js  <- NEW, three.js exporter-convention wrapper (class
                     with .parse(material, onDone/onError) or similar,
                     matching how other examples/jsm/exporters/*.js are
                     shaped) around encodeNTC() from NTCManifest.js —
                     this is genuinely new since ntc_16bit_tsl is
                     inference-only and never had an exporter class, just
                     the bare encodeNTC() function. This is what the
                     trainer example calls to produce a downloadable
                     .ntc file, and what replaces "export via
                     NeuralMaterialManifest + hand-rolled download button"
                     if the current trainer example does that inline.
```

Use `git mv` for every rename to preserve history; do content edits
(import paths, class names, `THREE.NeuralXxx` → `THREE.NTCXxx` in error
messages/JSDoc `@augments`/`@three_import`) as a second pass per file so
diffs stay reviewable.

**Open design decision to make during implementation, not here:**
`ntc_16bit_tsl`'s `NTCMLPNode.js` (decoder + `applyNTCToMaterial`) and this
branch's `NeuralMaterialNodeMaterial.js` (trainer-aware node material
class) overlap in the decode path but aren't identical — one is a
free function + material mutator, the other is a `MeshPhysicalNodeMaterial`
subclass constructed directly from a classification. Decide whether the
merged runtime keeps the subclass shape (probably right, since the
trainer's preview/debug-view code — `setDebugView`, `FRAME_VIEWS` — wants a
live material instance to mutate) with `applyNTCToMaterial` becoming
internal plumbing it calls, or the reverse.

## Phase 4 — Rename the trainer example + assets

- `examples/webgpu_materials_neural_material.html` →
  `examples/webgpu_materials_neural_texture_compression_trainer.html`
  (matches the `ntc_16bit_tsl` example's base name, `_trainer` suffix
  distinguishes it from the inference example)
- Update its imports to the new `ntc/` module paths and `NTCExporter`
- `examples/files.json`, `examples/tags.json`: rename entry, tags become
  `[ "neural", "material", "pbr", "tsl", "training", "compression", "ntc" ]`
  or similar
- `test/vitest/**/neural-material/*` → `test/vitest/**/ntc/*`, renamed
  per-file to `NTC*.test.js`, imports updated
- `test/unit/addons/neural-material/*` → mirrored under `ntc/`

## Phase 5 — Pull in the run-time inference example + assets from `ntc_16bit_tsl`

Bring across as-is (already self-contained, only needs the merged
`ntc/`+`loaders/NTCLoader.js` from Phase 3 to exist):
- `examples/webgpu_materials_neural_texture_compression.html`
- `examples/ntc/*.ntc` (5 sample assets) + `examples/ntc/README.md`
  (update its "this branch ships inference only" note — no longer true)
- `examples/screenshots/webgpu_materials_neural_texture_compression.jpg`
- `test/unit/addons/ntc/NTC.tests.js` — reconcile against whatever
  `test/unit/addons/neural-material/*` survives from Phase 4 (likely
  overlapping coverage; merge into one file rather than keeping both)
- `examples/jsm/Addons.js`: add `export * from './ntc/...'` lines,
  `export { NTCLoader } from './loaders/NTCLoader.js'`,
  `export { NTCExporter } from './exporters/NTCExporter.js'`
- `examples/files.json`/`tags.json`: add the new example entry

Do this via `git checkout ntc_16bit_tsl -- <paths>` or `git show
ntc_16bit_tsl:<path> > <path>` per file, not a branch merge (the two
branches share only an old common ancestor and have diverged a lot outside
neural/NTC code — SunLight moves, GaussianSplat renames, USD work, etc. — a
real merge would drag all of that in and conflict heavily).

## Phase 6 — Cross-check and cleanup

- `grep -riE "neuralmaterial|neuraltexture|neuralappearance"` across the
  whole repo (code, docs, `test/e2e`, `EXT_neural_textures.md` if kept,
  `.cursor/plans`) — should return nothing except historical plan docs
  intentionally left in place
- Run `test/unit` and `test/vitest` neural/ntc suites
- Load the two remaining examples (trainer + inference viewer) via the
  `run` skill / dev server, confirm: trainer trains, exports a `.ntc`
  matching the new `NTCExporter`, and that exact file loads correctly in
  the inference viewer via `NTCLoader` — this is the actual end-to-end
  proof the consolidation worked, not just "files compile"
- Update top-level docs referencing the old example/loader names (search
  `docs/` and root `README.md` for `NeuralMaterial`/`neural_material`)

---

## Open questions for you before I start executing

1. **`NTCExporter` shape** — should it match a typical three.js exporter
   API (e.g. `new NTCExporter().parse( material, onDone, onError, options )`
   returning/streaming the manifest object or a JSON string), or would you
   rather it stay a plain function export (`exportNTC(material, ...)`) like
   `encodeNTC` is today, with a thin class wrapper only for
   `Addons.js`/example-import consistency?
2. **`.cursor/plans/neural_appearance_*.md` and `neural_texture_*` docs** —
   delete alongside the code, or leave as historical record? (They're not
   shipped, purely dev planning artifacts.)
3. **`EXT_neural_textures.md`** at repo root — I haven't read it yet; if
   it documents the appearance/texture GLTF extension specifically it goes
   in Phase 1, if it documents the general NTC concept it should stay
   (and probably move into `examples/ntc/README.md` or get superseded by
   it).
4. Want this done as one branch-wide pass, or would you rather land it as
   a sequence of smaller PRs following the phases above (appearance
   removal → texture-demo removal → rename → NTC import → exporter)? Given
   the file count (~150+ touched), smaller PRs would review a lot easier.
