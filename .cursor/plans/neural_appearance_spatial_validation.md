# Neural Appearance Spatial Validation

## Goal

Build a linear-HDR validation path that separates teacher errors, latent bandwidth limits, optimization failures, mip filtering errors, and runtime evaluation errors. Land these diagnostics before changing the asset format.

## Current baseline

- `evaluateRuntimeValidation()` in `examples/jsm/materials/NeuralAppearanceTrainer.js` reports aggregate loss, three `wi.z` and `wo.z` bins, reciprocity, angular smoothness, and preview samples.
- The preview records UV, directions, mip, target RGB, and predicted RGB, but no code aggregates error by spatial frequency, seam distance, or mip.
- `evaluateNeuralAppearanceJson()` mirrors serialized CPU inference, including FP16 latent quantization and footprint-based mip selection.
- The E2E test compares tone-mapped screenshots. Its single spatial fixture, `neural_train_uv_grid.mtlx`, uses an 8² latent field and loose image thresholds.
- Training and runtime estimate footprints differently. Training supplies `2^mip / resolution`; runtime uses the longest screen-space UV derivative in latent texels.

## Deliverables

### 1. Structured validation samples

Extend `generateTrainingSamples()` or add a separate exported validation-sample generator in `examples/jsm/materials/NeuralAppearanceTrainer.js`.

Generate deterministic samples over:

- UV grids at texel centers and texel boundaries.
- Explicit mip levels and footprints.
- Repeat counts 1, 2, 4, and 8.
- Front-facing, grazing, mirror-pair, and off-specular direction pairs.
- UV seam neighborhoods at `u` and `v` values near 0 and 1.

Keep this set independent from training batches and seeded separately. Store the fixture label, spatial frequency, seam classification, and requested mip on each sample.

### 2. Linear-HDR metrics

Extend `evaluateRuntimeValidation()` with:

- Per-mip mean absolute error, relative absolute error, and maximum channel error.
- Per-spatial-frequency metrics using fixture metadata, not frequency inferred from rendered pixels.
- Seam versus interior metrics.
- Highlight peak position and peak RGB error for directional sweeps.
- FP32-manifest versus FP16-runtime latent error.
- Explicit CPU evaluator versus WebGPU runtime parity.

Preserve the current metric keys so existing consumers keep working. Add a schema tag to the validation object before E2E code depends on new fields.

Use an epsilon only in relative-error denominators. Report absolute error beside every relative metric so dark samples do not dominate.

### 3. Teacher readback

Add a diagnostic render mode to `examples/webgpu_materials_neural_appearance_train.html` that exposes linear, pre-tone-mapping teacher and neural pixels to the E2E harness.

The path must:

- Use `NoColorSpace`.
- Disable tone mapping and exposure changes.
- Preserve values above one in half-float readback.
- Fail when half-float teacher readback is unavailable.
- Render a plane with a known tangent frame for numerical comparisons.

Keep screenshot comparisons as presentation tests, not fidelity gates.

### 4. Sweep harness

Add a focused Node script under `test/e2e/` or `utils/neural-appearance/` that drives the existing training example and emits JSON records for:

- `sourceResolution`
- `resolution` and `latentDownsample`
- repeat count
- iteration count
- hidden width
- seed
- fixed mip and automatic LOD

Record training time, exported byte count, validation metrics, and active teacher target type. Do not add every sweep point to routine CI. Commit a small smoke matrix and document the larger local matrix.

### 5. Diagnostic attribution

Run each fixture through four checkpoints:

1. Teacher target repeatability for identical samples.
2. In-memory FP32 model prediction.
3. Serialized JSON plus FP16 latent prediction through `evaluateNeuralAppearanceJson()`.
4. WebGPU runtime output.

Attribute the first checkpoint where error crosses its threshold. Include the checkpoint name in failures.

## Files

- Modify `examples/jsm/materials/NeuralAppearanceTrainer.js`.
- Modify `examples/webgpu_materials_neural_appearance_train.html`.
- Modify `test/unit/addons/materials/NeuralAppearanceTrainer.tests.js`.
- Modify `test/e2e/neural-appearance-training.js`.
- Add a sweep runner only if the E2E harness cannot express the matrix cleanly.
- Reuse `examples/jsm/materials/NeuralAppearanceFilterUtils.js`; do not add a second footprint implementation.

## Test-first sequence

1. Unit-test metric aggregation with synthetic samples whose per-mip, seam, and frequency errors are known.
2. Unit-test footprint-to-mip decisions at exact level boundaries.
3. Unit-test FP32 versus serialized FP16 attribution.
4. Add a constant Lambert linear-HDR E2E baseline at resolution 1.
5. Add one spatial smoke fixture at two repeat counts and two mip levels.
6. Add CPU-to-WebGPU parity after the readback path exists.

## Pass criteria

- Constant Lambert at resolution 1 stays within `1e-3` absolute linear-RGB error for the CPU serialization path. Set the WebGPU tolerance from measured half-float noise and record it in the test.
- Every validation sample appears in exactly one requested-mip bin and one seam/interior bin.
- Fixed-mip CPU and WebGPU outputs agree within the measured precision tolerance.
- Automatic LOD selects the same mip as the shared footprint function at points away from integer LOD boundaries.
- The harness detects an intentionally corrupted mip, UV seam, and decoder weight and identifies the correct checkpoint.
- CI output includes raw metric values and fixture metadata on failure.

## Decisions this plan must settle

- Adopt one footprint scalar for teacher filtering, training mip selection, CPU validation, and runtime LOD.
- Decide whether the runtime target is nearest-mip, stochastic adjacent-mip, or trained trilinear behavior. Do not tune spatial thresholds until this contract is fixed.
- Keep FLIP or tone-mapped image metrics as secondary perceptual measures. Linear HDR remains the numerical source of truth.

## Non-goals

- No new neural output heads.
- No encoder or optimizer rewrite.
- No production-scale sweep in CI.
- No asset-format version change.
