# Neural Appearance Texture-Based Emission

## Goal

Represent spatially varying MaterialX emission as a latent-only RGB output evaluated once per fragment. Preserve the existing per-light BRDF decoder and direct-light behavior.

## Output contract

Define emission as linear scene-referred RGB radiance:

```text
outgoingLight = directNeuralLighting + decodedEmission
```

Emission does not depend on `wi`, `wo`, light count, or the geometric cosine. Renderer exposure and tone mapping remain downstream.

The head consumes the sampled 8D latent code. It must run once per fragment through the NodeMaterial emissive path, not once inside `NeuralAppearanceLightingModel.direct()`.

## Fixtures

Add MaterialX fixtures:

- Constant emission with a black reflective base.
- Repeating two-color emission at 1× and 4×.
- Emission plus diffuse reflection.
- Emission with no scene lights.
- HDR emission above one.

Use `standard_surface` emission inputs for broad loader coverage. Add an equivalent `gltf_pbr` fixture if the surface mappings produce different emissive semantics.

## Teacher extraction

Add an explicit teacher mode that isolates `material.emissiveNode`.

Requirements:

- No direct or indirect scene-light contribution.
- No cosine division.
- No display tone mapping or color-space conversion.
- Half-float HDR readback.
- The same UV override and footprint filtering used for BRDF targets.
- A clear error when the MaterialX surface cannot expose emission separately.

Do not derive emission by subtracting two rendered images. Build a teacher output path with defined graph ownership and test it directly.

## Model design experiment

Compare:

### Candidate A: latent-only MLP

- Input: 8 latent values.
- Output: RGB emission.
- Hidden width: start at 8 or 16.
- Activation: non-negative linear with a forward/backward-consistent derivative, or exponential if HDR measurements justify it.

### Candidate B: direct latent projection

- Input: 8 latent values.
- Output: one affine 8→3 projection.

Choose the smallest candidate that meets constant and repeating fixture gates. Do not store direct RGB texels unless both learned candidates fail or cost more at the target quality.

Train BRDF and emission jointly after each head passes alone. Measure gradient conflict and BRDF regression while sharing the latent field.

## Asset format

Add a versioned optional `outputs.emission` section containing:

- Head type and dimensions.
- Weights and biases.
- Output activation with exact semantics.
- Color-space and radiometric metadata.

Keep v1 loader compatibility. A v1 asset produces zero emission. The loader must reject partially defined heads, unknown activations, and mismatched dimensions.

Update reference evaluations with UV-only emission probes. Keep directional BRDF reference records unchanged.

## Runtime integration

In `NeuralAppearanceNodeMaterial.js`:

- Fetch the latent code once and share it between emission and BRDF evaluation.
- Decode emission outside the direct-light callback.
- Route the result through `emissiveNode` or the equivalent `NodeMaterial.setupLighting()` hook.
- Keep `setupOutgoingLight()` and alpha behavior unchanged.
- Ensure zero-light scenes still compile and render the head.

Verify that multiple direct lights do not multiply emission and that intensity controls either BRDF only or both outputs according to an explicit API. Prefer separate `intensity` and `emissiveIntensity` controls.

## Training and validation

Extend `NeuralAppearanceTrainer.js` with a separate target and loss:

- Sample emission once per UV/footprint, then reuse it across directional samples in the batch.
- Weight emission loss independently from BRDF loss.
- Report emission loss, BRDF loss, and combined loss.
- Validate emission by UV and mip through the serialized FP16 path.
- Add a shared-latent interference report: BRDF validation before and after enabling the emission head.

Avoid duplicating teacher evaluations for identical UV and footprint pairs.

## Files

- Modify `examples/jsm/materials/NeuralAppearanceTeacherEvaluator.js`.
- Modify `examples/jsm/materials/NeuralAppearanceTrainer.js`.
- Modify `examples/jsm/loaders/NeuralAppearanceLoader.js`.
- Modify `examples/jsm/materials/NeuralAppearanceNodeMaterial.js`.
- Add MaterialX fixtures under `examples/materialx/`.
- Modify loader and trainer unit tests.
- Modify `test/e2e/neural-appearance-training.js`.
- Update `utils/neural-appearance/convert_checkpoint.py` only if an upstream checkpoint contains a compatible emission head.

## Test-first sequence

1. Unit-test manifest parsing for absent, valid, and malformed emission heads.
2. Unit-test CPU emission evaluation and reference records.
3. Add zero-light constant-emission teacher and runtime E2E tests.
4. Add HDR and repeating fixtures at fixed mip levels.
5. Verify one, two, and zero direct lights produce the same emission.
6. Enable joint BRDF training and measure shared-latent interference.

## Pass criteria

- A v1 asset renders exactly as before and reports zero emission.
- Constant emission matches the linear-HDR teacher within 1% of the channel range, including values above one.
- Zero-light and multi-light renders return the same decoded emission within precision tolerance.
- Resolved repeating emission meets the spatial gates from the repeating-color plan.
- Enabling emission increases BRDF validation error by less than 5% relative. If it exceeds that limit, compare separate latent channels or staged optimization before finalizing the format.
- Serialized CPU and WebGPU emission agree within the measured FP16 tolerance.

## Non-goals

- Emission used as a light source for other objects.
- Indirect transport or path tracing.
- Directional emission.
- Fractional opacity or transmission.
