# Neural Appearance UV Normal Maps

## Goal

Determine whether the current 8-channel UV latent field, learned shading frames, and RGB BRDF decoder can reproduce tangent-space normal maps across directions and mip footprints. Change frame or filtering training only after tests isolate a failure.

## Representation hypothesis

A tangent-space normal map remains inside the v1 representation. UV selects a latent code, and the learned 8→12 frame layer can rotate the local directional basis. The decoder can then reproduce the teacher BRDF for the rotated lobe.

The hypothesis fails if the learned frame cannot preserve the required normal variation, the latent grid undersamples the map, or filtered normal distributions need information that a single frame and decoder response cannot retain.

## Fixtures

Add MaterialX fixtures under `examples/materialx/`:

- A constant flat normal control.
- Sinusoidal tangent-space normals along U.
- A two-axis checker of known tangent-space normals.
- The same patterns at 1×, 2×, 4×, and 8× UV repetition.
- A glossy isotropic base material so highlight movement reveals normal direction.

Use generated numeric patterns or a dedicated linear normal texture. Do not reuse a color-space texture. Document tangent convention, green-channel sign, wrap mode, and expected normal at probe UVs.

## Test matrix

For each fixture, evaluate:

- Latent resolutions 8, 16, and 32 where memory permits.
- Fixed mip levels from 0 through the smallest level.
- At least two latent downsample ratios.
- View rotations around the tangent normal and grazing views.
- Mirror-aligned and off-specular incident directions.

Compare a plane with an identity tangent frame first. Add the torus knot only after the plane passes so mesh TBN errors do not contaminate the result.

## Measurements

Extend the shared spatial validation output with normal-map-specific values:

- Teacher and neural highlight peak direction.
- Angular error between those peak directions.
- Peak intensity and lobe-width error.
- Error grouped by normal-pattern phase and mip.
- Rotation consistency for the same tangent-space sample.
- Filtered response width as footprint grows.

Do not infer a neural normal from RGB. Measure the response the model represents.

## Implementation sequence

### Phase 1: characterize v1 unchanged

1. Add fixtures and probe tests.
2. Train with the existing Gaussian footprint-filtered teacher.
3. Verify the MaterialX normal graph receives the intended UV and derivatives.
4. Compare teacher, in-memory model, serialized FP16 CPU evaluator, and WebGPU runtime.
5. Sweep latent resolution and repeat count to locate the bandwidth boundary.

If errors track the expected Nyquist limit and fall with resolution, retain the current representation and document the operating range.

### Phase 2: isolate learned-frame training

Add diagnostics for the learned frames in `NeuralAppearanceTrainer.js`:

- Decode both frame normals and tangents at fixture probe UVs.
- Check unit length, orthogonality, and handedness.
- Compare frame variation against highlight movement.
- Verify gradients reach both rotation weights and latent values through frame construction.

Normalize the learned bitangent in training and runtime if the parity test shows a mismatch. Keep frame math identical in the CPU evaluator and TSL runtime.

### Phase 3: evaluate filtering

The Gaussian teacher currently integrates radiance samples over a UV footprint. `prefilterLeanNormalRoughness()` in `NeuralAppearanceFilterUtils.js` computes normal moments but does not feed teacher targets.

Run an A/B study:

- Existing radiance-space footprint integration.
- LEAN-derived normal and roughness moments applied to the teacher material before directional evaluation.

Integrate LEAN only if it lowers held-out coarse-mip lobe position and width error without degrading base-mip parity. Do not combine both methods unless the target semantics are defined.

## Files

- Add normal fixtures under `examples/materialx/` and any required linear texture under `examples/materialx/resources/`.
- Modify `test/e2e/neural-appearance-training.js`.
- Modify `examples/webgpu_materials_neural_appearance_train.html` only for fixture selection or probe exposure.
- Modify `examples/jsm/materials/NeuralAppearanceTrainer.js` for frame diagnostics if Phase 1 requires them.
- Modify `examples/jsm/materials/NeuralAppearanceFilterUtils.js` only after the filtering A/B result.
- Add unit coverage in `test/unit/addons/materials/NeuralAppearanceTrainer.tests.js`.

## Pass criteria

- The flat-normal control matches the corresponding glossy fixture within the shared linear-HDR tolerance.
- For resolved patterns, median highlight-direction error stays below 3 degrees and the 95th percentile stays below 8 degrees across tested rotations.
- Coarser footprints broaden or average the lobe in the same direction as the teacher; they must not create new RGB peaks.
- Runtime rotation does not change tangent-space probe results beyond the WebGPU precision tolerance.
- Every learned frame remains finite, right-handed, and orthonormal within `1e-3`.
- The test reports a bandwidth-limit failure separately from an optimizer or runtime-parity failure.

Treat the angle limits as initial gates. Record baseline distributions before tightening them.

## Decision outcome

End this work with one explicit result:

- v1 supports resolved tangent-space normal maps within a measured repeat/resolution envelope;
- frame training needs a parity or gradient correction; or
- coarse normal distributions require an added representation such as moments.

Do not add latent channels or a normal output head without evidence for the third result.

## Non-goals

- Object-space procedural normals.
- Displacement or silhouette changes.
- Clearcoat-normal support.
- A new asset version unless tests prove v1 cannot represent the target.
