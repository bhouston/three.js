# Neural Appearance Cutout Opacity

## Goal

Represent spatially varying cutout opacity with a latent-only scalar head evaluated before lighting. Preserve the opaque depth pipeline for retained fragments and keep fractional blending outside this work.

## Output contract

The head predicts linear coverage `opacity` in `[0, 1]` from the sampled 8D latent code. Runtime applies a material alpha threshold:

```text
discard when decodedOpacity < alphaCutoff
```

The output must not depend on light direction, view direction, or light count. Use `opacityNode` and `alphaTestNode`; keep `transparent = false`.

Define alpha-to-coverage as an optional raster mode, not a different learned target.

## Fixtures

Add MaterialX fixtures:

- Constant opacity 0 and 1 controls.
- Binary checker mask at 1×, 2×, 4×, and 8×.
- Antialiased circle or stripe mask with known linear coverage at edges.
- A mask combined with a diffuse color texture.
- A mask whose pattern crosses the UV seam.

Prefer `gltf_pbr` with `alpha_mode = mask` because the loader already maps it to `alphaTestNode`. Add standard-surface coverage only after defining how blend opacity becomes a cutout target.

Record alpha cutoff, wrap mode, texture color-space handling, and expected probe values.

## Teacher extraction

Add a teacher mode that evaluates the MaterialX opacity graph as a scalar before discard.

Requirements:

- Return continuous source coverage for training, including antialiased edge values.
- Do not infer opacity from RGB or background compositing.
- Reuse semantic UV and footprint filtering.
- Expose the source alpha cutoff as metadata.
- Distinguish an absent opacity graph from a constant-one graph.

Use the continuous target to train the head. Apply cutoff only for coverage and silhouette metrics.

## Model experiment

Compare:

- An affine 8→1 latent projection with sigmoid output.
- A small 8→H→1 MLP with sigmoid output.

Use binary cross-entropy for binary masks and a bounded regression loss for antialiased coverage. Test one shared loss only if it handles both fixtures without degrading edge calibration.

Train opacity alone first. Then train it jointly with BRDF and emission heads and report shared-latent interference.

## Asset format

Add a versioned optional `outputs.opacity` section containing:

- Head type and dimensions.
- Weights and biases.
- Exact output activation.
- Default `alphaCutoff`.
- Coverage semantics and optional alpha-to-coverage recommendation.

V1 assets imply opacity 1. Preserve v1 loader and runtime behavior. Reject cutout heads with missing cutoff, invalid dimensions, or an output range that is not bounded.

Add UV-only opacity reference evaluations around both mask interiors and edges.

## Runtime integration

In `NeuralAppearanceNodeMaterial.js`:

- Fetch the latent code during fragment material setup.
- Decode opacity once.
- Assign the result through `opacityNode`.
- Apply `alphaTestNode` or `alphaTest` before lighting.
- Keep `transparent = false`, `depthWrite = true`, and the opaque render queue.
- Expose `alphaCutoff` and `alphaToCoverage` without recompiling decoder weights.
- Share latent sampling with BRDF and emission heads.

Verify shadow and depth passes use the same cutout graph. If NodeMaterial requires a separate depth or shadow material path, include it in the implementation before declaring support.

## Filtering and LOD

Train every mip against footprint-filtered continuous coverage. Coarse levels should preserve mean area coverage, not majority-vote binary labels.

Test:

- Fixed mip silhouette area.
- Automatic LOD transition stability.
- Threshold sensitivity.
- Alpha-to-coverage with and without MSAA.
- UV-hash stochastic LOD, if retained, for persistent holes or edge noise.

Do not use post-tone-mapping RGB differences to judge mask quality.

## Metrics

Report:

- Continuous opacity MAE.
- Binary intersection-over-union at the asset cutoff.
- False-positive and false-negative coverage.
- Silhouette area error.
- Boundary distance error in pixels on a fixed-resolution plane.
- Per-mip mean coverage.
- Seam versus interior error.
- Depth and color pass disagreement.

## Files

- Modify `examples/jsm/materials/NeuralAppearanceTeacherEvaluator.js`.
- Modify `examples/jsm/materials/NeuralAppearanceTrainer.js`.
- Modify `examples/jsm/loaders/NeuralAppearanceLoader.js`.
- Modify `examples/jsm/materials/NeuralAppearanceNodeMaterial.js`.
- Add MaterialX fixtures under `examples/materialx/`.
- Modify loader, trainer, and MaterialX loader unit tests.
- Modify `test/e2e/neural-appearance-training.js`.
- Add depth/shadow E2E coverage if the current harness cannot inspect those passes.

## Test-first sequence

1. Unit-test MaterialX mask extraction and absent-opacity behavior.
2. Unit-test manifest defaults, valid heads, malformed heads, and v1 compatibility.
3. Unit-test CPU head evaluation around the cutoff.
4. Add constant-zero, constant-one, and binary checker E2E tests.
5. Add antialiased edge and per-mip coverage tests.
6. Add depth, shadow, MSAA alpha-to-coverage, and automatic-LOD tests.
7. Enable joint training and measure BRDF and emission regression.

## Pass criteria

- V1 assets remain fully opaque and produce unchanged RGB.
- Constant controls classify every probe correctly.
- Resolved binary masks reach at least 0.99 intersection-over-union on the identity-frame plane.
- Antialiased fixtures stay below 2% continuous coverage MAE away from the screen rasterization boundary.
- Each coarse mip preserves mean teacher coverage within 1 percentage point.
- Color, depth, and shadow passes agree on retained fragments.
- Joint training increases BRDF or emission validation error by less than 5% relative. If it exceeds that limit, evaluate separate latent channels before freezing the format.

Treat the thresholds as initial gates and retain raw measurements for review.

## Non-goals

- Fractional alpha blending and order-dependent transparency.
- Refraction, physical transmission, or colored attenuation.
- Volumetric density.
- Learned view-dependent opacity.
