# Neural Appearance Repeating Color Textures

## Goal

Measure how accurately the v1 UV latent hierarchy reproduces repeating diffuse color and establish the relationship between source frequency, latent resolution, training budget, and mip footprint.

## Scope

Use diffuse-only MaterialX materials. Constant roughness and zero specular prevent angular error from hiding spatial reconstruction error. Keep the BRDF decoder, latent channel count, and asset format unchanged during characterization.

## Fixtures

Replace the single ambiguous UV-grid case with controlled fixtures:

- A two-color checker at 1×, 2×, 4×, and 8× repeats.
- Horizontal and vertical sinusoidal color ramps at matching frequencies.
- A non-periodic border marker to detect wrap mistakes separately from reconstruction quality.
- Constant-color controls for both checker colors.

Store textures in linear color space or generate values in MaterialX. Record exact RGB values, UV convention, wrap mode, and expected value at probe UVs.

Use separate fixture names in the training example and E2E harness. A query parameter may select repeat count if one MaterialX graph can expose it without adding hidden runtime state.

## Sampling and metrics

Generate a stratified UV probe grid dense enough to sample each source period at least eight times. Jittered training samples must remain separate from fixed validation probes.

Report:

- Linear-RGB MAE, relative error, and maximum channel error.
- Error by UV-frequency band and mip.
- Checker classification accuracy away from filtered boundaries.
- Edge position and transition-width error.
- Seam versus interior error.
- Mean predicted color to detect energy bias.
- FP32 model versus serialized FP16 runtime error.

Do not use tone-mapped screenshots as the pass criterion. Keep them as a visual regression check.

## Nyquist contract

Define `sourceCycles` as pattern cycles across UV `[0, 1]` and `latentResolution` as base-level texels per axis.

Classify cases before training:

- Resolved: at least four latent texels per cycle.
- Boundary: between two and four latent texels per cycle.
- Unresolved: fewer than two latent texels per cycle.

The exact quality boundary depends on nonlinear latent decoding. Use these groups to explain results, not to claim guaranteed reconstruction. For unresolved levels, compare against the footprint-filtered teacher rather than the base texture.

## Implementation sequence

### 1. Fixture and loader checks

Add the fixtures under `examples/materialx/` and unit-test their semantic UV and color-space assumptions where loader tests can inspect the compiled nodes.

Verify repeat wrapping in:

- MaterialX image sampling.
- CPU latent bilinear sampling.
- Exported texture metadata.
- `NeuralAppearanceLoader`.
- WebGPU latent sampling.

### 2. Structured validation

Use the shared spatial validation generator and metrics. Add fixture metadata for repeat count, source phase, expected class, and boundary distance.

Test fixed mip levels before automatic LOD. Validate each independently learned mip against a teacher footprint matching that level.

### 3. Training sweep

Run the small committed matrix:

- Repeats: 1, 4, 8.
- Latent resolution: 8 and 16.
- Fixed mip: base and one coarse level.
- Seeds: one routine CI seed and three local diagnostic seeds.

Run larger iteration, hidden-width, and latent-downsample sweeps outside routine CI. Save machine-readable results, not generated assets.

### 4. Runtime LOD

After fixed mips pass, compare:

- Deterministic nearest mip.
- Existing UV-hash stochastic adjacent mip.
- A train-for-trilinear candidate if the shared validation plan selects it.

Judge automatic LOD against a teacher filtered with the same footprint scalar. Do not accept a method based only on reduced screenshot noise.

## Files

- Add controlled fixtures under `examples/materialx/`.
- Modify fixture selection in `examples/webgpu_materials_neural_appearance_train.html`.
- Modify `test/e2e/neural-appearance-training.js`.
- Extend shared metrics in `examples/jsm/materials/NeuralAppearanceTrainer.js` only when fixture metadata requires it.
- Add unit coverage in `test/unit/addons/materials/NeuralAppearanceTrainer.tests.js` and, where applicable, `test/unit/addons/loaders/MaterialXLoader.tests.js`.

## Pass criteria

- Constant controls meet the shared linear-HDR tolerance at every mip.
- Resolved checker interiors classify the correct color for at least 99% of probes.
- Resolved sinusoidal fixtures stay below 3% mean absolute error relative to their channel range at mip 0.
- Seam error does not exceed interior error by more than the measured half-float tolerance for periodic fixtures.
- Coarse-mip edge width and mean color agree with the footprint-filtered teacher within 5% of the channel range.
- An unresolved fixture must fail as a labeled bandwidth limit or match its filtered target. It must not pass through a loose whole-image average.

Treat numerical thresholds as starting gates and record the baseline before tightening them.

## Decision outcome

Document:

- The minimum latent texels per source cycle that meets the test gates.
- The iteration and hidden-width sensitivity inside the resolved region.
- Whether failures first appear in teacher filtering, optimization, serialization, or WebGPU runtime.
- The selected runtime LOD contract.

Use those measurements to set example defaults. Do not raise default latent resolution without reporting training memory and time.

## Non-goals

- Glossy or anisotropic texture variation.
- Normal maps.
- UDIM or non-periodic virtual textures.
- Encoder bootstrap.
- Asset-format changes.
