---
name: Neural Appearance IBL
overview: Add environment lighting through a validation-first, GPU-trained compact PMREM approximation. The design replaces the unreleased format with an integrated BRDF and IBL contract and gates the optimized head on a brute-force integration baseline.
todos:
  - id: ibl-reference
    content: Build brute-force environment integration and white-furnace validation gates
    status: completed
  - id: ibl-targets
    content: Generate and validate constrained diffuse/specular lobe targets
    status: completed
  - id: ibl-gpu-head
    content: Train the 14→32→13 IBL head in a frozen GPU phase
    status: completed
  - id: ibl-runtime
    content: Wire learned two-sample PMREM lighting into the neural material
    status: completed
  - id: ibl-format-tests
    content: Replace the format, add an environment-lit main example, tests, and documentation
    status: in_progress
isProject: false
---

# Neural Appearance IBL Support

## Verified design direction

- Keep the strong parts of the sketch: a view-conditioned head, three.js PMREM reuse, a white-furnace constraint, and a brute-force reference before optimization.
- Replace the original underdetermined `specScale`/`specBias` split and unsupervised frame-0 diffuse normal with an identifiable 13-value contract:
  - `diffuseDirection` (3, normalized)
  - `diffuseReflectance` (3, nonnegative)
  - `specularDirection` (3, normalized)
  - `specularRoughness` (1, sigmoid)
  - `specularWeight` (3, nonnegative)
- Evaluate
  `PMREM(diffuseDirection, 1) * diffuseReflectance + PMREM(specularDirection, specularRoughness) * specularWeight`.
  Under a unit-white environment both PMREM samples are one, so the exact anchor becomes `diffuseReflectance + specularWeight = directionalAlbedo`. White-furnace constrains total energy but does not determine the split; constrained angular-kernel fitting supplies that supervision.
- Use a 14-input head (8 latents + view direction projected into both learned frames). Keep the existing BRDF/latents/frame weights frozen during the second IBL training phase so direct-light fidelity cannot regress.

## 1. Establish the reference and acceptance gates

- Add deterministic cosine-hemisphere integration utilities around the existing cosine-factored BRDF evaluator in [examples/jsm/neural/NeuralAppearanceTSL.js](examples/jsm/neural/NeuralAppearanceTSL.js) and its CPU validation mirror in [examples/jsm/neural/NeuralAppearanceRuntime.js](examples/jsm/neural/NeuralAppearanceRuntime.js).
- Provide a debug/reference environment path using the same scene PMREM and transformed canonical directions: 16 samples for interactive preview, 128–512 for validation. Treat 16 samples as a smoke test, not ground truth.
- Extend [examples/jsm/neural/NeuralAppearanceValidator.js](examples/jsm/neural/NeuralAppearanceValidator.js) with directional albedo, white-furnace residual, environment-integral error, and view-angle/mip bins.
- Compare the integrated neural BRDF against environment-only renders of the original MaterialX teacher. If this baseline is poor, stop and improve direct BRDF training before adding the compact head.

## 2. Generate identifiable IBL targets

- Add grouped `(uv, materialMip, wo)` queries to [examples/jsm/neural/NeuralAppearanceSampler.js](examples/jsm/neural/NeuralAppearanceSampler.js), each with a deterministic set of incident directions and quadrature weights.
- Reuse the GPU teacher atlas in [examples/jsm/neural/NeuralAppearanceTeacherEvaluator.js](examples/jsm/neural/NeuralAppearanceTeacherEvaluator.js) to obtain the angular response kernel; aggregate directional albedo and moments without reintroducing a CPU optimization backend.
- Fit the kernel to a normalized Lambert/cosine lobe plus one PMREM-compatible specular lobe, with nonnegative RGB weights and calibrated moment-spread-to-three.js-roughness mapping. Use known Lambert/GGX materials to test calibration and define deterministic tie-breaking for broad overlapping lobes.
- Store fitted directions, roughness, weights, fit residual, and directional albedo as IBL training records. Reject or flag poor one-lobe fits; keep the manifest representation extensible so a second specular lobe can be added later only if anisotropic, sheen, or clearcoat fixtures demonstrate the need.

## 3. Add a frozen second GPU training phase

- Extend [examples/jsm/neural/NeuralAppearanceModel.js](examples/jsm/neural/NeuralAppearanceModel.js) with a 14→32→13 IBL head and structured output transforms; reuse the existing learned frames but do not treat either frame normal as a physical diffuse normal.
- Add IBL weight/activation/sample ranges in [examples/jsm/neural/NeuralAppearanceGPUModel.js](examples/jsm/neural/NeuralAppearanceGPUModel.js), and dedicated forward/backward and Adam-range compute nodes in [examples/jsm/neural/NeuralAppearanceGPUCompute.js](examples/jsm/neural/NeuralAppearanceGPUCompute.js). Reuse `backwardNormalizeTSL` for both directions and add explicit sigmoid/nonnegative derivatives.
- Run this after the existing BRDF phase in [examples/jsm/neural/NeuralAppearanceTrainer.js](examples/jsm/neural/NeuralAppearanceTrainer.js), updating only IBL-head parameters. Combine supervised direction/roughness/weight losses with the white-furnace residual, and report direct validation before and after as a freeze regression check.

## 4. Integrate scene environments and PMREM at runtime

- Add IBL evaluation and canonical-to-world direction transforms in [examples/jsm/neural/NeuralAppearanceTSL.js](examples/jsm/neural/NeuralAppearanceTSL.js), sharing latent LOD/trilinear behavior with the BRDF path and evaluating once per fragment.
- Override `setupEnvironment()` in [examples/jsm/neural/NeuralAppearanceNodeMaterial.js](examples/jsm/neural/NeuralAppearanceNodeMaterial.js) so `material.envNode`, `material.envMap`, and `scene.environment` follow the same precedence as `MeshStandardNodeMaterial`.
- Implement a neural environment lighting node modeled on [src/nodes/lighting/EnvironmentNode.js](src/nodes/lighting/EnvironmentNode.js), but with the learned diffuse/specular directions and roughness supplied through custom `pmremTexture().context()` calls. This preserves PMREM caching, roughness mapping, environment rotation, Y-flip, and scene/material intensity semantics.
- Accumulate the two-sample result in `NeuralAppearanceLightingModel.indirect()`. Keep direct lights, emission, opacity, and AO behavior unchanged; when the scene has no environment, the integrated IBL head simply contributes nothing.

## 5. Version, demo, documentation, and verification

- Replace the unreleased manifest contract in [examples/jsm/neural/NeuralAppearanceFormat.js](examples/jsm/neural/NeuralAppearanceFormat.js) and [examples/jsm/neural/NeuralAppearanceManifest.js](examples/jsm/neural/NeuralAppearanceManifest.js) with a single new format version in which `outputs.brdf` and the structured `outputs.ibl` are required. Update [examples/jsm/loaders/NeuralAppearanceLoader.js](examples/jsm/loaders/NeuralAppearanceLoader.js), bundled neural assets, checkpoint conversion, runtime data, and tests directly; do not retain v1/v2 parsing or migration branches.
- Extend CPU validation output evaluation, packed runtime uniforms, hot-update compatibility, and reference evaluations for the optional head.
- Make IBL clearly visible in the main [examples/webgpu_materials_neural_appearance.html](examples/webgpu_materials_neural_appearance.html) workflow:
  - Load a reusable HDR environment and assign it to `scene.environment` so both the MaterialX teacher and neural material use the same PMREM source.
  - Add environment intensity and rotation controls, plus independent direct-light and environment toggles.
  - Add comparison modes for MaterialX teacher, brute-force neural reference, and compact neural IBL, with environment-only as the clearest default IBL demonstration and mixed direct+IBL as a secondary view.
  - Preserve the existing training/test query parameters and expose stable state hooks so e2e tests can select each lighting and comparison mode deterministically.
- Expand unit coverage for the replacement format, required IBL validation, 14-input projection, output transforms, GPU offsets/sample serialization, normalize gradients, frozen BRDF weights, white-furnace identity, and PMREM environment precedence. Extend [test/e2e/neural-appearance-training.js](test/e2e/neural-appearance-training.js) with diffuse, glossy dielectric, metallic, normal-map, and multi-lobe environment-lit comparisons.
- Update [.cursor/plans/neural_appearance_implementation_report.md](.cursor/plans/neural_appearance_implementation_report.md) to describe GPU-only training, the new IBL path, measured limitations, and the one-lobe-to-two-lobe decision result.