# Neural Appearance Object-Space 3D Surface Textures

## Goal

Choose a representation for MaterialX appearances driven by local or object-space position. Prove the current teacher limitation first, then compare a mesh-specific 2D bake with a portable 3D latent field.

## Current limitation

MaterialX procedural nodes such as `noise3d` compile against `positionLocal` or `positionWorld`. `NeuralAppearanceTeacherEvaluator` overrides UV, shading frame, view direction, and light direction on an atlas quad. It does not override local or world position.

The current asset indexes a 2D latent field by UV. It cannot distinguish two surface points that share UV but require different object-space values. A 3D procedural graph may produce an atlas-dependent target instead of the intended material response.

## Diagnostic fixture

Add `examples/materialx/neural_train_position_noise3d.mtlx` with:

- A low-frequency, deterministic object-space pattern whose expected value can be computed at probe positions.
- Axis-colored controls derived from local X, Y, and Z where MaterialX permits them.
- A diffuse-only surface.

Add teacher samples containing explicit `positionLocal` values. Before adding any override, show that changing the requested position does not change the teacher output as expected. The test should fail with a message that names the missing geometry semantic.

Then add a scoped `positionLocal` override to the teacher context and verify the fixture. Add `positionWorld` only if the product contract requires world-anchored materials; world space introduces object-transform and portability concerns.

## Design gate

Do not implement runtime storage until measurements and product requirements select one option.

### Option A: mesh-specific UV bake

Bake object-space MaterialX values into the target mesh's UV atlas, then train the existing 2D latent hierarchy.

Advantages:

- Reuses v1 latent textures, filtering, loader, and runtime.
- Keeps storage proportional to surface area rather than volume.
- Supports current WebGPU texture paths.

Costs:

- Asset becomes mesh and UV-layout specific.
- Overlaps need identical object-space values or must be rejected.
- Seams, chart padding, and texel density become bake inputs.
- Deformation invalidates a bake tied to undeformed local position.

### Option B: portable 3D latent field

Store a 3D latent volume sampled by normalized `positionLocal`.

Advantages:

- Preserves object-space procedural continuity without UV charts.
- One material can work across meshes that share a coordinate contract.

Costs:

- Dense storage grows cubically.
- Empty volume wastes memory for surface-only data.
- 3D footprints and mip selection require three-axis derivatives.
- Every asset needs bounds, normalization, wrap behavior, and out-of-bounds semantics.
- Object scaling changes effective frequency unless the coordinate contract handles it.
- The format, loader, CPU evaluator, checkpoint tools, and TSL runtime all change.

## Required measurements

For representative sphere, torus, and thin-shell meshes, record:

- UV atlas occupancy and seam count.
- 2D latent bytes at target surface texel density.
- Dense 3D latent bytes at matching smallest feature size.
- Teacher and runtime error near UV seams or volume boundaries.
- Training samples and time to reach the same linear-HDR error.
- Behavior under rigid transforms, non-uniform scale, and deformation.

Include one fixture with overlapping UVs that disagree in object space. Option A must reject or quantify it.

## Coordinate contract

Whichever option wins must define:

- Local versus world coordinate space.
- Asset bounds and normalization.
- Units and behavior under object scale.
- Wrap, clamp, and border semantics.
- Static, skinned, and morphed mesh behavior.
- Derivative source and anisotropic footprint reduction.
- Whether one asset can be shared by different meshes.

Store these fields in a versioned manifest for Option B. Store mesh identity, UV-set identity, and bake bounds as metadata for Option A.

## Implementation path after the gate

### Shared teacher work

- Extend sample records in `NeuralAppearanceTrainer.js` with position and position derivatives.
- Add position sample textures and scoped geometry overrides in `NeuralAppearanceTeacherEvaluator.js`.
- Add unit tests for override isolation and nested MaterialX graph use.
- Keep UV replacement active for graphs that combine 2D and 3D coordinates.

### Option A work

- Add a mesh bake entry point that renders or samples chart-aware UV targets.
- Detect UV overlap conflicts and insufficient chart padding.
- Train the existing independent mip grids against chart-filtered targets.
- Store mesh/bake metadata without changing decoder input size.

### Option B work

- Version the format and define 3D latent mip payloads.
- Add `Data3DTexture` loading and sampling.
- Add trilinear CPU evaluation and gradients to eight neighboring voxels.
- Define 3D LOD from position derivatives.
- Benchmark dense versus sparse/bricked storage before accepting dense 3D textures.

## Tests

- Unit: MaterialX position semantic selection and teacher override scoping.
- Unit: position normalization, bounds, wrap, and footprint math.
- Unit: manifest compatibility and malformed 3D payload rejection if Option B wins.
- E2E: object-space fixture on two meshes and under rigid transforms.
- E2E: non-uniform scale behavior according to the chosen contract.
- E2E: combined UV color and object-space procedural input.

## Gate criteria

Select Option A when mesh-specific assets are acceptable and its measured storage is at least four times smaller at equal error.

Select Option B only when material portability across UV layouts is required and a tested storage scheme fits the target GPU budget. A nominal dense volume estimate is not sufficient.

If neither option meets the product requirement, keep object-space MaterialX graphs unsupported and fail before training with the first unsupported node and coordinate space.

## Non-goals

- Participating media, density fields, or volume scattering.
- Displacement geometry.
- World-space effects that change when an object moves, unless selected as a separate product requirement.
- Silent fallback to atlas-quad positions.
