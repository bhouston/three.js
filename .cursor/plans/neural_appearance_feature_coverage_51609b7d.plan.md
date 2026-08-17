---
name: neural appearance feature coverage
overview: Documents the current representation limits and splits normal maps, repeating color, object-space 3D textures, emission, and cutout opacity into independent test-first implementation plans. The six implementation plans are complete and ready for separate review.
todos:
  - id: shared-validation
    content: Write the shared spatial fidelity and diagnostic plan.
    status: completed
  - id: normal-plan
    content: Write the UV normal-map test and implementation plan.
    status: completed
  - id: color-plan
    content: Write the repeating color-texture test and implementation plan.
    status: completed
  - id: object-space-plan
    content: Write the object-space 3D texture design and test plan.
    status: completed
  - id: emission-plan
    content: Write the texture-based emission output-head plan.
    status: completed
  - id: opacity-plan
    content: Write the cutout opacity output-head plan.
    status: completed
isProject: false
---

# Neural Appearance Feature Coverage Plans

## Produced plans
- [Shared spatial validation](neural_appearance_spatial_validation.md)
- [UV normal maps](neural_appearance_normal_maps.md)
- [Repeating color textures](neural_appearance_repeating_color.md)
- [Object-space 3D surface textures](neural_appearance_object_space_3d.md)
- [Texture-based emission](neural_appearance_emission.md)
- [Cutout opacity](neural_appearance_cutout_opacity.md)

## Current diagnosis
- The current asset is a **2D UV-indexed, opaque, reflective BRDF**: 8 latent channels plus two learned directional frames and a 20→H→H→3 RGB decoder.
- It understands **local tangent space for directions**, via the mesh TBN and learned shading frames. It has no local/object-space position input and no 3D latent volume.
- A UV normal map is representable in principle because the MaterialX teacher’s normal node changes the sampled BRDF and learned frames can encode local lobe rotation. Repetition and high-frequency detail fail when the latent grid/sample budget is too small; current defaults are only 8².
- A repeating color texture is also representable as spatial BRDF color. The existing UV-grid E2E case tests only 8² with loose image thresholds, so it does not establish robust texture fidelity.
- Emission and opacity require independent spatial outputs. Adding alpha to the existing per-light BRDF output would decode it once per light and allow direction dependence. Use latent-only heads evaluated once per fragment instead.

## Shared validation plan
Create [`neural_appearance_spatial_validation.md`](neural_appearance_spatial_validation.md):
- Add linear-HDR teacher/neural comparisons by UV, direction, mip, and spatial frequency.
- Sweep source resolution, latent resolution/downsample, repeats, iterations, and hidden width.
- Add per-mip, grazing, highlight-location, and CPU↔WebGPU parity metrics.
- Establish whether failures come from teacher sampling, latent bandwidth, optimization, or runtime evaluation before changing the format.

## Independent feature plans

### 1. UV normal maps
Create [`neural_appearance_normal_maps.md`](neural_appearance_normal_maps.md):
- Add a MaterialX tangent-space normal-map fixture with controlled sinusoidal/checker normals and explicit UV repetition.
- Verify that highlight direction follows the teacher across rotations and that mip filtering broadens unresolved normal variation.
- Test the current representation first; only change frame/filter training if the test isolates a representation or optimization failure.
- Inspect the unused LEAN helper and decide whether to integrate normal moments into teacher target generation.

### 2. Repeating color textures
Create [`neural_appearance_repeating_color.md`](neural_appearance_repeating_color.md):
- Replace the single loose UV-grid case with explicit 1×, 2×, 4×, and 8× repeat fixtures.
- Test diffuse-only color so angular/specular errors cannot hide spatial reconstruction errors.
- Define pass criteria in linear HDR and by UV-frequency band.
- Document the expected Nyquist relationship between repeat count and latent resolution.

### 3. Object/local-space 3D surface textures
Create [`neural_appearance_object_space_3d.md`](neural_appearance_object_space_3d.md):
- Add a diagnostic MaterialX `position`/`noise3d` fixture and prove that the current atlas teacher cannot supply meaningful object-space positions.
- Compare two designs: mesh-specific bake into a UV atlas versus a portable material with a 3D latent texture indexed by normalized `positionLocal`.
- Use a design gate before implementation because storage, mip/footprint rules, object transforms, seams, and portability differ substantially.
- Keep true participating-media density/scattering out of this plan.

### 4. Texture-based emission
Create [`neural_appearance_emission.md`](neural_appearance_emission.md):
- Add MaterialX constant and repeating emissive fixtures, including emission with zero scene lights.
- Extend the format with a latent-only RGB emission head or justify direct emissive texels after measuring quality/cost.
- Add a teacher path that isolates `emissiveNode` without cosine normalization.
- Connect the decoded value to NodeMaterial emission once per fragment, independent of direct lights.

### 5. Cutout opacity
Create [`neural_appearance_cutout_opacity.md`](neural_appearance_cutout_opacity.md):
- Add binary and antialiased repeating opacity-mask fixtures.
- Extend the format with a latent-only scalar opacity head evaluated during material setup, before lighting.
- Route it through `opacityNode`/`alphaTestNode`, preserving the opaque depth pipeline for cutouts.
- Test silhouette coverage, alpha-to-coverage behavior, mip stability, and teacher/neural mask agreement.
- Leave fractional blending and physical transmission for later plans.

## Plan boundaries
- Keep the BRDF decoder RGB-only and per-light.
- Share the 8D latent code initially, but measure interference between BRDF, emission, and opacity before fixing the final format.
- Version the manifest when adding output heads; retain v1 loader compatibility.
- Do not combine all features into one implementation PR. Land shared diagnostics first, then color/normal characterization, then emission and cutout, with 3D coordinates behind its design gate.