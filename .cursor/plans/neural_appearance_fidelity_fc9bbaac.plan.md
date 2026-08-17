---
name: Neural Appearance Fidelity
overview: Close the largest fidelity gaps between the browser trainer/runtime and NVIDIA’s SIGGRAPH 2024 method, prioritizing the 8×8 spatial bottleneck and correct footprint-filtered training before expanding the renderer.
todos:
  - id: resolution-baseline
    content: Measure artifact frequency and fidelity across substantially larger latent resolutions at fixed mip 0
    status: pending
  - id: filtered-training
    content: Implement Gaussian footprint-filtered teacher targets and paper-like mip sampling
    status: pending
  - id: encoder-bootstrap
    content: Add MaterialX parameter encoding, latent baking, and direct fine-tuning phases
    status: pending
  - id: angular-training
    content: Add directional mollification and reference-importance angular sampling
    status: pending
  - id: runtime-lod
    content: Implement unbiased adjacent-level stochastic LOD with non-UV-locked randomness
    status: pending
  - id: quality-gates
    content: Add supersampled footprint references and perceptual/spatial regression metrics
    status: pending
isProject: false
---

# Neural Appearance Fidelity Plan

## Assessment

The current implementation reproduces the paper’s basic BRDF-evaluation shape: an 8-channel hierarchical latent texture, two latent-dependent shading frames, a compact MLP, bilinear sampling within a selected mip, log-like L1 training, half-float latent export, and footprint-based mip selection. Recent code also trains the frame extractor and each mip independently, uses computed mesh tangents, rejects LDR teacher readback, and validates grazing angles—so older notes claiming those pieces are wholly absent are stale.

The dominant visible grid is expected from [`examples/webgpu_materials_neural_appearance_train.html`](examples/webgpu_materials_neural_appearance_train.html): the finest latent field defaults to 8×8 and the GUI caps it at 16×16. NVIDIA keeps the finest latent level at the source texture resolution, commonly 4K or higher. Linear filtering of 64 codes explains the block-shaped, interpolated appearance; it is not an antialiasing defect.

## Recommended work

1. **Establish resolution scaling and a fair visual baseline**
   - Extend [`examples/webgpu_materials_neural_appearance_train.html`](examples/webgpu_materials_neural_appearance_train.html) and [`examples/jsm/materials/NeuralAppearanceTrainer.js`](examples/jsm/materials/NeuralAppearanceTrainer.js) beyond the 8–16 texel demo regime, with explicit source-resolution/downsample controls and memory/time estimates.
   - Compare 8, 16, 32, 64, and higher latent resolutions on `neural_train_uv_grid.mtlx`, using fixed mip 0 first. This should verify that the visible grid frequency tracks latent resolution.
   - Keep the preview canvas’s `image-rendering: pixelated` clearly labeled as a sample-grid diagnostic so it is not mistaken for runtime filtering.

2. **Implement paper-equivalent appearance filtering**
   - Replace one teacher evaluation per UV/mip footprint with Gaussian spatial supersampling in [`examples/jsm/materials/NeuralAppearanceTeacherEvaluator.js`](examples/jsm/materials/NeuralAppearanceTeacherEvaluator.js), increasing sample count with filter area as described in Section 5.2.
   - Sample mip levels from an exponential distribution favoring fine levels rather than evaluating every mip equally for every direction pair.
   - Add LEAN-style or equivalent normal/roughness prefiltering for encoder inputs when normal maps are involved; ordinary texture derivatives/mipmaps do not produce a correctly filtered nonlinear BRDF.
   - Preserve independent latent grids per mip, which the current trainer now correctly optimizes.

3. **Add the encoder bootstrap phase**
   - Introduce a material-parameter encoder in [`examples/jsm/materials/NeuralAppearanceTrainer.js`](examples/jsm/materials/NeuralAppearanceTrainer.js), train encoder + decoder first, bake all texels/mips, then directly fine-tune latents.
   - Define a reliable MaterialX parameter extraction contract rather than the current `encodeInputs()` placeholder of UV plus canonical frame/directions.
   - Until the encoder exists, add a controlled neighboring-latent regularizer so equal or slowly varying materials do not retain unrelated random texel codes.
   - This is essential for scaling beyond tiny latent textures: the paper specifically shows direct latent optimization retaining initialization noise and assigning incompatible codes to equal appearances.

4. **Improve angular/highlight convergence**
   - Add directional mollification for narrow lobes: average teacher samples in a shrinking cone early in training.
   - Add reference-importance-sampled outgoing directions, not only the current uniform/cosine/mirror/Rusinkiewicz mixture.
   - Increase training budget substantially or move optimization to GPU compute. The browser default is roughly millions of BRDF records; the paper used about 40 billion over 300k iterations.
   - Evaluate a bounded positive output activation and smoother hidden activation with matching derivatives; retain FP32 master training and FP16 post-training validation.
   - Remove or formalize the current straight-through derivative for the per-channel non-negative output clamp, which can turn underfit angular regions into abrupt hue changes.

5. **Make runtime LOD selection match the paper**
   - In [`examples/jsm/materials/NeuralAppearanceNodeMaterial.js`](examples/jsm/materials/NeuralAppearanceNodeMaterial.js), keep bilinear filtering within a mip but replace nearest-level rounding with true Russian-roulette selection between adjacent levels.
   - Seed stochastic level choice from pixel/sample/frame state or blue noise, not a UV-locked sine hash, so variance can average instead of appearing as a stationary texture pattern.
   - Use projected footprint area, not only the maximum derivative axis, and test anisotropic/minified UV cases.

6. **Separate raster-preview scope from paper parity**
   - Document that the current runtime is direct-light rasterization only. The paper’s showcased results use a multi-bounce path tracer, usually high-SPP reference imagery, with `eval`, `sample`, and `pdf` hooks.
   - Treat the missing learned anisotropic GGX importance sampler, PDF evaluation, indirect/environment lighting, auxiliary albedo, and tensor-core/fused execution as separate parity gaps. They matter for paper-level rendering and performance but do not cause the present 8×8 grid artifact.
   - Tighten supported MaterialX diagnostics for transmission, position-dependent nodes, multiple normal layers, and other state that the atlas teacher cannot abstract faithfully.

7. **Add quality gates**
   - Extend [`test/e2e/neural-appearance-training.js`](test/e2e/neural-appearance-training.js) with spatial-frequency sweeps, forced-mip comparisons, UV-seam/minification views, and linear-HDR metrics before ACES tone mapping.
   - Add focused WebGPU tests for the real MaterialX teacher: UV variation, incoming/outgoing direction sensitivity, known constant materials, and agreement among teacher, serialized CPU evaluation, and the GPU runtime shader.
   - Compare filtered neural mips to supersampled teacher references at matching footprints, not merely screenshots of the normal rasterized teacher.
   - Track FLIP or SSIM alongside mean RGB error, angular-bin error, highlight peak/width, reciprocity, and seed variance; gate these metrics rather than only checking that they were reported.

## Priority

Address resolution scaling first, then filtered teacher targets, then encoder bootstrapping. Those three explain most of the quality difference visible in the paper. Importance sampling and path tracing are major feature gaps, but they will not remove a spatial 8×8 interpolation pattern in the current direct-light preview.

Sources: [SIGGRAPH 2024 paper](https://arxiv.org/html/2305.02678v2), [official project page](https://tizianzeltner.com/projects/Zeltner2024RealtimeNeural/), and [NVlabs reference implementation](https://github.com/NVlabs/neuralappearance).