# Neural Appearance Material Implementation in Three.js WebGPU

**Date:** August 2026  
**Status:** Implemented in `examples/jsm/neural/`, `examples/jsm/loaders/NeuralAppearanceLoader.js`, and `examples/webgpu_materials_neural_appearance.html`

---

## 1. Executive Summary

This report analyzes the neural appearance material system implemented in Three.js WebGPU. The system compresses complex, multi-node MaterialX surface representations into a compact, real-time neural material asset evaluated entirely inside WebGPU shaders using Three.js Shading Language (TSL).

The implementation consists of four primary components:
1. **Asset Format & Loader (`NeuralAppearanceFormat.js`, `NeuralAppearanceLoader.js`):** A JSON specification storing an 8-channel latent texture mip pyramid as two half-float (`RGBA16F`) textures, a learned shading frame transformation, compact MLP weights for BRDF and IBL, plus optional emission and cutout opacity heads.
2. **GPU Teacher Evaluator (`NeuralAppearanceTeacherEvaluator.js`, `NeuralAppearanceTeacherAtlas.js`, `NeuralAppearanceTeacherReadback.js`):** An offscreen WebGPU rendering engine that samples arbitrary MaterialX shader graphs over spatial coordinates, directional hemisphere pairs, and Gaussian footprint-filtered footprints.
3. **In-Browser Distillation Trainer (`NeuralAppearanceTrainer.js`, `NeuralAppearanceModel.js`, `NeuralAppearanceMLP.js`, `NeuralAppearanceSampler.js`):** An optimization engine running in JavaScript and WebGPU that jointly trains latent mip grids, shading-frame weights, and MLP decoders using Adam optimization and a power-transformed loss.
4. **Real-Time Runtime Material (`NeuralAppearanceNodeMaterial.js`, `NeuralAppearanceTSL.js`):** A custom `NodeMaterial` and `LightingModel` evaluating neural BRDFs per direct light, a two-sample PMREM IBL approximation per fragment, and auxiliary heads (emission, opacity) within standard raster passes.

---

## 2. System Architecture and How It Works

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   OFFLINE / IN-BROWSER TRAINING                        │
│                                                                                        │
│  ┌──────────────────────┐         ┌────────────────────────┐                           │
│  │ MaterialX Definition │ ──────> │  WebGPU GPU Teacher    │                           │
│  │ (MeshPhysicalNode)   │         │  (Atlas Render Target) │                           │
│  └──────────────────────┘         └───────────┬────────────┘                           │
│                                               │                                        │
│                                    Directional / Spatial Samples                       │
│                                    (UV, wi, wo, Footprint, Target HDR)                 │
│                                               │                                        │
│                                               ▼                                        │
│                                   ┌────────────────────────┐                           │
│                                   │ NeuralAppearanceTrainer│                           │
│                                   │ • Multi-mip Latents    │                           │
│                                   │ • Learned Frame Matrix │                           │
│                                   │ • MLP Decoders         │                           │
│                                   └───────────┬────────────┘                           │
│                                               │ Export                                 │
└───────────────────────────────────────────────┼────────────────────────────────────────┘
                                                │
                                                ▼
                                    ┌────────────────────────┐
                                    │ Neural Material JSON   │
                                    │ (Format v3 / 8D Mips)  │
                                    └───────────┬────────────┘
                                                │
┌───────────────────────────────────────────────┼────────────────────────────────────────┐
│                                   RUNTIME INFERENCE (WebGPU / TSL)                     │
│                                               │                                        │
│                                   ┌───────────▼────────────┐                           │
│                                   │ NeuralAppearanceLoader │                           │
│                                   │ (2x RGBA16F Textures)  │                           │
│                                   └───────────┬────────────┘                           │
│                                               │                                        │
│                                   ┌───────────▼──────────────────┐                     │
│                                   │ NeuralAppearanceNodeMaterial │                     │
│                                   └───────────┬──────────────────┘                     │
│                                               │                                        │
│                ┌──────────────────────────────┼──────────────────────────────┐         │
│                │ Per-Fragment Setup           │ Direct Light Callback        │         │
│                │                              │ (per light)                  │         │
│                ▼                              ▼                              ▼         │
│      ┌──────────────────┐           ┌──────────────────┐           ┌─────────────────┐ │
│      │ Cutout Opacity   │           │ Decoded Emission │           │ Neural BRDF     │ │
│      │ (alphaTestNode)  │           │ (emissiveNode)   │           │ × max(wi.z, 0)  │ │
│      └──────────────────┘           └──────────────────┘           └─────────────────┘ │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 The Representation and Asset Format

The material format (`three-neural-appearance` v3) stores surface properties in a factorized neural representation:

1. **8-Dimensional Latent Texture Pyramid:**
   - Eight continuous channels partitioned across two `DataTexture` instances (`RGBAFormat`, `HalfFloatType`).
   - Includes complete power-of-two mip hierarchies down to $1\times1$.
   - Quantized to IEEE 754 half-precision floats (16-bit float per channel), reducing texture memory to 16 bytes per texel at base resolution.

2. **Dual Learned Shading Frames:**
   - An affine mapping matrix ($8 \to 12$) transforming the 8D latent code into two coordinate frames:
     $$\mathbf{n}_k = \text{normalize}\left(\mathbf{W}_{n,k} \mathbf{z} + [0, 0, 1]^T\right), \quad \mathbf{t}_k = \text{normalize}\left(\mathbf{W}_{t,k} \mathbf{z} + [1, 0, 0]^T\right), \quad \mathbf{b}_k = \mathbf{n}_k \times \mathbf{t}_k$$
   - These frames capture anisotropic highlights, microscopic normal distributions, and clearcoat tilt without storing explicit normal maps.

3. **BRDF MLP Decoder:**
   - Input dimension: 20 floats (8 latent values + 6 projected directional components $\mathbf{w}_i, \mathbf{w}_o$ for Frame 1 + 6 components for Frame 2).
   - Architecture: 2 hidden layers (default 16 to 64 units each) with ReLU activations.
   - Output: 3 linear RGB values representing the cosine-factored bidirectional reflectance distribution function.

4. **IBL Head:**
   - A required $14 \to 32 \to 13$ view-conditioned MLP consumes the 8D latent code plus the view direction projected into both learned frames.
   - It predicts normalized diffuse and specular PMREM query directions, nonnegative diffuse/specular RGB weights, and a sigmoid roughness in the same perceptual space used by three.js PMREM sampling.
   - The white-furnace identity `diffuseReflectance + specularWeight == directionalAlbedo` is tracked as a validation metric.
5. **Auxiliary Output Heads:**
   - **Emission Head:** An $8 \to 3$ MLP decoding unlit linear HDR radiance $\mathbf{L}_e$ once per fragment.
   - **Cutout Opacity Head:** An $8 \to 1$ MLP with sigmoid activation outputting scalar coverage $a \in [0, 1]$ evaluated against `alphaCutoff`.

### 2.2 Teacher Evaluation Pipeline

The teacher evaluator (`NeuralAppearanceTeacherEvaluator`) distills the full shader node graph of `MeshPhysicalNodeMaterial`:

- **Offscreen Batch Atlas:** Packs training queries into a half-float render target using an orthographic plane mesh.
- **Shader Graph Overrides:** Replaces standard geometry inputs (`uv`, `normalView`, `tangentView`, `bitangentView`, `positionViewDirection`) with atlas-driven data textures using TSL context overrides.
- **Directional Light Simulation:** Injects incoming test direction $\mathbf{w}_i$ via `NeuralTeacherLightingModel` with unit white radiance, suppressing indirect lighting.
- **Gaussian Footprint Filtering:** For coarser mip levels, evaluates a footprint area $\det(\nabla \text{UV}) \cdot \text{res}^2$ and super-samples the tile with a Gaussian-weighted spatial kernel ($1$ to $64$ samples) matching the target mip's pixel coverage.
- **Strict Readback Contract:** Uses asynchronous half-float pixel readback (`readRenderTargetPixelsAsync`). Rejects low dynamic range fallbacks to prevent clipping HDR highlights.

### 2.3 Training and Optimization Engine

The browser-side trainer (`NeuralAppearanceTrainer`) optimizes the network and latent textures:

- **Stratified Sampling:** Generates stratified jittered UV samples paired with a mixture of direction distributions:
  - Uniform hemisphere ($35\%$)
  - Cosine-weighted hemisphere ($20\%$)
  - Uniform sphere ($10\%$)
  - Specular mirror pairs ($15\%$)
  - Rusinkiewicz power-sampled half-angle microfacet reflections ($20\%$)
- **Cosine Target Normalization:** Converts teacher outgoing radiance $\mathbf{L}_o$ into BRDF target $\mathbf{f}_r = \mathbf{L}_o / \max(\mathbf{w}_i \cdot \mathbf{n}, 10^{-4})$, weighting samples by $\mathbf{w}_i \cdot \mathbf{n} \cdot (1 + \beta \frac{\text{peak}}{1 + \text{peak}})$ to prioritize specular highlights.
- **Loss Formulation:** Computes generalized logarithmic L1 loss using a cube-root power transform:
  $$\mathcal{L}(\hat{y}, y) = |\phi_3(\hat{y}) - \phi_3(y)|, \quad \text{where } \phi_p(x) = p(x^{1/p} - 1)$$
  This stabilizes training across high-dynamic-range peaks without vanishing gradients on darker diffuse surfaces.
- **Joint Optimization:** Uses Adam optimization with cosine learning rate decay and gradient norm clipping ($||\mathbf{g}|| \le 1.0$) across direct-light weights, rotation parameters, and multi-resolution latent grids. The IBL head is fitted afterward from cached teacher angular kernels while the direct-light representation is held fixed.

### 2.4 WebGPU Shader Runtime (TSL)

At runtime, `NeuralAppearanceNodeMaterial` integrates into the Three.js WebGPU pipeline:

1. **Tangent Space Construction:** Derives an orthonormal TBN matrix from mesh vertex tangents (`geometry.computeTangents()`) or Gram-Schmidt orthogonalized screen-derivative fallbacks.
2. **LOD Computation:** Measures screen-space UV derivatives via `dFdx(uv)` and `dFdy(uv)`:
   $$\text{footprint} = \max\left(||\frac{\partial \text{UV}}{\partial x} \cdot \text{size}||, ||\frac{\partial \text{UV}}{\partial y} \cdot \text{size}||\right), \quad \text{lod} = \text{clamp}\left(\log_2(\text{footprint}), 0, \text{mips} - 1\right)$$
3. **Latent Fetch:** Bilinearly samples the two `RGBA16F` textures at the computed LOD level.
4. **Shading Frame Evaluation:** Evaluates the $8 \to 12$ linear rotation uniform matrix and normalizes the two basis coordinate frames.
5. **Direct Light Integration:** Inside `NeuralAppearanceLightingModel.direct()`, projects view vector $\mathbf{w}_o$ and light vector $\mathbf{w}_i$ into both frames, executes the MLP decoder via packed 4-vector uniform dot products, and multiplies the result by $\max(\mathbf{w}_i \cdot \mathbf{n}, 0) \cdot \mathbf{C}_{\text{light}}$.
6. **IBL Integration:** `setupEnvironment()` follows `MeshStandardNodeMaterial` environment precedence, then a neural environment lighting node samples the PMREM at learned diffuse and specular directions and accumulates indirect light once per fragment.
7. **Auxiliary Passes:** Assigns decoded emission to `emissiveNode` and decoded opacity to `opacityNode` / `alphaTestNode`.

---

## 3. What the Neural Material Captures

The implementation captures complex optical behavior in a fixed-cost shader:

| Surface Property | Capture Mechanism | Fidelity Characteristics |
| :--- | :--- | :--- |
| **Diffuse Reflectance & Albedo** | 8D latent code + MLP base outputs | Exact color matching; smooth transitions under varying illumination. |
| **Specular Roughness & Metallic** | Directional MLP mapping + power-loss fitting | Recreates isotropic and anisotropic specular lobes from dielectric and metallic surfaces. |
| **Multi-Lobe & Clearcoat Effects** | Dual learned local frames | Replaces multi-layer BSDF evaluations (base + clearcoat + sheen) with a single MLP pass. |
| **Normal Maps & Mesostructures** | Frame perturbation + multi-mip latent fitting | Captures high-frequency bump/normal maps without separate normal textures. |
| **Repeating Patterns & Textures** | Multi-resolution latent grid with repeat wrapping | Preserves high-frequency spatial detail up to the Nyquist limit of the chosen latent resolution. |
| **Multi-Scale Appearance (LOD)** | Footprint-filtered hierarchical latent pyramid | Pre-filters specular roughness and micro-geometry at distance, preventing aliasing and moiré. |
| **Surface Emission** | Latent-to-RGB emission head | Evaluates HDR self-illumination independently of scene light count. |
| **Cutout Opacity / Alpha Masking** | Latent-to-scalar sigmoid opacity head | Sharp cutout silhouettes integrated with depth and shadow passes. |

---

## 4. Relationship to the NVIDIA SIGGRAPH 2024 Paper

The implementation is inspired by the research paper:
> **Real-Time Neural Appearance Models**  
> *Tizian Zeltner, Brent Burley, Jonathan Bikover, Matt Jen-Yuan Chiang, Peter Shirley, Cem Yuksel, Matt Pharr (NVIDIA)*  
> *ACM Transactions on Graphics (SIGGRAPH 2024)*

### 4.1 Conversion from Raytracing to Rasterization

The NVIDIA paper focuses on **Monte Carlo Path Tracing** in engines like Falcor. Translating the approach to **real-time rasterization** in Three.js WebGPU introduces fundamental differences:

```
NVIDIA SIGGRAPH 2024 Pipeline (Path Tracing)
┌─────────────────────────────────────────────────────────────────────────┐
│ • eval(wi, wo)   -> Evaluates BSDF value                                │
│ • sample(wo, ξ)  -> Generates scattered ray wi via learned neural PDF   │
│ • pdf(wi, wo)    -> Computes sampling probability density               │
│ • High sample count, multi-bounce indirect global illumination          │
└─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼ Converted to Raster Context
Three.js WebGPU Implementation (Direct-Lighting Rasterizer)
┌─────────────────────────────────────────────────────────────────────────┐
│ • eval(wi, wo) only -> Light direction wi is fixed by scene lights      │
│ • No sample() / pdf() required (no stochastic secondary rays needed)    │
│ • Direct evaluation per punctual/directional light inside forward pass  │
│ • Integrated with standard raster depth, shadow, and tone mapping       │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Comprehensive Feature Comparison Matrix

| Aspect | NVIDIA SIGGRAPH 2024 Reference | Three.js WebGPU Implementation |
| :--- | :--- | :--- |
| **Rendering Context** | Monte Carlo Path Tracing (multi-bounce global illumination) | Real-time Rasterization (forward/deferred direct lighting) |
| **Required Interfaces** | `eval()`, `sample()`, `pdf()` | `eval()` only (per direct light source) |
| **Importance Sampling** | Learned anisotropic spherical Gaussian / GGX network | Not needed in raster context (punctual/directional lights) |
| **Latent Representation** | 8 to 16 channels, multi-level mip hierarchy | Exactly 8 channels packed into two `RGBA16F` textures |
| **Shading Frames** | Two learned local frames derived from latents via linear layer | Two learned local frames derived from latents via linear layer |
| **Decoder Architecture** | 20-input compact MLP with fused tensor execution | 20-input compact MLP executed via TSL matrix/vector uniforms |
| **Training Pipeline** | PyTorch offline pipeline, ~300k iterations, ~40B samples | In-browser JS/WebGPU trainer (~1k–10k iterations, ~1M–50M samples) |
| **Parameter Bootstrap** | Material parameter encoder pre-training before latent baking | Direct latent grid optimization from scratch |
| **Appearance Filtering** | Gaussian footprint prefiltering over material graph | Multi-tap Gaussian supersampling in offscreen GPU atlas |
| **Auxiliary Heads** | Focus on standard BSDF reflectance | Required IBL head plus optional independent heads for Emission ($8 \to 3$) and Cutout Opacity ($8 \to 1$) |
| **Format Interop** | Slang / Falcor checkpoint (model.json + EXR files) | Native JSON format + Python bridge (`convert_checkpoint.py`) |

---

## 5. Current Limitations and Constraints

1. **Single-Lobe IBL Approximation:**
   - Image-based lighting is approximated with one learned diffuse PMREM query and one learned specular PMREM query.
   - Strong multi-lobe, sheen, clearcoat, or highly anisotropic environments can still exceed this compact two-sample representation and should be measured against the brute-force reference path.
2. **Opaque and Cutout Surfaces Only:**
   - Physical transmission (glass, water), subsurface scattering, and blended fractional alpha transparency are outside the current representation.
3. **Browser Training Memory and Compute Bounds:**
   - While inference is fast in WebGPU shaders, in-browser training is bounded by JavaScript memory, GPU buffer sizes, and teacher readback cost. Training high-resolution textures (e.g., $1024\times1024$ or $2048\times2048$) requires several minutes and substantial RAM.
4. **Lack of an Analytical Parameter Encoder:**
   - The reference paper uses an encoder network mapping analytical material parameters (albedo, roughness, metallic) directly to latent codes before fine-tuning. Three.js currently optimizes random latent codes from scratch, requiring more iterations to resolve spatial patterns.
5. **Discrete LOD Transitions:**
   - Runtime LOD uses nearest-integer mip selection or UV-hash stochastic dithering rather than continuous dual-level trilinear blending in the shader.

---

## 6. Implementation File Reference

- `examples/webgpu_materials_neural_appearance.html`: Interactive viewer and in-browser training interface.
- `examples/jsm/loaders/NeuralAppearanceLoader.js`: Asset loader and validator for `.json` neural appearance files.
- `examples/jsm/neural/NeuralAppearanceNodeMaterial.js`: WebGPU `NodeMaterial` implementing the neural lighting pipeline.
- `examples/jsm/neural/NeuralAppearanceTSL.js`: TSL shader graph generating WGSL instructions for WebGPU.
- `examples/jsm/neural/NeuralAppearanceTrainer.js`: In-browser training orchestrator.
- `examples/jsm/neural/NeuralAppearanceTeacherEvaluator.js`: GPU teacher evaluating MaterialX node materials.
- `examples/jsm/neural/NeuralAppearanceTeacherAtlas.js`: Offscreen texture atlas manager for batch sample evaluation.
- `examples/jsm/neural/NeuralAppearanceTeacherReadback.js`: Asynchronous half-float GPU readback module.
- `examples/jsm/neural/NeuralAppearanceModel.js`: Model definitions, forward passes, and backward gradients.
- `examples/jsm/neural/NeuralAppearanceMLP.js`: Pure JavaScript MLP primitives, Adam optimizer, and activation functions.
- `examples/jsm/neural/NeuralAppearanceSampler.js`: Jittered UV, directional, and Rusinkiewicz sample generators.
- `examples/jsm/neural/NeuralAppearanceValidator.js`: Validation metrics across angular bins, reciprocity, and smoothness.
- `examples/jsm/neural/NeuralAppearanceFilterUtils.js`: Gaussian footprint kernels, area estimators, and LEAN normal/roughness estimators.
- `examples/jsm/neural/NeuralAppearanceFormat.js`: Constants and schema definitions for format version 3.
- `utils/neural-appearance/convert_checkpoint.py`: Python script converting official NVIDIA checkpoints to Three.js JSON assets.
