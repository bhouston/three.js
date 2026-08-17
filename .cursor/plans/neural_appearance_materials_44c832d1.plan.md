---
name: Neural Appearance Materials
overview: Add a WebGPU-only addon that loads a compact NVIDIA neural-appearance checkpoint and evaluates its learned BRDF for Three.js direct lights. Demonstrate it with a converted checkpoint trained from NVIDIA’s Apache-2.0 FauxLeather sample, while keeping training and importance sampling outside the browser runtime.
todos:
  - id: runtime-addon
    content: Implement the loader, GPU resources, neural decoder, and direct-light material integration
    status: completed
  - id: convert-asset
    content: Add the checkpoint adapter and produce a licensed compact FauxLeather runtime asset on compatible NVIDIA hardware
    status: completed
  - id: example
    content: Create and register the WebGPU neural appearance example and screenshot
    status: completed
  - id: verify
    content: Add parser/reference tests and run lint, unit, and WebGPU E2E checks
    status: completed
isProject: false
---

# Neural Appearance Material Support

## Runtime addon
- Add [`examples/jsm/loaders/NeuralAppearanceLoader.js`](examples/jsm/loaders/NeuralAppearanceLoader.js) to load a portable manifest, FP16 latent mip pyramid, and packed decoder weights. Validate channel counts, layer dimensions, activation names, and supported format version before constructing GPU resources.
- Add [`examples/jsm/materials/NeuralAppearanceNodeMaterial.js`](examples/jsm/materials/NeuralAppearanceNodeMaterial.js), extending `NodeMaterial` with `lights = true`. Build the paper’s raster path in TSL/WGSL: UV-footprint LOD, two RGBA latent samples for the 8D code, latent-derived shading frames, tangent-space direction transforms, configurable small MLP layers, and non-negative RGB BRDF output.
- Implement an internal `LightingModel.direct()` that accumulates `evalBRDF( latent, viewDirection, lightDirection ) * lightColor`. Support directional, point, and spot lights plus shadows. State the initial limits in JSDoc: WebGPU only, opaque reflective surfaces, direct lighting only, and no reciprocity/energy-conservation guarantee.
- Support integer stochastic MIP selection from the paper behind a material option, with deterministic nearest-level selection for stable screenshots. Do not implement the sampling decoder because Three.js has no in-core path tracer consuming `sample`/`pdf`.

## Checkpoint conversion and example asset
- Add [`utils/neural-appearance/convert_checkpoint.py`](utils/neural-appearance/convert_checkpoint.py) as an offline adapter for the official `NVlabs/neuralappearance` checkpoint. Convert `model.json` plus its multi-channel EXRs into a versioned browser manifest, two RGBA16F mip chains, and packed weight data; fail on unsupported architectures instead of silently changing the model.
- On a Linux or Windows host with a Vulkan NVIDIA GPU and CoopVec support, train NVIDIA’s bundled `FauxLeather.mtlx` with multiple latent MIP levels, then run the adapter. Check in only the compact runtime output and Apache-2.0 attribution under [`examples/models/neural-appearance/faux-leather/`](examples/models/neural-appearance/faux-leather/). Keep the asset resolution low enough for Three.js contribution limits.
- Keep the browser format independent of Slang/Falcor so loading only depends on Three.js. Preserve FP16 latent values and network weights; WebGPU executes ordinary shader arithmetic because browser APIs do not expose the paper’s tensor-core/SER optimizations.

## Example and registration
- Add [`examples/webgpu_materials_neural_appearance.html`](examples/webgpu_materials_neural_appearance.html) using a UV-mapped procedural showcase mesh, orbit controls, moving direct lights, and GUI toggles for neural material versus a conventional physical baseline, LOD mode, and fixed MIP inspection. Credit the 2024 paper and NVIDIA asset/code sources.
- Register the example in [`examples/files.json`](examples/files.json), add focused search terms in [`examples/tags.json`](examples/tags.json), and generate [`examples/screenshots/webgpu_materials_neural_appearance.jpg`](examples/screenshots/webgpu_materials_neural_appearance.jpg). Do not edit the already modified generated `build/` outputs.

## Verification
- Add [`test/unit/addons/loaders/NeuralAppearanceLoader.tests.js`](test/unit/addons/loaders/NeuralAppearanceLoader.tests.js) with a tiny synthetic manifest/buffer covering successful parsing and malformed dimensions, versions, offsets, and activations; register it in [`test/unit/three.addons.unit.js`](test/unit/three.addons.unit.js).
- Compare several CPU reference evaluations exported by the converter against the browser decoder at fixed UV/direction/MIP inputs, then run addon lint/unit tests and the WebGPU example E2E screenshot. Check close, grazing, and minified views for finite output, expected view dependence, and stable deterministic LOD.