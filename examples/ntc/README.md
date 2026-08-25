# `.ntc` sample assets

Five pre-trained Neural Texture Compression assets, used by
[`webgpu_materials_neural_texture_compression.html`](../webgpu_materials_neural_texture_compression.html):

| File | MaterialX source |
| --- | --- |
| `brick.ntc` | `neural_train_brick.mtlx` |
| `glass_dispersion.ntc` | `gltf_pbr_glass_dispersion.mtlx` |
| `gold.ntc` | `neural_train_glossy_gold.mtlx` |
| `rainbow_emissive.ntc` | `neural_train_emissive_grid.mtlx` |
| `wave_normal.ntc` | `neural_train_normal_map.mtlx` |

Each `.ntc` file is a JSON manifest (`format: "three-ntc"`) holding one
shared multiresolution latent grid (uint8-quantized) plus one small MLP
decoder (float16-packed), jointly fit against whichever PBR channels the
source MaterialX material actually varies spatially - see
[`../jsm/ntc/NTCFormat.js`](../jsm/ntc/NTCFormat.js) for the full channel
vocabulary and [`../jsm/loaders/NTCLoader.js`](../jsm/loaders/NTCLoader.js)
for the format itself.

They were trained against the same MaterialX samples using
[`webgpu_materials_neural_texture_compression_trainer.html`](../webgpu_materials_neural_texture_compression_trainer.html)
(`NTCTrainer.js`/`NTCNodeMaterial.fit()`), then exported via `NTCExporter`
(see [`../jsm/exporters/NTCExporter.js`](../jsm/exporters/NTCExporter.js)).
