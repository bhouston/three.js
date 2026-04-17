# Curated MaterialX Test Manifest

This manifest defines the curated local MaterialX test assets used by the MaterialX loader example and automated checks.

## Naming Convention

All curated files follow:

`surface_subject_feature.mtlx`

- `surface`: `open_pbr`, `gltf_pbr`, or `standard_surface`
- `subject`: primary material archetype (pearl, marble, velvet, etc.)
- `feature`: main capability being exercised (thinfilm, dispersion, anisotropy, logic, etc.)

## Balanced Surface Coverage

- `open_pbr_surface`: 4 files
- `gltf_pbr`: 4 files
- `standard_surface`: 4 files

Total curated files: 12

## Curated Asset Matrix

| File | Surface | Archetype | Primary Coverage | Inspired From |
| --- | --- | --- | --- | --- |
| `open_pbr_pearl_thinfilm.mtlx` | `open_pbr_surface` | Pearl | thin film, coat, subsurface | `materialx.js/open_pbr_pearl` |
| `open_pbr_soapbubble_transmission.mtlx` | `open_pbr_surface` | Soap bubble | transmission, thin walled, thin film | `materialx.js/open_pbr_soapbubble` |
| `open_pbr_brushed_metal_anisotropy.mtlx` | `open_pbr_surface` | Brushed metal | metalness, anisotropic roughness | `materialx.js/open_pbr_aluminum_brushed` |
| `open_pbr_velvet_fuzz.mtlx` | `open_pbr_surface` | Velvet | fuzz layer, rough specular | `materialx.js/open_pbr_velvet` |
| `gltf_pbr_carpaint_clearcoat.mtlx` | `gltf_pbr` | Car paint | clearcoat, roughness | `materialx.js/gltf_pbr_carpaint` |
| `gltf_pbr_glass_dispersion.mtlx` | `gltf_pbr` | Glass | transmission, ior, dispersion, attenuation | `materialx.js/gltf_pbr_dispersion` |
| `gltf_pbr_gold_metal.mtlx` | `gltf_pbr` | Gold | metallic response baseline | `materialx.js/gltf_pbr_gold` |
| `gltf_pbr_default_feature_sweep.mtlx` | `gltf_pbr` | Baseline sweep | broad glTF PBR input coverage | `materialx.js/gltf_pbr_default` |
| `standard_surface_marble_veins.mtlx` | `standard_surface` | Marble | procedural nodegraph, subsurface | `materialx.js/standard_surface_marble_solid` |
| `standard_surface_jade_translucent.mtlx` | `standard_surface` | Jade | anisotropy + subsurface style response | `materialx.js/standard_surface_jade` |
| `standard_surface_wood_grain.mtlx` | `standard_surface` | Wood | tiledimage texture-driven roughness/color | `materialx.js/standard_surface_wood_tiled` |
| `standard_surface_logic_composite_nodes.mtlx` | `standard_surface` | Logic composite | boolean/composite/color utility nodes | `materialx.js/standard_surface_logic_composite_nodes` |

## Duplication Rule

Archetypes are intentionally unique across surface families in this curated set (for example, only one car paint file, only one velvet file, only one glass file).
