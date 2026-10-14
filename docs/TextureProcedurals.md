# Experimental glTF texture procedurals

These explicitly registered addons evaluate pattern graphs through this fork's MaterialX compiler. They implement an experimental profile of [glTF proposal #2381](https://github.com/KhronosGroup/glTF/pull/2381), pinned to **b13ced8e5dda85ad6b982b1b27bc5fe570306d38** (April 16, 2025). This proposal is a draft, and this implementation does not claim Khronos conformance or ratification.

Both `KHR_texture_procedurals` and `EXT_texture_procedurals_mx_1_39` must appear in `extensionsUsed`. The latter declares the MaterialX 1.39 node library for the entire asset and has no separate payload. Graphs and functional definitions live in `extensions.KHR_texture_procedurals`.

## Registration and round trips

```js
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { GLTFTextureProceduralsLoaderExtension } from 'three/addons/loaders/GLTFTextureProceduralsLoaderExtension.js';
import { GLTFTextureProceduralsExporterExtension } from 'three/addons/exporters/GLTFTextureProceduralsExporterExtension.js';

const loader = new GLTFLoader();
loader.register( parser => new GLTFTextureProceduralsLoaderExtension( parser ) );
const gltf = await loader.loadAsync( 'pattern.gltf' );

const exporter = new GLTFExporter();
exporter.register( writer => new GLTFTextureProceduralsExporterExtension( writer ) );
const json = await exporter.parseAsync( gltf.scene );
const glb = await exporter.parseAsync( gltf.scene, { binary: true } );
```

Procedural materials are node materials. Use `WebGPURenderer`; its WebGL backend also supports these materials. Ordinary `GLTFLoader` without registration uses the standard fallback texture. No MaterialX dependency is added to ordinary glTF loading.

## Authoring and edits

Keep the source graph as the editable representation. Compiled TSL is not reversible into MaterialX.

```js
import { MeshBasicMaterial, Mesh, PlaneGeometry } from 'three';
import {
 createTextureProceduralMaterial,
 getTextureProcedurals,
 updateTextureProceduralMaterial
} from 'three/addons/utils/GLTFTextureProcedurals.js';

const source = {
 procedurals: [ {
  name: 'checker', nodetype: 'nodegraph', type: 'color3', inputs: {},
  nodes: [ {
   nodetype: 'checkerboard', type: 'color3',
   inputs: {
    color1: { nodetype: 'input', type: 'color3', value: [ 0.1, 0.4, 0.8 ] },
    color2: { nodetype: 'input', type: 'color3', value: [ 1, 0.8, 0.2 ] }
   },
   outputs: { out: { nodetype: 'output', type: 'color3' } }
  } ],
  outputs: { pattern: { nodetype: 'output', type: 'color3', node: 0, output: 'out' } }
 } ]
};
const material = await createTextureProceduralMaterial(
 new MeshBasicMaterial(), source,
 { baseColorTexture: { index: 0, output: 'pattern' } }
);
const mesh = new Mesh( new PlaneGeometry(), material );

getTextureProcedurals( material ).source.procedurals[ 0 ].nodes[ 0 ]
 .inputs.color1.value = [ 0.8, 0.1, 0.2 ];
await updateTextureProceduralMaterial( material );
const result = await exporter.parseAsync( mesh );
```

Resource indices in filename ports address the authoring `options.textures` array: `{ nodetype: 'input', type: 'filename', value: [ 0 ] }`. Supply ready Three.js textures in `{ textures: [ texture ] }`. Imported external images and GLB buffer-view images are resolved asynchronously before graph compilation. Export remaps resources to its own texture indices.

Supported source edits must be followed by `updateTextureProceduralMaterial()`. The exporter checks graph state and runtime nodes to diagnose uncompiled edits or unsupported node replacement, instead of silently saving a stale graph. Retain shared source representations to reuse graphs across materials. Do not serialize arbitrary shader code or a complete MaterialX surface material through this API.

Imported geometry material variants receive independent editable source, binding and resource arrays. Arbitrary authoring `Material.clone()` does not retain the external procedural state and is diagnosed during export. Recreate authored copies with `createTextureProceduralMaterial()` using an ordinary fallback material, the original state source/bindings and its ready texture array.

## Fallback and required content

Every procedural texture-info retains a valid ordinary texture `index`. Provide a real baked fallback as the material's corresponding map when its appearance matters. When no map is provided, export uses a valid embedded one-pixel dummy. **A dummy does not preserve the procedural appearance.** The demo deliberately contrasts visible procedural patterns with a plain dummy fallback.

The extension is optional by default: exporting registers both identifiers in `extensionsUsed`, without automatically requiring them. Authoring `{ required: true }` deliberately adds `KHR_texture_procedurals` to `extensionsRequired`; the MaterialX library identifier remains the asset-wide declaration in `extensionsUsed`. Unsupported optional graphs emit diagnostics and retain fallback materials; unsupported required content fails loading. Missing/conflicting MaterialX library declarations, malformed graphs, missing resources and unsupported nodes are diagnosed under that policy.

## Experimental graph profile

- Named `inputs` and `outputs` object maps, string output selectors, and numeric arrays are used, including `[ 1 ]` for a scalar. Legacy port arrays and integer selectors are not accepted.
- Graphs contain one flat indexed DAG. Connections may address graph interface inputs or processing-node outputs. Named graph outputs and multiple output nodes are supported. Cycles, invalid indices/ports and incompatible types are rejected.
- Pattern-node signatures/defaults come from the fork's MaterialX 1.39 interface registry and compiler. Math, procedural noise, checkerboard, explicit geometry streams, image and placement nodes are supported where the existing compiler implements them. Shading-model, BSDF and arbitrary source-code nodes are excluded.
- Functional `procedural_definitions` contain a `nodedef` and a graph implementing that definition. They are immutable declarations, preserved for round trips. Code-based implementations are unsupported and never executed.
- Functional implementations select their `nodedef` by its index in `procedural_definitions`; they use that definition's inputs, without a second input interface. One version per custom category is supported, and standard-library overrides are rejected. Unlike the pinned default-only declaration prose, a custom input may omit its default if every instance supplies it. All functional definitions are validated, including unused definitions, and their referenced textures must be available.
- `texcoord` selects numbered `TEXCOORD_n` (Three.js `uv` through `uv3`, then `texcoord_n`); `geomcolor` selects numbered `COLOR_n` (`color` or `color_n`). Indices must be nonnegative safe integers, and referenced streams must exist on the mesh. Three-component UVs are unsupported. The graph owns placement/sampling: texture-info `texCoord`, `KHR_texture_transform` and glTF samplers apply only to fallback texture evaluation. Use explicit MaterialX geometry/image/place2d nodes for procedural evaluation. Tangent-space procedural normal and clearcoat-normal channels currently require UV0; negative-X and nonuniform normal scales cannot be represented by glTF's scalar `scale` and are diagnosed on export.
- Numeric graph colors follow MaterialX's linear working interpretation. Image nodes default to linear data; explicitly specify `colorspace: 'srgb_texture'` on the image node or its file input for sRGB color decoding. Material factors remain outside graph evaluation. Packed metallic/roughness output uses G for roughness and B for metallic; occlusion uses R.
- Numeric graph types are `float`, `integer`, `color3`, `color4`, `vector2`, `vector3`, `vector4`, `matrix3x3` and `matrix4x4`; matrices are row-major on the wire. Booleans use JSON booleans, strings use strings, and filename inputs use `[textureIndex]`. Integer-vector types and unit conversion/annotations are unsupported. Image filtering is limited to `closest` and `linear`; addressing supports `constant`, `clamp`, `periodic` and `mirror`. Input-stream values use the glTF geometry coordinates directly. Ordinary glTF vertex-color modulation remains outside the graph, even when a graph also explicitly reads a color stream.
- Animation of graph interface values is currently unsupported. The existing glTF animation-pointer implementation is not treated as procedural animation support. Required procedural animation is rejected; optional unsupported animation is diagnosed. Custom-definition internals cannot be animated.

The proposal's schemas and companion checkerboard describe an older array/scalar format; several main examples are invalid JSON. This profile follows the pinned base prose instead. Corrected fixtures use valid glTF `images[].uri` and valid resource indices. The printed animation pointer `/extensions/procedurals/...` also omits the extension identifier and value leaf and is not implemented literally.

External/data-URI and GLB buffer-view resources use the normal GLTFLoader texture pipeline and its registered codec loaders. Export uses GLTFExporter's normal image serialization; compressed textures need its usual decompression utility. Graph sampling ignores resource `flipY` and uses glTF's top-first image orientation. Only graphs bound to exported materials are emitted, together with their complete functional-definition declarations. Unsupported optional bindings load as fallback and diagnostics, and are not emitted as procedural bindings on re-export.

## Material channel coverage

| Material | Texture-info channels | Graph components |
| --- | --- | --- |
| Core PBR | `baseColorTexture`, `metallicRoughnessTexture`, `normalTexture`, `occlusionTexture`, `emissiveTexture` | RGB[A], G/B, RGB, R, RGB |
| Unlit | `baseColorTexture` | RGB[A] |
| `KHR_materials_clearcoat` | `clearcoatTexture`, `clearcoatRoughnessTexture`, `clearcoatNormalTexture` | R, G, RGB |
| `KHR_materials_transmission` | `transmissionTexture` | R |
| `KHR_materials_volume` | `thicknessTexture` | G |
| `KHR_materials_specular` | `specularTexture`, `specularColorTexture` | A, RGB |
| `KHR_materials_sheen` | `sheenColorTexture`, `sheenRoughnessTexture` | RGB, A |
| `KHR_materials_iridescence` | `iridescenceTexture`, `iridescenceThicknessTexture` | R, G |
| `KHR_materials_anisotropy` | `anisotropyTexture` | RG direction, B strength |
| `EXT_materials_bump` | `bumpTexture` | R |

Material factors, emissive intensity, alpha mode/cutoff, normal scale, occlusion strength and physical material properties remain active. Iridescence thickness interpolates the material's thickness range using G. Anisotropy uses normalized RG direction with the material rotation and B strength. Bump uses R with the material bump scale; simultaneous procedural bump and normal bindings are diagnosed as unsupported. Legacy specular/glossiness `diffuseTexture` and extensions without a defined mapping above are outside this profile. Separate authored roughness and metallic values must be combined into the G/B channels of one graph output; the exporter never invents separate metallic or roughness glTF slots.

Sibling factors/ranges are retained even when their material lobe has zero strength. glTF has one `sheenColorFactor`, so export combines Three.js `sheen` strength with `sheenColor`; reload preserves that product, without inventing a separate strength property in glTF.

For a graph with separate scalar outputs, the authoring helper creates a serializable standard `combine3` node, preserving those outputs:

```js
import { packTextureProceduralChannels } from 'three/addons/utils/GLTFTextureProcedurals.js';

const packedGraph = packTextureProceduralChannels( graph, {
 roughness: 'roughnessOutput', metallic: 'metallicOutput', occlusion: 'occlusionOutput'
}, 'orm' );
// Bind the new 'orm' output to metallicRoughnessTexture and occlusionTexture.
// Without occlusionOutput, R defaults to 1.
```

## Examples and validation

[Demo](../examples/webgpu_loader_gltf_texture_procedurals.html) shows checkerboard, noise/math and texture/math graphs alongside ordinary fallbacks. Corrected valid fixtures are in `examples/models/gltf/TextureProcedurals/`. QUnit tests cover graph validation and loader/exporter semantics. GPU screenshots require a physical renderer; software GPU fallback results are not rendering evidence.

Run `node test/unit/puppeteer.texture-procedurals.js` for the focused browser suite. Pass an artifact-directory argument to also capture the demo, test results and verified physical WebGL renderer information. The runner uses source imports and closes its browser/server on completion.

Run `node --test test/unit/addons/loaders/GLTFTextureProcedurals.node.tests.js` for pure graph validation checks without a browser or GPU.
