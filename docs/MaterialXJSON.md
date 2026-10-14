# Shared MaterialX XML and JSON loading

`MaterialXLoader` accepts XML text, a plain JSON document object, or its JSON text. Both formats build the same normalized document and use the existing MaterialX interface registry, defaults, node compiler, surface mappings and resource lifecycle. XML documents and `.mtlx.zip` archives keep their existing loading behavior. `load()` and `parseBuffer()` detect JSON text from the document contents, including UTF-8 buffers and extensionless URLs.

The JSON format is an **addon document representation**, not an official MaterialX JSON interchange format and not the `KHR_texture_procedurals` wire format. A format adapter translates its own graph format into this representation; it does not need a separate MaterialX evaluator.

```js
import { MaterialXLoader } from 'three/addons/loaders/MaterialXLoader.js';

const document = {
 element: 'materialx',
 attributes: { version: '1.39' },
 children: [ {
  element: 'nodegraph', attributes: { name: 'pattern' },
  children: [ {
   element: 'constant', attributes: { name: 'color', type: 'color3' },
   children: [ {
    element: 'input',
    attributes: { name: 'value', type: 'color3', value: '0.2, 0.4, 0.8' }
   } ]
  }, {
   element: 'output',
   attributes: { name: 'out', type: 'color3', nodename: 'color' }
  } ]
 } ]
};

const loader = new MaterialXLoader();
const { materials } = loader.parse( document );
// loader.parse( JSON.stringify( document ) ) produces the same result.
```

Each element has an `element` string, optional `attributes` object containing string, finite number or boolean values, and optional ordered `children` array. Missing attributes and children mean empty collections. Attribute values normalize to XML strings; vectors use comma-separated numeric strings. The document root is `materialx`. The loader validates the representation before evaluating it; malformed JSON, invalid element shapes and non-scalar attribute values fail with a document diagnostic.

## Host node and resource policies

Pass `allowedNodes` and/or `blockedNodes` as arrays or sets of processing category names. Structural elements such as `materialx`, `nodegraph`, `input`, `output`, `nodedef` and `implementation` do not need to appear in an allowlist. A blocklist takes precedence over an allowlist, and an empty allowlist permits no processing categories. Policies apply before a host `nodeResolver` is called, so custom resolution cannot bypass a category restriction.

```js
const { materials } = loader.parse( document, {
 allowedNodes: [ 'constant', 'multiply', 'checkerboard', 'noise2d', 'texcoord' ],
 blockedNodes: [ 'standard_surface', 'surfacematerial' ]
} );
```

These policies restrict compiler categories. A glTF adapter must additionally validate its graph DAG, port types, resource indices and pattern-only scope. An XML surface document remains supported when no policy is supplied. Restricting categories does not introduce support for arbitrary shader/source implementations.

`nodeResolver( materialXNode, outputName )` can provide a host TSL node before built-in compilation. Return `null` to let the existing node library handle it. `textureResolver( filenameNode )` can return a ready TSL `TextureNode` before ordinary filename loading, or `null` to use normal resource loading. This lets container formats resolve textures through their own loader and reuse MaterialX image evaluation without duplicating it. These callbacks are application code; assets cannot supply executable callbacks.

## Reading a document without creating surface materials

Adapters that bind pattern outputs to a host material can call `parseDocument()` instead of `parse()`. It accepts the same XML/JSON input and options and returns a configured `MaterialXDocument` with its `rootNode`. Reading creates the normalized tree and applies document configuration without evaluating materials. Output nodes are evaluated on demand through the shared compiler:

```js
const normalized = loader.parseDocument( document, {
 allowedNodes: [ 'constant' ]
} );
const graph = normalized.rootNode.getChildByName( 'pattern' );
const colorNode = graph.getChildByName( 'out' ).getNode();
await normalized.waitForResources();
```

The glTF texture procedural addon uses this document entry point. Its remaining format-specific work is draft graph validation and normalization, glTF resource/geometry bindings, material channel bindings and fallback policy. It shares MaterialX compilation and image evaluation with ordinary XML loading.

## Export scope

JSON loading does not reverse compiled TSL into MaterialX or introduce a general MaterialX exporter. The experimental glTF exporter is a separate opt-in addon and serializes the preserved editable draft graph source. Loader-only users register only the loader plugin and import no exporter. Its source validation, indices and glTF packing remain separate from the shared MaterialX parser/compiler.

Run `node --test test/unit/addons/loaders/MaterialXLoader.node.tests.js` for DOM-free JSON regressions. The existing browser `MaterialXLoader` tests compare XML and JSON material/node semantics, policy restrictions, JSON text/buffer/URL detection, and unchanged XML loading.
