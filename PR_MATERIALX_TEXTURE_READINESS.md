# MaterialX Texture Readiness and Graph Metadata

## Purpose

This PR makes `MaterialXLoader` safer for tools that need to consume a parsed material immediately after loading. Image-backed MaterialX nodes currently start asynchronous texture loads during parsing, but callers have no supported way to wait for those images before doing one-shot work such as baking a material into render targets.

The change adds a `texturesReady` promise to the loader result. The promise resolves once every texture load started during parse has either loaded or failed, so callers can wait for stable material inputs without polling texture state or relying on a render loop to eventually catch up.

## Raw Graph Access

The PR also attaches the parsed `MaterialXDocument` and resolved surface shader node to the produced material as plain instance properties. This is useful for tools that need authoring-level information that is lost after compilation to TSL, such as inspecting the texture-coordinate chain behind an image node.

These properties are intentionally not stored in `material.userData`, because the raw MaterialX graph has parent/child/document references and is not JSON-cloneable.

## Scope

- Adds `texturesReady` to `MaterialXLoader` parse/load results.
- Records texture load failures in the existing MaterialX log instead of throwing asynchronously from the image callback.
- Adds a `getTexcoordNode()` compile-context hook so texcoord resolution has one central path.
- Exposes raw MaterialX graph handles on the original parsed material.
- Does not add new TSL node types or change texture-coordinate behavior.

## Validation

Run focused lint:

```sh
npx eslint examples/jsm/loaders/materialx/MaterialXDocument.js examples/jsm/loaders/materialx/MaterialXLog.js examples/jsm/loaders/materialx/compile/MaterialXCompileRegistry.js
```
