# MaterialX Semantic Geometry Override Refactor

## Purpose

Refactor MaterialX coordinate and geometry input translation so callers can override semantic inputs such as UV coordinates without traversing the compiled node graph or depending on generated `AttributeNode` instances.

The first consumer is `NeuralAppearanceTeacherEvaluator`, which needs controlled UV, normal, tangent, bitangent, view direction, and light direction inputs. The same mechanism should support baking, material previews, procedural geometry probes, and tests.

This document proposes the refactor. It does not prescribe changes to MaterialX vocabulary. MaterialX names such as `texcoord`, `defaultgeomprop`, and `UV0` should remain at the parser boundary.

## Current problem

Three.js already supports context-driven texture coordinates:

- `TextureNode.setup()` calls `builder.context.getUV( textureNode, builder )` when the texture has no explicit UV node or `forceUVContext` is enabled.
- `replaceDefaultUV()` installs that callback.
- `overrideNode()` and `overrideNodes()` replace known node instances.

MaterialX bypasses the default-UV path in several places because it compiles coordinate inputs into explicit TSL nodes.

### Explicit MaterialX texcoord

`MaterialXDocument.js` translates a MaterialX `texcoord` input directly to `uv( index )`:

```js
} else if ( this.element === 'input' && this.name === 'texcoord' && this.type === 'vector2' ) {

	let index = 0;
	const defaultGeomProp = this.getAttribute( 'defaultgeomprop' );

	if ( defaultGeomProp && /^UV(\d+)$/.test( defaultGeomProp ) ) {

		index = parseInt( defaultGeomProp.match( /^UV(\d+)$/ )[ 1 ], 10 );

	}

	node = this.materialX.compileContext.mxToBottomLeftUvSpace( uv( index ) );

}
```

### Implicit MaterialX texcoord

`MaterialXCompileRegistry.js` contains a second direct UV construction:

```js
const getDefaultUvNode = ( compileContext ) =>
	compileContext.mxToBottomLeftUvSpace( uv( 0 ) );
```

Texture nodes, procedural nodes, and explicit coordinate graphs can therefore hold UV attributes that `replaceDefaultUV()` does not replace.

### Current workaround

`NeuralAppearanceTeacherEvaluator` traverses the cloned material, finds nodes whose attribute name is `uv`, and installs an identity-based override for each one:

```js
const uvOverrides = collectUvNodes( sampleMaterial )
	.map( ( node ) => [ node, materialUv ] );

sampleMaterial.contextNode = TSL.replaceDefaultUV(
	materialUv,
	TSL.overrideNodes( [
		...uvOverrides,
		[ TSL.normalView, normal ],
		[ TSL.tangentView, tangent ],
		[ TSL.bitangentView, bitangent ],
		[ TSL.positionViewDirection, wo ]
	] )
);
```

This workaround has several weaknesses:

- It recognizes only the attribute name `uv`, not indexed attributes such as `uv1`.
- It depends on the implementation shape of the compiled graph.
- It traverses every enumerable node-valued material property.
- It can miss coordinate nodes hidden behind an unsupported traversal boundary.
- It combines two UV override systems: `replaceDefaultUV()` and explicit node replacement.
- Future MaterialX geometry inputs would require more traversal heuristics.

## Design goals

1. Resolve every MaterialX default or explicit UV set through one compiler abstraction.
2. Allow late overrides at node-build time, after the MaterialX document has been parsed.
3. Preserve MaterialX UV-set indices.
4. Preserve MaterialX coordinate-space conversion.
5. Preserve explicit coordinate processing such as `place2d`; replace only its source geometry coordinate.
6. Remove UV graph traversal from `NeuralAppearanceTeacherEvaluator`.
7. Keep existing TSL `replaceDefaultUV()` behavior compatible.
8. Provide a path for other semantic geometry inputs without forcing all of them into this patch.
9. Keep ordinary MaterialX output unchanged when no override is installed.

## Non-goals

- Renaming MaterialX `texcoord` inputs to Three.js terminology.
- Replacing arbitrary user-authored coordinate graphs with a global UV.
- Treating all `vector2` inputs as texture coordinates.
- Changing MaterialX top-left versus bottom-left semantics.
- Adding neural appearance behavior to `MaterialXLoader`.
- Replacing the generic `overrideNodes()` API.
- Refactoring light-direction overrides. Light direction belongs to the lighting model, not the MaterialX geometry input resolver.

## Recommended architecture

Use two layers:

1. A MaterialX-local semantic resolver that removes duplicate `uv( index )` construction.
2. A context-aware indexed UV accessor that can be overridden without finding concrete node instances.

Do not overload the existing `TextureNode` `getUV` callback with indexed geometry semantics. Its first argument is a `TextureNode`, and existing callbacks may inspect that node. Procedural MaterialX coordinates also exist outside `TextureNode`.

### Layer 1: MaterialX compile-context resolver

Add a single resolver to `MaterialXDocument.compileContext`:

```js
this.compileContext = {
	// Existing fields...
	getTexcoordNode: ( index = 0 ) => (
		mxToBottomLeftUvSpace( semanticUV( index ) )
	)
};
```

The exact names may change, but keep the responsibilities separate:

- `semanticUV( index )` resolves the Three.js geometry input and supports late override.
- `mxToBottomLeftUvSpace()` converts from the loader-declared source convention into MaterialX’s expected convention.

Replace both current direct constructions:

```js
compileContext.mxToBottomLeftUvSpace( uv( index ) )
```

and:

```js
compileContext.mxToBottomLeftUvSpace( uv( 0 ) )
```

with:

```js
compileContext.getTexcoordNode( index )
```

This ensures explicit `texcoord`, image-node fallback coordinates, and procedural fallback coordinates share one source.

### Layer 2: context-aware indexed UV accessor

Introduce a semantic UV node or equivalent helper in `src/nodes/accessors/UV.js`.

Conceptual API:

```js
semanticUV( index = 0 )
```

Default behavior:

```js
uv( index )
```

Override behavior:

```js
builder.context.getUVAttribute( index, semanticUvNode, builder )
```

The context key name is provisional. `getUVAttribute` distinguishes it from the existing texture-level `getUV( textureNode, builder )` callback.

The semantic node must:

- carry the UV-set index;
- return a `vec2`;
- use the ordinary `uv`, `uv1`, `uv2`, and later attribute naming when not overridden;
- remain safe across node hashing, serialization, cloning, and repeated builds;
- avoid recursively applying its own override while evaluating an override callback.

Possible implementation shapes:

#### Preferred: dedicated semantic accessor node

Create a small node class that stores `index` and resolves either the context callback or the existing `uv( index )` attribute.

Advantages:

- No behavior change for existing calls to `uv()`.
- The callback receives a clear UV index.
- MaterialX can migrate independently.
- Tests can exercise default and overridden behavior directly.

Costs:

- Adds one node type or helper.
- Requires export wiring through TSL.

#### Alternative: make `uv()` context-aware

Change `uv( index )` itself to honor an indexed context callback.

Advantages:

- All TSL UV access becomes semantically overrideable.
- No second public UV accessor.

Risks:

- `uv()` currently returns an `AttributeNode`; callers may depend on `getAttributeName()`, global hashing, or serialization shape.
- Existing `replaceDefaultUV()` callbacks receive a `TextureNode`, not an index.
- A broad core change would require more regression coverage than this MaterialX refactor.

Use this alternative only after auditing all `uv()` consumers and documenting compatibility.

#### Minimal alternative: canonical cached `uv( index )` nodes

Cache one `AttributeNode` per UV index so `overrideNodes( [[ uv( 0 ), replacement ]] )` can target a stable identity.

Advantages:

- Small change.
- Removes graph traversal for known UV sets.

Limitations:

- Does not unify with context-driven `getUV`.
- Callers must know which UV sets the material uses.
- Keeps identity replacement as the semantic API.
- Does not provide a clean path for other geometry inputs.

This can serve as a short-term fix, but it should not be the final abstraction.

## Override utility

Add a utility that applies both texture-default and semantic-attribute UV overrides when callers intend to replace all UV sources.

Conceptual API:

```js
replaceUV( callbackOrNode, flowNode = null )
```

It should install:

- the existing texture-level `getUV` callback for ordinary `TextureNode` defaults;
- the new indexed UV-attribute callback for MaterialX and other semantic accessors.

The callback needs the UV-set index. A possible normalized signature is:

```js
( { index, sourceNode, textureNode }, builder ) => replacement
```

Compatibility matters more than this exact signature. Do not silently change the existing `replaceDefaultUV()` callback contract. Keep `replaceDefaultUV()` as-is and implement `replaceUV()` as a new API, or provide a static-node overload that does not alter callback arguments.

For the neural teacher, all UV indices may initially map to the same sampled coordinate:

```js
sampleMaterial.contextNode = TSL.replaceUV(
	() => materialUv,
	TSL.overrideNodes( [
		[ TSL.normalView, normal ],
		[ TSL.tangentView, tangent ],
		[ TSL.bitangentView, bitangent ],
		[ TSL.positionViewDirection, wo ]
	] )
);
```

If preserving independent UV sets becomes necessary, the teacher sample payload should provide one coordinate and gradient pair per used set.

## Coordinate-space rule

Apply overrides at the source-geometry level, before MaterialX UV-space conversion.

The required order is:

```text
geometry UV or override
    → loader uvSpace conversion
    → MaterialX texcoord graph
    → place2d / transforms / address modes
    → texture or procedural evaluation
```

This rule prevents the teacher from bypassing `uvSpace: 'top-left'` conversion and avoids double-flipping.

The implementation and tests must state whether an override supplies Three.js/source-space UVs or already-normalized MaterialX bottom-left UVs. Source-space UVs are recommended because they preserve loader behavior.

## Geometry inputs beyond UV

Do not generalize prematurely, but design the UV accessor so a later semantic geometry API remains possible.

Potential future accessors:

```js
getGeometryAttribute( 'uv', index )
getGeometryAttribute( 'normal' )
getGeometryAttribute( 'tangent' )
getGeometryAttribute( 'bitangent' )
getPosition( 'local' | 'world' | 'view' )
getViewDirection()
```

Normals, tangents, positions, and view direction already have shared TSL nodes such as `normalView` and `positionViewDirection`, so `overrideNodes()` handles them without graph discovery. UV sets are the immediate gap because indexed attributes are created throughout translation and texture nodes have a separate default-coordinate hook.

A generic `getGeometryAttribute()` callback should be a later proposal unless implementing the UV-specific node reveals a clear, low-risk shared design.

## File-level implementation plan

### 1. `src/nodes/accessors/UV.js`

- Add the context-aware indexed semantic UV accessor.
- Preserve the existing `uv()` API.
- Document the context callback and fallback behavior.
- Add serialization support if a new node class requires it.

### 2. TSL exports

Update the relevant exports:

- `src/nodes/Nodes.js` or the current node export barrel;
- `src/Three.TSL.js`;
- generated build outputs only through the repository’s normal build process.

Use a name consistent with existing TSL conventions after checking maintainers’ preferences.

### 3. `src/nodes/utils/UVUtils.js`

- Add `replaceUV()` or a narrowly named semantic UV override helper.
- Keep `replaceDefaultUV()` backward compatible.
- Document callback arguments and coordinate-space expectations.

### 4. `examples/jsm/loaders/materialx/MaterialXDocument.js`

- Add `compileContext.getTexcoordNode( index )`.
- Route MaterialX `texcoord` input fallback through it.
- Keep `defaultgeomprop="UVn"` parsing unchanged.
- Keep UV-space conversion in this resolver.

### 5. `examples/jsm/loaders/materialx/compile/MaterialXCompileRegistry.js`

- Replace `getDefaultUvNode()`’s direct `uv( 0 )` construction with `compileContext.getTexcoordNode( 0 )`.
- Check every UV fallback category in `UV_FALLBACK_CATEGORIES`.
- Confirm image, tiled image, triplanar, height-to-normal, and procedural nodes use either explicit coordinates or the shared fallback.

### 6. `examples/jsm/materials/NeuralAppearanceTeacherEvaluator.js`

- Replace `replaceDefaultUV()` plus discovered `uvOverrides` with the new semantic override utility.
- Delete `collectUvNodes()`.
- Keep normal, tangent, bitangent, and view-direction overrides in `overrideNodes()`.
- Preserve affine tile gradients.

### 7. Tests

Add focused tests for the core accessor and MaterialX translation. The repository currently lacks dedicated MaterialX loader unit tests under `test/`.

## Required tests

### Core semantic UV behavior

1. `semanticUV( 0 )` resolves to `uv`.
2. `semanticUV( 1 )` resolves to `uv1`.
3. An indexed context callback receives the correct index.
4. An override applies to procedural use, not only `TextureNode`.
5. Existing `replaceDefaultUV()` behavior remains unchanged.
6. Nested contexts restore the original UV source outside their scope.
7. Override evaluation does not recurse into itself.

### MaterialX translation

Use small inline MaterialX documents or focused fixtures.

1. An image without explicit `texcoord` uses UV0.
2. `defaultgeomprop="UV1"` uses UV1.
3. An explicit `texcoord` node uses the semantic resolver.
4. A procedural `checkerboard` or `noise2d` fallback uses the semantic resolver.
5. `place2d` still transforms the overridden source coordinate.
6. `uvSpace: 'top-left'` flips exactly once.
7. `uvSpace: 'bottom-left'` does not flip.
8. Address modes still operate after coordinate replacement.
9. No-override output matches the pre-refactor output.

### Neural teacher integration

1. Remove `collectUvNodes()` and verify the UV-grid fixture still produces varying teacher targets.
2. Verify changing `duvDx` and `duvDy` changes filtered texture output.
3. Verify UV1 can be overridden when a fixture requests it.
4. Run constant Lambert and UV-grid neural training cases.
5. Compare teacher and neural output at multiple rotations.

## Acceptance criteria

- `NeuralAppearanceTeacherEvaluator` contains no traversal that searches for UV attributes.
- All MaterialX fallback and explicit UV-set resolution passes through one compile-context method.
- UV0 and UV1 can be overridden after parsing and before shader generation.
- MaterialX coordinate transforms remain downstream of the override.
- `replaceDefaultUV()` remains source-compatible.
- Existing MaterialX examples render unchanged without overrides.
- The UV-grid neural training test passes with the traversal workaround removed.
- Addon lint and unit tests pass.
- Focused WebGPU neural appearance E2E tests pass.

## Risks and mitigations

### Callback contract collision

The existing `getUV` callback receives a `TextureNode`. Reusing it for indexed UV attributes could break callers.

Mitigation: use a distinct context key or a new utility that adapts both mechanisms without changing `replaceDefaultUV()`.

### Double UV-space conversion

An override installed after `mxToBottomLeftUvSpace()` may bypass or duplicate Y flipping.

Mitigation: resolve semantic geometry UV first, then apply MaterialX conversion in `getTexcoordNode()`.

### Explicit coordinate graph replacement

Replacing an image node’s final UV can accidentally bypass `place2d` and other MaterialX transforms.

Mitigation: override only the source accessor. Keep graph operations downstream.

### Multiple UV sets

Mapping every index to one teacher coordinate hides UV1-specific behavior.

Mitigation: preserve indices in the API now. Extend teacher sample payloads when fixtures require independent sets.

### Node identity and hashing

A new semantic node may share or cache incorrectly across indices or builders.

Mitigation: include the index in hashing and serialization; add UV0/UV1 tests in the same graph.

### Broad core API expansion

A generic semantic geometry framework could make this patch difficult to review.

Mitigation: land the UV-specific abstraction first. Generalize only after another concrete consumer needs it.

## Suggested implementation sequence

1. Add failing tests that demonstrate `replaceDefaultUV()` does not replace explicit MaterialX `texcoord` and procedural fallback coordinates.
2. Add the context-aware indexed semantic UV accessor.
3. Add the MaterialX compile-context resolver and migrate both direct `uv()` sites.
4. Add MaterialX UV0, UV1, procedural, and UV-space tests.
5. Add the combined UV override utility without changing `replaceDefaultUV()`.
6. Simplify `NeuralAppearanceTeacherEvaluator` and delete graph traversal.
7. Run neural unit and focused WebGPU E2E tests.
8. Consider a separate proposal for generalized semantic geometry inputs.

## Validation commands

Use the repository’s exact test placement to select the narrowest commands after implementation. At minimum:

```sh
npm run lint-core
npm run lint-addons
npm run test-unit
npm run test-unit-addons
TEST_CASE=uvGrid npm run test-e2e-neural-appearance
```

Run the full neural appearance E2E suite if the focused UV-grid case passes.

## Handoff summary

The refactor should not translate MaterialX terminology into Three.js terminology throughout the parser. It should translate MaterialX geometry inputs through one semantic compiler boundary.

The preferred implementation adds:

- a MaterialX `getTexcoordNode( index )` compile-context resolver;
- a context-aware indexed UV source;
- a backward-compatible utility for replacing both default texture UVs and semantic UV attributes.

This removes the neural teacher’s UV graph traversal while preserving indexed UV sets, MaterialX coordinate transforms, and existing TSL behavior.
