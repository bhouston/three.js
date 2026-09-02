# TSL TempNode Flow-Code Hierarchy Fix

## Purpose

This PR fixes a TSL code-generation edge case where a cached `TempNode` can be reused from inside a conditional code block before its assignment has been emitted along that block's control path.

When a temp node is first materialized, its assignment is added to the current flow scope. Later references usually reuse the cached `propertyName`. If that later reference happens inside an `If`/`Else` branch, the cached assignment may not exist in that branch even though the variable name is reused there. The generated shader can then read the default-initialized value instead of the intended expression.

## Change

- `TempNode.build()` now mirrors the non-temp cache path by asking `NodeBuilder.addFlowCodeHierarchy()` to re-flow the cached assignment when a cached temp is referenced inside a node block.
- `NodeBuilder.addFlowCodeHierarchy()` now handles nodes whose original flow code was emitted at top level. In that case there is no `flowCodeBlock` map to inspect, and no re-flow is needed because the assignment is already visible to nested blocks.

## Scope

This is a narrow code-generation fix. It does not add new TSL syntax, node types, precision modes, or backend features.

## Why Standalone

The behavior is independent of any particular example. Any TSL graph that materializes a reusable temp outside a conditional and later references it from inside a conditional can hit the same control-flow visibility issue.

## Validation

Run focused lint:

```sh
npx eslint src/nodes/core/TempNode.js src/nodes/core/NodeBuilder.js
```

Regenerate builds after changing `src/`:

```sh
npm run build
```
