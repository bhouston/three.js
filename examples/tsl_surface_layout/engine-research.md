# Engine resource-graph research for issue 121

Research date: 2026-10-17. Primary documentation and source only. These are architecture findings, not performance measurements.

## Recommendation

Keep a surface layout/codec separate from multipass resource ownership. First prototype a function returning ordinary producer-carrying TSL nodes, backed by explicit physical attachment descriptors. Use distinct immutable application bindings for `reflectionRaw`, `reflectionTemporal`, `reflectionSpatial`, and `previousOpaque`. Add a resource registry only if the runnable composition demonstrates a real declaration/binding problem. A dictionary or codec is not a resource graph: lifetime inference, native pass merging, access validation, transient pooling, and barriers need substantially more information and renderer support.

## Unreal RDG: documented UE 5.8

[Epic RDG programming guide](https://dev.epicgames.com/documentation/en-us/unreal-engine/render-dependency-graph-in-unreal-engine?application_version=5.8), sections “Render Graph Builder”, “RDG Resources and Views”, “External Resources”, “Transient Resources”. The fetched page identifies Unreal Engine 5.8.

RDG separates setup from execution. Creating a texture initially creates a descriptor; pass parameter structures declare actual resource uses. Compilation derives dependencies and resource lifetimes, culls dead work, inserts transitions/fences, and plans transient allocations. Its SRV/UAV views describe GPU access, not whether encoded RGB represents a shading normal. External textures are registered; extraction retains a pooled texture beyond execution and enables import into a subsequent graph. Extraction extends the end lifetime; registration extends the beginning. Transient memory aliasing is a backend allocator feature for disjoint lifetimes, not a consequence of naming resources. This evidence supports separating a packed-surface semantic layer from a broader resource graph. It does not establish an automatic public SSA-style immutable handle for every RDG write, and should not be described that way. (Approximately 145 words attributed to this guide.)

[FRDGBlackboard API, UE 5.8](https://dev.epicgames.com/documentation/en-us/unreal-engine/API/Runtime/RenderCore/FRDGBlackboard?application_version=5.8) identifies `RenderGraphBlackboard.h`. It is graph-allocator-scoped typed struct storage. Creation asserts if the type already exists; checked immutable retrieval asserts when absent. Epic recommends this for data initialized once and fetched across the pipeline, and ordinary function arguments for more constrained data. Mutable retrieval exists, so “blackboard” does not itself guarantee immutable stage versions. The graph's pass declarations still supply dependencies. Transfer the scoped identity and checked access if needed; do not make a global bag of currently selected results.

## Unity RenderGraph: SRP Core 17.0.4 / Unity 6000.0

[Render graph fundamentals, Core 17.0](https://docs.unity3d.com/Packages/com.unity.render-pipelines.core@17.0/manual/render-graph-fundamentals.html) currently identifies **17.0.4** in its title and metadata. It describes lightweight graph-specific handles, explicit read/write declarations, and setup/compile/execute phases rebuilt each frame. Resources created inside an execution do not persist to the next execution. Persistent resources are externally owned and imported. Allocation is delayed until a resource is needed; last use allows memory reuse, and culled producers need no allocation. Surviving passes execute in declaration order. This is not a promise to resolve arbitrary forward producer declarations through topological reordering. (Approximately 100 words attributed to fundamentals.)

The user-supplied [7000.0 URP introduction](https://docs.unity.com/en-us/engine/7000.0/manual/render-pipelines/universal-render-pipeline/customizing-urp/render-graph/introduction) is currently **Unity 7.0 Alpha**, not the Core 17 stable package version. It additionally describes compatible raster-pass merging, asynchronous compute synchronization, imported resource ownership and unsafe-pass optimization restrictions. Those capabilities belong to that engine's compiled render graph; a TSL layout helper must not claim them merely because it returns handles.

Exact source version: Unity Graphics `6000.0/staging` resolved through `git-dedup ls-remote` to [`feb4de2d9a93a4ae10d287d3a6d7003d08ea3e53`](https://github.com/Unity-Technologies/Graphics/tree/feb4de2d9a93a4ae10d287d3a6d7003d08ea3e53). Its [Core package.json](https://github.com/Unity-Technologies/Graphics/blob/feb4de2d9a93a4ae10d287d3a6d7003d08ea3e53/Packages/com.unity.render-pipelines.core/package.json#L1-L5) confirms 17.0.4 / Unity 6000.0.

The pinned [RenderGraphBuilders.cs](https://github.com/Unity-Technologies/Graphics/blob/feb4de2d9a93a4ae10d287d3a6d7003d08ea3e53/Packages/com.unity.render-pipelines.core/Runtime/RenderGraph/RenderGraphBuilders.cs#L145-L227) supplies concrete version/access behavior:

- `UseResource` records reads of an explicit version or the latest version; a preserving write also creates an implicit dependency on prior contents.
- Writes create a new resource version. Validity checks reject writing an already versioned handle and writing a resource twice in one pass; the rationale is avoiding branching version histories.
- [Attachment validation](https://github.com/Unity-Technologies/Graphics/blob/feb4de2d9a93a4ae10d287d3a6d7003d08ea3e53/Packages/com.unity.render-pipelines.core/Runtime/RenderGraph/RenderGraphBuilders.cs#L235-L388) rejects sampled-use/attachment conflicts and distinguishes depth attachment use from ordinary sampled texture use.
- These internal access versions are not automatically different persistent physical copies or application meanings such as raw versus filtered radiance. Temporal history still needs persistent storage and invalidation. (Approximately 125 words across this source.)

[RenderGraphResourceRegistry.cs](https://github.com/Unity-Technologies/Graphics/blob/feb4de2d9a93a4ae10d287d3a6d7003d08ea3e53/Packages/com.unity.render-pipelines.core/Runtime/RenderGraph/RenderGraphResourceRegistry.cs#L334-L372) implements latest/new version handles. [CreateTexture](https://github.com/Unity-Technologies/Graphics/blob/feb4de2d9a93a4ae10d287d3a6d7003d08ea3e53/Packages/com.unity.render-pipelines.core/Runtime/RenderGraph/RenderGraphResourceRegistry.cs#L758-L766) stores a descriptor and optional transient-pass index. Logical identity and physical allocation are separate mechanisms.

Unity's [ContextContainer, Core 17.0.4](https://docs.unity3d.com/Packages/com.unity.render-pipelines.core@17.0/api/UnityEngine.Rendering.ContextContainer.html) is a dictionary keyed by a generic type. Like RDG's blackboard, this is typed CPU data plumbing; it does not replace recording each pass's resource accesses.

## Surface layout is an independent concern

[URP G-buffer layout, Unity 6000.0](https://docs.unity3d.com/6000.0/Documentation/Manual/urp/rendering/g-buffer-layout.html) documents encoded world normals, smoothness, sRGB albedo, material flags, baked occlusion, emission/GI/lighting, and conditional targets. Complex Lit uses forward rendering because its lighting model does not fit that layout. Deferred lighting reuses the emission/GI target for lighting results. This is the renderer's chosen data contract, not the RenderGraph API. A Three.js prototype should borrow explicit space/encoding conventions and retain custom material/forward escape hatches, rather than import Unity's fixed field list. Application-level stage names remain valuable even where a graph permits in-place storage reuse.

## What transfers to Three.js

| Concern | Small transferable pattern | Requires broader graph/renderer work |
| --- | --- | --- |
| Normal/roughness share one texture | Codec descriptor, semantic accessors, one physical target identity | Automatic global packing/repacking |
| Lazy output production | Preserve original PassTextureNode in decode expression; compile only selected configuration | Arbitrary producer discovery for raw textures/depth attachments |
| Raw/filtered/temporal identity | Separate immutable returned nodes and explicit previous imports | General write-version/hazard/lifetime compiler |
| Optional features | Build a new owned composition when preset changes, dispose old composition, reset histories | Automatic target removal from existing PassNode and transient allocation |
| Depth reuse | Explicit producer ordering, same depth texture, preserving clear/depth state, compatible samples | Generic access-declaration validation and barrier scheduling |
| Debug views | Explicit selected views in same composition; account for their graph roots | Native graph resource-lifetime visualization |
| Histories | Per-camera/view owner, validity bit, resize/format/cut invalidation | Persistent storage lifecycle integrated into a graph compiler |

Suggested staged API sketch (illustrative, not a proposal for core exports):

```js
const surface = makeSurfacePass( scene, camera, layout, requestedFields );
const aoVisibility = ao( surface.depth, surface.normal, camera );
const reflectionRaw = ssr( opaqueColor, surface.depth,
  surface.normal, surface.roughness, camera );
const reflectionSpatial = filterReflection( reflectionRaw, surface.normal );
const composed = composeLighting( opaqueColor, aoVisibility, reflectionSpatial );
// Ownership separately records pass/effect disposal and temporal invalidation.
// No mutable resource named merely "reflection" is rebound under old consumers.
```

Declaring descriptors before assigning a producer can reduce construction order friction, but freeze and validate bindings before shader compilation. Required errors: missing producer, duplicate producer, same-frame cycle, incompatible shared attachment size/samples/format, illegal channel overlap. A previous-frame handle is an explicit temporal edge with external/persistent ownership. Neither engine's docs justify promising a small descriptor helper will provide all of these checks automatically.

## Fork/upstream provenance

Pinned fork dev: [`d7848108d3c2b93c322b99fe41b4c3dbafa620b5`](https://github.com/bhouston/three.js/commit/d7848108d3c2b93c322b99fe41b4c3dbafa620b5), commit timestamp 2026-10-14T17:15:50+02:00. Upstream `mrdoob/three.js` dev resolved on 2026-10-17 to [`c46746f581ee5db57aaa3c359ad74be6864248da`](https://github.com/mrdoob/three.js/commit/c46746f581ee5db57aaa3c359ad74be6864248da).

Compared upstream raw files pinned to that SHA against `git-dedup show <forkSHA>:<path>`, normalizing CRLF and trailing whitespace only. All nine match: `src/nodes/display/PassNode.js`, `src/nodes/core/MRTNode.js`, `src/nodes/core/OutputStructNode.js`, `src/renderers/common/RenderPipeline.js`, `src/nodes/core/NodeFrame.js`, `examples/webgpu_postprocessing_ao.html`, `examples/webgpu_postprocessing_ssr.html`, `examples/webgpu_postprocessing_ssgi.html`, `examples/webgpu_deferred.html`. Their audited behavior is not fork-only at these revisions. This is deliberately a scoped comparison, not a whole-tree equivalence claim. No fetch, branch switch, or repository mutation was performed.
