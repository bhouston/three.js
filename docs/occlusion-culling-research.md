# Visibility and streaming research spike

Research date: 2026-10-23. Repository baseline: `087f01f5d20fb8ea527395e04780bbbe7d1345a0` on this fork's `dev` branch. Recommendations below are design proposals, not existing Three.js guarantees.

## Recommendation

For a large first-person world, use a shared spatial index, per-view visibility selection, and a separate streaming scheduler. Start with hierarchical CPU frustum culling and coarse room/portal or precomputed visibility for appropriate indoor content. Add asynchronous cluster occlusion queries where measurement justifies them. For large instance populations, investigate WebGPU depth pyramids and GPU-generated indirect instance counts.

There is a common architecture, but no universally best culling algorithm. Dense interiors, open terrain, destructible worlds, and many repeated instances need different policies. Keep those policies in addons/application code initially; introduce core integration only when a measured prototype identifies a missing renderer contract.

## Industry practice

| Technique | Useful case | Main cost or limitation |
| --- | --- | --- |
| Distance, screen size, LOD | Broadly applicable, especially outdoors | Needs content-specific thresholds; changes detail rather than proving occlusion |
| BVH/grid/octree frustum traversal | Large populations with spatial locality | Bounds maintenance, output enumeration; does not reject objects behind walls |
| Cells, portals, baked potentially visible sets (PVS) | Rooms, corridors, static buildings | Offline data/authoring; open doors and destructible walls require conservative handling |
| Hierarchical asynchronous occlusion queries | Expensive clusters behind large occluders | Query submission and delayed results; avoid one query per small mesh |
| Hierarchical depth (HZB) plus GPU-driven draws | Many instances, dynamic worlds | Depth passes/pyramid/compaction overhead and renderer integration |
| CPU software occlusion | CPU submission bottleneck on suitable platforms | Rasterization cost, conservative coverage, browser worker/WASM complexity |

[Unreal's documented visibility pipeline](https://dev.epicgames.com/documentation/en-us/unreal-engine/visibility-and-occlusion-culling-in-unreal-engine) applies distance, frustum, precomputed visibility, then dynamic occlusion. Its conventional hardware queries return a frame later; HZB is another documented option. These are the conventional visibility paths, not a complete account of Nanite's separate GPU cluster pipeline.

[Unity's baked occlusion](https://docs.unity.com/en-us/engine/6000.3/manual/cameras/occlusion-culling/occlusion-culling) trades preprocessing, memory and CPU work for reduced rendering in occluded environments. [Unity 6 URP GPU occlusion](https://docs.unity3d.com/6000.0/Documentation/Manual/urp/gpu-culling.html) works with GPU Resident Drawer, bounding spheres and depth textures. Its temporal policy renders objects unoccluded in either current or previous frame. That policy should not be confused with simply hiding objects using yesterday's visibility bits. Unity also documents that testing can cost more than it saves in low-occlusion scenes.

[Coherent Hierarchical Culling (CHC)](https://developer.nvidia.com/gpugems/gpugems2/part-i-geometric-complexity/chapter-6-hardware-occlusion-queries-made-useful) combines hierarchy, front-to-back traversal, prior visibility and asynchronous queries to reduce query overhead and waiting. Previous visibility guides scheduling; newly visible regions still need discovery. [Cesium's hierarchical plane masking](https://cesium.com/blog/2015/08/04/fast-hierarchical-culling/) is a useful CPU counterpart: children skip frustum planes already proven to contain their parent.

## What this Three.js branch already provides

- Both renderers perform frustum tests while traversing objects: [WebGLRenderer](../src/renderers/WebGLRenderer.js) and [common Renderer](../src/renderers/common/Renderer.js). The current branch calls `object.intersectsFrustum( frustum )`; inspect this branch rather than assuming an older release's API.
- The common renderer exposes `isOccluded( object )`, with query handling in [WebGPUBackend](../src/renderers/webgpu/WebGPUBackend.js) and [WebGLBackend](../src/renderers/webgl-fallback/WebGLBackend.js). [The existing example](../examples/webgpu_occlusion.html) sets `mesh.occlusionTest = true` and uses results to change a color. This is query infrastructure, not automatic scene-wide culling. It belongs to WebGPURenderer and its fallback, not the separate legacy WebGLRenderer.
- Queries are issued around rendered objects and resolved asynchronously. Removing a queried object from submission prevents fresh queries for it: a hidden object cannot discover its own reappearance. Keep a separate conservative proxy query or periodic/current-view retesting strategy.
- The scene hierarchy is an ownership/transform hierarchy, not automatically a spatial acceleration structure. A spatially arbitrary Group does not become a BVH merely by setting `frustumCulled`.

For WebGL2, [the specification](https://registry.khronos.org/webgl/specs/2.0/) supplies `ANY_SAMPLES_PASSED_CONSERVATIVE` and availability polling. A prototype can issue cluster-bound queries after opaque occluders, disable color/depth writes for bounds, poll without blocking, and restore renderer state. Raw context state changes must be coordinated with Three.js state caching; an explicit pass is preferable to sprinkling query calls through mesh callbacks.

For WebGPU, [the specification](https://www.w3.org/TR/webgpu/) provides compute, storage buffers, occlusion queries and indirect draws. An HZB path should retain results on the GPU and compact visible instances or write indirect counts. Reading thousands of flags back to JavaScript can erase the benefit. API capabilities alone do not mean that an arbitrary existing Three.js scene can be submitted this way without batching/material constraints and pass integration.

## Incremental updates between frustums

The previous visible list is insufficient: it contains no record of objects just outside the old frustum. Camera translation, rotation or projection changes can reveal them. Retest a hierarchy covering the complete resident population, or maintain both an accepted/rejected partition and a conservative boundary frontier.

Start with these low-risk forms of coherence:

1. Propagate an inside-plane bitmask **within the current traversal**. Parent containment proves descendant containment if bounds are valid.
2. Cache the last rejecting plane and try it first next frame. This changes test order only; every result still uses the new frustum. It remains correct after teleports and FOV changes.
3. Retain stable object IDs and compute `entered = current - previous`, `exited = previous - current` for application/render-proxy updates. This saves downstream work but does not remove discovery cost.
4. Refit moving bounds, rebuild degraded trees when needed, and consider separate static/dynamic indexes. Include transform propagation in measurements.

A stronger, conservative temporal certificate is possible. For a static AABB with center `c` and half-extents `e`, plane `(n,d)` has maximum signed distance `sMax = n·c + d + |n|·e` and minimum `sMin = n·c + d - |n|·e`. If the new plane is `(n+Δn,d+Δd)`, every point's signed-distance change is bounded by:

```text
E = |Δn·c + Δd| + |Δn|·e
```

An old outside classification remains valid when `sMax + E < 0`; an old inside classification remains valid when `sMin - E >= 0`. Apply this per plane, with numerical tolerances and plane coefficients from the cached reference view, not merely the previous incremental delta. New bounds require new certificates. The per-node bookkeeping can exceed the saved tests; benchmark before adopting it. Revalidating boundaries is mandatory when a certificate fails. Rotation often invalidates many distant nodes even for a small angular change.

Do not carry old plane masks into a changed view without such proof. Likewise, temporal occlusion is harder than frustum coherence: a moved camera or door can invalidate the depth evidence even if the object's bounds stay unchanged. Camera cuts, moving occluders, near-plane intersections, camera-inside bounds and missing query results should fall back to drawing/retesting.

## Executable CPU experiment

Run from the repository root, with Node.js:

```sh
node test/benchmarks/frustum-coherence.js
```

[The script](../test/benchmarks/frustum-coherence.js) compares a flat AABB scan, binary spatial hierarchy with per-traversal plane masks, and the same hierarchy trying each node's previous rejecting plane first. All modes enumerate candidate IDs. It checks all candidate sets against `Frustum.intersectsBox` over 720 views covering walking, fast turns, camera cuts and FOV/aspect changes: 2,160 set comparisons over 4,096 static boxes.

An initial Windows001 run produced:

| Mode | CPU ms/update | Plane evaluations/update | Candidate IDs/update |
| --- | ---: | ---: | ---: |
| Flat | 0.0473 | 14,278 | 1,073 |
| Hierarchy and masks | 0.0167 | 772 | 1,073 |
| Hierarchy and cached rejection order | 0.0165 | 767 | 1,073 |

The large reduction comes from spatial hierarchy and masks. Temporal ordering adds little on this scene. Plane evaluations are not equal-cost operations: hierarchy classification computes both support vertices. These are illustrative single-process CPU measurements, not statistically established speedups. The benchmark warms each mode for 300 updates and measures 1,200 updates sequentially, excluding view construction, correctness checks, tree building and sorting. It includes candidate enumeration and temporary allocations; it does not model dynamic refits, scene matrix updates, renderer traversal, draw calls or GPU work. Internal nodes retain ID arrays for cheap accepted-subtree enumeration, trading memory for simplicity. No GPU context is created and no rendering performance or visual correctness is claimed.

This intentionally establishes the value of the simpler hierarchy before implementing temporal boundary certificates. It does **not** implement occlusion, a production addon, or an incrementally maintained visible scene.

Validation: the experiment's candidate-set assertions and ESLint check pass. `npm test` passes core lint, 1,982 core unit tests (one TODO), and 139 addon unit tests with zero failures. The browser runner printed a resource 404 and Windows already-exited-process cleanup messages, but exited successfully. This JavaScript repository has no dedicated typecheck script. Browser unit tests are not used as evidence of GPU rendering correctness; this spike does not perform GPU visual validation.

## Unified design and core integration

Use a shared world index with stable IDs and conservative world bounds. Compose coarse stages followed by renderer-specific refinement:

```text
world metadata / spatial index
  ├─ streaming interest → prioritized loads → resident assets
  └─ per-view distance → frustum → PVS/portals → optional occlusion → draw candidates
```

Proposed addon contracts should accept a view/pass identifier, current frustum, world/bounds revision and previous view state, and return candidates plus entered/exited IDs and counters. Occlusion backends additionally need depth provenance, query age and invalidation state. Unknown must mean potentially visible. Index residency, geometric visibility, query scheduling and asset lifetime are separate concerns even when sharing cells.

For CPU integration, select spatial chunk roots before rendering, so rejected branches avoid traversal. Toggling thousands of leaves can leave the full traversal and matrix-update costs intact. A production integration should measure those costs explicitly. Global `visible` flags also affect other passes; maintain per-view results or conservatively include the union of required passes when adapting through scene membership.

Keep policy (BVH vs grid vs portals, query budgets, streaming priority) outside core. Possible future core primitives are a supported per-pass candidate provider before render-list construction, conservative aggregate bounds for subtree pruning, and explicit occluder/query/refinement passes. Any such API must define matrix updates, lights, transparent sorting, nested groups/LOD, shadows, reflections, XR views, clipping, disposal and query lifetime. Existing query and frustum extension points should be evaluated first. Avoid a global `object.isVisibleThisFrame` property: visibility is view- and pass-dependent.

## The proposed two-scene FPS architecture

The distinction is useful, but model it as **world database + resident render representation**, rather than two Scenes sharing Object3D membership. An Object3D has one parent; moving it between scenes reparents it and can change world transforms. Cloning a complete scene also duplicates transform/state maintenance and complicates lifetime management.

Keep complete lightweight world metadata separately. Maintain stable resident chunk roots in a render scene, or explicit render proxies sharing geometry/materials with controlled ownership. Change membership at cell/residency transitions rather than rebuilding it for every slight camera turn. Then compute per-view candidates inside the resident region. Shadow casters, mirrors, secondary cameras and picking can require objects outside the main-camera candidate list; physics, audio and gameplay should not depend on main-camera visibility.

Streaming can share the spatial partition and use visibility as a priority signal, but **occluded does not mean safe to unload**. Load a larger region than the current frustum, using movement prediction, nearby portal/navigation adjacency and explicit teleport destinations. Keep hidden nearby assets resident, add distance/time hysteresis, cap concurrent requests and per-frame decode/upload work, cancel stale priorities, and apply an explicit memory budget. Retain coarse fallback geometry where useful. Turning the camera should not cause a network round trip.

[Unreal World Partition](https://dev.epicgames.com/documentation/en-us/unreal-engine/world-partition-in-unreal-engine) is a relevant model: spatial cells are loaded around streaming sources, including destination preloading. Rendering culling and streaming use related spatial knowledge while solving different scheduling problems.

## Suggested next experiments

1. A renderer example with selectable flat/hierarchical modes, frozen culling camera and a top-down candidate view. Include candidate counts, matrix/traversal time, draw calls, triangles and total frame time. Test static interiors and open terrain, with bounds refits as a separate workload.
2. An indoor cell/portal prototype supporting open/closed doors, followed by cluster-bound asynchronous queries using the existing common-renderer infrastructure. Render occluders first; bounds queries must not write depth or occlude each other. Transparent surfaces are not general occluders; alpha-tested/deformed occluders require matching coverage. Keep hidden proxies testable and invalidate stale evidence.
3. A WebGPU HZB/indirect-instance prototype. For ordinary depth use farthest-depth reduction (max); reversed depth needs the opposite reduction and comparison. Cover the entire projected bound conservatively, including mip footprint and screen edges. Recheck disocclusion using current-view depth rather than treating old depth as proof.

Use real hardware adapters for visual/GPU claims. Benchmark open views where culling can lose, dense interiors where it should win, dynamic doors, fast turns, teleports, multiple cameras, shadow passes and camera-inside bounds. For GPU comparisons disable frame caps/vsync, warm variants, run sequentially for several seconds, count GPU-completed frames and drain work before switching. Report CPU selection, GPU pass costs, latency/artifacts and total frame time separately. This spike has only validated CPU candidate selection.
