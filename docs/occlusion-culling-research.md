# Visibility and streaming research spike

Research date: 2026-10-23. Repository baseline: `087f01f5d20fb8ea527395e04780bbbe7d1345a0` on this fork's `dev` branch. Recommendations below are design proposals, not existing Three.js guarantees.

## Recommendation

For a large first-person world, use a shared spatial index, per-view visibility selection, and a separate streaming scheduler. Start with hierarchical CPU frustum culling and coarse room/portal or precomputed visibility for appropriate indoor content. Add asynchronous cluster occlusion queries where measurement justifies them. For large instance populations, investigate WebGPU depth pyramids and GPU-generated indirect instance counts.

There is a common architecture, but no universally best culling algorithm. Dense interiors, open terrain, destructible worlds, and many repeated instances need different policies. Keep those policies in addons/application code initially; introduce core integration only when a measured prototype identifies a missing renderer contract.

Following the supplied [maintainer review preferences](https://gist.github.com/bhouston/1b66f04b2838229a36844cd66ea31c7e), this spike pairs research with one small renderer example. Keep benchmarks external (jsbench/jsfiddle or task artifacts), reuse existing abstractions, and propose one small implementation at a time. The summary records review preferences rather than a formal project policy.

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

For WebGL2, [the specification](https://registry.khronos.org/webgl/specs/2.0/) supplies `ANY_SAMPLES_PASSED_CONSERVATIVE` and availability polling. Reuse the common renderer's query infrastructure for a prototype: query cluster bounds after opaque occluders, disable color/depth writes for bounds, and poll without blocking. Keep new rendering work in the node-based pipeline; do not propose a legacy EffectComposer pass family or bypass renderer state management with raw context calls.

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

## External CPU experiment

The task artifacts for [the research issue](https://github.com/bhouston/agent-swarm/issues/151) preserve `frustum-coherence.mjs` and `frustum-benchmark.txt`; no benchmark suite is added to Three.js. In the retained task workspace, run `node .swarm/artifacts/frustum-coherence.mjs` from the task root alongside its `three.js` checkout. The artifact imports that checkout's source files.

The experiment compares a flat AABB scan, binary spatial hierarchy with per-traversal plane masks, and the same hierarchy trying each node's previous rejecting plane first. All modes enumerate candidate IDs. It checks all candidate sets against `Frustum.intersectsBox` over 720 views covering walking, fast turns, camera cuts and FOV/aspect changes: 2,160 set comparisons over 4,096 static boxes.

An initial Windows001 run produced:

| Mode | CPU ms/update | Plane evaluations/update | Candidate IDs/update |
| --- | ---: | ---: | ---: |
| Flat | 0.0473 | 14,278 | 1,073 |
| Hierarchy and masks | 0.0167 | 772 | 1,073 |
| Hierarchy and cached rejection order | 0.0165 | 767 | 1,073 |

The large reduction comes from spatial hierarchy and masks. Temporal ordering adds little on this scene. Plane evaluations are not equal-cost operations: hierarchy classification computes both support vertices. These are illustrative single-process CPU measurements, not statistically established speedups. The benchmark warms each mode for 300 updates and measures 1,200 updates sequentially, excluding view construction, correctness checks, tree building and sorting. It includes candidate enumeration and temporary allocations; it does not model dynamic refits, scene matrix updates, renderer traversal, draw calls or GPU work. Internal nodes retain ID arrays for cheap accepted-subtree enumeration, trading memory for simplicity. No GPU context is created and no rendering performance or visual correctness is claimed.

This intentionally establishes the value of the simpler hierarchy before implementing temporal boundary certificates. It does **not** implement occlusion, a production addon, or an incrementally maintained visible scene.

The original experiment passed candidate-set assertions and ESLint. The initial PR passed `npm test`: core lint, 1,982 core unit tests (one TODO), and 139 addon unit tests, with zero failures. The browser runner printed a resource 404 and Windows already-exited-process cleanup messages but exited successfully. This JavaScript repository has no dedicated typecheck script. These checks establish no GPU rendering correctness or performance result.

## Interactive city example

[webgpu_culling_occlusion.html](../examples/webgpu_culling_occlusion.html) replicates the seeded 32 by 32 chunk layout and binary hierarchy from [the sibling hierarchical-culling example](https://github.com/bhouston/three.js/blob/research-hierarchical-culling/examples/webgpu_culling_hierarchical.html). Each of 1,024 InstancedMesh chunks contains 16 buildings: 16,384 opaque box skyscrapers in total. Heights are increased to 40-160 m; the camera starts at 1.7 m with a 700 m far plane. Inspector controls select no culling, flat chunk frustum tests, hierarchical frustum tests, or hierarchy plus occlusion. Automatic walking follows four connected streets at eye level. A 2D overview shows candidate and surviving chunks without rendering the entire city in another GPU view.

The city replaces the earlier stationary-query demonstration, which remains available at [its historical commit](https://github.com/bhouston/three.js/blob/8964bfd0110fe26c28f0ffc964533e72cccb1788/examples/webgpu_culling_occlusion.html). This version uses **current-view CPU geometric occlusion**, not asynchronous GPU query results. For each of the nearest 48 candidate buildings, camera-facing opaque box faces define shadow volumes: four side planes through the camera and face edges, plus a plane behind the face. A chunk or building is rejected only when its entire AABB lies strictly inside all five positive half-spaces of one volume, using a 0.02 m margin. Faces intersecting the near plane and nearly coplanar camera/face configurations are skipped.

The volumes are rebuilt every frame, so camera motion and turns need no stale-result recovery. Surviving instances are compacted within their original chunks; non-occlusion modes restore static instance matrices when necessary. The method is conservative for this exact solid-box geometry. An enclosing AABB of an irregular, hollow, displaced or transparent building is **not** a valid occluder. This is an application-specific experiment rather than a general Three.js occlusion API or a substitute for HZB. It does not combine multiple occluders to cover one bound, so some fully hidden objects can remain submitted.

At the initial 1440 by 1000 street view, no culling submits 16,384 buildings in 1,024 chunks (1,027 draw calls). Flat and hierarchical frustum selection submit the same 5,184 buildings in 324 chunks (327 calls). Occlusion reduces that to 454 buildings in 50 chunks (53 calls). A few dozen submitted buildings is not guaranteed: the long open street, distant rooftops and conservative bounds keep additional buildings. These are submission counts, not counts of pixels or buildings actually contributing to the image.

Hardware verification on NVIDIA RTX 3060 Ti / WebGPU Ampere found identical canvas pixels (no channel difference above two) between no culling and occlusion at six fixed views and ten walking positions, including street turns, an elevated view, a camera inside a building, and oblique views. Mode switching, flat/hierarchical candidate equivalence and viewport resizing were also checked. External throughput measurements compare stationary and moving cameras with warmups, sequential multi-second trials and GPU queue completion; results and methodology appear in the PR description and task artifacts. CPU sorting, volume construction, culling and instance packing must be included in any claimed benefit.

Measured median completed-frame cost with 48 candidate occluders was 4.709 ms stationary and 3.619 ms walking, versus 2.376 ms and 2.084 ms for hierarchy alone. With no culling, the same workloads took 6.728 ms and 6.861 ms. Thus this CPU technique reduces submissions substantially but is slower than frustum culling for these cheap instanced boxes. A short stationary budget sweep from 8 to 48 occluders traded 110 down to 53 calls and roughly 3.16 up to 4.61 ms/frame; it did not beat hierarchy alone. These results motivate GPU depth/occlusion work rather than a claim that fewer buildings automatically means a faster frame.

All visibility is owned by this one main view. Static buildings, opaque faces and known geometry are required; dynamic geometry needs updated bounds and occluder faces. Shadows, reflections, streaming, LOD and general meshes remain separate follow-ups. No renderer core change or public API is added.

## Unified design and core integration

Use a shared world index with stable IDs and conservative world bounds. Compose coarse stages followed by renderer-specific refinement:

```text
world metadata / spatial index
  ├─ streaming interest → prioritized loads → resident assets
  └─ per-view distance → frustum → PVS/portals → optional occlusion → draw candidates
```

An application prototype can track view/pass identity, current frustum, bounds revisions, candidate IDs and previous state internally. Occlusion additionally needs depth provenance, query age and invalidation state. Unknown must mean potentially visible. Avoid exposing these implementation details as new public result fields before a concrete use case requires them. Keep shared spatial/loading code independent of TSL/WebGPU; register renderer-specific integration explicitly at application level.

For CPU integration, select spatial chunk roots before rendering, so rejected branches avoid traversal. Toggling thousands of leaves can leave the full traversal and matrix-update costs intact. A production integration should measure those costs explicitly. Global `visible` flags also affect other passes; maintain per-view results or conservatively include the union of required passes when adapting through scene membership.

Keep policy (BVH vs grid vs portals, query budgets, streaming priority) outside core. First evaluate existing query and frustum extension points with one small example. If measurements then demonstrate a missing primitive, discuss that specific core change separately with the relevant maintainers, including sunag for node-system architecture. Its contract must cover matrix updates, lights, sorting, LOD, shadows, reflections, XR, clipping and lifetime. Reuse math/object abstractions rather than introducing selective direct-access fast paths. Keep pass-specific setup local; document restrictions without adding validation throws. A new public visibility API or helper module is not justified by this spike alone.

## The proposed two-scene FPS architecture

The distinction is useful, but model it as **world database + resident render representation**, rather than two Scenes sharing Object3D membership. An Object3D has one parent; moving it between scenes reparents it and can change world transforms. Cloning a complete scene also duplicates transform/state maintenance and complicates lifetime management.

Keep complete lightweight world metadata separately. Maintain stable resident chunk roots in a render scene, or explicit render proxies sharing geometry/materials with controlled ownership. Change membership at cell/residency transitions rather than rebuilding it for every slight camera turn. Then compute per-view candidates inside the resident region. Shadow casters, mirrors, secondary cameras and picking can require objects outside the main-camera candidate list; physics, audio and gameplay should not depend on main-camera visibility.

Streaming can share the spatial partition and use visibility as a priority signal, but **occluded does not mean safe to unload**. Load a larger region than the current frustum, using movement prediction, nearby portal/navigation adjacency and explicit teleport destinations. Keep hidden nearby assets resident, add distance/time hysteresis, cap concurrent requests and per-frame decode/upload work, cancel stale priorities, and apply an explicit memory budget. Retain coarse fallback geometry where useful. Turning the camera should not cause a network round trip.

[Unreal World Partition](https://dev.epicgames.com/documentation/en-us/unreal-engine/world-partition-in-unreal-engine) is a relevant model: spatial cells are loaded around streaming sources, including destination preloading. Rendering culling and streaming use related spatial knowledge while solving different scheduling problems.

## Suggested next experiments

1. Profile the city example across streets, intersections and elevated views. Compare CPU volume construction and candidate processing with draw savings before expanding its scope. Keep throughput benchmarks external.
2. An indoor cell/portal prototype supporting open/closed doors, followed by cluster-bound asynchronous queries using the existing common-renderer infrastructure. Render occluders first; bounds queries must not write depth or occlude each other. Transparent surfaces are not general occluders; alpha-tested/deformed occluders require matching coverage. Keep hidden proxies testable and invalidate stale evidence.
3. A WebGPU HZB/indirect-instance prototype. For ordinary depth use farthest-depth reduction (max); reversed depth needs the opposite reduction and comparison. Cover the entire projected bound conservatively, including mip footprint and screen edges. Recheck disocclusion using current-view depth rather than treating old depth as proof.

Use real hardware adapters for visual/GPU claims. Benchmark open views where culling can lose, dense interiors where it should win, dynamic doors, fast turns, teleports, multiple cameras, shadow passes and camera-inside bounds. For GPU comparisons disable frame caps/vsync, warm variants, run sequentially for several seconds, count GPU-completed frames and drain work before switching. Report CPU selection, GPU pass costs, latency/artifacts and total frame time separately. This spike validates CPU candidate selection and selected hardware-rendered views, with external stationary and walking throughput measurements.
