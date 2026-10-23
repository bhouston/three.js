# Hierarchical culling research

For a large outdoor first person scene, start with a spatial hierarchy of chunks, hierarchical frustum tests, distance limits and spatially partitioned batching. Add occlusion where buildings or terrain actually hide substantial work. Share the hierarchy with streaming, but keep resource residency separate from visibility. This is a design study and reproducible CPU experiment, not a renderer API proposal or measured rendering optimization.

## Current Three.js behavior

Code reviewed at dev commit `087f01f5d20fb8ea527395e04780bbbe7d1345a0`:

- [WebGLRenderer.projectObject](../../src/renderers/WebGLRenderer.js) recursively visits children. Failing a mesh's frustum test skips that mesh's render-list entry, not its descendants. `visible = false` does prune a subtree. Layer mismatch does not prune children, since they may use other layers.
- [Renderer._projectObject](../../src/renderers/common/Renderer.js) has the same broad behavior. Its `intersectsFrustum` object hook is an individual object test, not a spatial candidate enumeration interface. BundleGroup can reuse render lists; it is not a general spatial hierarchy.
- Groups do not acquire enclosing bounds automatically. A transform hierarchy is not necessarily spatially coherent: a parent's geometry need not contain its children's geometry.
- Matrix updates occur before candidate selection and can themselves walk the scene. Hierarchical culling alone will not eliminate that cost. Static transforms, explicit dirty updates, animation, sorting, shadows and draw submission need separate profiling.
- InstancedMesh culls a whole batch; BatchedMesh can cull individual instances. Neither substitutes for a world spatial index. Split huge batches into spatial chunks, trading additional draws for tighter rejection.

Thus the linear-traversal premise is substantially correct for ordinary visible scene graphs, with exceptions for invisible subtrees and cached bundles. It is not accurate to say every Object3D always incurs a geometry/frustum intersection.

## Practical hierarchy choices

| Structure | Appropriate use | Main cost |
| --- | --- | --- |
| Static object BVH | Irregular object/chunk distribution | Build cost; refit or rebuild after edits |
| Loose quadtree or grid | Mostly ground-based outdoor world | Vertical overlap; large objects need careful ownership |
| Loose octree | Volumetric worlds | More nodes and update bookkeeping |
| Dynamic AABB tree | Moving actors | Refits/reinsertions; fat bounds increase false positives |
| Mesh triangle BVH | Ray and geometry queries | Does not itself accelerate renderer object traversal |

Recommended starting design: static chunk BVH plus a separate dynamic index. Leaves contain spatially bounded batches or a small set of renderables. Maintain world-space conservative bounds. Bounds must include animation, skinning, shader displacement, instance changes and descendants represented by that leaf. Objects without trustworthy bounds stay in an always-considered list. Refit only dirty ancestor chains; rebuild when repeated movement makes tree quality poor. Account for index update time and memory in any performance claim.

Classify a node AABB against six frustum planes. Reject the subtree when wholly outside any plane. Pass a six-bit mask to children, clearing planes for which the entire node is inside. With a zero mask, descendants need no more plane tests, but still need enumeration and other visibility rules. Worst-case work remains linear, especially when nearly everything is visible; there is no general logarithmic guarantee.

Plane masking and plane coherence are established techniques in [Assarsson and Moller, Optimized View Frustum Culling Algorithms](https://www.tandfonline.com/doi/abs/10.1080/10867651.2000.10487517). A cheap temporal improvement is to test the node's last rejecting plane first. Its cached index is an ordering hint, not a cached visibility answer.

## Incremental views and cubemaps

Rechecking only last frame's visible objects is incorrect: newly entering objects would never be discovered. Reuse the index, traversal order and rejecting-plane hints; restart containment masks for each query. More ambitious updates can maintain a conservative expanded-frustum candidate cache and refresh it when the view exits its validity region. Small angular motion can expose a large distant strip, so frustum-difference queries are not automatically cheaper. Camera cuts, projection changes and dynamic bounds invalidate relevant proofs.

Use separate state for every camera, cube face, cascade, eye and probe. Query all six faces against the same index; optionally traverse once with a six-view bitset and propagate per-face plane masks. Large objects can belong to multiple faces. This reduces repeated traversal, not the required six views' shading. A capture-range broad phase can first reject distant chunks; a sphere enclosing each face's truncated frustum is a conservative option. The camera far value is not itself a valid sphere radius for a perspective frustum's corners.

[PMREMGenerator.fromScene](../../src/extras/PMREMGenerator.js) captures scene views before filtering. Hierarchical selection can help that capture stage. PMREM from an existing environment texture filters a texture and does not traverse scene objects, so this optimization does not accelerate that path's scene selection. Diffuse probe captures similarly benefit during scene rendering; spherical harmonic integration or filtering is separate work. Probe scheduling, static capture caches and lower update frequency may save more than culling alone.

## Occlusion and a shared visibility pipeline

Frustum culling answers whether an object could be in view; occlusion asks whether nearer geometry hides it. Both can use node bounds and per-view state. A useful pipeline is distance and frustum rejection, LOD selection, optional occlusion, then draw generation. Large opaque buildings and hills are useful occluders; foliage and thin/transparent objects require conservative treatment.

[CHC](https://www.cg.tuwien.ac.at/research/vr/chcull/) combines hierarchy traversal, front-to-back ordering and asynchronous hardware queries. [CHC++](https://www.cg.tuwien.ac.at/research/publications/2008/mattausch-2008-CHC/) adds adaptive prediction and query batching. The lesson for WebGL2 is to query selected coarse bounds, overlap useful work and poll availability later, rather than block on each object's result. A previously occluded object must be retested with a proxy or another conservative recovery mechanism; suppressing its draw forever prevents discovering disocclusion.

Three.js already has query plumbing in [WebGPUBackend](../../src/renderers/webgpu/WebGPUBackend.js), `object.occlusionTest`, `renderer.isOccluded()` and [webgpu_occlusion](../../examples/webgpu_occlusion.html). The example visualizes query results; this is not automatic hierarchical occlusion culling. Query outcomes reflect the depth/order of the pass that issued them and arrive asynchronously. A stale result is not a current-frame proof. Keep view identity, bounds versions and camera-cut invalidation explicit, with visible fallback for unknown results. Even delayed conservative policies need disocclusion tests and visual validation.

For WebGPU, a depth pyramid (Hi-Z/HZB), conservative projected bounds, GPU compaction and indirect rendering are a promising later stage. This needs renderer integration to avoid reading every object's visibility back to the CPU. Handle reversed depth, near-plane intersections, multisampling and depth holes correctly. Current-frame depth from selected occluders gives stronger evidence than stale depth; temporal prediction needs recovery and camera-cut handling. On sparse open terrain, depth generation and testing may cost more than they save.

## Unreal and Unity

[Unreal's documented Actor visibility pipeline](https://dev.epicgames.com/documentation/unreal-engine/visibility-and-occlusion-culling-in-unreal-engine) orders distance, frustum, precomputed visibility and dynamic occlusion tests. It describes hardware queries and an optional mipmapped HZB approach. This describes conventional Actor visibility, not the whole Nanite renderer.

Unity offers both [baked occlusion data](https://docs.unity.com/en-us/engine/6000.3/manual/cameras/occlusion-culling/occlusion-culling) used at runtime and [GPU occlusion with GPU Resident Drawer in URP](https://docs.unity.com/en-us/engine/6000.3/manual/analysis/graphics-performance-profiling/in-urp/gpu-culling). The latter uses current and previous frame depth and retains objects tested visible in either frame. The baked and GPU systems have different constraints; neither should be summarized as simply an octree that rejects invisible objects.

## Core integration and streaming

An initial addon can query chunks and set group visibility before each render invocation, provided it owns those flags and restores them afterward. Player-camera visibility must not leak into reflection or shadow passes. Updating all flags or attaching all objects can reintroduce linear work; chunk-level selection is more useful. This approach does not automatically handle renderer-internal shadow views.

A future opt-in core boundary should enumerate candidates per view/pass while preserving render-list construction semantics. The default enumerator remains scene traversal. An indexed provider owns bounds and updates and supplies a conservative candidate superset. Core must still preserve inherited visibility, layers, group renderOrder, LOD behavior, clipping groups, lights, material groups, callbacks, bundles, ArrayCamera behavior and unbounded objects. Shadows require their own candidates; main-camera rejection must not remove offscreen shadow casters. A hook after full scene traversal would be too late to solve the CPU problem. CPU and GPU providers may need different submission paths rather than one callback pretending they are interchangeable.

Share chunk IDs, conservative bounds, HLOD metadata and dirty versions with streaming. Keep loaded, GPU-resident, render-candidate and occluded states separate. Prefetch in an expanded region using velocity and expected turns; retain nearby content behind the player and use hysteresis, request cancellation, concurrency limits and CPU/GPU byte budgets. Keep coarse parents until detailed children are resident. Shadows, probes, gameplay and physics can need resources outside the main frustum. Do not unload based on a transient occlusion result.

[3D Tiles](https://github.com/CesiumGS/3d-tiles/blob/main/specification/README.adoc) is a useful reference for hierarchical refinement and screen-space error; [Unreal World Partition](https://dev.epicgames.com/documentation/unreal-engine/world-partition-in-unreal-engine) illustrates loading cells around streaming sources. For an FPS, share spatial metadata but let gameplay and streaming policies remain application-owned.

## Reproducible experiment and next validation

Run `node utils/culling/experiment.js` from the repository root. It uses source modules, seeded static AABBs, a median-split BVH with eight-object leaves and inherited plane masks. It compares exact sorted candidate IDs with Three.js `Frustum.intersectsBox` for every view. The baseline deliberately uses the same AABBs rather than default renderer bounding spheres, isolating hierarchy effects. It is a synthetic operation-count experiment with no graphics context, not a renderer benchmark or visual demo.

Observed counts for 32,768 objects:

| Query workload | Flat box tests | Hierarchy node tests | Leaf visits | Flat plane tests | Hierarchy plane tests |
| --- | ---: | ---: | ---: | ---: | ---: |
| 120 walking views | 3,932,160 | 48,380 | 126,456 | 8,218,562 | 148,074 |
| Six cube faces | 196,608 | 2,162 | 5,568 | 401,377 | 6,989 |
| 80 teleports and wide views | 2,621,440 | 111,284 | 419,064 | 6,541,272 | 80,233 |

These counts establish less plane-testing work for this distribution, not an equivalent speedup: containment tests do extra arithmetic, tree building allocates and sorts, and accepted leaves still require output processing. Static data means there is no refit cost here. The experiment implements neither temporal caching nor occlusion nor renderer integration.

Next step: a visual chunk-selection example with identical flat and indexed outputs, dynamic refits, camera cuts, shadows and six-view captures. Record index maintenance, candidate construction, matrix updates, draw calls, triangles, memory and GPU cost separately. Include clustered, all-visible, sparse and moving scenes. For rendering throughput, verify a physical GPU adapter, disable vsync/frame caps, warm each implementation, run sequential multi-second trials, count GPU-completed frames and drain queued work between implementations. No rendering throughput or rendering-quality verification is claimed by this CPU experiment.
