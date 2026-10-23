# Hierarchical culling research

For a large outdoor first person scene, start with a spatial hierarchy of chunks, hierarchical frustum tests, distance limits and spatially partitioned batching. Add occlusion where buildings or terrain actually hide substantial work. Share the hierarchy with streaming, but keep resource residency separate from visibility. The accompanying example demonstrates application-level selection before considering renderer changes.

## Current Three.js behavior

Code reviewed at dev commit `087f01f5d20fb8ea527395e04780bbbe7d1345a0`:

- [WebGLRenderer.projectObject](../../src/renderers/WebGLRenderer.js) recursively visits children. Failing a mesh's frustum test skips that mesh's render-list entry, not its descendants. `visible = false` does prune a subtree. Layer mismatch does not prune children, since they may use other layers.
- [Renderer._projectObject](../../src/renderers/common/Renderer.js) has the same broad behavior. Its `intersectsFrustum` object hook is an individual object test, not a spatial candidate enumeration interface. BundleGroup can reuse render lists; it is not a general spatial hierarchy.
- Groups do not acquire enclosing bounds automatically. A transform hierarchy is not necessarily spatially coherent: a parent's geometry need not contain its children's geometry.
- Matrix updates occur before candidate selection and can themselves walk the scene. Hierarchical culling alone will not eliminate that cost. Static transforms, explicit dirty updates, animation, sorting, shadows and draw submission need separate profiling.
- InstancedMesh culls a whole batch; BatchedMesh can cull individual instances. Neither substitutes for a world spatial index. Split huge batches into spatial chunks, trading additional draws for tighter rejection.

Thus the linear-traversal premise is substantially correct for ordinary visible scene graphs, with exceptions for invisible subtrees and cached bundles. It is not accurate to say every Object3D always incurs a geometry/frustum intersection.

## Interactive example

[webgpu_culling_hierarchical.html](../../examples/webgpu_culling_hierarchical.html) renders 16,384 procedural buildings in 1,024 static InstancedMesh chunks. Settings switches between flat chunk tests and a binary hierarchy with inherited plane masks. Both select conservative chunk AABBs; the example disables the meshes' built-in frustum tests to isolate application selection. Drag to orbit or enable automatic walking. The overview draws every chunk, highlights selected chunks in green, and displays the main camera frustum and coarse hierarchy bounds.

At the initial 1600 by 1000 desktop view, both modes select 82 chunks. Flat selection visits 1,024 bounds and performs 2,792 plane tests; hierarchical selection visits 267 bounds and performs 348 plane tests. Counts vary with the viewport aspect ratio and camera pose. The counters describe CPU selection work, not measured rendering speedup. The overview updates every chunk and draws the entire world. The renderer also still visits scene children; no core traversal optimization is claimed. Static transforms are initialized once, then automatic scene matrix updates are disabled.

The example owns chunk visibility independently for its main and overview passes. It supports static bounds, one main view and no shadows, probe captures or occlusion. Moving geometry would require bounds updates and ancestor refits; shadows and captures would require their own queries. It introduces no renderer API or reusable spatial-index module.

Hardware verification used a non-fallback NVIDIA Pascal WebGPU adapter. External checks compared actual main-pass candidates with `Frustum.intersectsBox` after camera and projection changes in both modes, and found identical primary-view pixels for the initial flat and hierarchical views. Screenshots illustrate the controls and counts; Inspector FPS is incidental, not a benchmark result.

## Independent spatial index for large scenes

The first example already builds a spatial tree independently of Object3D parentage. Its small throughput gain has two main causes: batching reduces the selection problem to only 1,024 chunks, and toggling chunk visibility still leaves the renderer visiting every scene child. Both selection techniques also submit the same draws. Fewer plane tests alone therefore save little total frame time.

[webgpu_culling_spatial.html](../../examples/webgpu_culling_spatial.html) instead creates 100,000 independent Mesh spheres randomly distributed on a plane. Its uniform scene has 100,000 direct children. Its clustered scene distributes the spheres among 100 deliberately non-spatial groups: every group contains objects from distant clusters. No instancing or authored spatial grouping reduces the node count. A 10,000-object all-visible view is a control where traversal cannot reject anything.

[SceneSpatialIndex](../../examples/jsm/utils/SceneSpatialIndex.js) extracts world-space bounding spheres, then builds a balanced binary BVH over AABBs enclosing those spheres. Splits follow the longest spatial axis and median center, with up to eight entries per leaf. This is a practical balanced tree, not a claim of globally optimal visibility partitioning. It does not follow the authored parent/child structure.

The experimental common-renderer hook `getRenderChildren( camera, frustum )` lets the helper return only a subset of each node's existing direct children. The helper reconstructs the ancestor paths of spatially selected objects and restores original sibling order in those candidate lists. The renderer continues ordinary traversal of those paths, preserving parent ownership, inherited visibility, layers and group state. It does not flatten, clone, reparent, swap children arrays or mutate object visibility. The index query and candidate-path construction occur for each render view. Fully contained root bounds take an ordinary-traversal fast path.

The new example compares stock rendering, a flat scan of cached world-space spheres using the same child-selection hook, and BVH selection. This separates the gain from skipping renderer traversal from the additional gain of accelerating the visibility query. Every mode uses the renderer's sphere test at leaves; node AABBs enclose those spheres. Unlike the earlier chunk experiment, tighter leaf bounds do not explain differences in draw counts.

The prototype indexes plain static meshes. Lights, uncullable/custom renderables and unsupported dynamic or bundled branches retain ordinary traversal. Unsupported roots retain their existing traversal too. Rebuild after changing indexed world transforms, geometry bounds, frustumCulled or child membership; visibility and layer changes still follow ordinary rendering rules. Bounds must conservatively include shader displacement. Rebuild cost and retained index memory are additional costs, and the prototype does not provide incremental refitting or automatic change tracking.

The hook currently applies to the common renderer used by WebGPURenderer; WebGLRenderer continues ordinary traversal. It is an experimental core boundary for review, not a finalized cross-renderer API. The example imports source modules so its raw GitHub demo exercises the hook without committing generated build files. FrustumArray and transformed/non-spatial hierarchies have unit coverage; production shadow, probe, bundle and custom-renderer integration needs further review and rendering coverage.

## Large node count measurements

Hardware: NVIDIA GTX 1050, driver 32.0.15.6094, Chrome 155.0.8059.86, non-fallback WebGPU, 1600 by 1000 pixels, DPR 1. All three modes produced identical draw-object IDs in the same order and identical image pixels in each case. Draws include the ground and background. The flat world-space index and BVH both use the new child-selection hook; stock mode traverses the original graph and performs ordinary renderer sphere tests.

| Scene | Draws per frame, all modes | Stock ms/frame | Flat index ms/frame | BVH index ms/frame | Stock / BVH |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100,000 uniform spheres, flat graph | 662 | 13.135 | 3.961 | 2.066 | 6.36x |
| 100,000 clustered spheres, 100 non-spatial groups | 1,655 | 14.124 | 5.606 | 3.650 | 3.87x |
| 10,000 uniform spheres, flat graph | 56 | 1.561 | 1.373 | 1.081 | 1.44x |
| 10,000 spheres, all visible | 10,002 | 26.036 | 29.252 | 24.841 | Inconclusive |

These are medians of three sustained-throughput averages, each computed as elapsed wall time divided by GPU-completed frames over a five-second trial. Implementations run sequentially, with rotated trial order, a one-second warmup before each trial, no vsync/frame cap, and GPU queue completion after each four-frame batch. Queues are drained before switching modes. No individual frames are timed. Camera updates, selection, ancestor-path construction, renderer traversal, sorting, draws and GPU completion are included. Inspector and HUD are excluded for every mode. Static scene matrix updates are disabled equally after initial setup; a dynamic application needs a separate update strategy.

| Scene | Stock trial range, ms | Flat trial range, ms | BVH trial range, ms |
| --- | ---: | ---: | ---: |
| Uniform 100,000 | 12.979–13.158 | 3.923–3.971 | 2.003–2.080 |
| Clustered 100,000 | 13.570–14.198 | 5.475–5.627 | 3.551–3.904 |
| Uniform 10,000 | 1.536–1.570 | 1.358–1.378 | 1.066–1.081 |
| All-visible 10,000 | 24.682–26.465 | 27.134–30.596 | 24.753–28.995 |

In the uniform 100,000 case, renderer projection visits fall from 100,004 to 664; flat selection tests 100,000 cached spheres, while the BVH visits 430 tree nodes and tests 334 spheres. In the clustered case, visits fall from 100,104 to 1,757 despite every authored group spanning distant clusters. Draw submission is unchanged. The roughly 1.9x uniform-case gain from flat index to BVH isolates the additional benefit of hierarchical selection after avoiding the renderer's full scan.

The all-visible case cannot reject renderables. BVH root containment falls back to ordinary traversal, and its timing range overlaps stock rendering. Flat indexing adds enumeration, candidate-path construction and sorting without saving visits. This control does not establish a BVH speedup. Large improvements require many offscreen nodes and a meaningful CPU visibility/traversal cost; GPU-bound or all-visible scenes need batching, LOD or other techniques.

Index construction is excluded from steady-state timings. Single observed builds took approximately 575 ms for uniform 100,000, 560 ms for clustered 100,000 and 40–44 ms for the 10,000 cases, including world-bound extraction. Source Mesh allocation is separate setup work. Extra index memory and rebuild/refit costs must be budgeted; these measurements cover static scenes, not moving-object maintenance. Raw trials and the external runner are task artifacts rather than a repository benchmark suite.

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

## Integration scope

An initial addon can query chunks and set group visibility before each render invocation, provided it owns those flags and restores them afterward. Player-camera visibility must not leak into reflection or shadow passes. Updating all flags or attaching all objects can reintroduce linear work; chunk-level selection is more useful. This approach does not automatically handle renderer-internal shadow views.

A production candidate-enumeration hook remains a design discussion. The large-node prototype demonstrates the traversal bottleneck and preserves ordinary ancestor paths, but it does not settle a public API. Integration must preserve inherited visibility, layers, group renderOrder, LOD, clipping groups, lights, material groups, callbacks, bundles and ArrayCamera behavior. Shadows need their own candidates, including offscreen casters. Filtering after full traversal would not solve the CPU cost. CPU enumeration and GPU-driven submission may need different designs; this note does not propose a shared CPU/GPU provider API.

The original chunk selection stays local to its example. The requested large-node follow-up adds a separately tested helper and the small experimental traversal hook, using existing bounds classes. The helper has no TSL/WebGPU dependency; acceleration depends on the common renderer consuming its hook. Supported bounds and update ownership are documented rather than enforced with defensive runtime throws. Tests check rendering candidates and source-graph semantics rather than validation failures.

For an eventual Three.js example, use small procedural assets, a default WebGPURenderer, Inspector controls and an `animate()` loop. Keep comments short and avoid test-only globals or backend-selection toggles. Treat WebGL2 occlusion queries as research context, not a proposal for a new EffectComposer pass family. A later node-based occlusion feature should be reviewed independently from the CPU hierarchy foundation.

## Streaming

Share chunk IDs, conservative bounds, HLOD metadata and dirty versions with streaming. Keep loaded, GPU-resident, render-candidate and occluded states separate. Prefetch in an expanded region using velocity and expected turns; retain nearby content behind the player and use hysteresis, request cancellation, concurrency limits and CPU/GPU byte budgets. Keep coarse parents until detailed children are resident. Shadows, probes, gameplay and physics can need resources outside the main frustum. Do not unload based on a transient occlusion result.

[3D Tiles](https://github.com/CesiumGS/3d-tiles/blob/main/specification/README.adoc) is a useful reference for hierarchical refinement and screen-space error; [Unreal World Partition](https://dev.epicgames.com/documentation/unreal-engine/world-partition-in-unreal-engine) illustrates loading cells around streaming sources. For an FPS, share spatial metadata but let gameplay and streaming policies remain application-owned.

## External experiment

The [original experimental script](https://github.com/bhouston/three.js/blob/26ee982277a/utils/culling/experiment.js) remains available at its historical commit, and a standalone copy accompanies the task artifacts. To reproduce the original, place the script at `utils/culling/experiment.js` in a checkout of the reviewed commit and run `node utils/culling/experiment.js`. The standalone artifact accepts the checkout path: `node culling-experiment.mjs <three.js-checkout>`.

The experiment uses source modules, seeded static AABBs, a median-split BVH with eight-object leaves and inherited plane masks. It compares exact candidate IDs with Three.js `Frustum.intersectsBox` across 206 views and checks fully contained, fully outside and plane-touching cases. The baseline uses the same AABBs rather than renderer bounding spheres. It counts operations without a graphics context. It is retained externally rather than introducing a maintained benchmark suite into Three.js.

Observed counts for 32,768 objects:

| Query workload | Flat box tests | Hierarchy node tests | Leaf visits | Flat plane tests | Hierarchy plane tests |
| --- | ---: | ---: | ---: | ---: | ---: |
| 120 walking views | 3,932,160 | 48,380 | 126,456 | 8,218,562 | 148,074 |
| Six cube faces | 196,608 | 2,162 | 5,568 | 401,377 | 6,989 |
| 80 teleports and wide views | 2,621,440 | 111,284 | 419,064 | 6,541,272 | 80,233 |

These counts establish less plane-testing work for this distribution, not an equivalent speedup: containment tests do extra arithmetic, tree building allocates and sorts, and accepted leaves still require output processing. Static data means there is no refit cost here. The experiment implements neither temporal caching nor occlusion nor renderer integration.

## Small follow-up steps

1. Use the accompanying static example as the foundation. Profile the complete path, including matrix updates, visibility changes and draw submission, before seeking a core hook. Keep throughput experiments external, for example via jsfiddle.
2. Extend validation to dynamic refits, camera cuts, all-visible scenes, shadows and six-view captures in separate follow-ups before proposing wider integration.
3. Discuss the smallest necessary renderer change with the relevant maintainers, keeping occlusion, GPU draw generation and streaming policy out of that foundation PR. Discuss node-system architecture with its maintainers before implementing it.

Record index maintenance, candidate construction, matrix updates, draw calls, triangles, memory and GPU cost separately. Keep performance trials external rather than adding `test/benchmark/` or a replacement utility suite. For rendering throughput, verify a physical GPU adapter, disable vsync/frame caps, warm each implementation, run sequential multi-second trials, count GPU-completed frames and drain queued work between implementations. No rendering throughput or rendering-quality verification is claimed by this CPU experiment.

The scope follows the supplied [summary of Mugen87's review preferences](https://gist.github.com/bhouston/1b66f04b2838229a36844cd66ea31c7e): small foundation changes, minimal API surface, existing abstractions and external performance experiments. The summary informs this proposal; it does not imply maintainer endorsement of a culling API or the research-note location.
