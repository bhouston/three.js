# Optional Hi-Z SSR investigation

Studied October 24, 2026 on `interior-ssr-review`, starting at `d303775b6a43`. Sources are [PR #76](https://github.com/bhouston/three.js/pull/76) (head `faa7a27dcb30`) and [PR #79](https://github.com/bhouston/three.js/pull/79) (head `8e72a8b525e7`). Both PRs are merged remotely, but their implementations were absent from this local branch. This is a selective port, preserving the newer local SSR fixes, rather than a merge of their older SSRNode.

## Follow-up: online research and optimized construction

The user subsequently stopped another concurrent render. The first investigation's absolute FPS values below are therefore historical, potentially contended measurements; use the new clean runs for performance conclusions. Quality equality is unaffected by that contention.

Primary sources reviewed:

- [AMD SSSR integration and pipeline](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/stochastic-screen-space-reflections/): depth hierarchy, classification, indirect ray dispatch, roughness/variance driven ray density, hit confidence, and reconstruction.
- [Pinned AMD SSSR traversal and validator source](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/blob/v1.1.4/sdk/include/FidelityFX/gpu/sssr/ffx_sssr_intersect.h): exact cell-boundary advancement, initial cell exit, depth-direction guard, mirror leaf selection, and validation.
- [AMD SPD implementation design](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/single-pass-downsampler/): local shared-memory reductions, optional wave operations, and the cross-workgroup synchronization requirements of its final reduction.
- [McGuire and Mara's implementation and paper](https://casual-effects.blogspot.com/2014/08/screen-space-ray-tracing.html): perspective-correct screen-space DDA and finite depth intervals.
- [Stachowiak's original stochastic SSR work](https://h3.gd/stochastic-ssr/): importance sampling coupled to spatiotemporal reconstruction, including its ratio-estimator interpretation.

### Are the implementations correct?

**The hierarchy is correct for its stated nearest/farthest footprint contract.** Normal depth reduces nearest with min and farthest with max; reversed depth swaps these operations. Odd dimensions retain every trailing texel. The hardware reference checks both channels and exact footprints rather than relying on a pretty rendered image.

**Conservative skipping preserves the existing marcher; it is not the SSSR traversal algorithm.** Four corner samples conservatively cover a padded batch rectangle at the selected mip, since its extent is at most one nominal cell width per axis. The margin permits skipping only when all original ray samples are in front of the nearest surface. Updating refinement history at the skipped endpoint is necessary. This preserves the original sparse exponential sample schedule and its existing limitations; it does not give the complete pixel coverage of DDA.

**Adaptive implements the central cell-boundary traversal idea, but is not a complete AMD SSSR implementation or a proven reference tracer.** It advances to an XY exit or eligible depth plane, coarsens after empty-space traversal, and descends for leaf acceptance. Projected depth is affine for standard/reversed depth, and depth-plane advancement is disabled for rays moving toward the camera. These are consistent with the published algorithm. Its one-texel initial step differs from AMD's first-cell exit. Leaf acceptance retains our permissive infinite-line distance test; stochastic tracing lacks AMD's explicit self-hit and hit-backface validation. These differences can affect correctness and must not be dismissed as pure optimization. The current adaptive tests establish finite execution and actual hits across camera/depth modes, not nearest-intersection correctness or physical fidelity.

Adaptive also caps all hierarchy visits, including descents and ascents. Its 64 iterations are not 64 leaf tests. Coarse leaves, starts, iteration caps and changed hit validation all need reference-image assessment. Perfect mirrors should retain mip-zero leaf resolution; raising the minimum mip is a quality tradeoff.

### Why the expected large gain is absent

1. **The comparison workload is different.** The original fast mirror demo replaced a potentially very long pixel march. Interior's stochastic baseline already uses at most 63 coarse samples at quality one. Replacing those cheap, incomplete samples with hierarchy visits is not guaranteed to save work or retain the same intersections.
2. **Construction and representation have real costs.** Our RG32F atlas stores both nearest and farthest bounds, whereas SSSR consumes a single-channel hierarchy. The original local builder dispatched once per level, copied full-resolution depth, and retained intermediate textures. Construction must be repaid by tracing savings every frame.
3. **Conservative failures add work.** An attempted skip performs four hierarchy loads and rectangle/depth math. If any foreground object makes the bound inconclusive, the original leaf tests still run. Low quality can leave only one useful batch opportunity.
4. **Only part of the published pipeline is present.** AMD classifies rays, compacts them into compute work, reduces ray density on rough surfaces, uses temporal variance, and can terminate low-occupancy rough-ray waves. Ours is a fragment shader over the output image. Divergent stochastic rays and expensive lanes remain. A roughness fade applied after tracing saves no ray work.
5. **The benchmark includes the entire application.** Geometry, SSGI, AO, fallback and denoising still run. A substantial builder-only gain can yield only a small full-frame change. The new isolated builder benchmark and untimed ray counters distinguish these costs; full-frame FPS alone cannot identify their cause.

The first three points follow directly from the implementation; the relative costs of memory bandwidth, divergence and instruction scheduling are hardware-dependent inferences, not GPU-profiler measurements.

### Quality-preserving improvement retained

The builder now reduces three successive levels per workgroup using 64 threads and shared memory, then uses a dispatch boundary before the next group of levels. It writes the same atlas, copies mip zero during the first reduction's depth read, and stores only the final local level as an intermediate. There are no cross-workgroup atomic completion tricks or spin waits. Explicit two-dimensional dispatches keep 4K group addressing correct. This is inspired by SPD's local-reduction design, not a full port of SPD.

Standalone GPU-completed throughput, before/after/after/before order, one-second warmup and five-second windows, hardware Apple Metal:

| Size | Previous builds/s | New builds/s | Throughput improvement | Dispatches before → after | Intermediate storage before → after |
| --- | ---: | ---: | ---: | --- | --- |
| 801×603 | 4,344 | 4,788–4,808 | 10–11% | 10 → 3 | 1.28 MB → 0.061 MB |
| 1920×1080 | 1,374–1,378 | 1,917–1,920 | 39–40% | 11 → 4 | 5.53 MB → 0.263 MB |
| 3840×2160 | 371–374 | 553–562 | 48–51% | 12 → 4 | 22.12 MB → 1.053 MB |

Intermediate storage excludes the unchanged atlas. All active atlas texels match the previous builder exactly at these sizes, including non-power-of-two sizes. This benchmark uses a synthetic R32F source; full application depth formats/workloads differ. It includes submission overhead and all construction work, not GPU timestamps. [Raw builder measurements](ssr-hi-z-data/research/tiled-builder.json).

A rejected prototype fused pairs serially within one thread. Despite fewer dispatches and less memory, it regressed standalone throughput. It was not retained. This is evidence that dispatch count alone is insufficient: parallelism and per-thread work matter. [Rejected prototype measurements](ssr-hi-z-data/research/rejected-serial-pairs.json).

### Clean complete-frame comparison and ray diagnostics

These are fresh 1920×1080 runs after the concurrent render stopped. Each row runs linear, conservative, adaptive, then linear, with the same completion/warmup rules as above. The previous-builder runs explicitly inject the saved source at `6f2c0f703b35` instead of relying on branch state.

| Scene / builder | Linear before FPS | Conservative FPS | Adaptive FPS | Linear after FPS |
| --- | ---: | ---: | ---: | ---: |
| Floor / previous | 65.03 | 61.79 | 64.82 | 63.05 |
| Floor / tiled | 64.94 | 62.20 | 65.42 | 64.99 |
| Cornell / previous | 44.00 | 42.89 | 40.68 | 44.23 |
| Cornell / tiled | 44.42 | 42.98 | 40.79 | 44.35 |

The builder improves, but the complete-frame changes are small and do not establish a meaningful Hi-Z win over linear. The floor improvement versus the prior builder is within run drift; Cornell's tiny change is inconclusive. Conservative still loses to linear, and adaptive only ties the floor while losing on Cornell. Every conservative and repeated-linear raw comparison remains exact, including coverage. All runs record zero browser/GPU errors. [Previous floor](ssr-hi-z-data/research/previous-floor-1080p.json), [tiled floor](ssr-hi-z-data/research/tiled-floor-1080p.json), [previous Cornell](ssr-hi-z-data/research/previous-cornell-1080p.json), [tiled Cornell](ssr-hi-z-data/research/tiled-cornell-1080p.json).

The untimed diagnostic compiles counters into an intercepted SSR shader, renders a fixed camera/noise frame, and reads out loop counts. It never times an instrumented shader. Counts include loop entries that terminate at an endpoint or screen boundary; they are not counts of GPU instructions, texture accesses, or wave occupancy.

| Scene | Active traced pixels | Mean loop entries linear / conservative / adaptive | Conservative checks / scheduled samples skipped | Adaptive cap reached without a hit |
| --- | ---: | --- | --- | ---: |
| Floor | 1,433,663 | 44.78 / 44.43 / 20.34 | 3,651,471 / 528,418 | 4.22% |
| Cornell | 1,583,667 | 36.15 / 30.70 / 31.01 | 3,409,009 / 9,203,309 | 12.01% |

Floor conservative pays four atlas samples for each of 3.65 million checks but skips only 0.53 million original sample positions. Its loop-entry count falls by less than 1%. Cornell skips more, but 3.41 million four-load checks remain expensive. Adaptive averages fewer visits on the floor, but each visit computes boundaries, depth planes, mip transitions and conditionals. Execution time depends on slow lanes within GPU groups as well as averages; divergence/register effects remain plausible explanations rather than profiler-proven causes. The capped-miss counts show that the iteration budget still affects adaptive coverage. Increasing it is not free.

The guide-buffer readback also shows **79.56% of Cornell traced surface pixels and 5.08% of floor pixels are at or above the existing downstream roughness cutoff**. About 70.8% of Cornell's linear loop entries belong to those roughness-rejected pixels. This is a concrete opportunity for classification. It cannot be fixed by blindly discarding them: the current before-filtering resolve still carries the original raw ray length into reconstruction even when roughness fades RGB coverage to zero. A safe change must first define the fallback hit-distance/history contract and verify filtered lighting and motion. This investigation therefore retains the existing integration and records that next target explicitly. [Floor counters](ssr-hi-z-data/research/floor-ray-counters.json), [Cornell counters](ssr-hi-z-data/research/cornell-ray-counters.json).

### Reproducing the follow-up

```sh
git-dedup show 6f2c0f703b35:examples/jsm/tsl/display/HiZNode.js > /tmp/HiZNode-before-research.js
node test/benchmarks/hi-z-builder.mjs /tmp/HiZNode-before-research.js /tmp/hiz-builder.json
HIZ_NODE_SOURCE=/tmp/HiZNode-before-research.js HIZ_WIDTH=1920 HIZ_HEIGHT=1080 HIZ_OUTPUT=/tmp/hiz-before node test/benchmarks/interior-ssr-hi-z.mjs
HIZ_WIDTH=1920 HIZ_HEIGHT=1080 HIZ_OUTPUT=/tmp/hiz-after node test/benchmarks/interior-ssr-hi-z.mjs
HIZ_DIAGNOSTIC=1 HIZ_WIDTH=1920 HIZ_HEIGHT=1080 HIZ_OUTPUT=/tmp/hiz-counters node test/benchmarks/interior-ssr-hi-z.mjs
```

Set `HIZ_MODEL='Cornell Box (Mirror Wall)'` to repeat the scene comparison/counters on Cornell. `HIZ_NODE_SOURCE` and `HIZ_SSR_SOURCE` support exact saved-source comparisons; reports include those paths and the browser version. Run all GPU workloads sequentially with other renders stopped. The standalone benchmark compares both channel bounds exactly at every active atlas texel before timing, and reverses mode order.

Final hardware validation passes the 200-case pyramid suite, 192 traversal/coverage cases and the local endpoint/refinement unit regression, with no skipped GPU tests. [Final validation](ssr-hi-z-data/research/validation.json), [unit result](ssr-hi-z-data/research/units.json). The standalone comparison additionally covers full 1080p and 4K atlases; 2D workgroup addressing is necessary at 4K. Changed-file lint and whitespace checks pass. Root build outputs are not included.

## Decision

Keep **Linear** as the interior default. **Conservative Hi-Z** is an optional, quality-preserving candidate for workloads with useful empty-space skips. **Adaptive Hi-Z (experimental)** changes hit sampling and is explicitly a speed/quality experiment. Neither alternative establishes a meaningful speed improvement in the two locally measured interior scenes, including a 1080p floor check. Do not automatically enable either based on the presence of mirrors.

The existing PR evidence already rejects a universal speedup. PR #76 reduced hierarchy construction from repeated full-resolution scans to successive reductions, cutting its standalone RTX 3060 Ti 1080p build from 7.459 to 0.254 ms. Complete 1080p demo frames improved about 9% for the steampunk mirror, but regressed about 9% for castle GGX. Those are the PR author's measurements, not measurements made here.

PR #79 reported about 2.75× throughput for the 1080p steampunk mirror, but a 15% GGX regression. Its traversal starts a texel from the origin, solves cell depth crossings, caps iterations, and bypasses legacy binary refinement. Coarse leaves additionally risk missing thin geometry. Its linked path-traced comparison documents small average fidelity losses with visible local stair steps, and its separate 480×360 suite shows performance regressions for all alternate modes. See the PR's [algorithm report](https://github.com/bhouston/three.js/blob/adaptive-hi-z-ssr/test/webgpu/SSRHiZAdaptive.md) and [fidelity results](https://github.com/bhouston/three-ssr-fidelity/blob/compare-hi-z-ssr/docs/SSR-HIZ-RESULTS.md).

That large mirror speedup does not transfer directly to Interior: Interior always uses stochastic GGX, including nominal mirrors, and already limits the original marcher to at most 63 coarse samples at quality 1. The old non-stochastic mirror can march far more projected texels. Adaptive's 64 iterations also pay for cell transitions and descents, so they are not equivalent to 64 original leaf samples. Background, short rays, roughness, resolution, hierarchy dispatches, and the rest of the frame determine whether skipping repays its cost.

## Local measurements

Verified non-fallback Apple Metal hardware (`apple`, `metal-3`), hardware Chrome, 960×720 and a 1920×1080 floor check, pixel ratio 1, quality 1, otherwise Interior defaults. A 1024 bake atlas replaces the interactive default of 4096 to make the diagnostic reproducible. All lighting, SSGI, reconstruction, fallback, TRAA and hierarchy construction remain in the timed Beauty frame. Each mode warms up for one second, then runs sequentially for at least five seconds, counting frames only after a batch of four has completed on the GPU. Vsync and frame limits are disabled. Queues drain between modes. These measure sustained whole-frame throughput including CPU submission, not isolated GPU execution times.

| Scene | Linear before, FPS | Conservative, FPS | Adaptive, FPS | Linear after, FPS |
| --- | ---: | ---: | ---: | ---: |
| Primitives Floor (Metallic Mirror), 960×720 | 94.83 | 91.47 | 94.58 | 96.55 |
| Cornell Box (Mirror Wall), 960×720 | 53.09 | 50.98 | 46.81 | 49.78 |
| Primitives Floor (Metallic Mirror), 1920×1080 | 39.38 | 37.03 | 38.90 | 39.07 |

The floor's conservative result is about 3.5% below the initial linear control; adaptive is within linear drift and supplies no demonstrated speed win. Cornell's conservative result lies within the two linear controls, so a small difference is inconclusive. Adaptive is below both Cornell linear controls. At 1080p conservative is about 6% slower, while adaptive remains slightly below both linear controls. Single windows and repeated linear controls are sufficient to reject a claimed large gain here, but not to establish small differences. No claim about other hardware, higher resolutions, moving cameras or other viewpoints follows from this table.

[Floor raw results](ssr-hi-z-data/floor-960x720.json), [Cornell raw results](ssr-hi-z-data/cornell-960x720.json), [1080p floor raw results](ssr-hi-z-data/floor-1920x1080.json). The runner asserts SSR executes once per completed frame, linear constructs zero pyramids, and each Hi-Z mode constructs one pyramid per frame. Repeated linear after Hi-Z also constructs zero pyramids; switching modes does not leave construction running.

## Quality and integration

Quality captures freeze the linear HDR source and camera, align the 32-slice FAST sequence, and compare the raw trace and separate coverage attachment. They run without TRAA camera jitter. The timing phase restores live lighting and full Beauty including TRAA. Comparing jittered raw buffers between modes is invalid even if stochastic indices match.

| Scene | Conservative maximum RGBA error / changed coverage pixels | Adaptive linear RGB RMSE / changed coverage pixels |
| --- | --- | --- |
| Floor, 960×720 | 0 / 0 | 0.04036 / 10,544 |
| Cornell, 960×720 | 0 / 0 | 0.26733 / 234,055 |
| Floor, 1920×1080 | 0 / 0 | 0.03645 / 23,508 |

The 960×720 images contain 691,200 pixels each; 1080p contains 2,073,600. Adaptive differences are substantial in Cornell. Difference from linear alone cannot establish physical fidelity; both tracers retain the limitations in the [existing correctness review](interior-ssr-review.md). A path-traced or reflected-camera reference and moving-camera/thin-geometry tests remain necessary before recommending adaptive for quality-sensitive use. These raw, fixed-source comparisons deliberately exclude denoiser history as a confound; they are not a full animated reconstruction-quality assessment.

The local port preserves coverage output, FAST sampling, HDR clamping, material reflectance, the grazing denominator guard, ray endpoint bounds, background rejection and the corrected refinement bracket/validation. One additional fix is essential: when conservative traversal proves a batch is entirely in front, it updates `previousS` and `previousInFront` to the batch endpoint. Otherwise this branch's newer refinement logic can refine from an older bracket and change quality. Adaptive never invokes legacy binary refinement. As in #79, its budget has an eight-iteration minimum and `hiZMinMip > 0` is a lossy option.

The hierarchy owns its resources separately from SSR, allowing sharing. Dispose both when finished. It requires WebGPU compute; linear tracing does not allocate GPU hierarchy storage. Very tall depth textures can exceed the atlas dimension limit and report an error. Standard, reversed and logarithmic depth are validated separately; combined reversed plus logarithmic depth is not covered.

## Using the options

Interior's **Screen Space Reflections → Traversal** selector exposes Linear, Conservative Hi-Z, and Adaptive Hi-Z (experimental). Changing modes also resets reconstruction history. Existing SSR debug views inspect the selected tracer. The default stays Linear.

```js
import { hiZ } from 'three/addons/tsl/display/HiZNode.js';
const pyramid = hiZ( depth );
const reflection = ssr( color, depth, normal, {
    camera,
    hiZ: pyramid,
    hiZTraversal: 'conservative'
} );
reflection.hiZEnabled = false; // Original linear tracing, no hierarchy construction.
reflection.hiZEnabled = true;
reflection.hiZTraversal = 'adaptive'; // Experimental changed sampling.
reflection.hiZMaxSteps.value = 64;
reflection.hiZMinMip.value = 0;
// Later: reflection.dispose(); pyramid.dispose();
```

`hiZTraversal` and `hiZEnabled` rebuild the tracing shader. Adaptive ignores `stepExponent` and `binaryRefine`; budget and minimum mip are uniforms. Supply the hierarchy at construction, using the same raw depth texture as SSR.

## Reproduction and checks

No extra scene application is needed: the existing Interior example and ported procedural fixtures provide the rig. From the repository root:

```sh
node test/benchmarks/ssr-hi-z-validation.mjs
node test/benchmarks/interior-ssr-hi-z.mjs
HIZ_MODEL='Cornell Box (Mirror Wall)' HIZ_OUTPUT=/tmp/three-hiz-cornell node test/benchmarks/interior-ssr-hi-z.mjs
HIZ_WIDTH=1920 HIZ_HEIGHT=1080 HIZ_OUTPUT=/tmp/three-hiz-floor-1080p node test/benchmarks/interior-ssr-hi-z.mjs
```

`CHROME_PATH` selects hardware Chrome. `HIZ_WIDTH`, `HIZ_HEIGHT`, `HIZ_MODEL`, and `HIZ_OUTPUT` configure the interior run. The script saves raw metrics and diagnostic screenshots; it fails on nonfinite traces, unequal conservative output, unexpected pyramid/frame counts, or browser/GPU errors. Both runners reject software/fallback adapters. The test pages import source directly, so generated repository-root build artifacts are unnecessary.

The procedural suite checks 200 pyramid cases (1,936,440 exact bounds comparisons and 2,400 reference trace cases), plus perspective/orthographic × standard/reversed/log depth × mirror/GGX × refinement on/off × qualities 0.25/1 for 48 conservative, 48 adaptive and 48 coarse adaptive cases. Conservative hit masks must match and float32 output has a 1e-4 absolute roundoff tolerance; the interior half-float captures match exactly; adaptive checks require finite output, actual hits and changed-hit coverage. An additional 48-case run exercises stochastic coverage output, and the local endpoint/refinement unit regression also runs. [Raw validation](ssr-hi-z-data/validation.json) and [unit results](ssr-hi-z-data/units.json) accompany this report. The maximum observed conservative float32 channel difference was 0.00006104, with zero changed hit-mask pixels. Changed-file lint passes. All three interior runs have zero recorded browser/GPU errors and exactly matching repeated linear captures.
