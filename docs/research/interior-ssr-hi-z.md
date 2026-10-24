# Optional Hi-Z SSR investigation

Studied October 24, 2026 on `interior-ssr-review`, starting at `d303775b6a43`. Sources are [PR #76](https://github.com/bhouston/three.js/pull/76) (head `faa7a27dcb30`) and [PR #79](https://github.com/bhouston/three.js/pull/79) (head `8e72a8b525e7`). Both PRs are merged remotely, but their implementations were absent from this local branch. This is a selective port, preserving the newer local SSR fixes, rather than a merge of their older SSRNode.

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
