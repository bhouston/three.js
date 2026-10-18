# Adaptive Hi-Z SSR

`SSRNode` now offers an opt-in cell-boundary traversal. The original linear marcher and its conservative Hi-Z batch optimization remain available and remain the defaults. The new traversal changes sampling and must be judged with both throughput and image comparisons.

```js
const depth = scenePass.getTextureNode( 'depth' );
const pyramid = hiZ( depth );
const reflections = ssr( color, depth, normal, {
    hiZ: pyramid,
    hiZTraversal: 'adaptive',
    // Other material, camera, and stochastic options are unchanged.
} );
reflections.hiZMaxSteps.value = 64; // full-quality iteration budget; clamped to 8..512
reflections.hiZMinMip.value = 0; // optional coarse leaf; 1 trades thin-detail accuracy
// Set hiZEnabled=false for the original linear trace.
// Set hiZTraversal='conservative' for exact scheduled-sample rejection.
```

Adaptive tracing starts one projected texel from the origin, then advances to the next hierarchy cell exit or nearest-depth plane. Empty-cell crossings coarsen the hierarchy; depth crossings descend. At the selected leaf it applies the existing thickness, mirror normal, distance, and shading rules. Standard/reversed projected depth is affine in the ray parameter, so the hot hierarchy loop avoids per-cell perspective divides. Direction reciprocals are hoisted outside the loop. Logarithmic depth uses the general view-space conversion path. Exact power-of-two footprints account for odd trailing rows and columns.

The traversal is inspired by [AMD's SSSR algorithm](https://github.com/GPUOpen-Effects/FidelityFX-SSSR/blob/master/ffx-sssr/ffx_sssr.h), implemented against this node's atlas and existing shading rather than transplanted shader code. It does not include AMD's compute ray classification, ray compaction, subgroup occupancy termination, or denoiser.

Quality changes are intentional:

- The quality uniform scales the iteration budget, with an eight-iteration floor. The default full-quality budget is 64; a legacy mirror ray may march many more projected samples. Exhausted rays use the existing miss/fallback behavior.
- Adaptive leaves solve depth planes instead of following the old uniform/exponential schedule. `stepExponent` and `binaryRefine` apply to the original/conservative paths; adaptive tracing does not bisect unrelated legacy sample indices.
- The start offset, tiny boundary offset, and solved-plane rounding tolerance can change isolated hits and self-intersections.
- Minimum mip greater than zero may miss thin details or a later surface within a tile when another texel supplies its nearest depth.
- The tested depth configurations are standard, reversed, and logarithmic separately, with perspective and orthographic cameras. Combined reversed+logarithmic depth inherits the existing SSR conversion limitation and is not covered by these tests.

On Windows001's physical NVIDIA RTX 3060 Ti, Chrome 155.0.8059.79, 1920×1080, quality=1, fixed demo views and otherwise default demo settings:

| Traversal | Steampunk mirror ms/frame | Castle GGX ms/frame |
|---|---:|---:|
| Original linear | 5.606 | 5.026 |
| Conservative Hi-Z | 5.077 | 5.608 |
| Adaptive, max budget 64 | 2.039 | 5.937 |
| Adaptive, max budget 64, minimum mip 1 | 2.056 | 5.931 |

This is about 2.75× mirror throughput versus linear in this view, but about 15% lower GGX throughput. It is not a universal win. The coarse option provides no meaningful speed gain in these two views and should not be selected on performance grounds here. An exploratory 128-iteration implementation cost 2.746 ms mirror / 6.907 ms GGX; lowering the budget is part of the speed/quality tradeoff, so the entire gain must not be attributed to the hierarchy itself. Hoisting projected-depth math changed the 64-budget mirror result from 2.136 to 2.039 ms, while the GGX change was within noise.

The retained numbers count GPU-completed frames over at least five seconds, following one-second warmup, with vsync/frame caps disabled and queue drains between sequential modes. They include CPU submission and full-frame rendering rather than isolated GPU timestamps. A checked-in-core-build smoke run also rendered both adaptive demos. Raw results are in `SSRHiZAdaptiveResults.json`; the previous builder investigation is in `SSRHiZPerformance.md`.

Quality comparisons are integrated in [three-ssr-fidelity PR #9](https://github.com/bhouston/three-ssr-fidelity/pull/9): linear, conservative, adaptive, and coarse adaptive for both mirror and GGX, across nine diagnostic scenes. The matched controls use the same fork, materials, camera, settings, seed, frame count, and TRAA pipeline. GGX controls isolate traversal; they do not reproduce the castle demo's recurrent denoising/multi-bounce pipeline. Matching-linear PSNR/delta comparisons should be considered alongside the path-traced references, not treated as proof that looking different from the original is worse. The suite's existing custom NewSSRNode research variants are preserved separately.

The completed 72-image hardware capture run (480×360, scene quality 0.5, Chrome 148, RTX 3060 Ti, nine scenes × eight variants) gives exact lossless equality for all 18 conservative controls. Against path-traced references, adaptive mirror changes mean PSNR by −0.038 dB (worst −0.144 dB), and adaptive GGX by −0.332 dB (worst −0.997 dB). Coarse leaves change the means by −0.026 dB and −0.314 dB respectively. These whole-image averages can hide local artifacts: the offscreen mirror fixture has visible stair-stepped reflection edges. GGX differences are larger, particularly in the metal-hit fixture. This is a useful experimental mirror speed/quality option, not a demonstrated net fidelity improvement. See the suite report for per-scene PSNR and amplified deltas. Its Chrome 148 measurements must be kept separate from the Chrome 155 demo timings above.

The full fidelity throughput matrix also completed: 72 ten-second windows at 480×360, scene quality 0.5, Chrome 148. Geometric-mean FPS ratios versus matched linear are mirror conservative 0.877×, adaptive 0.865×, coarse adaptive 0.847×; GGX conservative 0.834×, adaptive 0.819×, coarse adaptive 0.826×. All six alternatives regress at this preset. Mirror adaptive ranges from 0.752× to 1.014× across scenes; GGX from 0.701× to 0.935×. These full-frame measurements do not isolate the cause, but show that hierarchy construction/traversal overhead and reduced useful skipping can outweigh savings for this workload. No universal speedup follows from adding Hi-Z. [Full quality and throughput report](https://github.com/bhouston/three-ssr-fidelity/blob/compare-hi-z-ssr/docs/SSR-HIZ-RESULTS.md).

Branch-based live demo URLs (the branch is retained after merging):

- [Steampunk, original linear](https://raw.githack.com/bhouston/three.js/adaptive-hi-z-ssr/examples/webgpu_postprocessing_ssr.html)
- [Steampunk, conservative Hi-Z](https://raw.githack.com/bhouston/three.js/adaptive-hi-z-ssr/examples/webgpu_postprocessing_ssr.html?hiZ=conservative)
- [Steampunk, adaptive Hi-Z](https://raw.githack.com/bhouston/three.js/adaptive-hi-z-ssr/examples/webgpu_postprocessing_ssr.html?hiZ=adaptive)
- [Castle, original linear](https://raw.githack.com/bhouston/three.js/adaptive-hi-z-ssr/examples/webgpu_postprocessing_ssr_denoise.html)
- [Castle, conservative Hi-Z](https://raw.githack.com/bhouston/three.js/adaptive-hi-z-ssr/examples/webgpu_postprocessing_ssr_denoise.html?hiZ=conservative)
- [Castle, adaptive Hi-Z](https://raw.githack.com/bhouston/three.js/adaptive-hi-z-ssr/examples/webgpu_postprocessing_ssr_denoise.html?hiZ=adaptive)

Append `&hiZMinMip=1` for coarse leaves or `&hiZMaxSteps=128` for a larger budget. Controls are also in the Hi-Z panel. Raw source is available from [raw.githubusercontent.com](https://raw.githubusercontent.com/bhouston/three.js/adaptive-hi-z-ssr/examples/webgpu_postprocessing_ssr.html). RawGitHack returns HTTP 403 from the runner, so CDN access cannot be confirmed here; the same branch pages render successfully on localhost with the checked-in core build.

Hardware validation: 48 conservative comparisons have identical RGBA outputs; 48 adaptive and 48 coarse comparisons produce finite outputs, actual reflection hits (excluding environment-miss sentinels), and intentional changed-hit coverage. They cover both cameras, three separate depth configurations, mirror/GGX, binary-refinement flags, and quality settings. A low-quality 65×49 mirror fixture changes 113 of 3,185 hit-mask pixels, with linear-RGB RMSE 0.00821; this is a contract change, not an equality optimization. Core units: 1,957 passed / one TODO / zero failures; addon units: 102 passed. Core/addon/test/changed-file lint passed; no configured typecheck.

Further work should prioritize coherent compute ray queues, roughness/material classification, and reduced rays per quad with suitable reconstruction. [AMD's SSSR documentation](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/stochastic-screen-space-reflections/) describes a broader pipeline combining adaptive ray density and temporal variance with traversal. Those changes can save more than another small atlas-fetch optimization, but require an explicit reconstruction/fallback contract. Subgroup occupancy termination requires optional GPU capabilities and a portable fallback. Tiled local pyramid reductions could reduce dispatch overhead further; the current builder already avoids repeated full-resolution scans.
