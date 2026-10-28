# Denoiser metrics on the existing baking scenes

Measured 2026-10-28 on Build001: NVIDIA GeForce GTX 1050 (Pascal), Windows, Chrome 156. The physical WebGPU adapter was verified; no software renderer was used.

These are the actual Cornell Box, Courtyard and Space Ship Hallway loaders from [webgpu_lightmap_baking.html](../../examples/webgpu_lightmap_baking.html), including their original geometry, materials and lighting. They replace the earlier Cornell-like fixture as the evidence for these scenes.

## Reconstruction error

All methods use **16 accumulated samples**, compared with a **512-sample raw reference**. Values are energy-normalized linear RGB irradiance RMSE on covered texels; lower is better. The ± value is the standard deviation of relative RMSE across five sequential ray-phase trials, not a confidence interval. Aggregate RMSE pools mean squared error across trials.

| Existing scene | Actual atlas pages | None | Variance à-trous | OIDN RTLightmap |
| --- | --- | --- | --- | --- |
| Cornell Box | 1 × 256² | 1.87 ± 0.01% | 1.27 ± 0.05% | 1.25 ± 0.02% |
| Courtyard | 1 × 256² | 6.97 ± 0.15% | 6.11 ± 0.16% | 4.91 ± 0.05% |
| Space Ship Hallway | 4 × 512² | 41.67 ± 1.80% | 42.35 ± 6.09% | 29.42 ± 1.20% |

Variance filtering improves Cornell Box and Courtyard. Hallway shows no clear variance-filter benefit: its trial spread is large and the aggregate error is slightly higher than None. OIDN has the lowest measured aggregate RMSE in all three scenes at the same sample count.

## Warm processing cost

These are amortized means from at least five seconds of sequential GPU-completed 16-sample bake pipelines, not timings of individual render frames. Sampling, normalization, the chosen filter, and OIDN readback/inference/upload are included. Initial scene preparation, bounce-cache construction, shader/model warmup, downloads, camera rendering and final metric/preview readback are excluded.

| Existing scene | None | Variance à-trous | OIDN RTLightmap |
| --- | --- | --- | --- |
| Cornell Box | 123.3 ms | 127.2 ms | 225.1 ms |
| Courtyard | 142.2 ms | 144.4 ms | 213.5 ms |
| Space Ship Hallway | 1033.2 ms | 1050.3 ms | 2085.2 ms |

## Spending the filter budget on more raw samples

Raw sample counts are estimated from warmed throughput and rounded to integers. The last column is actual raw time divided by filtered time; all these comparisons were within about 3% of the target budget. This reports attained time rather than assuming that the budgets match exactly.

| Existing scene | Filter | Filter error / cost | Raw samples | Raw error / cost | Raw/filter time |
| --- | --- | --- | --- | --- | --- |
| Cornell Box | variance | 1.27% / 127.2 ms | 16 | 1.88% / 123.4 ms | 0.971 |
| Cornell Box | oidn | 1.25% / 225.1 ms | 29 | 1.30% / 221.9 ms | 0.986 |
| Courtyard | variance | 6.11% / 144.4 ms | 16 | 7.11% / 144.2 ms | 0.999 |
| Courtyard | oidn | 4.91% / 213.5 ms | 24 | 5.93% / 215.1 ms | 1.008 |
| Space Ship Hallway | variance | 42.35% / 1050.3 ms | 16 | 41.19% / 1031.3 ms | 0.982 |
| Space Ship Hallway | oidn | 29.42% / 2085.2 ms | 32 | 33.38% / 2078.8 ms | 0.997 |

OIDN gives lower measured aggregate RMSE than the similar-cost extra-ray baseline in Courtyard and Hallway. The Cornell advantage over extra rays is marginal, and Hallway has substantial trial variation. These are scene-specific measurements, not a universal denoising speedup. Variance filtering gives a larger improvement per added cost in Cornell; OIDN achieves the lowest measured fixed-sample error.

## Settings and limits

- Ray radiosity uses the example's default sampler with a 256-pixel bounce cache. The scene-loading code is shared directly with the interactive example; Cornell's emissive panel is enabled and its alternate live point light is disabled, matching the example's Ray Radiosity mode.
- Requested resolution is 256, with a maximum page edge of 512 to fit the experimental browser OIDN integration. Hallway auto-grows into four 512² pages. These are **reduced-resolution experiments**, not measurements at the example's normal 1024-or-larger defaults. Error aggregates all pages; preview screenshots show only page one.
- Directional output and seam stitching are disabled to isolate RGB irradiance filtering. This does not measure dominant-direction quality, seam mismatch, or runtime shaded-image error. Texels are weighted equally rather than by world surface area.
- The reference is still a Monte Carlo estimate, especially in the difficult hallway. It shares the same fixed bounce cache, so cache bias and cache convergence are outside the metric. Trials use subsequent ray-sequence phases, not explicitly independent seeds or matched paths. Their spread is descriptive, not a formal uncertainty bound.
- OIDN uses the pinned RTLightmap weights and FP32 browser inference documented in [the integration guide](lightmap-denoising-experiment.md). It has no geometry/chart guides and can blur real detail or unrelated islands. Only one hardware GPU and one settings combination per scene were measured.

## Reproduce

Run a local build with npm run build, then run each scene sequentially:

```sh
node utils/benchmark-lightmap-denoising.js --scene "Cornell Box" --output ./results/cornell --resolution 256 --samples 16 --reference 512 --oidn
node utils/benchmark-lightmap-denoising.js --scene "Courtyard" --output ./results/courtyard --resolution 256 --samples 16 --reference 512 --oidn
node utils/benchmark-lightmap-denoising.js --scene "Space Ship Hallway" --output ./results/hallway --resolution 256 --samples 16 --reference 512 --oidn
```

The runner disables vsync/frame caps, verifies the physical adapter, warms each option, counts GPU-completed bakes, drains queues between options, saves JSON/previews, and closes its browser/server. Without --scene it runs the original small fixture. [Raw measurements](lightmap-denoising-scenes.json) preserve every quality trial, completed-work count, elapsed batch duration and adapter metadata.

Validation: build passed; core lint and changed-file ESLint passed. Browser core tests: 1,982 passed, one todo; addon tests: 164 passed, using a launcher without software-rendering configuration. Raw suite results are recorded in the task artifacts. No typecheck script exists in this repository. Root build outputs are excluded from commits.
