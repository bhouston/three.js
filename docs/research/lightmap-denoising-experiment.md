# Browser lightmap denoiser experiment

> **Removed:** the OIDN denoiser was removed because it does not run on large light maps. `LightBaker` is back to the boolean `denoise` option (variance à-trous, on by default). The notes below are kept as a record of the experiment.

For tables measured on the actual Cornell Box, Courtyard and Space Ship Hallway examples, see [existing-scene metrics](lightmap-denoising-scenes.md). The standalone fixture below remains useful for small controlled experiments.

`LightBaker` now has a `denoiser` option: `'none'` (default), `'variance'`, or `'oidn'`. The baking example exposes all three. The legacy `denoise: true` selects `'variance'`; `false` selects `'none'`. Avoid specifying both options. All methods preserve raw accumulated samples, and seam stitching remains an independent option.

```js
const baker = new LightBaker( renderer, {
  algorithm: 'ray',
  resolution: 256,
  denoiser: 'variance'
} );
await baker.bake( scene ); // Also awaits final denoising.
baker.apply();

baker.denoiser = 'oidn';
await baker.processAsync(); // Reprocess existing samples, without baking again.
```

`process()` remains synchronous for None and variance. OIDN requires `processAsync()`. Failed imports, downloads and GPU operations reject instead of silently switching methods. Changing the source samples or options during inference rejects stale results. Resetting/disposal does not write filtered results back into the radiosity solver.

## Browser OIDN

There are JavaScript ports of OIDN's neural networks. This experiment uses [oidn-webgpu 0.6.0](https://github.com/pissang/oidn-webgpu) because its browser guide explicitly supports **RTLightmap**, including that model's logarithmic HDR transfer. This is WebGPU inference, not a WASM build of the native OIDN runtime. No API key or inference server is involved.

The optional package is imported only when OIDN is selected. Install it with your bundler, or add the example's import-map entry:

```json
"oidn-webgpu": "https://esm.sh/oidn-webgpu@0.6.0?bundle"
```

Weights are fetched from the official [oidn-weights](https://github.com/RenderKit/oidn-weights) repository at commit `28883d1769d5930e13cf7f1676dd852bd81ed9e7`, file `rtlightmap_hdr.tza` (1,830,792 bytes; SHA256 `5f9757cb6a3d6c966cbd50dcd0cbffef88fe78f80cb7bf45ca99bcbd4ff1292d`). Code is MIT; weights are Apache-2.0. The dependency and weights are remote optional downloads, not vendored build outputs.

The adapter reads the normalized HDR atlas through a float render target, runs FP32 inference on the renderer's device, and uploads a float texture. This deliberately favors a verifiable public texture/readback path over accessing renderer-internal GPU textures. Both transfers are part of inference cost. The first call additionally downloads weights and compiles shaders.

Current limits:

- Whole-image inference is limited to **512×512 per page**. Larger atlases reject with a clear error. This avoids independently normalized HDR tiles and bounds memory; it is an experimental limit, not proof every device can allocate 512×512 inference buffers.
- The model takes HDR color only, without geometry or chart guides. Unrelated UV islands can influence one another. It is not a topology-aware replacement for the existing filter or seam stitching.
- Only RGB irradiance is filtered. Dominant direction textures are not filtered by either option in this experiment.
- Network/download failure and GPU device loss are surfaced to the caller. No software rendering fallback is enabled.

## Variance filter

`'variance'` selects the branch's existing `LightMapDenoiser`, not a newly named Gaussian blur. It uses accumulated luminance moments, world position, normals, chart rejection and multiscale à-trous filtering with local gradient fitting. It is now allocated lazily. This provides the requested browser variance-based experiment without duplicating an existing implementation. The low-level class still exposes filter strength/pass controls for further tuning.

## Reproduce the comparison

Build locally with `npm run build`, then run:

```sh
node utils/benchmark-lightmap-denoising.js --output ./results --resolution 128 --samples 16 --reference 2048 --oidn
```

Or open `examples/webgpu_lightmap_denoising_benchmark.html` on a local server. Query parameters are `resolution`, `samples`, `reference`, `seconds` (at least five), and `oidn=1`. The runner verifies a physical adapter, disables vsync/frame limits, writes JSON and atlas preview screenshots, and closes its browser/server.

The benchmark warms each method, then runs it sequentially for at least five seconds and counts whole GPU-completed bake pipelines. It includes sampling, normalization, selected filtering and its transfers, and drains the GPU before switching. It does not time individual render frames. Model downloads/shader warmup and final metric readback/previews are excluded from steady-state throughput.

For each method, additional raw samples are chosen from the measured cost ratio. That raw variant is also warmed and measured over a five-second completed-work batch. `budgetRatio` reports the actual raw/filtered time ratio: a large mismatch is not an equal-time comparison. Integer rounding, fixed overhead and a cap below the reference can prevent exact matches.

Metrics are linear RGB RMSE, energy-normalized RMSE and MAE on covered reference texels. Five quality trials use subsequent ray-sequence phases for each method and raw comparator, outside throughput timing. The report includes the relative-RMSE spread and pools mean squared error across trials. Preview tone mapping is excluded from metrics. Seam stitching and directional output are disabled to isolate irradiance denoising. The fixture is a Cornell-like room with emissive ceiling panel, colored walls and two occluders.

Interpret results conservatively: the high-sample reference is still noisy; the fixed bounce cache is shared and its bias is not measured. Reset repeats UV jitter, but the ray source's render-pass sequence continues, so trials and reference use subsequent phases rather than matched rays or explicitly independent seeds. The fixture does not measure world-area weighting, cross-chart seam quality, independent-seed confidence intervals or large production scenes. Cold download/compilation costs make one-off bakes more expensive than warmed results. A lower raw-noise estimate alone is not evidence of better reconstruction.
