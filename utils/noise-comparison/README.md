# Procedural noise comparison

This focused comparison extends the uniform-field chart and table in [Three.js PR 35244](https://github.com/mrdoob/three.js/pull/35244), pinned to `f14867640c96eb324665cb68948fccb12318231e`. FAST Product is the original **FAST Spatial** texture; FAST Separate is **FAST Spatio-Temporal**. The underlying texture metadata identifies the product and separate filters. All four original series are reproduced from the original textures and NumPy seed, rather than copied or fitted. Their values match the published table at its displayed precision.

![Spatially filtered convergence](convergence-blur.png)

![Unblurred convergence](convergence-raw.png)

![Texture download cost](texture-cost.png)

See [TABLES.md](TABLES.md) for numerical tables and nonlinear/spectral checks, and [results.json](results.json) for full precision, input hashes, arithmetic details, and small CPU reference fixtures. PNGs are ordinary raster charts; their SVG sources are included.

## Reproduce

From the repository root, with Python 3, NumPy, Pillow, Node.js and the repository npm dependencies installed:

```sh
python utils/noise-comparison/compare.py
node utils/noise-comparison/chart.js
```

`CHROME_PATH` can select installed Chrome for the chart rasterizer. Chart generation uses 2D SVG and does not measure rendering performance. It does not change repository-root build outputs.

The benchmark preserves the original 128×128 R-channel viewport, 32-slice FAST indexing, 37/71-pixel cycle shift, midpoint byte decoding, and seven-tap separable wrapped sigma=1 Gaussian. White is NumPy `default_rng(0)`, as in the original published benchmark. Procedural methods use seed 1, continuous frame indices 0 through 1023, and no artificial 32-frame restart or texture quantization. IGN explicitly rounds each arithmetic operation to float32. 3LHW uses unsigned 32-bit arithmetic, then converts its upper 24 bits to a unit float. Accumulation and statistics use float64. These are CPU reference quality numbers; GPU float fusion may cause small IGN differences.

## Interpretation

**3LHW** builds three scrambled hierarchy levels per queried pixel, combines them with a hashed 8×8 tile phase, and adds a 32-bit golden-ratio Weyl increment each frame. No tile is generated or stored. Its low-discrepancy temporal sequence eventually competes with FAST Separate on this particular uniform-value running-average metric, but it has substantially worse single-frame and short-history spatial quality. The 1024-frame crossing is not evidence of a general state-of-the-art advance. The threshold and spectrum diagnostics expose tradeoffs that a raw-value average alone misses.

**Animated IGN** evaluates interleaved gradient noise with a 5.588238-pixel diagonal translation per frame. Its spatial neighborhood stratification gives useful filtering behavior with very little code and no lookup. It is a practical screen-space dithering alternative, rather than a replacement for optimized spatiotemporal masks in arbitrary sampling domains. The translation is continuous here: restarting it at 64 frames would change the convergence result. Floating-point precision degrades at large coordinates/frame indices; these measurements cover only the stated 1024-frame range.

The additional nonlinear check thresholds each sample at probabilities 0.1, 0.5 and 0.9 before accumulation and reports RMSE against the true probability, including bias. The spectrum diagnostic summarizes the first 32 contiguous frames without wrapping the procedural sequence. Neither test covers moving objects, reprojection/disocclusion, multidimensional integration, importance sampling, scene-dependent visibility, or a production denoiser. The Gaussian wraps at viewport edges for exact comparability with the original chart; actual example blur has its own edge treatment.

## Download and startup cost

| Method | Texture bytes | Intrinsic mask construction at runtime |
|---|---:|---|
| White | 0 | None |
| FAST Product | 3,019,696 | None; preoptimized PNG must be fetched, decoded and uploaded |
| FAST Separate | 3,025,821 | None; preoptimized PNG must be fetched, decoded and uploaded |
| Golden Ratio | 3,019,696 | Uses the first slice of the Product PNG in this example |
| 3LHW | 0 | None; arithmetic per sample |
| Animated IGN | 0 | None; arithmetic per sample |

Zero intrinsic mask construction does **not** imply zero shader compilation, first draw, or per-sample cost. Texture byte counts are exact repository payload sizes, not estimated HTTP transfer sizes. Network latency and browser caching depend on deployment. The example validation report separately measures local selection-to-GPU-completion latency on a physical adapter; it is not a portable download-time benchmark. The procedural choices require no FAST PNG request.

## Provenance

- [EA FAST / Donnelly et al.](https://github.com/electronicarts/fastnoise): original optimized textures and filter metadata; existing texture license remains beside them.
- [Jimenez, Next Generation Post Processing in Call of Duty: Advanced Warfare](https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare/): interleaved gradient noise.
- [Demofox, Interleaved Gradient Noise: A Different Kind of Low Discrepancy Sequence](https://blog.demofox.org/2022/01/01/interleaved-gradient-noise-a-different-kind-of-low-discrepancy-sequence/): neighborhood behavior and animated translation.
- 3LHW is the experimental `tiled-bayer-weyl` construction from the preceding PR 117 runtime exploration, renamed for this focused example. It combines established stratification, scrambling and Weyl-sequence ideas; no novelty or universal dominance is claimed.
