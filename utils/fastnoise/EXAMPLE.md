# Precomputed experimental noise in the example

The blue-noise example offers six dropdown choices: the existing FAST
Spatio-Temporal, FAST Spatial, Golden Ratio and White options, plus two selected
experimental scalar volumes:

- **Light seam 3% (FAST polish)**: PR117's strongest confirmed schedule for a low
  regression count across the frozen evaluation suite.
- **PR116 annealed stratification**: PR116's promising temporal-stratum-preserving
  construction, with broad improvement and substantially faster construction.

These are complementary tradeoffs, not a universal ranking or a claim that
either wins on every metric. See [FRONTIER.md](FRONTIER.md) and the combined
comparison chart for the measurements. All other research methods remain in the
offline tools and analysis; they are omitted from the example's dropdown.

Both experimental volumes are generated offline and shipped as PNGs. Selecting
one loads its atlas and does not run an optimizer. Each is a **single scalar
field**, not independent vector components. All RGB values read from a grayscale
PNG represent the same sample.

## Repacking offline volumes

```sh
node utils/fastnoise/generate-example-noises.js RAW_DIR
node utils/fastnoise/export-example-atlases.js RAW_DIR
```

Generation takes several minutes and requires a physical WebGPU adapter. It
runs only on the developer machine; the example imports no generator code.
`CHROME_PATH` selects an installed Chrome. `--resume` reuses intact outputs from
unchanged sources. Light seam generation includes its FAST and continuation
prerequisites. The pinned PR116 runner also reproduces the other frozen profiles
on both seeds for research reproducibility; only annealed seed 1 is exported.
The generator verifies both selected volume hashes against the committed
`results/frontier.json` research records. PR116 source is pinned to commit
`d8b1acffaada09223b5133d965f3a7cf9028fe5b`; execution adaptations and published
hash checks are documented in [pr116-frontier/README.md](pr116-frontier/README.md).

The exporter requires `light-seam-03-seed1.f32` and
`pr116-annealed-seed1.f32`. An optional second argument selects a destination
instead of `examples/textures/noises/experimental/`. Each input contains exactly
128 x 128 x 32 little-endian Float32 values in z/y/x order, finite and in [0, 1).
It does not generate volumes or silently substitute another method.

The output is lossless 8-bit grayscale PNG, 128 x 4096: slice z occupies rows
`128*z` through `128*z+127`. Quantization is `floor(value * 256)`. The example
reconstructs bin midpoints `(byte + 0.5) / 256`, as in the evaluation. Every PNG
is decoded immediately and every pixel is checked against the exact quantized
input. The manifest records source-volume and PNG SHA-256, seed, dimensions,
layout and provenance. Repacking the same inputs produces deterministic assets.

The example uses seed 1, fixed scalar playback and the existing cycle-offset
scheme. This permits visual comparison, not a claim of general sampling
superiority. Frame repetition, seam offsets and the finite 32-frame cycle remain
relevant limitations. FAST-derived refinements retain the upstream licensing
obligations; keep the asset directory's FAST license notice with redistributed
assets.

## Browser validation

Run `node utils/fastnoise/check-noise-example.js --root . --artifacts OUTPUT`
with installed Chrome and a physical WebGPU adapter. The check requires exactly
two experimental entries, checks all six choices in Sphere and Uniform modes,
exact PNG midpoint readback, frames 1/31/32/1024, download retry, stale request
handling and caching. Access to example state is injected by the test server,
not shipped in the example. These checks verify playback and integration, not
scene-level sampling superiority.

All six choices pass on physical NVIDIA Pascal / WebGPU, including both modes,
atlas decoding across cycle boundaries, retries, stale requests and caching.

