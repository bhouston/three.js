# Precomputed experimental noise in the example

The blue-noise example's dropdown exposes the experimental scalar volumes from
this branch, including the PR116 comparison and the later FAST refinements and
controls. All experimental volumes are generated offline and shipped as PNGs,
including the quick constructors. Selecting one loads an atlas; it does not run
an optimizer or require WebGPU generation in the browser.

These are research alternatives, not uniformly better blue noise. In particular,
filter/rank and PR116 relaxation target filtered scalar values and can perform
poorly after nonlinear transformations. No frontier candidate consistently beat
FAST on the full evaluation suite. See [PR116-COMPARISON.md](PR116-COMPARISON.md)
and [FRONTIER.md](FRONTIER.md) for measured tradeoffs and generation schedules.

Each experimental atlas is a **single scalar field**, not independent vector
components. All RGB values read from the grayscale PNG represent the same
sample. These assets cannot be interpreted as a vector-noise texture.

## Repacking offline volumes

After generating the seed-1 research volumes, run:

```sh
node utils/fastnoise/export-example-atlases.js RAW_DIR
```

An optional second argument selects a destination instead of
`examples/textures/noises/experimental/`. The exporter requires all 19 named
`*-seed1.f32` inputs listed in its definitions. Each input contains exactly
128 × 128 × 32 little-endian Float32 values in z/y/x order, finite and in [0, 1).
It does not generate volumes or silently substitute another method.

The output is lossless 8-bit grayscale PNG, 128 × 4096: slice z occupies rows
`128*z` through `128*z+127`. Quantization is `floor(value * 256)`. The example
reconstructs bin midpoints `(byte + 0.5) / 256`, as in the evaluation. Every PNG
is decoded immediately and every pixel is checked against the exact quantized
input. The manifest records the source-volume and PNG SHA-256, seed, dimensions,
layout and provenance. Repacking the same inputs produces deterministic assets.

The example uses seed 1, fixed scalar playback and the same cycle-offset scheme
as the existing example. This permits visual comparison, not a claim of general
sampling superiority. Frame repetition, seam offsets and the finite 32-frame
cycle remain relevant limitations. The +2000/+3000/+4000 entries are original-
loss FAST restart controls, not distinct scratch generators.

FAST-derived refinements retain the upstream FAST licensing obligations. Keep
the existing FAST license and attribution with redistributed derived assets;
the asset directory's documentation links that notice. PR116 provenance and
the exact offline generation schedules are recorded in the comparison reports
and generation tooling, rather than inferred from the PNG contents.
