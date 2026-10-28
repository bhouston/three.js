# FAST blue noise generator and analysis tools

Tools behind `examples/textures/noises/fast/` and `examples/jsm/tsl/utils/FASTNoise.js`.

- `generator.html`: WebGPU port of EA's [FAST](https://github.com/electronicarts/fastnoise) generator (`fastnoise/shaders/*.hlsl`, `main.cpp`). Real sample space, uniform distribution, Gaussian spatial filter, weighted exponential temporal filter, product or separate combine.
- `generate.js`: runs `generator.html` in headless Chrome and writes the volume as raw float32 (z, y, x).
- `compare.py`: scores volumes (ours as `.f32`, or EA's PNG slices) on spectra and filtered error.
- `make_atlas.py`: packs EA's PNG slices into the 256×4096 atlas `fastNoise()` reads.
- `convergence.py` / `chart.js`: error of a running average after N frames, and the PR chart.
- `procedural.py`: code-only noises (interleaved gradient noise, R2 + golden ratio) for comparison.

## Usage

```sh
# serve this folder, then generate from the repository root (so puppeteer resolves)
( cd utils/fastnoise && node ../server.js -p 8732 & )
node utils/fastnoise/generate.js "mode=separate&steps=10000" separate.f32
python3 utils/fastnoise/compare.py "ours=separate.f32" "EA=ea:noise/real/temporal/exp/real_uniform_gauss1_0_exp0101_separate05"
```

Query parameters: `mode` (`product` | `separate`), `steps` (10000), `size` (128), `depth` (32), `sigma` (1), `alpha` (0.1), `beta` (0.1), `weight` (separate spatial weight, 0.5), `seed` (1).
EA's textures come from `noise.zip` in their repository: `https://media.githubusercontent.com/media/electronicarts/fastnoise/main/noise.zip`.

## Algorithm

Each 128×128 slice starts with all 16384 strata once, jittered and shuffled. Every iteration pairs each texel with a random partner in the same slice (a Feistel network involution), computes the change in the filtered energy distance loss `Σ F(i - j) · -|v_i - v_j|`, and swaps pairs whose combined change is negative. A random fraction of swaps is suppressed early on, halving whenever few swaps happen. `F` is the σ = 1 Gaussian convolved with itself (13×13 taps) combined with the α = β = 0.1 weighted exponential over the 32 wrapped slices: multiplied for product (5408 taps), added for separate (200 taps).

## Results (Apple M-series GPU, Metal)

Matching EA at 10000 steps:

| | temporal low-freq power | spatial low-freq power | per-pixel error @ 32 | blurred @ 1 | blurred @ 32 | EMA 0.1 + blur |
|---|---|---|---|---|---|---|
| EA separate | 0.154 | 0.0030 | 0.0040 | 0.0252 | 0.00058 | 0.0023 |
| ours separate (31 s) | 0.153 | 0.0031 | 0.0040 | 0.0252 | 0.00058 | 0.0023 |
| EA product | 0.965 | 0.0005 | 0.0482 | 0.0204 | 0.00228 | 0.0036 |
| ours product (17 min) | 0.969 | 0.0005 | 0.0484 | 0.0203 | 0.00233 | 0.0036 |

Separate quality vs. steps:

| steps | time | per-pixel error @ 32 | blurred @ 1 | blurred @ 32 | EMA 0.1 + blur |
|---|---|---|---|---|---|
| 0 | 0 s | 0.0508 | 0.0823 | 0.0143 | 0.0193 |
| 100 | 0.3 s | 0.0100 | 0.0330 | 0.0022 | 0.0042 |
| 300 | 1.0 s | 0.0055 | 0.0268 | 0.0011 | 0.0027 |
| 1000 | 3.5 s | 0.0042 | 0.0256 | 0.00063 | 0.0023 |
| 10000 | 31 s | 0.0040 | 0.0252 | 0.00058 | 0.0023 |

Code-only noise, error after a running average and σ = 1 blur:

| frames | white | FAST spatio-temporal | IGN, animated | R2 + golden ratio |
|---|---|---|---|---|
| 1 | 0.079 | 0.022 | 0.029 | 0.028 |
| 4 | 0.040 | 0.0068 | 0.0100 | 0.0145 |
| 32 | 0.014 | 0.00064 | 0.0022 | 0.0031 |
| 256 | 0.0051 | 0.00022 | 0.00135 | 0.00048 |
| 1024 | 0.0026 | 0.00012 | 0.00135 | 0.00009 |

Metrics: temporal low-freq power is the mean of temporal spectrum bins 1–3 relative to all bins (1 = white); spatial low-freq power is radius < 0.1 relative to radius > 0.3. "Blurred @ N" is the std. dev. of the N-frame running average after a wrapped σ = 1 Gaussian. All volumes are quantized to 8 bits before scoring, like EA's PNGs.
