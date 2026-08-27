# Gap 2 — Learned interpolation of grid corners + positional encoding

## Current state (code)

- `NTCDecoderTSL.js` samples the packed mip-chain texture with a single hardware call — `textureLevel(mipChainTexture, uvNode, resolvedLodNode)` — relying on the GPU's fixed-function bilinear/trilinear filtering to interpolate between grid cells and between mips. The decoder MLP receives exactly one already-interpolated feature vector plus a normalized LOD scalar (`inputSize = channels + 1`).
- There is **no positional encoding** anywhere in the decoder input — no triangle-wave/Fourier terms encoding the texel's sub-cell position, unlike the paper's Section 4.3.2 (12 values: 6 horizontal + 6 vertical triangle-wave terms).
- The paper instead sends the MLP **four raw, unfiltered corner feature vectors** from the high-res grid `G0` (`4×C0` values) concatenated together, plus a bilinearly-interpolated tap from the low-res grid `G1` (`C1` values), plus 12 positional-encoding values, plus 1 LOD value — letting the MLP itself learn how to reconstruct high-frequency detail between texels ("learned interpolation," Section 4.3.1), specifically because hardware bilinear filtering of heavily quantized low-bit latents can't represent sub-texel high-frequency content well.

## Why this matters — and why it's a genuine tradeoff, not a pure win

The paper adopts learned interpolation **because** its latents are quantized to 2–4 bits; hardware bilinear filtering of that few quantization levels produces visible blocking/banding, and the MLP compensates by learning the interpolation function jointly with reconstruction. If gap 1 (lower bit depth) is adopted, this implementation will likely hit the same problem and want the same fix.

However, learned interpolation is **more expensive at decode time**: 4 corner taps instead of 1, a wider MLP input layer (`4*C0 + C1 + 12 + 1` vs today's `channels + 1`), and loss of the "free" hardware trilinear blend the current code relies on (`NTCHalfFloatTexture.js`'s box-filter-synthesized mip chain + a single `textureLevel` call is deliberately cheap and mobile-friendly — see its own comments about avoiding a visible LOD "pop" cheaply). For **mobile decode-speed**, this is a real cost/quality tradeoff, not a strict improvement, and should not be adopted wholesale without measuring on-device.

## Proposed changes (staged, each independently toggleable)

1. **Add positional encoding as an opt-in decoder input** (`NTCDecoderTSL.js`, `NTCGridPyramidModel.js`): implement the triangle-wave variant the paper cites (Müller et al. 2021, cheaper than sin/cos) for `log2(maxUpsample)` octaves, tiled per the paper's 8×8 pattern. This is additive to the input vector size and can be tested in isolation with today's *single-grid + hardware-bilinear* sampling first — a positional-encoded MLP might already improve reconstruction sharpness even without learned interpolation, at modest extra ALU cost (a handful of triangle-wave ops, no extra texture fetch).
2. **Add "learned interpolation" mode as a separate opt-in**, gated behind the same `gridPairing`-style option as plan 1's G0/G1 split (they're naturally linked — G0 is the grid learned interpolation applies to): fetch the 4 texel-aligned corners of `G0` via 4 explicit point-sampled texture fetches (`textureLevel` with nearest filtering, or manual `textureGather`-style 4-tap fetch) instead of 1 filtered fetch, concatenate raw, and let `G1` continue to use ordinary bilinear (matching the paper's asymmetric treatment of the two grids).
3. **Benchmark decode cost delta on target mobile GPUs before defaulting it on**: measure ms/frame at 1080p/4K fullscreen quad on at least one Android WebGPU device and one Apple Silicon device (iPad/iPhone Safari WebGPU or Mac) comparing: (a) current hardware-trilinear single-tap, (b) +positional encoding only, (c) +learned interpolation (4-tap) + positional encoding. Use this to decide the default profile for glTF/mobile presets (plan 03) — the paper's own Table 4 shows NTC decode cost is already 2–4× BC7's; a further 4× tap-count increase for the highest-res grid could meaningfully hurt fill-rate-bound mobile scenes.

## Suggested sequencing

1. Positional encoding alone (cheapest, isolated change, likely a straightforward win for sharpness at any bit depth).
2. Wire up on-device profiling harness (could reuse `webgpu_materials_neural_texture_compression.html` with a stats overlay) before touching corner sampling.
3. Learned interpolation, gated behind the G0/G1 split from plan 01, only if profiling shows acceptable mobile cost — likely worth exposing as a "desktop/high-quality" profile distinct from a "mobile/fast" profile that keeps hardware bilinear (see plan 03).

## Risks

- Four independent texture fetches per shaded pixel (vs 1 today) is the single biggest risk to mobile frame time in this whole comparison; tile-based mobile GPUs are often bandwidth/fetch-latency sensitive, not just ALU-sensitive, so this needs real device measurement, not just FLOP counting.
- WGSL/TSL currently builds one `textureLevel` call cleanly; sourcing 4 unfiltered corners portably across WebGPU (and any future WebGL2 fallback) needs care around texel-center addressing and edge/wrap behavior — worth a dedicated unit test similar to `NTCDecoderTSL.test.js`.
- Increases decoder input width and thus MLP parameter count (first hidden layer receives more inputs) — interacts with plan 03's activation/size decisions and plan 01's storage budget (bigger MLP eats into the total-size budget the same way more latent bits do).
