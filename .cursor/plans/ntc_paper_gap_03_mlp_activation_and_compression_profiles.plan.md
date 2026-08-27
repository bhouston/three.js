# Gap 3 — MLP activation/size and named compression profiles

## Current state (code)

- MLP defaults to **2 hidden layers × 32 channels, ReLU activation, linear output** (`NTCGridPyramidModel.js`'s `hiddenSizes: [32, 32]`, `createMLP(..., 'relu', 'linear')`; `NTCMLPTSL.js` only implements `relu`/`linear` in `evaluateLinearLayerMat4`). The paper uses **2 hidden layers × 64 channels**, with **GELU** (and a hand-derived cheaper approximation, "hardGELU," a 3-piece function similar to hard-Swish) for the 3 non-output layers, and no activation on the final layer.
- Per-channel output range-matching (sigmoid/tanh/softplus in `NTCOutputActivations.js`) is a difference in *purpose*, not a gap — the paper doesn't need this because it targets a small fixed material-channel set and normalizes textures to `[0,1]` uniformly; this implementation supports an open ~24-channel PBR vocabulary with physically different ranges (HDR emissive, signed normal offsets, etc.), so this is a deliberate and reasonable divergence, not something to "fix."
- **No named compression-profile presets** exist. The paper ships four (`NTC 0.2/0.5/1.0/2.25` bpp, Table 2), each a fixed bundle of grid resolution/channel-depth/bit-depth. This repo instead exposes `levels`, `baseResolution`, `hiddenSizes`, `mipsPerLevel`, and quantization `mode`/`range` as five independent GUI/API knobs with no bundled recommendation, so there's no "pick mobile-low/mobile-high/desktop" one-liner for a user authoring a glTF asset.
- `EXT_neural_textures.md` (draft, unimplemented) already proposes a **mobile-specific** default distinct from the paper's desktop-oriented profiles: 1 hidden layer × 16 width, ReLU, 3-level grid (32/64/128, 4 ch/level), targeting <800 FLOPs/pixel — i.e., smaller than even this repo's current default, prioritizing decode speed over the paper's quality-first defaults.

## Why this matters for mobile + glTF

Today, a developer must hand-pick 5+ knobs correctly to land a good size/quality/speed tradeoff — most won't, and mobile decode cost scales directly with hidden width and layer count (each hidden neuron is one more `mat4` row group evaluated per shaded pixel). Providing curated, benchmarked presets is low-effort relative to gaps 1/2 and directly serves "how do I ship this on mobile" — it's the single most actionable near-term deliverable for glTF authors.

## Proposed changes

1. **Add `hgelu` (hardGELU) activation to `NTCMLPTSL.js`** alongside the existing `relu`, implementing the paper's exact piecewise formula:
   ```
   hardGELU(x) = 0            if x < -3/2
               = x            if x > 3/2
               = x/3 (x+3/2)  otherwise
   ```
   and to the CPU training-side (`NTCMLP.js`, forward/backward in `NTCGPUKernelsTSL.js`) so QAT and export match. Keep `relu` as the default for existing configs (backward compatible); expose `hgelu` as an opt-in for quality-oriented profiles since it costs a few more ALU ops per neuron than `relu`'s single `max`.
2. **Increase `hiddenSizes` default only inside new presets, not globally** — don't change the existing default `[32,32]`/`relu` for backward compatibility of already-tuned configs; instead ship the paper-matching `[64,64]`/`hgelu` combo as an explicit "desktop/high-quality" preset.
3. **Introduce an `NTC_PROFILES` table** (new file, e.g. `examples/jsm/ntc/training/NTCProfiles.js`) with at minimum:
   - `mobile-fast`: 1 hidden layer × 16, `relu`, small grid (matches `EXT_neural_textures.md`'s recommendation) — prioritizes FLOPs/pixel and draw-call-friendly small weight buffers.
   - `mobile-balanced`: today's existing default (2×32, relu) — already a reasonable mobile-safe middle ground, just needs to be *named* and documented as such.
   - `desktop-quality`: 2×64, `hgelu`, larger grid — closest match to the paper's own default configuration.
   - Each preset bundles `levels`, `baseResolution`, `hiddenSizes`, `activation`, `mipsPerLevel`, and (once plan 01 lands) bit-depth/gridPairing, plus a documented expected bpp/size range so users can pick by target file size, mirroring paper Table 2's presentation.
4. Surface these presets in both example HTML pages (`webgpu_materials_neural_texture_compression_trainer.html`'s GUI as a profile dropdown that sets the underlying knobs; `webgpu_materials_neural_texture_compression.html` docs/comments referencing which profile a sample `.ntc` file was trained with).

## Suggested sequencing

1. `hgelu` activation (self-contained, testable via existing `NTCMLPTSL.test.js`/`NTCMLP.test.js` patterns, no format changes).
2. `NTC_PROFILES` table + GUI dropdown — purely additive, no breaking changes, immediately useful.
3. Benchmark each preset's decode cost on a representative mobile device (reuse plan 02's profiling harness) to attach real ms/frame numbers to the presets' docs, not just theoretical FLOPs.
4. Once plan 01's bit-depth/G0-G1 work lands, extend presets to include those knobs and re-benchmark size/PSNR to produce paper-Table-2-style numbers for this codebase's own README/docs.

## Risks

- `hgelu`'s extra ALU cost (division/multiply vs `relu`'s single `max`) needs measuring — the paper adopted it as a *cheaper approximation* of GELU for their fused CUDA kernel context, which doesn't automatically mean it's cheaper than plain ReLU in a WGSL fragment shader; benchmark before defaulting any preset to it.
- Adding a profiles table creates a second source of truth alongside the raw knobs (`GRID_LEVELS_OPTIONS` etc. in `NTCGridModel.js`) — keep profiles as thin composition over the existing constants rather than duplicating validation logic.
