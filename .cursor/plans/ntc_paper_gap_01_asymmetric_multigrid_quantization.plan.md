# Gap 1 — Asymmetric multi-grid feature pyramid + low-bit-depth quantization

## Current state (code)

- One grid tensor per pyramid level, all sharing the same channel count (default 4), built by `computeGridLevels`/`createLatentGrid` in `examples/jsm/ntc/training/NTCGridModel.js`. There is no per-level split into a high-resolution/low-channel grid (`G0`) and a low-resolution/high-channel grid (`G1`) as in the paper's Section 4.1 / Table 1.
- Latent quantization is fixed at 8-bit uint8, symmetric, one `[min,max]` range per level (or global), via `examples/jsm/ntc/training/NTCQuantization.js`. There is no per-tensor variable bit depth (the paper uses 2–4 bits per grid, chosen per profile) and no asymmetric quantization range that centers a zero bin exactly (paper Section 4.2).
- `NTCMipBands.js` already documents this divergence in comments: uniform-width `floor(lod / mipsPerLevel)` banding instead of the paper's asymmetric per-level widths (Table 1: 256², 64², 16², 4² for `G0`, each paired with a `G1` at half that resolution).
- `NTCModelSize.js` assumes 1 byte/channel/texel storage (uint8) and 2 bytes/channel runtime (fp16) — there's no accounting for sub-byte bit-packing.

## Why this matters for mobile + glTF

Storage is the primary lever for glTF delivery size on mobile (download bandwidth, parse time, VRAM). The paper's headline result — 16× more texels at 30% less storage than BC-high — comes specifically from combining **more, thinner grids** (2–4 bit) rather than **fewer, fatter grids** (8 bit). At fixed total storage, the current implementation's 8-bit-only quantizer forces a choice between fewer levels (worse mip range) or coarser base resolution (worse peak fidelity), whereas the paper's approach keeps both by shrinking bits/latent instead of latent count. For mobile-targeted glTF assets, where every extra 100KB matters for time-to-first-frame over cellular, this is the highest-leverage single change for compression ratio.

## Proposed changes

1. **Sub-byte quantization**: extend `NTCQuantization.js` to support `bits: 2|4|6|8` per grid/level (packing multiple sub-byte values per byte, or storing as uint8 with a declared `bits` field so the *decoded* value only occupies `2^bits` levels — this is the low-risk first step: keep byte alignment, just reduce the effective quantization levels, matching the paper's approach of `Q_k = 1/N_k` with `N_k = 2^{B_k}`). Bit-packing to save additional storage (2 or 4 values per byte) is a follow-up once quality/bit-depth tradeoffs are validated.
2. **Asymmetric quantization range**: implement `[-((N_k-1)/2)*Q_k, (N_k/2)*Q_k]` per the paper (Section 4.2) instead of the current symmetric `[min,max]` — ensures a zero-valued latent quantizes without error, which matters a lot at 2–4 bits.
3. **G0/G1 paired grids per level**: add an optional second grid per pyramid level in `NTCGridModel.js`/`NTCGridPyramidModel.js` — `G0` at the level's native resolution with few channels (2-bit), `G1` at half resolution with more channels (4-bit), bilinearly upsampled and concatenated before the MLP tap. This is the bigger structural change; land it behind an option (`gridPairing: 'single' | 'g0g1'`) so existing single-grid configs keep working.
4. **Named compression profiles**: once (1)-(3) land, add profile presets analogous to paper Table 2 (`NTC_0_2`, `NTC_0_5`, `NTC_1_0`, `NTC_2_25`-style bpp targets) that set `levels`, `baseResolution`, per-level bit depths, and `hiddenSizes` together — see plan 03 for where these presets should live (`NTCGridModel.js` constants + a new `NTC_PROFILES` table), so users targeting mobile don't have to hand-tune 5 independent knobs.

## Suggested sequencing

1. Land asymmetric range + configurable bit depth (2/4/6/8) on the *existing* single-grid architecture first — this is additive, testable in isolation (`NTCQuantization.test.js`, `NTCTrainer.quantization.test.js` already exist and should extend cleanly), and immediately compresses shipped `.ntc` files without touching the grid/decoder architecture.
2. Measure PSNR/size tradeoff on the existing 8 sample materials (`examples/ntc/*.ntc`) at 2/4/6/8 bits to confirm the paper's claim that "above two channels, quality is roughly flat" holds here too.
3. Only then attempt the G0/G1 structural split — it touches `NTCGridPyramidModel.js`, `NTCDecoderTSL.js` (multi-grid sampling + concatenation), `NTCManifest.js` (new per-grid-pair schema, format version bump), and `NTCLoader.js`/`NTCExporter.js`. Treat as a `NTCFormat` v3 bump, keep v2 loadable for backward compatibility.

## Risks

- Format version bump means old `.ntc` files need a migration path or permanent v2 support in the loader.
- Lower bit depth increases banding risk on smooth gradients (paper Section 6.4.1/Fig 19 checks this explicitly) — should add a synthetic-gradient regression test mirroring the paper's Appendix D check.
- Sub-byte bit-packing (if pursued beyond "uint8 storing few levels") complicates the WGSL/TSL sampling path since GPU texture formats don't have native 2-bit/4-bit channels — likely needs a manual bit-unpack in the shader or an 8-bit texture with 2-4 packed sub-values per texel, adding ALU cost that should be profiled on a real mobile device before committing.
