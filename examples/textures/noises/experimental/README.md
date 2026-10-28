# Experimental scalar noise atlases

The blue noise example loads these precomputed textures lazily through its
noise dropdown. No optimizer runs in the example. Each PNG is a 128 by 4096,
8-bit grayscale atlas of 32 frames, stored in z/y/x order. Sampling uses the
same 32-frame cycle shifts as FAST and decodes a byte as `(byte + .5) / 256`.

The two selected experimental profiles are GPU Stratified Annealed and PR117
Light Playback. Both use seed 1. `manifest.json` records GPU generator options,
source-volume hashes, quantized-value hashes and PNG hashes. The PR117
profile imports the published quantized PNG pinned to commit
`2a065526980261bcf5b129f07ddb12ff7eda1f91`; its decoded Float32
hash differs from the original unquantized volume hash by design.
The annealed generator is in `utils/fastnoise/experiments`. The pinned PR117
original hash and generation stages are recorded in
`utils/fastnoise/results/frontier/pr117-import.json`.
See `utils/fastnoise/FRONTIER.md` for measured tradeoffs; the dropdown names do not imply universal quality improvements.

Regenerate from the repository root with installed dependencies and a
verified physical WebGPU adapter:

```sh
node utils/fastnoise/example-textures.js /path/to/raw-volumes --generate-gpu
```

The GPU runner defaults to installed Windows Chrome with D3D11 and disables
software rasterization. Other platforms require an appropriate hardware
backend. Fetch the pinned PR117 commit with git-dedup before regeneration.
Omit `--generate-gpu` to reuse previously generated GPU `.f32` volumes.
Both methods are precomputed for consistent runtime behavior. Each atlas is
approximately 0.53 MB and is fetched only when selected, then cached for later
selections.
