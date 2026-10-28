# Experimental scalar noise atlases

The blue noise example loads these precomputed textures lazily through its
noise dropdown. No optimizer runs in the example. Each PNG is a 128 by 4096,
8-bit grayscale atlas of 32 frames, stored in z/y/x order. Sampling uses the
same 32-frame cycle shifts as FAST and decodes a byte as `(byte + .5) / 256`.

All profiles use seed 1. `manifest.json` records exact generator options,
source-volume hashes, quantized-value hashes and PNG hashes. The PR117
profiles are pinned to commit `502a026ac59540f305d6f05fa2e8a53c1301815a`.
The other generators are in `utils/fastnoise/alternatives` and
`utils/fastnoise/experiments`. See `utils/fastnoise/FRONTIER.md` for measured
tradeoffs; the dropdown names do not imply universal quality improvements.

Regenerate from the repository root with installed dependencies and a
verified physical WebGPU adapter:

```sh
node utils/fastnoise/example-textures.js /path/to/raw-volumes --generate-gpu
```

The GPU runner defaults to installed Windows Chrome with D3D11 and disables
software rasterization. Other platforms require an appropriate hardware
backend. Fetch the pinned PR117 commit with git-dedup before regeneration.
Omit `--generate-gpu` to reuse previously generated GPU `.f32` volumes.
All methods are precomputed for consistent runtime behavior, including the
fast constructors. Each experimental atlas is approximately 0.4–0.53 MB
and is fetched only when selected, then cached for later selections.
