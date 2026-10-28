# Experimental scalar noise atlases

These seed-1 textures are the exact quantized research volumes used in this
branch's comparisons, not independent RGB or vector noise. All channels of the
grayscale PNG represent one scalar sample.

Each atlas is 128 x 4096 (32 vertical 128 x 128 slices). Decode byte values with
`(byte + 0.5) / 256`. The example loads only the selected texture and caches it.
The manifest records labels, provenance and source-volume/PNG SHA-256 hashes.

See [the example guide](../../../../utils/fastnoise/EXAMPLE.md) for the complete
catalog, limitations and reproducible offline generation/export commands. The
PR116 constructors are pinned to commit
`54a6dfe336cd33fc1fe129e4e1e6b10900395832` of bhouston/three.js.

No candidate consistently improves on FAST across the complete research suite.
FAST refinements and original-loss restart controls are labeled explicitly.
The preview uses a finite 32-frame cycle with spatial offsets between cycles.

The repository's [MIT license](../../../../LICENSE) applies to the new code.
FAST-derived assets retain the upstream Electronic Arts notice in
[LICENSE-FAST.txt](LICENSE-FAST.txt).
