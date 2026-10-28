# Selected experimental scalar noise atlases

The example includes two experimental methods alongside its four existing
options: PR117 Light seam 3% FAST refinement and PR116 annealed stratification.
They represent complementary measured tradeoffs, not a universal ranking. The
other research methods remain available in the offline tools and reports.

These seed-1 textures are exact quantized research volumes, not independent RGB
or vector noise. Every channel of a grayscale PNG represents one scalar sample.
Each atlas is 128 x 4096 (32 vertical 128 x 128 slices). Decode bytes with
`(byte + 0.5) / 256`. The example loads only the selected texture and caches it.
The manifest records labels, provenance and source-volume/PNG SHA-256 hashes.

See [the example guide](../../../../utils/fastnoise/EXAMPLE.md) for limitations
and reproducible offline generation/export commands. The selected PR116
constructor is pinned to commit
`d8b1acffaada09223b5133d965f3a7cf9028fe5b` of bhouston/three.js.

No candidate consistently improves on FAST across the complete research suite.
The preview uses a finite 32-frame cycle with spatial offsets between cycles.

The repository's [MIT license](../../../../LICENSE) applies to the new code.
FAST-derived assets retain the upstream Electronic Arts notice in
[LICENSE-FAST.txt](LICENSE-FAST.txt).
