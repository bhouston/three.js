# Native HDR fixtures

These original synthetic neutral patches are dedicated to the public domain under CC0-1.0. No third-party image pixels are included.

Eight equal patches encode PQ luminance from 0 to 10,000 cd/m², with BT.2020 primaries. The manifest contains exact 12-bit AVIF codes and expected linear values at 203 cd/m² reference white. JXL adds a small 12-to-16-bit rounding step, documented as `jxlCode16`.

Generate the source PNG, PPM and manifest from the repository root:

```sh
node utils/generate-hdr-fixtures.js examples/textures/hdr-native
```

Then, inside this directory, encode with the official [libavif](https://github.com/AOMediaCodec/libavif/releases) 1.4.2 and [libjxl](https://github.com/libjxl/libjxl/releases) 0.12.0 tools:

```sh
avifenc --lossless --depth 12 --yuv 444 --cicp 9/16/0 --range full --clli 10000,10000 neutral-pq.png neutral-pq.avif
cjxl neutral-pq.ppm neutral-pq.jxl -d 0 -x color_space=RGB_D65_202_Rel_PeQ --intensity_target=10000
```

The generated PNG intentionally has no color profile: the encoder arguments explicitly label the samples as PQ. Do not use that intermediate PNG as an sRGB reference. The PPM and PNG are regenerated inputs, not required runtime assets.

Independent reference checks used `avifdec --depth 12 neutral-pq.avif reference.y4m` to verify the exact 12-bit channel values, and `djxl neutral-pq.jxl reference.ppm` to verify the exact source PPM bytes. MaxCLL/MaxPALL 10,000 are conservative metadata upper bounds, not a measured frame average. The committed binaries are 404 bytes (AVIF) and 135 bytes (JXL).
