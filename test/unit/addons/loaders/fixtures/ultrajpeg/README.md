# Ultra HDR JPEG fixture

Original synthetic fixture by Ben Houston, licensed under MIT as part of Three.js.

`linear-hdr.jpg` is a 16×16 constant image, 2,259 bytes, containing an sRGB-profiled primary JPEG (RGB bytes 128), a three-channel gain JPEG (equal channel bytes 255), Ultra HDR XMP version 1.0, GContainer metadata, and MPF indexing. Full reconstruction is linear RGB (4,4,4), alpha one. Gamma is one; offsets and minimum gain/capacity are zero. Maximum gain/capacity in log2 units is `log2(4 / sRGBToLinear(128/255))`, approximately 4.2118288.

Generated with sharp 0.35.5 at JPEG quality 100 and explicit XMP/MPF metadata. The reproducible generator and specification are available in https://github.com/bhouston/glTF/tree/ultrajpeg/extensions/2.0/Vendor/EXT_ultrajpeg/examples .

The test verifies full gain application and catches sRGB decoding after gain application, display-adapted reconstruction, peak normalization, and double sRGB decoding. It does not cover the entire image profile.
