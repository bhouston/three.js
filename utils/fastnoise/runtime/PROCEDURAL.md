# Zero-asset arithmetic candidates

`procedural.js` exports `proceduralSamplers[id](x, y, frame, seed)` and `proceduralWGSL[id]`, a WGSL module defining `sample(x:u32,y:u32,frame:u32,seed:u32)->f32`. All methods need zero texture bytes and no initialization. This does not establish a blue-noise spectrum or state-of-the-art quality.

* **animated-ign**: interleaved gradient noise with the familiar 5.588238 diagonal frame translation. A seed supplies a global coordinate translation. A cheap screen-space reference, with structured spatial correlations.
* **r2-weyl**: fixed-point R2 spatial lattice plus a golden-ratio temporal rotation and seeded phase. Highly structured low discrepancy, rather than isotropic blue noise.
* **hashed-digital**: reverse the frame bits and XOR with a pixel hash. Each pixel has stratified aligned power-of-two temporal prefixes, but the spatial distribution is hash noise. A useful control separating temporal discrepancy from spatial spectrum.
* **hierarchical-digital** (experiment): recursively affine-permute each quadtree parent's four children, pack fine-to-coarse child ranks, add hashed low bits, then XOR a common bit-reversed frame. At a fixed frame, every aligned 2^k square up to 256 contains every leading 2k-bit rank. The global temporal XOR preserves these strata. Tests must check shifted windows, arbitrary thresholds, anisotropy and moving/reprojected histories: aligned-stratum guarantees alone do not establish general rendering quality.
* **houston-stratified-noise** (experiment): use three scrambled quadtree levels (8x8 local stratification), an independently hashed tile phase, and a global golden temporal rotation. This trades global ordered patterns for tile boundaries and inter-tile phase noise. No tuned weights or scene-specific objective are used.

Inputs are integer pixel coordinates, uint32 frame and uint32 seed. Integer
methods retain the top 24 bits for an exactly representable f32 result below
one. R2, tiled Bayer and phase tiles use integer fixed-point Weyl rotation;
they avoid a growing float frame-times-golden product. The integer sequences
repeat after 2^32 frames. CPU references and exported GPU streams agree exactly
over every measured sample, not just selected points. Tests use actual frame
indices rather than repeating a 32-frame procedural volume.

IGN explicitly models f32 arithmetic but its unbounded translated coordinates
still lose precision as frame/coordinates grow. The tested 159-frame sequence
does not establish long-running stability. A bounded animation or precision
strategy requires its own quality evaluation; it must not silently reuse these
continuous-animation scores. Large-coordinate and moving-history behavior are
not established for any candidate.

The eight-level hierarchy performs eight hashes per sample and is a quality experiment, not necessarily a GPU speed winner. Three-level tiling costs three hierarchy hashes plus a tile hash. No GPU performance or quality claims are made here; compare threshold integration, spectra and realistic filtered estimators before recommending either experiment.
