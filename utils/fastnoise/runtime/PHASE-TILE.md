# Runtime phase tile

`phase-tile.js` adapts PR116's circular-phase lattice objective to startup generation of a **2D phase tile only**, rather than constructing a 32-frame volume. `generate({size:32,seed:1,passes:8})` returns 4 KiB; sizes 16 and 64 use 1 KiB and 16 KiB. `sample(tile,x,y,frame)` adds a golden-ratio Weyl progression on demand. There is no image download or finite 32-frame playback loop.

Spatial repetition is exactly the tile width/height. The integrated sampler uses
uint32 fixed-point golden progression, followed by an explicit f32 phase sum.
It repeats after 2^32 frames, avoiding precision loss from a growing floating
frame-times-golden product. The shader and CPU agree over all measured samples.
Size 128 is also supported (64 KiB persistent phases). This method optimizes
circular phase separation in a small neighborhood, not a full spatiotemporal
spectrum. A common temporal rotation retains the optimized circular distances,
but does **not** guarantee every threshold image has a blue-noise spectrum.
It should be described as spatially optimized low-discrepancy phases, not an
exact STBN replacement.

Passes 0 gives shuffled stratification and is a useful control. Passes 2 and 8
use greedy swaps with the same radius-2 Gaussian circular energy as PR116.
The optimization has no temporal precomputation. Scratch neighbor indices use
`size*size*24*4` bytes (96 KiB at size32), plus phases and approximately 17 KiB
lookup table; scratch is released after generation. Sampling needs one phase
fetch and arithmetic. See `gpu-results.json` for measured scalar compute
throughput, including its submission overhead and scope limits.

`phase-tile-timing.json` records sequential CPU completed generations after one second warmup, over at least five seconds per profile. First-call measurements are included to avoid hiding startup/JIT cost. These are CPU generation costs, not GPU/rendering benchmarks; quality belongs in the shared frozen scorer and must not be inferred from timing alone.
