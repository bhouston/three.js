# Optimized shift tile with rank-1 temporal progression

This experimental generator constructs a shuffled, uniformly stratified 2D phase
tile, then performs eight greedy random-swap attempts per pixel. Similar phases
repel inside a radius-two Gaussian neighborhood, with periodic spatial and phase
boundaries. Every frame adds the same golden-ratio rotation modulo one to the
tile. There are no precomputed textures or native dependencies.

The inspiration is [Belcour and Heitz, *Lessons Learned and Improvements when
Building Screen-Space Samplers with Blue-Noise Error Distribution* (2021),
supplemental material](https://belcour.github.io/blog/supp/2021-sampling-bluenoise/).
Their screen-space sampler combines an optimized shift tile with a rank-1
sequence using Cranley-Patterson rotations. They assess blurred integration
error, including Heaviside integrands. This implementation shares that structure
but uses a deliberately cheaper local circular-phase repulsion surrogate;
it does **not** reproduce their integrand-trained optimization or guarantees.

## Cost and local checks

For N = size squared and D = depth, generation takes O(N + ND) time with fixed
eight optimization passes and 24 neighbors, and O(N + ND) memory. The neighbor
table stores 24 unsigned indices per pixel. Initialization is independent of
depth, so deeper volumes amortize its cost. The output is Float32 z/y/x order.

On the swarm MediaPC Node process, one cold 128 x 128 x 32 call took 116 ms.
The local smoke check passed deterministic same-seed output, different-seed
variation, finite [0,1) values, and sizes 1, 2, 3, 16. For frame zero, threshold
0.5 and a normalized five-tap sigma-one Gaussian separable blur, error RMSE was
0.0741 versus 0.1426 for seeded white noise (48% lower). This is a synthetic CPU
check, not a rendered-scene or hardware-GPU verification. Compare against the
reference generator and other methods using the common benchmark before
interpreting a speedup.

## Expected tradeoffs

Every frame preserves the circular repulsion objective and uniform spatial
histogram. Per-pixel golden-ratio progression reduces temporal discrepancy
without optimizing a full 3D volume. However, a fixed shift tile creates strong
cross-pixel temporal dependence, and this is not an independently optimized
spatiotemporal blue-noise construction. A depth-periodic texture introduces a
temporal seam because depth is generally not a period of the irrational
sequence. Rare thresholds, nonlinear integrands, motion, EMA filtering, and
repeated volume playback can expose correlated artifacts. The finite local
kernel and low swap budget also leave residual low spatial frequencies.

The spatial histogram has one sample in each of N strata before Float32
rounding, but the 3D histogram is not a permutation of ND ranks. Seed controls
the shuffle and optimization pairs. The same seed and dimensions always
reproduce the output in browser and Node JavaScript.
