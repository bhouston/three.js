# Playback-aware FAST objective experiment

`research-generator.html` is an isolated derivative of `generator.html`; the baseline file is unchanged.

Parameters in addition to the baseline:

- `seam=0..1` blends the original time torus with signed temporal taps that follow playback's spatial offset `(37,71)` at each depth boundary. `seam=1` uses the playback topology exclusively. Positive crossings shift forwards; negative crossings use the inverse shift.
- `mix=0..1` blends the existing separate objective with a joint spatial-temporal product kernel. `radius=1` truncates its spatial autocorrelation footprint, normalized to unit spatial mass. Larger radii increase GPU work. The temporal coefficients retain the original FAST weighted-exponential autocorrelation before wrapping.
- `initial=URL` loads a raw little-endian Float32 volume of exactly `size*size*depth` elements in `[0,1)`, after consuming the ordinary initialization RNG so subsequent keys remain reproducible. This allows refinement of a baseline volume.
- `submissionBatch=10` drains small command submissions to avoid Windows GPU watchdog resets for the heavier kernels. Logical 100-iteration boundaries, keys, swap-counter accumulation and suppression updates are unchanged. Both the original and drained zero-option 100-step controls matched baseline bytes on hardware.

`seam=0&mix=0` preserves the original coefficients, tap order, RNG sequence and update schedule. The shader changes only integer coordinate wrapping in this configuration; byte identity should be checked on hardware. With seam enabled there are additional signed taps; this is an objective experiment, not a claim of faster generation. Joint-kernel blending targets filtered integration error, while the remaining separate component protects unfiltered spatial/temporal projections. No claim of general superiority is made before held-out evaluation.

The pair correction still uses the original same-slice weight lookup: the swap permutation only exchanges samples within a slice, and signed temporal taps do not cross a whole period (their magnitudes are less than depth), so its same-slice weights remain exact for the new objective.

Validation: JavaScript module syntax parsed with Node's VM module parser. CPU reference verified both explicit seam directions and all 2,016 combinations of start slice and signed temporal tap round-tripped to the source pixel. GPU compilation, baseline byte identity and effectiveness require the parent's sequential hardware evaluation. Errors and device loss are exposed through `window.failure`; output remains base64 `window.result`.
