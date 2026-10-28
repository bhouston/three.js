# FAST optimization experiments

Measured on Windows with Chrome 156, a physical NVIDIA GeForce GTX 1050,
and WebGPU reporting `nvidia` / `pascal`, `isFallbackAdapter: false`.
Baseline: `bf71e556b63c20af6b5f7f61ba30bcb11461e9e6`.

## Method

Use the default 128×128×32 volume and seed 1. Warm each variant with 100
iterations before a separate measured run. Run variants sequentially, with
`--disable-frame-rate-limit --disable-gpu-vsync`. Measure the complete iteration
loop through final mapped output readback, including loss, swaps, suppression
readbacks and CPU submission overhead. Thus the numerator counts completed
iterations, not submitted dispatches; no individual iteration timings are taken.
Finish readback and navigate away before starting the next implementation.
These compute workloads have no display frames to count.

The original product-mode command buffer lost its device on this Windows setup.
For product comparisons only, submit and drain every five iterations, while
preserving the original 100-iteration counter and suppression boundaries in
both baseline and candidate. This benchmark accommodation is not a production
optimization and is not included in the generator change.

Acceptance requires a repeatable speed improvement and no metric worsening by
more than 1%. Exact float32 byte equality is stronger evidence of unchanged
quality than comparing rounded scores. Approximate candidates are scored using
all ten `compare.py` metrics at full precision after its 8-bit quantization.

## Ten ideas tested

Initial separate-mode screening, 1,000 completed iterations. Small differences
around 1% are treated as inconclusive, rather than claimed as improvements.

| Idea | Time | Result |
| --- | ---: | --- |
| Baseline | 7.975–8.094 s | Reference |
| 1. Power-of-two wrap masks | 7.952 s | Identical bytes; inconclusive speed |
| 2. Compile-time tap count | 7.876 s | Identical bytes; small gain not retained alone |
| 3. Prepacked integer tap coordinates | 7.949 s | Identical bytes; inconclusive speed |
| 4. Wider workgroups | 7.489–7.514 s | Identical bytes; retained |
| 5. One compute pass per batch | 8.152 s | Identical bytes; no improvement |
| 6. Workgroup aggregation of swap counter | 8.177 s | Identical bytes; no improvement |
| 7. Explicitly hoist center flat index | 8.100 s | Identical bytes; no improvement |
| 8. **Off the wall:** skip loss for Feistel self-pairs | 8.116 s | Identical bytes; no improvement |
| 9. **Off the wall:** retire swap counter after suppression reaches one | 8.158 s | Identical bytes; no improvement, including removal of readback |
| 10. **Off the wall:** remove tiny kernel weights | 6.886 s | Faster but failed quality gate; rejected |

Workgroup sweep: 16×16, 64×1, 128×1, 32×4, and 64×4 took 7.522,
7.535, 7.531, 7.524, and 7.564 seconds respectively, with identical bytes.
32×2 keeps the original 64-invocation size. Combining masks and constant tap
count with 32×2 took 7.411 seconds; the incremental benefit was too small to
justify retaining the extra changes without further evidence.

Pruning weights <= 1e-6 reduced 200 taps to 168. Temporal low-frequency power
increased from 0.1625370596 to 0.1643222043 (+1.10%); blurred 256-frame error
increased from 0.0002189523061 to 0.0002264966584 (+3.45%). Both exceed the
quality budget despite the speed benefit. No approximate change is retained.

## Retained change

Use 32×2 workgroups for both shaders when the slice width is at least 32,
retaining the original 8×8 layout for smaller supported slices. The loss
arithmetic, traversal order within each invocation, random numbers, suppression
boundaries and iteration count are unchanged. Dispatch boundaries still
separate loss calculation from in-place swaps.

In the product-mode 100-iteration screen, time fell from 22.712 to 20.404 seconds
(10.2% lower time, 11.3% higher throughput). Both outputs had SHA-256
`845bdde1b3e280d9a8820b8129109394a2a1ca5cdac5dbda182aa25038fb8a46`.
This short run establishes throughput and output equality, not convergence.

Separate-mode 1,000-iteration baseline and exact candidates had SHA-256
`01dfd7c4ac052b73d466bb23da21ecdee5823c2bdddfd4d9fd132c6d49e0690c`.
At the full default 10,000 iterations, separate mode fell from 81.309 to
75.007 seconds (7.75% lower time, 8.40% higher throughput), with identical
SHA-256 `ad64707c4d4551ac6eacd4ce7c732e48d1ca5b83e3fc2c9cdc678fe5b5983df9`.
Consequently all ten quality metrics are identical at the default iteration count.
Timings are specific to this GPU and browser; other GPU architectures need
their own throughput measurements.
