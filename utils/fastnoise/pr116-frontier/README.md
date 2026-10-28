# Pinned PR #116 frontier comparison

These inputs come from bhouston/three.js PR #116, commit
`d8b1acffaada09223b5133d965f3a7cf9028fe5b`. Files under `original/` are
unaltered Git blobs, including the upstream runner for provenance; use `run.js`
for this branch's reproduction. The original generator uses the repository MIT
license and retains its EA FAST algorithm attribution.

```powershell
node utils/fastnoise/pr116-frontier/run.js OUTPUT_DIRECTORY
# --resume skips only a file whose bytes match its recorded hash.
```

Requires the repository's installed Puppeteer and Chrome. The runner checks a
physical WebGPU adapter, disables software rasterization, vsync and frame caps,
warms up each profile for 100 iterations, and runs all six volumes sequentially.
Do not run competing GPU jobs or CPU scoring during generation. The final full
volume readback proves completed GPU work; these are construction timings, not
rendering throughput benchmarks.

Three frozen profiles, each at 128×128×32 and seeds 1 and 2:

| Output prefix | PR #116 profile | Parameters beyond common settings |
| --- | --- | --- |
| `pr116-annealed` | `latin-anneal001-3000` | mixed .2, temperature .001 falling to zero at step 1500 |
| `pr116-greedy` | `latin-hybrid-3000` | mixed .2, no annealing |
| `pr116-autocorrelation` | `latin-psd-3000` | mixed .02, binomial autocorrelation |

Common settings: 3000 iterations, temporal stratification, identity weight .25,
swap floor 2. The annealed and greedy profiles use the upstream truncated mixed
kernel. The autocorrelation profile uses the upstream binomial kernel. Its seed
2 is an additional confirmation run; PR #116 only published seed 1 for it.

The upstream specialization function is used unchanged. Two execution changes
are explicit in `run.js`: allow Chrome's physical default backend instead of
forcing D3D11, and drain the GPU every ten iterations to avoid Windows watchdog
timeouts. Logical 100-iteration parameter generation, RNG order, swap counter
accumulation and suppression updates are preserved. No WGSL arithmetic,
initialization, pairing, objective, acceptance or temperature code is changed.
Cross-adapter floating-point/parallel-update differences can still change bytes;
`pr116-generation.json` records each output hash and whether it matches the
upstream published hash rather than assuming equality.

The upstream FAST 10000-step hashes are identical to the existing references:
seed 1 `ad64707c4d4551ac6eacd4ce7c732e48d1ca5b83e3fc2c9cdc678fe5b5983df9`;
seed 2 `c71a155de2239b39b12e1c089e8992e3b53197e6ec971c1ce90c9ab8fac0ae95`.
All upstream published output hashes are retained in `original/volume-hashes.json`.
The common scorer should use this branch's existing 74-metric protocol; PR #116's
318 diagnostics are a different suite and their win counts cannot be merged.
