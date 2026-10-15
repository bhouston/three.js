Hi-Z GPU validation
==================

Serve the repository root over localhost (for example `node utils/server.js -p 8080`),
then open http://localhost:8080/test/webgpu/hi-z.html in a WebGPU browser.
No build is required: the page imports source modules directly.

The page prints a JSON result, also available as `window.hiZTestResult` for browser
automation. Status is `running`, `passed`, or `failed`. It requires an identified
hardware WebGPU adapter and explicitly rejects software/fallback adapters.

The tests read every populated rg32float atlas texel (both nearest and farthest bounds) back from the GPU and compare
it exactly against an independent recursive CPU reduction. Coverage includes both
depth conventions (explicit override and automatic renderer detection), 1D/2D odd sizes through 257x129, deterministic random input,
constant input, impulses on trailing rows/columns/corners, and repeated resizing
and source replacement on a single node. Readback waits for GPU completion.
An additional shader test samples the public node at every original pixel
across multiple levels of a 7x9 buffer, including a clamped out-of-range level.
This checks UV footprint ownership and automatic render-graph dispatch.

Tiny sampled screen-space rays compare linear leaf hits with hierarchical
rejection using the GPU-produced bounds. These test conservative rejection and
odd-edge ownership, including misses and edge hits. They do not validate
SSRNode's production ray marcher.

Actual SSR shader validation
----------------------------

Open http://localhost:8080/test/webgpu/ssr-hi-z.html for production SSR
comparisons with Hi-Z disabled and enabled. The procedural scene contains a
reflective floor and five colored boxes. All inputs are local source modules.
The test covers perspective/orthographic cameras, standard/reversed/logarithmic depth,
mirror/stochastic rays, binary refinement on/off, low/full quality and a 65x49
odd-sized render target (48 cases). Noise indices reset before each render.
Both paths read back float RGBA reflection data after GPU completion; tests
require nonempty hits, identical hit masks and maximum channel error <= 0.002.
This is a correctness comparison, not a performance benchmark.
