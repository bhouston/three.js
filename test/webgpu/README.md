Hi-Z GPU validation
==================

Serve the repository root over localhost (for example `node utils/server.js -p 8080`),
then open http://localhost:8080/test/webgpu/hi-z.html in a WebGPU browser.
No build is required: the page imports source modules directly.

The page prints a JSON result, also available as `window.hiZTestResult` for browser
automation. Status is `running`, `passed`, or `failed`. It requires an identified
hardware WebGPU adapter and explicitly rejects software/fallback adapters.

The tests read every populated r32float atlas texel back from the GPU and compare
it exactly against an independent recursive CPU reduction. Coverage includes both
depth conventions (explicit override and automatic renderer detection), 1D/2D odd sizes through 257x129, deterministic random input,
constant input, impulses on trailing rows/columns/corners, and repeated resizing
and source replacement on a single node. Readback waits for GPU completion.

Tiny sampled screen-space rays compare linear leaf hits with hierarchical
rejection using the GPU-produced bounds. These test conservative rejection and
odd-edge ownership, including misses and edge hits. They do not validate
SSRNode's production ray marcher: Hi-Z traversal in SSR is a separate component.
