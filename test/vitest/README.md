# NTC test suite (vitest + playwright)

Isolated, incremental tests for the NTC (Neural Texture Compression)
framework (`examples/jsm/ntc/`, `examples/jsm/loaders/NTCLoader.js`),
separate from the main `test/unit` src/addons suites since these exercise
the public `examples/jsm` surface directly (not `src/`) and some need real
WebGPU.

```
npm run test-ntc          # everything (unit + browser)
npm run test-ntc-unit     # pure-JS project only, fast, no browser
npm run test-ntc-browser  # real WebGPU project only
npm run test-ntc-watch    # interactive watch mode, both projects
```

Two vitest projects (see `vitest.config.js`):

- **`unit`** (`test/vitest/unit/ntc/`) - plain Node, no browser. Pure-JS
  logic: CPU model construction, quantization, manifest encode/decode,
  channel classification/layout, training orchestration with GPU calls
  mocked.
- **`browser`** (`test/vitest/browser/ntc/`) - real Chromium + WebGPU
  (`--enable-unsafe-webgpu`). GPU-side TSL node graphs cross-checked
  against a plain-JS CPU reference implementation of the same math (the
  `evaluateNeuralTextureRaw` vs. `forwardMLP` pattern most of these tests
  follow), plus render-based regression tests for specific bugs.

A handful of tests still carry the pre-consolidation `Neural*` module names
in `describe()` labels or comments (this suite predates the branch's
neural-appearance/neural-texture removal and NTC renaming) - the imports
and behavior under test are current even where a label lags.

`test/unit/addons/ntc/` (in the *other*, `test/unit`-rooted suite) covers
`NTCMLP.js`/`NTCMLPTSL.js` with the older QUnit-style `test/unit/addons`
harness, alongside every other addon.
