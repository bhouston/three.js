# Unit tests

Install dependencies with `npm ci`. Tests import current source files; unit tests do not require a build or an HTTP server.

| Command | Environment |
| --- | --- |
| `npm run test-unit` | CPU core and TSL code-generation tests in Node |
| `npm run test-unit-addons` | CPU addon tests in Node |
| `npm run test-unit-browser` | Browser-dependent image and loader tests in Chromium |
| `npm run test-unit-gpu` | Native hardware WebGL and WebGPU tests using vitest-gpu |
| `npm test` | Core lint, CPU units and browser units |

Install Chromium once with `npx playwright install chromium` for browser tests. `npm run test-unit-headful` starts interactive browser tests. Vitest discovers `*.tests.js` automatically; no import manifest is needed. Use `npx vitest run --project unit test/unit/src/math/Vector3.tests.js` to select a CPU file. Use `npm run test-unit-gpu -- --project webgpu` to select a GPU backend.

Tests use native Vitest `describe`, `test`, hooks and assertions. Existing numerical and geometry comparisons live in `utils/assertions.js` and `utils/test-utils.js`. An unfinished behavior can use `test.fails` when it should execute and report an unexpected pass; use `test.skip` for an explicitly unsupported case.

GPU tests use the WebGL and WebGPU Node environments from [vitest-gpu](https://github.com/bhouston/vitest-gpu). They test real shader results and renderer pixels through completed GPU readbacks. Missing contexts or software adapters fail hardware verification. WebGL skips tests requiring WebGPU-only workgroups, barriers or atomics; unsupported optional adapter features are reported as skips. The native GPU command is separate so CPU work can run on machines without a GPU; CI runs both backends on macOS using Metal and checks the actual adapter identity. Node 22 or later is required by the native environments.

Browser tests are reserved for APIs such as DOMParser, image decoding and HTTP loader integration. Browser examples and screenshot maintenance are documented in [../e2e/README.md](../e2e/README.md). Use `debugger` with Vitest's browser UI or Playwright traces to inspect a failure.
