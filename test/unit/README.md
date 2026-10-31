# Unit tests

Install dependencies with `npm ci`. Tests import current source files. No build or manually started server is required.

| Command | Environment |
| --- | --- |
| `npm run test-unit` | CPU core and TSL code-generation tests in Node |
| `npm run test-unit-addons` | CPU addon tests in Node |
| `npm run test-unit-browser` | Browser-dependent image and loader tests in Chromium |
| `npm run test-unit-gpu` | Native hardware WebGL and WebGPU tests using vitest-gpu |
| `npm test` | Core lint, CPU units and browser units |

Install Chromium once with `npx playwright install chromium` for browser tests. `npm run test-unit-headful` starts interactive browser tests. Vitest discovers `*.tests.js` automatically; no import manifest is needed. Use `npx vitest run --project unit test/unit/src/math/Vector3.tests.js` to select a CPU file. Use `npm run test-unit-gpu -- --project webgpu` to select a GPU backend.

Tests use native Vitest `describe`, `test`, hooks and assertions. Existing numerical and geometry comparisons remain in `utils/test-utils.js`. An unfinished behavior can use `test.fails` when it should execute and report an unexpected pass; use `test.skip` for an explicitly unsupported case.

GPU tests use the WebGL and WebGPU Node environments from [vitest-gpu](https://github.com/bhouston/vitest-gpu). They test real shader results and renderer pixels through completed GPU readbacks. Missing contexts or software adapters fail hardware verification. WebGL skips tests requiring WebGPU-only workgroups, barriers or atomics; unsupported optional adapter features are reported as skips. The native GPU command is separate so CPU work can run on machines without a GPU; CI runs both backends on macOS using Metal and checks the actual adapter identity. Node 22 or later is required by the native environments.

Browser tests are reserved for APIs such as DOMParser, image decoding and HTTP loader integration. Browser examples and screenshot maintenance are documented in [../e2e/README.md](../e2e/README.md). Use `debugger` with Vitest's browser UI or Playwright traces to inspect a failure.

## Development workflow

`npm run test-unit-watch` runs core and addon files in parallel (up to four workers) and reruns affected tests as source changes. Select a file with `npm run test-unit-watch -- test/unit/src/math/Vector3.tests.js`. Tests within a file remain sequential, preserving existing shared-state assumptions. GPU files are limited to one worker per backend; example tests default to one worker and CI uses five Playwright shards.

The [official Vitest VS Code extension](https://github.com/vitest-dev/vscode) provides Test Explorer discovery, individual test runs and debugging using the Vitest configurations. No repository-specific editor integration is needed. Browser and native GPU projects still require their documented browser/hardware prerequisites.

The migration removes the custom QUnit HTML/import manifests and Puppeteer test lifecycle. Vitest supplies file discovery, filtering, watch mode, assertion diffs, source locations and per-test results. Playwright supplies browser lifecycle, retries, sharding and traces. CPU tests run without starting a browser or server. These remove setup work; an overall speedup has not been established by a controlled comparison with the old complete suite.

## Async failures and diagnostics

| Check | Behavior |
| --- | --- |
| Unhandled errors | Unhandled promise rejections and uncaught errors fail the run by default, including browser error events. |
| Unresolved work | A promise returned or awaited by a test reaches the test timeout: 5 seconds in Node, 15 seconds in Browser Mode, or the configured 30 seconds for native GPU tests. |
| Forgotten assertion awaits | Vitest 5 fails unawaited asynchronous `expect` assertions such as `resolves` and `rejects`. Arbitrary unreturned promises are not universally detectable. |
| Isolation | Test files get isolated environments by default. This does not isolate individual tests within a file or guarantee disposal of GPU resources. |

These are [Vitest defaults](https://vitest.dev/guide/features), not additional checks implemented by this PR. See [async assertion migration notes](https://vitest.dev/guide/migration/#unawaited-asynchronous-assertions-fail-the-test), [timeouts](https://vitest.dev/config/testtimeout) and [parallelism](https://vitest.dev/guide/parallelism).

For Node async-resource diagnostics, use `npx vitest run --project unit --detectAsyncLeaks`. For a process that will not exit, use `npx vitest run --project unit --reporter=default --reporter=hanging-process`. [Async leak detection](https://vitest.dev/config/detectasyncleaks) is opt-in, adds overhead and does not support Browser Mode. The [hanging-process reporter](https://vitest.dev/guide/reporters#hanging-process-reporter) diagnoses open handles. Neither is a general JavaScript heap-leak detector, and neither automatically fixes a leak. Tests still need to await work and clean up resources.
