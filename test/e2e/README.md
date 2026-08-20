# Three.js end-to-end testing

Driven by [Playwright](https://playwright.dev) under [vitest](https://vitest.dev)
(`test/e2e/e2e.test.js`, the `e2e` project in `vitest.config.js`). Screenshots
are pixel-diffed against the reference JPGs in `examples/screenshots/` using
the same hand-rolled comparator as before (`image.js`) - those files are the
test baseline *and* the source of the public examples gallery's thumbnails,
so they're not managed by Playwright's or vitest's own snapshot tooling.

### Motivation
Simplify code reviews with quick pixel testing inside CI. The same screenshots are used for thumbnails.

### Local usage
If you get an error in e2e test after PR and you sure that all is correct,
just make a new screenshot to example. As a last resort increase timeouts or add it to exception list.

```shell
# generate new screenshots for exact examples
E2E_ONLY=<example1_name>,<example2_name> npm run make-screenshot

# check exact examples (or use vitest's own -t <pattern> filter)
E2E_ONLY=<example1_name>,<example2_name> npm run test-e2e

# check all examples
npm run test-e2e

# check only webgpu_* examples
npm run test-e2e-webgpu
```

Merge only those commits that pass the tests, otherwise all next commits will also fail.

### How it works
- examples are queued onto a small pool of "lanes", each lane a full
  Chromium process reused sequentially for whichever example is next in the
  queue - `test.concurrent` lets vitest start all example tests at once, and
  each blocks until a lane is free. This gives real local parallelism
  (previously the script only ran one example at a time locally; parallelism
  only existed as a 5-way CI job matrix). Lane count defaults to
  `min(cpuCount - 1, 8)`, override with `E2E_WORKERS=<n>`.
- CI still shards across a job matrix (`CI=0..4`) on top of the lane pool
  within each job - the two are independent axes of parallelism.
- Chromium is launched with `--use-gl=angle --use-angle=swiftshader-webgl
  --enable-unsafe-swiftshader`, required since Chromium ~136 to get software
  WebGL in headless mode at all (without it, `getContext('webgl')` silently
  fails). WebGPU relies on a system Vulkan driver (lavapipe, installed via
  `mesa-vulkan-drivers` in CI).
- deterministic random/timer/rAF/video for screenshots
- increased robustness with hided text, datgui, different flags and timeouts.
- pipeline: turn off rAF -> 'networkidle0' -> networkTax -> turn on rAF -> render promise
- any `console.warn`/`console.error` logged by the page while its screenshot is
  being generated fails that example, in addition to the pixel diff. Fix the
  example (or, if the message is expected and harmless, add it to the
  exception list in `test/e2e/exception-list.js`) to get it passing again.
- on a `WebGPU Device Lost` error the affected lane's browser process is
  restarted and the example gets one retry before failing.

### Development progress

|           Travis                        |               Attempts               |
|-----------------------------------------|--------------------------------------|
| 61 from 362 failed, time=21:14          | networkidle0 timeout                 |
| 26 from 362 failed, time=16:22          | with rAF hook                        |
| 13=1+1+7+4 failed, time=4:26            | with render promise and parallelism  |
| 4=0+0+2+2 failed, time=5:13             | with network tax and other settings  |
| 4=0+0+2+2 failed, time=3:26             | with progressive attempts            |

### Status
97% examples are covered with tests. Check exception list for more information.
