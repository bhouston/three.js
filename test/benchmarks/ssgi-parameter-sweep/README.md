# Interior SSGI parameter sweep

This isolated benchmark branch changes no rendering defaults. It exports fifteen cases: nine combinations of relative radius `0.1 / 0.15 / 0.25` and thickness `0.0025 / 0.005 / 0.01`, followed by independent trials of `3 × 8` samples, `2 × 12` samples, full resolution, 8 history frames, 32 history frames, and denoise radius 1.

Every case starts from the same `2 × 8`, half-resolution, radius `0.25`, thickness `0.01`, 16-history-frame, denoise-radius-1.5 settings. Other options, including the selected sampling technique, retain their caller-provided values. Models are loaded/baked once by the common harness before the sweep.

Run the common serial harness from its worktree with its experiment manifest:

```sh
node test/benchmarks/interior-ssgi-experiments.mjs /path/to/experiment-manifest.json
```

The common harness must route this branch's module to `/test/benchmarks/ssgi-parameter-sweep.js`, serve its CPU reference at `/test/benchmarks/ssgi-geometry-reference.js`, and invoke the sweep after scene loading and hardware verification. An ordinary manifest alone does not invoke a module: the common harness's sweep hook supplies the callback below.

```js
const { runSSGIParameterSweep } = await import( '/test/benchmarks/ssgi-parameter-sweep.js' );
const results = await runSSGIParameterSweep( {
  state: window.__experiment,
  step: window.__step,
  drain: window.__drain,
  read: window.__read,
  width: 960,
  height: 720,
  freezeLighting: value => { window.__freezeLighting = value; },
  getFreezeLighting: () => window.__freezeLighting
} );
```

`state` supplies `model` (or `scene`), `camera`, `params`, `ssgiParams`, `ssgiPass`, `ssgiDenoise`, `lightingHistory`, `updateOutput()` and `updateSSGIParameters()`. `read(target, index)` returns decoded float RGBA pixels; it must handle source and GI targets. Stop automatic animation first. Use a real hardware adapter with vsync and frame limits disabled.

Each implementation warms for 64 whole frames, then runs for at least five seconds in batches of four frames followed by GPU completion. A GI update counter verifies that the completed frame count represents actual rendering. Source, raw GI and filtered GI are read only after timing. CPU triangle tracing then estimates geometry visibility independently of the screen-space integrator, using the exact world radius reported by the pass. Results contain settings, GPU-completed throughput and compact geometry coverage/irradiance error statistics. After geometry evaluation, a separate 32-frame capture freezes source copying through the provided setter/getter callbacks. Welford accumulation measures per-pixel premultiplied irradiance and coverage variation, returning only aggregate temporal deviation and means. These readback-heavy observations are outside throughput timing. Source-freeze state, original settings and output are restored even if a case fails. Without the optional freeze callbacks, temporal statistics are explicitly marked `sourceFreezeControlled: false`.

The CPU reference uses static triangle geometry and screen radiance. It does not provide full path-traced lighting, missing illumination, normal-map shading, transparency, or a converged temporal ground truth. Fifteen cases on each of Cornell Box and Dual Rooms provide a controlled comparison; they do not establish universal best defaults.

CPU-only harness verification (uses simulated time and injected frame/reference helpers, and launches no GPU):

```sh
node test/benchmarks/ssgi-parameter-sweep.test.mjs
```

For a reference-only repair run, pass `geometryOnly: true`. Each case still resets settings, warms 64 whole frames and drains before the three readbacks, but returns `throughput: null` and `temporal: null`; it neither retimes throughput nor captures 32 temporal frames. Such rows can replace separately captured reference statistics without claiming a new performance measurement. The caller exposes `diffuseHistory` when the selected source is `Diffuse + emission`: a callback SampleNode has no render target, so source selection must use that history explicitly. A missing selected history throws rather than comparing against the wrong beauty input. Output metadata records the chosen source and reference version.
