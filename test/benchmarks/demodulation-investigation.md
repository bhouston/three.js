# Demodulation performance investigation

Investigated 2026-10-17 on Build001, Windows, Chrome 155, hardware WebGPU on NVIDIA GeForce GTX 1050 (2 GB; driver 560.94). Baseline: `demodulation-helpers` at `fe6f83e505157de8863cb438ed7fe106fbfc96bc` (PR #61).

## Finding

Demodulation is a standard technique. The example is a deliberately exaggerated, spatial-only texture-preservation demonstration, and its frame rate is not a good estimate of the cost of adding demodulation to a production temporal denoiser. It has avoidable extra render passes and silently follows the display's device pixel ratio.

The immediate fix shares one demodulated texture between `recurrentDenoise`'s filtered and raw inputs. Previously each call to `toTextureNode()` independently created an RTT node from the same expression. At 1080p, removing the duplicate pass raises the single demodulated view from about 55 to 80 completed frames/s. The corresponding single unmodulated view is about 85 frames/s. The helper's division/multiplication is not intrinsically responsible for the original large performance gap.

The demo now defaults to pixel ratio 1 and exposes a pixel-ratio control. This trades display-pixel sharpness for speed on high-DPI screens. It keeps the existing default Split comparison and Inspector controls.

## Measurements

These are rendering throughput measurements, not display refresh rates or individual-frame GPU timings. Values below are arithmetic means of two sustained runs; raw counts and elapsed durations are in the issue artifacts.

| Drawing-buffer resolution | View | Original FPS | Shared-texture FPS |
| --- | --- | ---: | ---: |
| 1280 x 720 | Noisy signal, no denoiser | 1271.3 | 1270.3 |
| 1280 x 720 | Without demodulation | 277.3 | 276.4 |
| 1280 x 720 | With demodulation | 163.9 | 218.7 |
| 1280 x 720 | Split | 111.7 | 134.2 |
| 1920 x 1080 | Noisy signal, no denoiser | 698.7 | 702.1 |
| 1920 x 1080 | Without demodulation | 85.2 | 84.7 |
| 1920 x 1080 | With demodulation | 54.6 | 80.5 |
| 1920 x 1080 | Split | 34.8 | 43.9 |
| 3840 x 2160 | With demodulation | 8.7 | 12.9 |
| 3840 x 2160 | Split | 5.3 | 6.6 |

At 1080p this is approximately +47% single-view throughput and +26% Split throughput. At DPR 2, a 1920 x 1080 CSS window has a 3840 x 2160 drawing buffer: four times the pixels. The observed slowdown at 4K is even larger than simple pixel scaling on this GPU; these measurements do not isolate cache, bandwidth, or residency effects, so they should not be used to attribute that extra slowdown to one particular GPU bottleneck.

Split is `mix(without, withDemodulation, step(...))`. Both inputs are rendered over the entire screen before that final selection; it does not halve either denoiser's work. Measured render calls per completed frame:

| View | Original | Shared texture |
| --- | ---: | ---: |
| Noisy | 2 | 2 |
| Without | 3 | 3 |
| With | 5 | 4 |
| Split | 6 | 5 |

Each call includes its rendering work; the scene call draws the floor and both spheres. Splitting the final display is not a single production denoising path.

## Additional experiments

These use the shared-texture version at the same physical resolution and the same two-run protocol:

| Experiment | Resolution | Single demodulated FPS | Split FPS |
| --- | --- | ---: | ---: |
| Shared texture, MSAA on | 1920 x 1080 | 80.5 | 43.9 |
| Emit illumination in an extra scene MRT attachment | 1920 x 1080 | 81.8 | 44.3 |
| Shared texture, renderer MSAA off | 1920 x 1080 | 108.1 | 59.2 |
| Shared texture, MSAA on | 3840 x 2160 | 12.9 | 6.6 |
| Shared texture, renderer MSAA off | 3840 x 2160 | 15.6 | 8.0 |

The MRT experiment computes this synthetic illumination directly, removes the remaining conversion pass, and adds an attachment to the scene. It achieves only a small net improvement here: eliminating a pass does not eliminate its replacement's bandwidth cost. It is an experiment, not a change to the shipped example or a general claim about real ray-traced lighting producers.

Disabling MSAA improves throughput materially on this GPU, but changes antialiasing quality. The PR retains MSAA. In a production pipeline with temporal AA, benchmark whether scene MSAA and renderer-wide MSAA are useful for the desired quality. MSAA is not a required part of demodulation.

## Parameter audit

- `mode: 'diffuse'`, linear albedo demodulation, remodulation after filtering, `alphaSource: 'none'`, and `accumulate: false` are appropriate for this specific isolated diffuse/spatial demonstration. They do not demonstrate a complete temporal pipeline.
- Alpha is always 1. The filter derives `frameNum = 1 / alpha`, then variance 1 and aggressivity 0. Luminance edge stopping is multiplied by aggressivity, so the example's `lumaPhi = 0.01` had **no effect**. The unused override was removed. The filter also retains its full spatial radius. Simply setting `accumulate: true` will not introduce a correctly reprojected history; history, motion, depth/normal rejection, and reciprocal-history alpha must be supplied by the temporal pipeline.
- Radius 1.5 becomes a view-distance-scaled world radius (`radius * 0.1 * abs(viewZ)` for this diffuse path). It is intentionally large enough to blur the weave on the left. It is not a radius in pixels. The filter executes eight taps regardless of radius; radius can still affect cache behavior and quality.
- `normalPhi = 5` is the node's existing default, not a separately validated physical tuning value. Its implementation uses `(1-normalPhi)^2` inside a clamp when selecting the normal lobe. Here 5 saturates the clamp, and values above about 1.995 produce the same broad normal filtering. Treat its UI meaning with care; this report does not retune the shared denoiser.
- Noise strength 0.4 and texture frequency 36 are synthetic presentation choices, not Monte Carlo sampling parameters. There is no ray tracing or real GI workload here. Eight spatial taps without accumulated history leave visible noise; this is not evidence that a temporal production denoiser cannot converge.
- The helpers' per-channel `1e-4` floor prevents division by zero and preserves alpha. All demo albedos are at least 0.025, so the floor does not activate. Near-black HDR materials require separate stability testing. For specular signals use a compatible pre-integrated BRDF factor, not base color; never demodulate combined beauty containing unrelated lighting/emission.

## Research and expected usage

[NVIDIA NRD's integration guidance](https://github.com/NVIDIA-RTX/NRD#material-demodulation) explicitly removes material factors before denoising and reapplies them afterward. Its published native RTX 4080/1440p reference reports 2.55 ms for combined REBLUR diffuse/specular and 3.25 ms for RELAX with the documented settings. Those hardware, algorithms, signal types, APIs and timings are not comparable benchmarks for this GTX 1050 browser demo; they show that material demodulation is used in practical real-time denoisers, rather than imply a target FPS here.

[SVGF](https://research.nvidia.com/labs/rtr/publication/schied2017spatiotemporal/) combines temporal accumulation and variance-guided spatial filtering. That is the relevant production context, while this example deliberately removes temporal accumulation to expose albedo blurring.

[NRD's material-factor shader](https://github.com/NVIDIA-RTX/NRD/blob/master/Shaders/NRD.hlsli) uses a different near-black stabilization policy (material factors interpolated toward a 0.02 floor). That is a numerical/quality design choice, not a reason to change this generic helper's semantics without dedicated tests.

Recommended next production integration: denoise one separated lighting signal in a consistent demodulated history domain, remodulate once at resolve, avoid redundant intermediate textures (or emit illumination directly when the lighting producer can do so), and evaluate temporal quality under motion. Start at a controlled drawing-buffer resolution. Reduced-resolution denoising needs geometry-aware reconstruction and its own quality tests; this investigation does not add that feature.

## Reproduction and limits

Run from the repository root after `npm ci` and `npm run build`:

```powershell
$env:BENCH_OUTPUT = 'demodulation-benchmark.json'
node test/benchmarks/demodulation-investigation.js
```

The harness serves the real example with a hook replacing its animation loop. It uses visible Chrome, disables vsync/frame caps, verifies hardware adapter information and the WebGL renderer, warms each implementation for at least 1.5 seconds, measures for at least 5 seconds, and runs implementations sequentially. It submits eight complete frames, waits for `GPUQueue.onSubmittedWorkDone()`, and only then counts those eight as completed. It drains the queue before switching. The final batch may extend the interval beyond five seconds. No individual-frame timings are taken.

The harness stops Three.js's internal animation timer and manually advances its NodeFrame and resets renderer info for **every** completed pipeline frame. Without this, once-per-frame scene/RTT/denoiser passes can be reused across submissions, yielding invalid high FPS. All required scene, conversion, filtering, MSAA and final output work is included. Inspector UI construction remains, but animation-loop profiler collection and browser presentation/compositing cadence are excluded. Thus this measures sustained rendering throughput, not exactly the FPS shown by the interactive Inspector.

Options (environment variables): `CHROME_PATH`, `BENCH_SOURCE` (saved example HTML baseline), `BENCH_OUTPUT`, `BENCH_SHOT`, `BENCH_SIZES` (JSON pairs), `BENCH_REPEATS`, `BENCH_VIEWS` (comma-separated output names), `BENCH_RADIUS`, and `BENCH_VARIANT` (`mrt`, `no-msaa`, or `mrt-no-msaa`). Put screenshots in the swarm artifact directory when running under the swarm. No software renderer is permitted.

Baseline and fixed runs used identical drawing-buffer sizes, camera, geometry, shader implementation and MSAA settings. The default DPR change therefore does not contribute to the same-resolution speedup. Two repeats are a useful local check, not a cross-device performance guarantee. Runs were sequential, with no concurrent rendering benchmarks. Root `build/` outputs are generated locally and are not part of this change.



## Validation

- `npm run build`: passed.
- `npm test`: passed; 1957 core tests passed, one existing TODO, zero failed; 108 addon tests passed, zero failed. The stock runner sets Linux Vulkan/lavapipe-related launch settings. A local Puppeteer preload replaced those launch settings with the machine's existing Chrome/D3D11 configuration and verified a physical NVIDIA WebGL renderer before each suite. No repository test code was changed. The runner emitted Chrome process-shutdown warnings and a resource 404, but returned exit code 0 with all tests passing; full output is in `test-results.log`.
- Core lint and ESLint for the changed example and benchmark: passed. Benchmark syntax check: passed.
- Interactive visible Chrome validation on physical NVIDIA WebGPU: passed, no page errors. At device pixel ratio 2, a 1280 x 720 CSS viewport keeps a 1280 x 720 drawing buffer with the new default. Switching With demodulation, Albedo and Split through the actual select control passed. The screenshot retains the visible texture-preservation contrast. Inspector startup FPS shown in that screenshot is not a throughput measurement.
- No typecheck script exists in this JavaScript repository. No TypeScript changes were made.

Issue artifacts include `benchmark-baseline.json`, `benchmark-baseline-4k.json`, `benchmark-shared.json`, `benchmark-mrt.json`, `benchmark-no-msaa.json`, interactive validation JSON, screenshots and test output. The benchmark JSON stores completed-frame counts and elapsed durations, not just rounded FPS.
