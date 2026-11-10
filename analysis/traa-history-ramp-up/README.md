# TRAA history ramp-up measurements

These files accompany the test PR on `bhouston/three.js`, based on upstream `dev` at `c0acf54779ed`.
They live on a separate evidence branch so the implementation PR changes only TRAA and its existing SSR example.

## Method

Adapted from `bhouston/houston-stratified-noise/analysis/gpu/` (`run.mjs`, `measure_common.js`, `measure_ssr.html`, `measure_vxgi.html`, `analyze_realtime.py`). Noise helpers and effect sampling are unchanged. Linear Float32 RGB is read **after TRAA**, with tone mapping and output color conversion disabled. The rendered scenes and effect settings come from that harness. Resolution: 640 × 480, pixel ratio 1.

- Reference: 512-frame temporal mean after 512 warm-up frames, fixed minimum weight, current unmodified sampler and denoiser. This is a finite post-TRAA reference, not ground truth and not an independent noise sampler.
- Reset convergence: warm 256, 293, or 330 frames (three different sample/jitter phases), reset TRAA and, for SSR, its temporal denoiser. Keep the camera stationary. Read frames 1, 2, 4, 8, 16, 32, 64. VXGI retains its voxel volume. Old and new use the same phases.
- Steady state: after each convergence run, warm another 256 frames and capture 256 consecutive frames. Bias is RMS of the temporal mean minus the reference. RMSE = sqrt(temporal variance + bias²). Flicker is RMS consecutive-frame RGB difference.
- Motion: 256 warm-up frames, then an orbit of 0.001 radians per rendered frame for 128 frames around the scene's fixed look-at target. Capture frames 32, 64, 96, 128. No per-frame converged reference: judged visually from crops, not a quantitative ghosting test.
- Throughput: original upstream TRAA, modified TRAA with ramp off, modified TRAA with ramp on; sequentially, three rounds. 60 warm-up frames, then five seconds per variant. Disable vsync and frame-rate limits. Wait for `queue.onSubmittedWorkDone()` after each batch of four renders and before switching variants. Count GPU-completed frames, not submissions. No individual frame timing.
- The current renderer automatically restarts its internal animation loop during on-demand rendering. The driver stops and suppresses that loop on this renderer instance and sets the initial frame ID to zero, then uses the companion harness's manual `tick()` per rendered frame. This ensures reproducible sample phases without wall-clock RAF ticks between readbacks.
- Log the actual renderer device's `GPUAdapter.info` and reject software/fallback adapters. `verification.json` also tests the hardware WebGL fallback separately.

Three independent phase repeats are summarized as mean [minimum, maximum], not confidence intervals. FPS is from a shared, otherwise uncontrolled desktop GPU; the ranges matter. One GPU and one scene per effect cannot establish general equivalence or perceptual quality.

Images use the same display mapping for both variants: `pow(max(linear, 0) / (1 + max(linear, 0)), 1 / 2.2)`. This mapping is for previews only; metrics use unmodified linear floats.

## Reproduce

Install Puppeteer 25.16 and Sharp 0.35 in the directory containing these scripts, and NumPy for Python. Use hardware Chrome with Metal; adjust the executable and backend flag for other hardware platforms.

1. In a dedicated Three.js worktree containing the proposed change, copy `harness/measure_*.{js,html}` to `examples/`.
2. Save the upstream TRAA module for the untouched throughput control (never change the main checkout):
   ```sh
   git-dedup show c0acf54779ed:examples/jsm/tsl/display/TRAANode.js > examples/jsm/tsl/display/MeasureOriginalTRAANode.js
   ```
3. Run Three.js's `npm run dev` in that worktree on port 8141 (temporarily adjust the script's `-p` argument locally). The regular dev script ignores a trailing `--port` argument. Do not commit generated `build/` files.
4. Put these scripts under `analysis/traa-history-ramp-up/` in that worktree and run:
   ```sh
   node analysis/traa-history-ramp-up/run.mjs
   python3 analysis/traa-history-ramp-up/analyze.py
   node analysis/traa-history-ramp-up/crops.mjs
   node analysis/traa-history-ramp-up/verify.mjs
   ```
   Run verification only after the performance measurements, to avoid concurrent GPU work.

Raw RGB float readbacks are retained locally under `raw/` and omitted from this evidence branch due to their size. `runs.json` includes adapter information, per-run statistics, and all completed-frame counts and durations. `metrics.json` includes all repeat measurements and summary tables. `verification.json` records reset, resize, count cap, runtime-switch, output-alpha and hardware-backend checks.
