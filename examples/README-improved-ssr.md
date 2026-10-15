# Experimental glossy reflections

Run `npm install` and `npm start`, then open:

- [Steampunk camera](webgpu_postprocessing_ssr_improved.html)
- [Dungeon room](webgpu_postprocessing_ssr_improved_denoise.html)

The models, camera transforms, lights, environment and material roughness match the corresponding existing SSR examples. The room uses AgX exposure 1.57 without the existing example's additional contrast/gamma grading; both experimental comparison modes share that presentation. Original asset credits remain on each page.

The mode dropdown also accepts `?mode=baseline`, `scene`, `environment`, `raw`, `denoised`, `coverage`, `variance`, or `history`. `baseline` uses the existing deterministic SSR/blur node for the camera, and stochastic GGX SSR with TemporalReprojectNode + RecurrentDenoiseNode for the room. This is a comparison of node pipelines in the replicated scene, not a pixel-identical copy of every postprocessing setting of the existing examples. Both experimental outputs use SMAA rather than the room example's final TRAA.

## Pipeline

1. The scene MRT records beauty, decoded material inputs, NDC motion, and the existing indirect-specular contribution. Normal XYZ and perceptual roughness are quantized to four bytes packed in one uint attachment, decoded before filtering. RGB HDR environment specular and RG16F velocity keep the scene within the default WebGPU color-attachment budget. The example requires renderable `rg11b10ufloat` support.
2. `SSRTraceNode` traces a biased glossy half-vector distribution, with two rays and 32 projected-depth steps by default. It is deliberately **not** GGX VNDF/PDF integration. Rays vary their reach by 1, 0.5 and 1.5, use temporal noise with a 64-frame phase and slower sequence offset, interpolate depth crossings, offset origins and clamp gathered radiance. Reach and thickness are explicit scene-unit controls rather than a bounding-sphere heuristic.
3. Half resolution means half width and height. A four-frame rotating phase covers all full-resolution pixel positions. Four depth-plane/normal-weighted taps reconstruct full-resolution radiance and mean positive hit distance separately. Mirrors below roughness 0.1 receive an additional full-resolution trace and bypass denoising, preserving sharp detail. Material transparency/exclusion is governed by the scene pass; no separate inline material ray tracer is installed.
4. `ReflectionDenoiseNode` prefilters moving pixels with 16 rotating Vogel-disk samples. Radius derives from `roughness² * hitDistance / pixelWorldSize`, capped at 3% of image height. Normal, plane distance and roughness reject unrelated taps. Static pixels bypass the prefilter.
5. Temporal reprojection uses a virtual reflected point plus four individually validated history taps. World normals and prior depth reject incompatible surfaces. Parallax relative to the lobe footprint limits history. Moving pixels receive RGBA neighborhood clipping and a coverage cap. History age is capped at four on the transition to camera stillness. Per-pixel velocity keeps animated surfaces responsive even when the camera is stationary.
6. Persistent moments track `luminance(RGB) + coverage`. Color uses inverse history age; moments retain at least 20% current contribution. Young history gets a spatial variance floor; luminance disagreement shortens history. Default age budgets are 16 moving and 256 stationary frames.
7. Four à-trous iterations use pixel strides 1, 2, 4, 8. Luminance weights use measured temporal variance. Plane, normal and roughness weights preserve boundaries. Strides beyond the estimated reflection-lobe footprint are bypassed. Temporal feedback stores the color **before** these spatial passes, avoiding repeated wide blur.
8. The compositor removes the captured environment specular according to reflection coverage and adds incident SSR weighted once by EnvironmentBRDF. Misses retain the original environment; intensity zero restores the original scene. A per-material lighting hook captures indirect specular without changing any global lighting prototype.
9. Optional feedback, enabled by default, stores the linear composite before tone mapping. Hit-point motion vectors reproject that previous composite as the next frame's radiance source. The source is read before its RTT is overwritten, so there is no shader read/write feedback loop. Debug-mode transitions, resize, projection changes and large camera cuts invalidate feedback. Disable `feedback` to isolate single-bounce reconstruction.

The raw and denoised RGB are premultiplied by **hit coverage**, with alpha zero for misses. A multi-ray raw pixel averages successful hits and reports alpha one when any hit succeeds; it does not estimate an unbiased hit probability. Hit distance is separate. This alpha contract differs from the existing SSRNode's ray-distance alpha and the recurrent denoiser's inverse-history alpha. Do not interchange them.

## Limits

This remains a screen-space experiment. Offscreen/hidden surfaces are absent, depth thickness is approximate, and ray-source feedback is delayed. The biased sampler, radiance clamp and finite history are not physically unbiased. Environment replacement currently uses a single-scatter BRDF while the captured environment can include multiscattering/AO; rough metals can change brightness. Clearcoat, sheen, transmission, skinned motion and disocclusions need more focused quality work. The current examples use perspective cameras and standard materials. Logarithmic depth is not supported by these new nodes.

There is no assertion of improved image quality or throughput over the existing pipeline. Raw/denoised views demonstrate reconstruction but do not quantify temporal error or ghosting. Inspector FPS is not a benchmark.

## Validation

After starting with a development build (`node utils/build/dev.js`), run:

```powershell
$env:CHROME_EXECUTABLE = 'C:/Program Files/Google/Chrome/Application/chrome.exe'
node test/e2e/improved-ssr.js
```

The test creates and closes its own local server and Chrome instance. `CHROME_EXECUTABLE` may point to an existing Chrome installation on other platforms. `SWARM_ARTIFACTS` overrides the screenshot/report output folder. It refuses fallback/software adapters, checks actual WebGPU execution, reads GPU-completed reflection data for valid hits/NaNs, exercises debug and stock comparison modes, and checks camera movement, odd-sized resize and full-resolution tracing. GPU moment readback verifies that reset/camera cuts start at age one, stationary history accumulates, and moving geometry leaves finite statistics. This is a rendering smoke test, not a throughput benchmark.

The implementation was tested on a physical NVIDIA GTX 1050 (Pascal). Core unit tests passed 1,957 tests with one existing todo; addon unit tests passed 102. Core/addon lint and targeted example/test lint passed. Full example lint has existing errors in unrelated Ammo, multiple-elements and shadertoy examples. The repository is JavaScript and has no TypeScript check script. Generated root `build/` files are not part of this change.
