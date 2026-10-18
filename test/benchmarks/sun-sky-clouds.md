# Static clouds in the physical sun-sky example

The existing `webgpu_sky.html` example now composites a static spherical thin cloud layer into the atmosphere bake. `CloudLayerNode` uses five-octave MaterialX noise with coverage, altitude (metres), density (vertical optical depth), positive feature size (metres) and static arrangement offset controls. Coverage is an artistic threshold control rather than an exact covered-area fraction. The layer is clamped above the observer.

Cloud radiance uses the same atmosphere model's solar irradiance at cloud altitude, broad dual-lobe scattering, and approximate ambient/base lighting. This is an artistic thin-layer approximation, not volumetric transport or a multiple-scattering atmosphere model. Distant cloud footprints flatten toward the horizon.

The solar disc stays outside the sky/IBL input. A 1×1 GPU pass evaluates the same cloud density field in the sun direction. The meter carries its unscaled transmittance in alpha; the ground shader samples the same GPU result. The existing asynchronous meter readback then updates the directional light, CSM light, disc transmittance, exposure and readout together. There is no extra CPU readback barrier. Bake failure releases the baking flag in `finally`.

Clouds, ground and PMREM rebuild only when sun, atmosphere or cloud settings change. Orbiting and static frames reuse the textures. The scalar sunlight attenuation applies uniformly to this small scene; it is not a spatial cloud shadow map. Solar attenuation samples the central ray rather than averaging the finite disc. Static clouds have no animation, binary assets or new dependencies.

## Reproduce

From the repository root after `npm ci`:

```sh
node test/benchmarks/sun-sky-clouds.js cloud-results
```

The harness uses a physical WebGPU adapter and rejects software adapters. It captures elevations 90°, 30°, 5°, −2° with coverage 0, 0.3, 0.7 and 1. Full cover uses density 10; other states use density 2. It also captures views looking directly at the clear and covered sun. Output includes screenshots and JSON validation/throughput data. There is no configured TypeScript typecheck in this JavaScript repository.

## Validation on Build001

Hardware: NVIDIA GeForce GTX 1050, WebGPU adapter vendor `nvidia`, architecture `pascal`. Tests also ran against the unchanged checked-in build files used by raw.githack.

- All 16 sky images contained finite radiance.
- Coverage zero matched the original atmosphere shader bit-for-bit across 524,288 half-float components.
- Density zero returned exactly unit transmission.
- Thick full coverage reduced solar transmission below 0.001 at positive elevations.
- Directional light intensity matched atmosphere illuminance × cloud transmission × storage scale in every state; CSM matched directional intensity.
- Thirty calls to the example's actual animation function caused no new bakes with unchanged settings.
- Build and core lint passed; changed addon/example/benchmark ESLint passed.
- Core unit tests: 1,957 passed, one todo. Addon unit tests: 107 passed. The addon suite logged one resource 404; it had no assertion failures.

Complete rebuild throughput was about 73 GPU-completed updates/second with either clear or 50% cloud coverage, at 512×256 sky resolution. These counts include sky, meter, ground, PMREM, scene rendering and shadows, with three warmups and separate sequential runs lasting at least five seconds. Vsync/frame caps/software rasterization were disabled, and GPU queue completion was awaited for every counted update and before switching cases. The difference was within run-to-run noise; this is update throughput, not steady-state render FPS or a claim of a speedup.

## Screenshots

All captures use fixed manual exposure EV100 13.6 so changing cloud cover also reveals the lighting change.

Clear / overcast:

![Clear](screenshots/sky-clouds-clear.jpg)
![Overcast](screenshots/sky-clouds-overcast.jpg)

Golden hour:

![Golden hour](screenshots/sky-clouds-sunset.jpg)

Views toward the sun, clear / covered:

![Clear sun](screenshots/sky-clouds-sun-clear.jpg)
![Covered sun](screenshots/sky-clouds-sun-covered.jpg)
