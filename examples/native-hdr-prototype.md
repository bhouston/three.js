# Native HDR AVIF and JPEG XL prototype

This branch explores browser-native HDR image decoding into a Three.js `DataTexture`. It does not add a glTF extension or change `TextureLoader` behavior.

```js
import { NativeHDRLoader } from 'three/addons/loaders/NativeHDRLoader.js';
const texture = await new NativeHDRLoader().loadAsync( 'image.avif' ); // or .jxl
```

The loader uses native image decoding, float16 Canvas2D, `globalHDRHeadroom = Infinity`, float16 readback and CPU extended-sRGB linearization. It returns a half-float `LinearSRGBColorSpace` texture. An unsupported browser fails explicitly instead of silently producing an SDR texture. No WASM decoder or API key is needed.

## Run

From the repository root:

```sh
npm ci
node utils/build/dev.js
node utils/server.js
```

Open `/examples/webgl_loader_texture_native_hdr.html`. The page has WebGL2/WebGPU links, exposure adjustment, format selection and local image input. It reports native default-canvas controls, CPU HDR values and values sampled from a GPU render target. It checks GPU identity and rejects recognized software adapters. Both synthetic image fixtures have independent reference decoder checks; see [fixture provenance](textures/hdr-native/README.md).

Chrome 155 currently needs **Experimental Web Platform features** enabled for `globalHDRHeadroom`. Native format support and float16 canvas support alone are insufficient. This is a prototype requirement, not a production deployment recommendation.

Automated hardware validation, using an existing browser installation:

```powershell
$env:CHROME_PATH = 'C:/Program Files/Google/Chrome/Application/chrome.exe'
node utils/test-native-hdr.js --output ../hdr-results
node utils/test-native-hdr.js --experimental --output ../hdr-results
```

The default run expects explicit loader rejection; the experimental run requires both formats to preserve PQ reference values through WebGL2 and WebGPU within 2%. JSON and screenshots are optional outputs. Browser processes and the local server close at completion. No software renderer is selected. Hardware availability is required; this is not part of the ordinary unit-test CI suite.

## Measured results — October 14, 2026

Windows, Chrome 155.0.8059.79, NVIDIA GeForce RTX 3060 Ti. WebGL identified ANGLE/NVIDIA/D3D11; the explicit WebGPU adapter identified NVIDIA/Ampere and supplied the device used by Three.js. Both modes ran without software rendering flags.

| Format | WebGL2 | WebGPU | Largest RGB patch error | Brightest GPU red sample |
|---|---|---|---|---|
| AVIF | Passed | Passed | 0.213% | 49.15625 |
| JPEG XL | Passed | Passed | 0.317% | 49.15625 |

The brightest reference patch is approximately 49.26108 at 203-nit reference white. CPU and GPU samples agree; values above 1 survive upload and actual shader sampling. All eight patches, RGB channels and opaque alpha are checked. This is numeric HDR texture validation on a physical GPU, not an HDR monitor presentation test or compression benchmark.

With default browser settings, native decoding succeeds but float16 canvas drawing maps the brightest patch to approximately 1. The loader deliberately rejects this configuration because HDR headroom control is unavailable. Float16 storage by itself does not preserve source HDR.

Validation on the final code: core lint, addon lint, utility lint and lint of changed test/example files pass. The core unit suite reports 1,957 passed and one existing TODO; addon suite reports 104 passed, including two new asynchronous loader lifecycle tests. Unit suites were run with the existing Chrome through a separate harness because the stock unit runner specifies Linux Vulkan/lavapipe configuration. No software renderer was used for the HDR proof. The repository has no typecheck script. The fixture generator reproduces the committed reference sample codes.

**Conclusion:** a shared native AVIF/JXL loader is technically viable in the tested experimental Chrome configuration, and the same half-float texture works in both Three.js rendering backends. Native HDR extraction remains unsuitable as a universal glTF baseline until its API and color/radiance semantics are portable. A production path still needs a controlled decoder fallback and wider image/browser qualification.

## Limits

- Browser-defined reference-white scaling is not a physical radiance contract. The measured Chrome PQ path uses approximately 203 cd/m² = 1; other browsers and profiles need numerical qualification.
- Only neutral, opaque PQ fixtures are currently reference-tested. HLG, gain maps, color/ICC variants, alpha, orientation and photographic images need further qualification. The loader rejects non-finite/overflow pixels but cannot infer whether every browser/profile silently tone-mapped the source.
- Recovered linear samples remain uncompressed on the GPU. This is not AVIF/JXL-to-BC6H transcoding or a zero-copy path.
- CPU readback and per-pixel conversion are intentionally simple, with no performance claim. The preview tone maps to SDR; numerical testing does not require an HDR display.
- Safari/Firefox are not certified by this Windows-only experiment. Availability of the experimental headroom API is checked at runtime.
- WebGPU uses the same CPU-decoded DataTexture as WebGL2. Direct native-image imports have separate unresolved HDR semantics and are not represented as proven here.
