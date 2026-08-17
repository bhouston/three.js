---
name: Neural Training Tests
overview: Add deterministic unit and browser tests proving that synthetic MaterialX/Physical teachers can be distilled into the neural appearance format and that the trained result visually matches the teacher within defined tolerances.
todos:
  - id: synthetic-fixtures
    content: Add constant opaque MaterialX fixtures for Lambert and glossy training tests
    status: pending
  - id: unit-convergence
    content: Add deterministic trainer unit tests for sampling invariants, loss decrease, canonical JSON export, and CPU BRDF point matching
    status: pending
  - id: test-mode
    content: Add URL-parameter test mode to the training example with no-rotation, auto-train, and Puppeteer-readable globals
    status: pending
  - id: visual-script
    content: Create a focused Puppeteer script that screenshots teacher and trained neural views and compares them numerically
    status: pending
  - id: verify-docs
    content: Run addon unit tests and the focused browser training test; document commands and thresholds
    status: pending
isProject: false
---

# Neural Appearance Training Test Plan

## Goal

Prove the trainer works on known-simple materials before trusting arbitrary MaterialX. Use two test layers:

- **Unit/convergence tests** for fast deterministic validation of sampling, training, export, and numeric BRDF agreement.
- **Puppeteer visual tests** for the full browser flow: load test MaterialX, train, switch to neural result, screenshot, compare to teacher.

This plan does not require prior context beyond the files named here.

## 1. Add deterministic synthetic materials

Create tiny opaque MaterialX fixtures under [`examples/materialx/`](examples/materialx/):

- `neural_train_lambert_constant.mtlx`: `standard_surface` with constant base color, high roughness, no metal, no coat.
- `neural_train_glossy_constant.mtlx`: constant base color, low roughness, nonzero specular, no textures.
- Optional later: `neural_train_checker_roughness.mtlx` once the trainer truly evaluates shader-side UV-dependent nodes.

Start with constant fixtures because the current trainer can overfit them quickly and the expected BRDF is analytically simple.

## 2. Strengthen addon unit tests

Extend [`test/unit/addons/loaders/NeuralAppearanceLoader.tests.js`](test/unit/addons/loaders/NeuralAppearanceLoader.tests.js), or split a new [`test/unit/addons/materials/NeuralAppearanceTrainer.tests.js`](test/unit/addons/materials/NeuralAppearanceTrainer.tests.js), registered from [`test/unit/three.addons.unit.js`](test/unit/three.addons.unit.js).

Add tests that do not need WebGPU rendering:

- **Canonical export shape**: train a tiny constant material for a few iterations, assert all manifest arrays are real JSON arrays, then parse with [`NeuralAppearanceLoader`](examples/jsm/loaders/NeuralAppearanceLoader.js).
- **Sampling invariants**: generated samples have valid `wi.z > 0`, `wo.z >= 0`, UVs in `[0,1]`, 14 encoder inputs, and finite RGB targets.
- **Loss decreases**: run a fixed seed, constant material, small `resolution: 1`, `batchSize: 16`, `iterations: 50`; assert final loss is lower than the first reported loss by a meaningful margin.
- **BRDF point match**: after training, evaluate the exported decoder on CPU for a fixed set of directions and compare to `evaluatePhysicalBRDF()` with a loose but explicit tolerance.

Use deterministic seeds and small batches so these tests remain stable and fast in `npm run test-unit-addons`.

## 3. Add a visual training test page mode

Modify [`examples/webgpu_materials_neural_appearance_train.html`](examples/webgpu_materials_neural_appearance_train.html) so it can run unattended from Puppeteer:

- Add URL params, for example:
  - `?test=lambert`
  - `&iterations=200`
  - `&batchSize=256`
  - `&resolution=1`
  - `&noRotate=1`
  - `&autoTrain=1`
- Disable mesh auto-rotation whenever `noRotate=1` or while training.
- Expose browser globals for tests:
  - `window.__neuralAppearanceTrainingReady`
  - `window.__neuralAppearanceTrainingDone`
  - `window.__neuralAppearanceSetView('teacher' | 'neural')`
  - `window.__neuralAppearanceLastLoss`
  - `window.__neuralAppearanceExportJson`

This keeps the interactive example intact while giving Puppeteer a stable API.

## 4. Add focused Puppeteer comparison script

Create a dedicated script, for example [`test/e2e/neural-appearance-training.js`](test/e2e/neural-appearance-training.js), instead of forcing the generic screenshot harness to handle training time.

Flow:

1. Start `utils/server.js` or reuse the e2e server helper.
2. Open:
   `http://localhost:<port>/examples/webgpu_materials_neural_appearance_train.html?test=lambert&autoTrain=1&noRotate=1&iterations=200&batchSize=256&resolution=1`
3. Wait for `window.__neuralAppearanceTrainingDone === true`.
4. Set teacher view, wait one frame, screenshot only the canvas.
5. Set neural view, wait one frame, screenshot only the canvas.
6. Compare the two images with [`test/e2e/image.js`](test/e2e/image.js), but use a looser neural-training tolerance than the normal example screenshot harness, e.g. mean RGB error plus max-different-pixels.
7. Fail on console errors, non-finite loss, missing export JSON, or invalid loader parse.

Why a dedicated script: the existing [`test/e2e/puppeteer.js`](test/e2e/puppeteer.js) compares examples to checked-in screenshots. Here we want teacher-vs-trained-neural comparison from the same run, so there is no fragile golden image.

## 5. Add optional static screenshot coverage

Once the training page is stable, add a normal screenshot for the default non-training state:

- [`examples/screenshots/webgpu_materials_neural_appearance_train.jpg`](examples/screenshots/webgpu_materials_neural_appearance_train.jpg)

This only verifies the example loads. The real correctness signal remains the dedicated teacher-vs-neural Puppeteer test.

## 6. Test commands

Add package scripts only if desired:

- `npm run test-unit-addons` for unit/convergence tests.
- `node test/e2e/neural-appearance-training.js` for the focused browser training test.
- Optionally wire the focused script into `npm run test-e2e-webgpu` later if runtime is acceptable on CI.

## Pass Criteria

- Unit tests are deterministic and pass without screenshots.
- Training export loads in [`webgpu_materials_neural_appearance.html`](examples/webgpu_materials_neural_appearance.html).
- Puppeteer test proves a trained neural material visually matches its teacher for at least one constant Lambert-style MaterialX fixture.
- The visual test uses fixed camera, fixed lighting, disabled rotation, fixed seed, and explicit numeric image thresholds.