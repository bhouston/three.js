# QUnit -> Vitest migration

Status: **unit test migration complete.** Every `test/unit/src/**/*.tests.js`
and `test/unit/addons/**/*.tests.js` file has a converted `.js` sibling, and
the full suite passes: 1377 real tests + 29 `test.todo` placeholders (for
QUnit modules that had zero tests), 0 failures, eslint clean, matching the
original QUnit `QUnit.test(` count exactly.

This was additive throughout the conversion (QUnit originals left in place,
untouched) so it could be verified safely; QUnit removal and the CI cutover
are tracked separately (see the repo root for current status — this file is
the historical record of how the conversion was done and the conventions it
established, kept for reference and as the playbook for any future test
file).

## What's here

- `vitest.config.js` (repo root) — two vitest "projects":
  - **`unit-node`** — plain Node, matches `test/unit/src/**/*.js` and
    `test/unit/addons/**/*.js` other than the exclusions below. This is the
    default lane: fast (no browser boot), watch-mode-friendly. It ended up
    covering the vast majority of the suite.
  - **`unit-browser`** — real Chromium via Playwright (vitest browser mode),
    matches `**/*.browser.js` in either tree.
  - Both exclude the old `*.tests.js` QUnit files (they'll be deleted once
    QUnit itself is retired), and the node lane also excludes `*.browser.js`
    so a file only runs in one lane.
  - `resolve.alias` defines `@src` -> `src/` and `@test-utils` ->
    `test/unit/utils/`, so imports don't need `../../../../`.
  - `exclude` also lists specific shared non-test helper modules (see below).
- `test/unit/vitest-setup.js` — `expect()` matchers with the same names/
  semantics as QUnit assertions, so a conversion is a mechanical rename, not
  a rewrite:
  - `toNumEqual`, `toEqualKey`, `toSmartEqual` — same names/semantics as the
    custom asserts in `test/unit/utils/qunit-utils.js`.
  - `toEqualLikeQUnit` — for `assert.deepEqual`/`assert.propEqual` calls that
    compare two three.js **class instances** (not plain object/array
    literals). Vitest's plain `toEqual` is stricter than QUnit's `deepEqual`
    in two ways that come up often in this codebase: numbers compare via
    `Object.is` (so `-0 !== 0`, whereas QUnit's `===`-based compare treats
    them as equal — and three.js code legitimately produces `-0`, e.g.
    rotation round-trips through quaternions), and two different function
    references are never equal (whereas QUnit always treats two functions as
    equal — relevant because things like `Euler`'s `onChange` callback are
    bound per-instance in a class's constructor, so two otherwise-identical
    instances have different callback closures). Use plain `toEqual` when
    comparing plain object/array literals (no `-0`/function-identity gap
    there); use `toEqualLikeQUnit` when comparing class instances.
- Shared, hand-ported (not mechanically converted) test helpers:
  - `test/unit/utils/std-geometry-tests.js` — vitest port of
    `qunit-utils.js`'s `runStdGeometryTests` (the original assigns onto the
    global `QUnit.assert` object, which doesn't exist under vitest).
  - `test/unit/utils/light-tests.js` — same idea, for `runStdLightTests`.
  - Both are excluded from the node lane's test-file glob in
    `vitest.config.js` (they're helpers, not test files themselves).

Run them:
```
npm run test-unit-vitest          # both lanes
npm run test-unit-vitest-node     # node lane only
npm run test-unit-vitest-browser  # browser lane only
npm run test-unit-vitest-watch    # watch mode, both lanes
```

## Naming and file conventions

- **Lane** is chosen by filename suffix, no separate allowlist to maintain:
  `*.browser.js` -> browser lane, everything else -> node lane. Reserve the
  browser lane for tests that genuinely exercise `document`,
  `canvas`/`Image`/`createImageBitmap`, a real WebGL/WebGPU context, or
  similar — not just for files that import a class that touches those things
  in a code path the test doesn't exercise. In the end, out of 1406 total
  test cases across 249 files, only 4 files needed the browser lane:
  `loaders/ImageBitmapLoader.browser.js`, `addons/loaders/FBXLoader.browser.js`,
  `addons/loaders/GLTFLoader.browser.js` (real fixture loading /
  `ProgressEvent`/`createImageBitmap` paths Node doesn't have), plus whatever
  the addons segment found - see its file listing under `test/unit/addons/`.
- **No `.test.js`/`.tests.js` suffix** — these already live under
  `test/unit/src/` or `test/unit/addons/`, so the plain `.js` name matching
  the class under test (`Vector3.js`, `ImageBitmapLoader.browser.js`) is
  enough. The old QUnit files keep the `.tests.js` suffix until they're
  deleted.
- **Imports use the `@src`/`@test-utils` aliases** for `test/unit/src/**`
  files, not relative `../../../../` paths. **Addons tests** (`test/unit/addons/**`)
  are the one exception: they keep bare `'three'` / `'three/webgpu'` /
  `'three/tsl'` imports unchanged (these resolve via the package's own
  `exports` field / Node self-reference resolution, verified working under
  both plain Node and Vite/vitest) — that's intentional, since addons tests
  are meant to exercise the public package surface, not internal `src/`
  paths.
- **Don't add comments that just restate the next line** (`// Instancing`
  above `test( 'Instancing', ...)`). Keep comments that add information the
  code doesn't already say.
- **Empty QUnit modules** (zero tests inside, legal in QUnit) become a single
  `test.todo( 'no tests yet' )` placeholder — vitest errors with "No test
  found in suite" on a `describe` block with nothing in it, so an empty stub
  needs this to stay a valid (skipped, not failing) test file.

**Shared non-test helper modules** (e.g. `test/unit/addons/utils/GaussianSplatTestUtils.js`,
`test/unit/utils/std-geometry-tests.js`, `test/unit/utils/light-tests.js`)
must be added to the `exclude` list in `vitest.config.js` explicitly — the
node lane's `include` glob otherwise assumes every `.js` file under
`test/unit/src/` and `test/unit/addons/` is a test file, and errors with "No
test suite found" if one isn't.

## Findings from surveying all pre-migration `*.tests.js` files

Grepped every test file for `document.`, `window.`, `new Image(`,
`navigator.`, `createElementNS`, `HTMLCanvasElement`, `OffscreenCanvas`, and
separately for tests that import renderer/canvas-ish classes
(`WebGLRenderer`, `CanvasTexture`, `WebGLExtensions`, etc.) to see if they
exercise real GPU/DOM paths, before any conversion started.

**Result confirmed by the full conversion: the overwhelming majority of the
unit suite never needed a browser.** Things that looked suspicious turned
out not to need it:
- `renderers/WebGLRenderer.tests.js` is an empty stub (no body).
- `textures/CanvasTexture.tests.js` only checks inheritance/instancing, never
  calls `getContext()` or draws.
- `renderers/webgl/WebGLExtensions.tests.js` uses a hand-rolled mock GL
  context object, not a real one.

Real GPU rendering correctness is covered by the separate e2e screenshot
suite (`test/e2e/`), not the unit tests — so this isn't a gap the migration
introduces, it's how the suite already divides responsibility.

## Conversion playbook (kept for any future test file)

For a new `X.tests.js` (or converting one that's still QUnit-only):

1. Create `X.js` next to it (do **not** delete or edit the `.tests.js`
   original until QUnit itself is being retired).
2. Mechanical rewrites:
   - `import { describe, test, expect } from 'vitest';` at the top.
   - Rewrite `../../../../src/...` imports to `@src/...`, and
     `../../utils/...` imports to `@test-utils/...` (skip this for
     `test/unit/addons/**` files — see above).
   - `QUnit.module( 'X', () => {` → `describe( 'X', () => {`; drop the
     `export default` on the outermost one.
   - `QUnit.test( 'name', ( assert ) => {` → `test( 'name', () => {` (drop
     the `assert` param).
   - `assert.ok( EXPR, 'msg' )` → `expect( EXPR ).toBeTruthy()`
   - `assert.notOk( EXPR, 'msg' )` → `expect( EXPR ).toBeFalsy()`
   - `assert.strictEqual( A, B, 'msg' )` / `assert.equal( A, B, 'msg' )` →
     `expect( A ).toBe( B )`
   - `assert.notStrictEqual( A, B, 'msg' )` / `assert.notEqual( A, B, 'msg' )`
     → `expect( A ).not.toBe( B )`
   - `assert.deepEqual( A, B, 'msg' )` / `assert.propEqual( A, B, 'msg' )` →
     `expect( A ).toEqual( B )` for plain object/array literals, or
     `expect( A ).toEqualLikeQUnit( B )` when `A`/`B` are three.js class
     instances (see above for why).
   - `assert.notDeepEqual( A, B, 'msg' )` → `expect( A ).not.toEqual( B )`
     (or `.not.toEqualLikeQUnit(...)` for class instances).
   - `assert.throws( fn, /regex/, 'msg' )` → `expect( fn ).toThrow( /regex/ )`
   - `assert.numEqual/equalKey/smartEqual(...)` → `expect(...).toNumEqual/
     toEqualKey/toSmartEqual(...)` (matchers already provided by
     `test/unit/vitest-setup.js`, no logic changes needed)
   - `assert.expect( 0 )` (QUnit's "declare zero assertions expected", used
     for dispose()-only smoke tests) → just omit it; a test with no
     `expect()` calls still passes in vitest, and still fails if the call
     under test throws.
   - Drop section comments that just restate the test name right below them
     (`// Instancing` above `test( 'Instancing', ...)`), and drop any
     console-suppression workaround that existed only because Node lacked a
     browser API you're now running against for real.
3. If (and only if) the test genuinely exercises a DOM/canvas/GPU API Node
   doesn't have: name it `X.browser.js` instead.
4. If the original QUnit module has zero tests inside it, add a single
   `test.todo( 'no tests yet' )` so the (still valid, still empty) suite
   doesn't error under vitest.
5. Run `npx vitest run <file>` and diff test counts/names against the QUnit
   original to make sure nothing was silently dropped.
6. `npx eslint <file>` — should be clean with no config changes needed
   (`test/unit/**` is already covered by `npm run lint-test`).
