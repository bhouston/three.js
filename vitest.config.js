import { defineConfig } from 'vitest/config';
import { playwright } from '@vitest/browser-playwright';

// Vitest test suite for the neural-texture / neural-material / neural-appearance
// framework (examples/jsm/neural*). Deliberately separate from the repo's
// existing QUnit + puppeteer harness (test/unit/**) - see test/vitest/README.md
// for why, and for the conventions each project below expects tests to follow.
//
// Two projects:
//  - "unit": plain Node, for pure-JS logic (math, MLP forward pass, format/
//    manifest encode-decode, byte-level (de)serialization) that never touches
//    a GPU or a TSL node graph. Fast, no browser needed.
//  - "browser": real Chromium + real WebGPU (via the playwright provider),
//    for anything that builds or evaluates a TSL node graph, drives a
//    WebGPURenderer, or runs a compute kernel. These are the pieces the old
//    QUnit suite could only ever mock around - here they actually execute on
//    the GPU and get read back, so a broken kernel fails a test instead of
//    silently shipping.
const webgpuLaunchArgs = [
	// Playwright's bundled Chromium ships WebGPU behind these flags; without
	// them navigator.gpu.requestAdapter() resolves to null on every platform.
	'--enable-unsafe-webgpu',
	'--enable-features=Vulkan,Metal',
	'--use-angle=metal' // no-op (ignored) on non-macOS hosts; picks the Metal ANGLE backend on macOS runners.
];

export default defineConfig( {
	test: {
		projects: [
			{
				test: {
					name: 'unit',
					environment: 'node',
					include: [ 'test/vitest/unit/**/*.test.js' ]
				}
			},
			{
				resolve: {
					// examples/jsm/**/*.js consistently import the bare 'three'
					// specifier (matching the repo's own examples/*.html importmaps,
					// which all point "three" at build/three.webgpu.js) even for
					// WebGPU-only classes like NodeMaterial - see e.g.
					// NeuralTextureSource.js / NeuralTextureNodeMaterial.js /
					// NeuralMaterialSource.js. Outside an importmap-driven page,
					// though, 'three' resolves via package.json's own "exports"
					// field to build/three.module.js (the WebGPU-less core build),
					// so any addon module reached from a test - even transitively,
					// just by importing it - throws "Class extends value undefined"
					// or "X is not a constructor" at *import* time, before any test
					// in the file even runs. Re-point bare 'three' at the same
					// build/three.webgpu.js the real examples use, for this project
					// only (the 'unit' project's pure-JS-only files don't need this,
					// and aliasing it there too would be a needless behavior change
					// for tests that never touch three/webgpu at all).
					alias: [
						{ find: /^three$/, replacement: new URL( './build/three.webgpu.js', import.meta.url ).pathname }
					]
				},
				test: {
					name: 'browser',
					include: [ 'test/vitest/browser/**/*.test.js' ],
					browser: {
						enabled: true,
						provider: playwright( { launchOptions: { args: webgpuLaunchArgs } } ),
						headless: true,
						instances: [ { browser: 'chromium' } ]
					}
				}
			}
		]
	}
} );
