import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';
import { playwright } from '@vitest/browser-playwright';

const source = file => fileURLToPath( new URL( `./src/${ file }`, import.meta.url ) );
const browserTests = [
	'test/unit/src/loaders/ImageBitmapLoader.tests.js',
	'test/unit/addons/loaders/FBXLoader.tests.js',
	'test/unit/addons/loaders/MaterialXLoader.tests.js',
	'test/unit/addons/postprocessing/GTAOPass.tests.js'
];
const alias = {
	'three/tsl': source( 'Three.TSL.js' ),
	'three/webgpu': source( 'Three.WebGPU.js' ),
	three: source( 'Three.js' )
};
const setupFiles = [ './test/unit/utils/console-wrapper.js' ];

export default defineConfig( {
	test: {
		projects: [
			{
				resolve: { alias },
				test: {
					name: 'unit',
					environment: 'node',
					include: [ 'test/unit/src/**/*.tests.js', 'test/unit/tsl/*.tests.js' ],
					exclude: [ ...browserTests, 'test/unit/src/nodes/tsl/**' ],
					setupFiles
				}
			},
			{
				resolve: { alias },
				test: {
					name: 'addons',
					environment: 'node',
					include: [ 'test/unit/addons/**/*.tests.js' ],
					exclude: browserTests,
					setupFiles
				}
			},
			{
				resolve: { alias },
				test: {
					name: 'browser',
					include: browserTests,
					setupFiles,
					browser: {
						enabled: true,
						headless: true,
						provider: playwright(),
						instances: [ { browser: 'chromium' } ]
					}
				}
			}
		]
	}
} );
