import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';
import { playwright } from '@vitest/browser-playwright';

const source = file => fileURLToPath( new URL( `./src/${ file }`, import.meta.url ) );
const browserTests = [
	'test/unit/src/loaders/ImageBitmapLoader.tests.js',
	'test/unit/src/loaders/ImageLoader.tests.js',
	'test/unit/addons/loaders/GLTFLoader.tests.js',
	'test/unit/addons/loaders/USDLoader.tests.js',
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
const gpuProjects = [ 'webgl', 'webgpu' ].map( backend => ( {
	resolve: { alias },
	test: {
		name: backend,
		include: [ 'test/gpu/suite.tests.js' ],
		environment: `${ backend }-node`,
		env: { THREE_GPU_BACKEND: backend },
		maxWorkers: 4,
		testTimeout: 30000,
		hookTimeout: 30000
	}
} ) );

export default defineConfig( {
	test: {
		projects: [
			{
				resolve: { alias },
				test: {
					name: 'unit',
					environment: 'node',
					isolate: false,
					maxWorkers: 4,
					include: [ 'test/unit/src/**/*.tests.js', 'test/unit/tsl/*.tests.js' ],
					exclude: [ ...browserTests, 'test/unit/src/nodes/tsl/!(TSLCore).tests.js', 'test/unit/src/nodes/lighting/EnvironmentNode.tests.js' ],
					setupFiles
				}
			},
			{
				resolve: { alias },
				test: {
					name: 'addons',
					environment: 'node',
					isolate: false,
					maxWorkers: 4,
					include: [ 'test/unit/addons/**/*.tests.js' ],
					exclude: browserTests,
					setupFiles
				}
			},
			{
				resolve: { alias },
				test: {
					name: 'browser',
					maxWorkers: 4,
					include: browserTests,
					setupFiles,
					browser: {
						enabled: true,
						headless: true,
						provider: playwright( { launchOptions: { args: process.platform === 'win32' ? [ '--use-angle=d3d11' ] : process.platform === 'darwin' ? [ '--use-angle=metal' ] : [] } } ),
						instances: [ { browser: 'chromium' } ]
					}
				}
			},
			...gpuProjects
		]
	}
} );
