import { defineConfig } from 'vitest/config';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath( new URL( '../../', import.meta.url ) );
const gpuTests = [ 'test/unit/src/nodes/tsl/*.tests.js', 'test/gpu/*.tests.js', 'test/unit/src/nodes/lighting/EnvironmentNode.tests.js' ];

export default defineConfig( {
	resolve: {
		alias: {
			'three/tsl': `${ root }src/Three.TSL.js`,
			'three/webgpu': `${ root }src/Three.WebGPU.js`,
			'three': `${ root }src/Three.js`
		}
	},
	test: {
		projects: [ 'webgl', 'webgpu' ].map( backend => ( {
			extends: true,
			test: {
				name: backend,
				include: gpuTests,
				exclude: [ '**/TSLCore.tests.js' ],
				environment: `${ backend }-node`,
				env: { THREE_GPU_BACKEND: backend },
				maxWorkers: 1,
				testTimeout: 30000,
				hookTimeout: 30000
			}
		} ) )
	}
} );

