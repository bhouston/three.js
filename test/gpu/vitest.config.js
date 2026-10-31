import { defineConfig } from 'vitest/config';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath( new URL( '../../', import.meta.url ) );
const alias = {
	'three/tsl': `${ root }src/Three.TSL.js`,
	'three/webgpu': `${ root }src/Three.WebGPU.js`,
	'three': `${ root }src/Three.js`
};

export const gpuProjects = [ 'webgl', 'webgpu' ].map( backend => ( {
	resolve: { alias },
	test: {
		name: backend,
		include: [ 'test/gpu/suite.tests.js' ],
		environment: `${ backend }-node`,
		env: { THREE_GPU_BACKEND: backend },
		maxWorkers: 1,
		testTimeout: 30000,
		hookTimeout: 30000
	}
} ) );

export default defineConfig( { test: { projects: gpuProjects } } );
