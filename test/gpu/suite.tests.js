// Discover tests automatically and share imports and a renderer per backend.
import.meta.glob( [
	'../unit/src/nodes/tsl/*.tests.js',
	'!../unit/src/nodes/tsl/TSLCore.tests.js',
	'../unit/src/nodes/lighting/EnvironmentNode.tests.js'
], { eager: true } );
