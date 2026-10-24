import { vec3, vec4 } from 'three/tsl';
import { demodulate, remodulate } from '../../../../examples/jsm/tsl/display/RecurrentDenoiseNode.js';
import { gpuTest } from '../../src/nodes/tsl/gpu-test-utils.js';

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'TSL', () => {

		QUnit.module( 'RecurrentDenoise', () => {

			gpuTest( 'demodulation removes modulation and preserves metadata', ( { assert } ) => {

				const signal = vec4( 2, 6, 20, 37 );
				const modulation = vec3( 0.25, 0.5, 2 );
				assert.closeAbs( demodulate( signal, modulation ), vec4( 8, 12, 10, 37 ), 1e-5, 'RGB is divided componentwise; alpha remains the hit distance' );
				assert.closeAbs( remodulate( vec4( 8, 12, 10, 37 ), modulation ), signal, 1e-5, 'RGB is multiplied componentwise without clamping HDR values or alpha' );

			} );

			gpuTest( 'black and near-black modulation remains finite and reversible', ( { assert } ) => {

				const modulation = vec3( 0, 1e-8, 0.5 );
				const signal = vec4( 0.125, 0, 16, 0 );
				const demodulated = demodulate( signal, modulation );
				assert.closeAbs( demodulated, vec4( 1250, 0, 32, 0 ), 1e-3, 'black channels use the epsilon floor, including zero divided by black' );
				assert.closeAbs( remodulate( vec4( 1, 2, 3, - 7 ), modulation ), vec4( 1e-4, 2e-4, 1.5, - 7 ), 1e-7, 'remodulation uses the same floor and preserves signed metadata' );
				assert.closeAbs( remodulate( demodulated, modulation ), signal, 1e-5, 'HDR signal survives the round trip even for black modulation' );

			} );

			gpuTest( 'filtering demodulated lighting preserves colored surface detail', ( { assert } ) => {

				// Two adjacent texels share lighting but have strongly different albedo.
				// A two-tap box filter should average lighting without blending albedo.
				const red = vec3( 1, 0.125, 0.25 );
				const blue = vec3( 0.125, 0.25, 1 );
				const redSignal = vec4( 4, 0.25, 0.25, 3 );
				const blueSignal = vec4( 0.5, 0.5, 1, 3 );
				const filtered = demodulate( redSignal, red ).add( demodulate( blueSignal, blue ) ).mul( 0.5 );
				assert.closeAbs( filtered, vec4( 4, 2, 1, 3 ), 1e-6, 'filter operates on shared illumination' );
				assert.closeAbs( remodulate( filtered, red ), redSignal, 1e-6, 'red texel retains its original detail after filtering' );
				assert.closeAbs( remodulate( filtered, blue ), blueSignal, 1e-6, 'blue texel retains its original detail after filtering' );

			} );

		} );

	} );

} );
