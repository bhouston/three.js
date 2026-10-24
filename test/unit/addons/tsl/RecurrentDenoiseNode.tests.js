import { vec3, vec4 } from 'three/tsl';
import { demodulate, remodulate } from '../../../../examples/jsm/tsl/display/RecurrentDenoiseNode.js';
import { gpuTest } from '../../src/nodes/tsl/gpu-test-utils.js';

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'TSL', () => {

		QUnit.module( 'RecurrentDenoiseNode', () => {

			gpuTest( 'demodulation preserves HDR lighting and ray metadata', ( { assert } ) => {

				const signal = vec4( 2, 6, 20, 37 );
				const modulation = vec3( 0.25, 0.5, 2 );
				const lighting = demodulate( signal, modulation );
				assert.closeAbs( lighting, vec4( 8, 12, 10, 37 ), 1e-5, 'only RGB is divided by the material response' );
				assert.closeAbs( remodulate( lighting, modulation ), signal, 1e-5, 'lighting resolve restores the HDR signal' );

			} );

			gpuTest( 'black modulation stays finite and reversible', ( { assert } ) => {

				const modulation = vec3( 0, 1e-8, 0.5 );
				const signal = vec4( 0.125, 0, 16, 0 );
				const lighting = demodulate( signal, modulation );
				assert.closeAbs( lighting, vec4( 1250, 0, 32, 0 ), 1e-3, 'near-black BRDF channels use the floor' );
				assert.closeAbs( remodulate( lighting, modulation ), signal, 1e-5, 'the same floor preserves the round trip' );

			} );

			gpuTest( 'filtering lighting preserves surface detail', ( { assert } ) => {

				const red = vec3( 1, 0.125, 0.25 );
				const blue = vec3( 0.125, 0.25, 1 );
				const redSignal = vec4( 4, 0.25, 0.25, 3 );
				const blueSignal = vec4( 0.5, 0.5, 1, 3 );
				const filtered = demodulate( redSignal, red ).add( demodulate( blueSignal, blue ) ).mul( 0.5 );
				assert.closeAbs( filtered, vec4( 4, 2, 1, 3 ), 1e-6, 'filter averages illumination' );
				assert.closeAbs( remodulate( filtered, red ), redSignal, 1e-6, 'red surface detail survives filtering' );
				assert.closeAbs( remodulate( filtered, blue ), blueSignal, 1e-6, 'blue surface detail survives filtering' );

			} );

		} );

	} );

} );
