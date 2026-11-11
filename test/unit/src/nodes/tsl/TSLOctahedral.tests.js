import {
	float, vec2, vec3, abs, length, normalize, cos, sin, sqrt,
	octEncode, octDecode, packHalf2x16, unpackHalf2x16
} from 'three/tsl';
import { gpuTest, gpuFuzzTest } from './gpu-test-utils.js';

QUnit.module( 'Nodes', () => {

	QUnit.module( 'TSL', () => {

		QUnit.module( 'octahedral normals', () => {

			gpuTest( 'signed axis encodings and decoded poles', ( { assert } ) => {

				const cases = [
					[[ 0, 0, 1 ], [ 0, 0 ]],
					[[ 1, 0, 0 ], [ 1, 0 ]],
					[[ - 1, 0, 0 ], [ - 1, 0 ]],
					[[ 0, 1, 0 ], [ 0, 1 ]],
					[[ 0, - 1, 0 ], [ 0, - 1 ]],
					[[ 0, 0, - 1 ], [ 1, 1 ]]
				];

				for ( const [ normal, encoded ] of cases ) {

					assert.closeAbs( octEncode( vec3( ...normal ) ), vec2( ...encoded ), 1e-6, `encode ${ normal }` );
					assert.closeAbs( octDecode( vec2( ...encoded ) ), vec3( ...normal ), 1e-6, `decode ${ encoded }` );

				}

				// All four corners represent -Z; zero components must fold with +1,
				// rather than sign(0) collapsing the south pole onto the north pole.
				for ( const x of [ - 1, 1 ] ) {

					for ( const y of [ - 1, 1 ] ) {

						assert.closeAbs( octDecode( vec2( x, y ) ), vec3( 0, 0, - 1 ), 1e-6, `south-pole corner ${ x },${ y }` );

					}

				}

			} );

			gpuTest( 'independent octant values, folds, and equator seams', ( { assert } ) => {

				for ( const x of [ - 1, 1 ] ) {

					for ( const y of [ - 1, 1 ] ) {

						for ( const z of [ - 1, 1 ] ) {

							const encoded = vec2( x * ( z > 0 ? 1 / 3 : 2 / 3 ), y * ( z > 0 ? 1 / 3 : 2 / 3 ) );
							const normal = vec3( x / Math.sqrt( 3 ), y / Math.sqrt( 3 ), z / Math.sqrt( 3 ) );
							assert.closeAbs( octEncode( normal ), encoded, 1e-6, `octant encode ${ x },${ y },${ z }` );
							assert.closeAbs( octDecode( encoded ), normal, 1e-6, `octant decode ${ x },${ y },${ z }` );

						}

					}

				}

				for ( const z of [ - 1e-6, 0, 1e-6 ] ) {

					const normal = normalize( vec3( 3, - 4, z ) );
					assert.closeAbs( octDecode( octEncode( normal ) ), normal, 1e-6, `equator seam z=${ z }` );

				}

				assert.closeAbs( octEncode( vec3( 0, - 3, - 3 ) ), vec2( 0.5, - 1 ), 1e-6, 'lower hemisphere with zero X' );
				assert.closeAbs( octEncode( vec3( - 3, 0, - 3 ) ), vec2( - 1, 0.5 ), 1e-6, 'lower hemisphere with zero Y' );

			} );

			// Deterministic spherical Fibonacci samples cover both hemispheres
			// without a pole singularity. Each assertion is evaluated in generated
			// WGSL and GLSL and checked using GPU storage-buffer readback.
			const sphereNormal = ( index ) => {

				const z = index.toFloat().add( 0.5 ).mul( 2 / 4096 ).oneMinus();
				const radius = sqrt( float( 1 ).sub( z.mul( z ) ) );
				const angle = index.toFloat().mul( 2.399963229728653 );
				return vec3( cos( angle ).mul( radius ), sin( angle ).mul( radius ), z );

			};

			gpuFuzzTest( 'full-sphere round trip and signed bounds', 4096, ( { instanceIndex, assert } ) => {

				const normal = sphereNormal( instanceIndex );
				const encoded = octEncode( normal );
				assert.closeAbs( octDecode( encoded ), normal, 2e-6, 'round trip preserves direction' );
				assert.lessThanOrEqual( abs( encoded ), vec2( 1 ), 'encoding stays inside signed unit square' );

			}, { maxSitesPerInstance: 2 } );

			gpuFuzzTest( 'half-float storage preserves normal accuracy', 4096, ( { instanceIndex, assert } ) => {

				const normal = sphereNormal( instanceIndex );
				const stored = unpackHalf2x16( packHalf2x16( octEncode( normal ) ) );
				const decoded = octDecode( stored );
				assert.closeAbs( decoded, normal, 0.002, 'RG16F quantization error remains small' );
				assert.closeAbs( length( decoded ), float( 1 ), 2e-6, 'decoded normal has unit length' );

			}, { maxSitesPerInstance: 2 } );

		} );

	} );

} );
