import { OrthographicCamera, PerspectiveCamera, WebGPUCoordinateSystem } from 'three/webgpu';
import { Fn, float, getViewPosition, instanceIndex, instancedArray, uniform, vec2, vec4 } from 'three/tsl';
import { getSSRPixelWidth, validateSSRRayDistance, validateSSRViewDistance, validateSSRDepthSlab } from '../../../../examples/jsm/tsl/utils/SSRValidation.js';
import { rawComputeTest } from '../../src/nodes/tsl/gpu-raw-test-utils.js';
import { randomRays } from './ssr-trace-test-utils.js';

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'TSL', () => {

		QUnit.module( 'SSRValidation', () => {

			rawComputeTest( 'three geometric validators match independent CPU predicates', {}, async ( { assert, renderer } ) => {

				const rays = randomRays( 1024 );
				const samples = randomRays( 1024, 54321 );
				const origins = instancedArray( new Float32Array( rays.flatMap( ( ray, i ) => [ ...ray[ 0 ], [ 0.05, 0.1, 0.5, 1 ][ i % 4 ] ] ) ), 'vec4' );
				const ends = instancedArray( new Float32Array( rays.flatMap( ray => [ ...ray[ 1 ], 0 ] ) ), 'vec4' );
				const points = instancedArray( new Float32Array( samples.flatMap( ray => [ ...ray[ 0 ], 0 ] ) ), 'vec4' );
				const output = instancedArray( rays.length, 'vec4' );
				const kernel = Fn( () => {

					const a = origins.element( instanceIndex );
					const b = ends.element( instanceIndex );
					const p = points.element( instanceIndex );
					output.element( instanceIndex ).assign( vec4(
						float( validateSSRRayDistance( p.xyz, a.xyz, b.xyz.sub( a.xyz ), a.w ) ),
						float( validateSSRViewDistance( p.xyz, b.xyz, a.w ) ),
						float( validateSSRDepthSlab( a.z.mul( - 10 ), b.z.mul( - 10 ), float( 0.1 ) ) ), 0
					) );

				} )().compute( rays.length );
				try {

					await renderer.computeAsync( kernel );
					const actual = new Float32Array( await renderer.getArrayBufferAsync( output.value ) );
					const mismatches = [];
					for ( let i = 0; i < rays.length; i ++ ) {

						const a = Array.from( origins.value.array.slice( i * 4, i * 4 + 4 ) );
						const b = Array.from( ends.value.array.slice( i * 4, i * 4 + 3 ) );
						const p = Array.from( points.value.array.slice( i * 4, i * 4 + 3 ) );
						// Original point-to-line formula, with two point offsets and norms.
						const u = p.map( ( v, c ) => v - a[ c ] ), v = p.map( ( v, c ) => v - b[ c ] );
						const cross = [ u[ 1 ] * v[ 2 ] - u[ 2 ] * v[ 1 ], u[ 2 ] * v[ 0 ] - u[ 0 ] * v[ 2 ], u[ 0 ] * v[ 1 ] - u[ 1 ] * v[ 0 ] ];
						const distanceToLine = Math.hypot( ...cross ) / Math.hypot( ...b.map( ( v, c ) => v - a[ c ] ) );
						const separation = Math.hypot( ...p.map( ( v, c ) => v - b[ c ] ) );
						const rayZ = - a[ 2 ] * 10, surfaceZ = - b[ 2 ] * 10;
						const tolerance = Math.max( Math.abs( surfaceZ ), 1 ) * 1e-6;
						const expected = [ distanceToLine <= a[ 3 ], separation <= a[ 3 ], rayZ <= surfaceZ + tolerance && rayZ >= surfaceZ * 1.1 - tolerance ];
						for ( let c = 0; c < 3; c ++ ) {

							if ( actual[ i * 4 + c ] !== Number( expected[ c ] ) && mismatches.length < 8 ) mismatches.push( { i, validator: c, expected: expected[ c ], actual: actual[ i * 4 + c ] } );

						}

					}

					assert.deepEqual( mismatches, [], '3072 candidate decisions match independent CPU geometry' );

				} finally {

					kernel.dispose();

				}

			} );

			for ( const kind of [ 'narrow', 'wide', 'offset', 'orthographic' ] ) {

				rawComputeTest( `analytic pixel width matches inverse-projected neighbors: ${ kind }`, {}, async ( { assert, renderer } ) => {

					const camera = kind === 'orthographic' ? new OrthographicCamera( - 3, 7, 2, - 2, 0.1, 1000 ) : new PerspectiveCamera( kind === 'narrow' ? 20 : 100, 1.6, 0.1, 1000 );
					camera.coordinateSystem = WebGPUCoordinateSystem;
					if ( kind === 'offset' ) camera.setViewOffset( 1920, 1080, 13, 17, 1280, 720 );
					camera.updateProjectionMatrix();
					const inverse = uniform( camera.projectionMatrixInverse );
					const output = instancedArray( 65, 'vec4' );
					const kernel = Fn( () => {

						const uv = vec2( instanceIndex.mod( 8 ), instanceIndex.div( 8 ) ).add( 0.5 ).div( 9 );
						const depth = float( instanceIndex ).add( 1 ).div( 65 );
						const point = getViewPosition( uv, depth, inverse );
						const neighbor = getViewPosition( uv.add( vec2( 1 / 1280, 0 ) ), depth, inverse );
						const analytic = getSSRPixelWidth( inverse, point.z, float( 1 / 1280 ), camera.isPerspectiveCamera );
						output.element( instanceIndex ).assign( vec4( analytic, neighbor.x.sub( point.x ), 0, 0 ) );

					} )().compute( 65 );
					try {

						await renderer.computeAsync( kernel );
						const data = new Float32Array( await renderer.getArrayBufferAsync( output.value ) );
						let error = 0;
						for ( let i = 0; i < 65; i ++ ) error = Math.max( error, Math.abs( data[ i * 4 ] - data[ i * 4 + 1 ] ) / Math.max( Math.abs( data[ i * 4 ] ), 1e-10 ) );
						assert.ok( Number.isFinite( error ) && error < 0.0005, `maximum relative width error ${ error }` );

					} finally {

						kernel.dispose();

					}

				} );

			}

		} );

	} );

} );
