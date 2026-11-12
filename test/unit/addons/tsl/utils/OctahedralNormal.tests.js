import { float, vec2, vec3, hash, length, normalize } from 'three/tsl';
import { packNormalToOctahedral, unpackOctahedralToNormal } from '../../../../../examples/jsm/tsl/utils/OctahedralNormal.js';
import { gpuTest, gpuFuzzTest } from '../../../src/nodes/tsl/gpu-test-utils.js';

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'TSL', () => {

		QUnit.module( 'OctahedralNormal', () => {

			gpuTest( 'packNormalToOctahedral known values', ( { assert } ) => {

				assert.closeAbs( packNormalToOctahedral( vec3( 0, 0, 1 ) ), vec2( 0, 0 ), 1e-6, '+Z is the center' );
				assert.closeAbs( packNormalToOctahedral( vec3( 1, 0, 0 ) ), vec2( 1, 0 ), 1e-6, '+X' );
				assert.closeAbs( packNormalToOctahedral( vec3( - 1, 0, 0 ) ), vec2( - 1, 0 ), 1e-6, '-X' );
				assert.closeAbs( packNormalToOctahedral( vec3( 0, 1, 0 ) ), vec2( 0, 1 ), 1e-6, '+Y' );
				assert.closeAbs( packNormalToOctahedral( vec3( 0, - 1, 0 ) ), vec2( 0, - 1 ), 1e-6, '-Y' );

				// x == y == 0 on the lower hemisphere: sign( 0 ) must count as positive, otherwise the pole collapses to ( 0, 0 ) == +Z.
				assert.closeAbs( packNormalToOctahedral( vec3( 0, 0, - 1 ) ), vec2( 1, 1 ), 1e-6, '-Z is a corner, not the center' );

				// lower hemisphere fold with one zero component: p = ( 0, 3/7 ) folds to ( 1 - 3/7, 1 - 0 )
				assert.closeAbs( packNormalToOctahedral( vec3( 0, 0.6, - 0.8 ) ), vec2( 4 / 7, 1 ), 1e-6, 'x == 0, z < 0 keeps a positive x sign' );
				assert.closeAbs( packNormalToOctahedral( vec3( 0, - 0.6, - 0.8 ) ), vec2( 4 / 7, - 1 ), 1e-6, 'x == 0, y < 0, z < 0' );
				assert.closeAbs( packNormalToOctahedral( vec3( - 0.6, 0, - 0.8 ) ), vec2( - 1, 4 / 7 ), 1e-6, 'y == 0, x < 0, z < 0' );

				// the input does not need to be normalized
				assert.closeAbs( packNormalToOctahedral( vec3( 0, 0, 5 ) ), vec2( 0, 0 ), 1e-6, 'unnormalized +Z' );
				assert.closeAbs( packNormalToOctahedral( vec3( 3, - 4, 0 ) ), vec2( 3 / 7, - 4 / 7 ), 1e-6, 'unnormalized equator' );
				assert.closeAbs( packNormalToOctahedral( vec3( 0.001, 0.002, - 0.002 ) ), packNormalToOctahedral( vec3( 1, 2, - 2 ) ), 1e-5, 'scale invariant' );

			} );

			gpuTest( 'unpackOctahedralToNormal known values', ( { assert } ) => {

				assert.closeAbs( unpackOctahedralToNormal( vec2( 0, 0 ) ), vec3( 0, 0, 1 ), 1e-6, 'center is +Z' );
				assert.closeAbs( unpackOctahedralToNormal( vec2( 1, 0 ) ), vec3( 1, 0, 0 ), 1e-6, '+X' );
				assert.closeAbs( unpackOctahedralToNormal( vec2( 0, - 1 ) ), vec3( 0, - 1, 0 ), 1e-6, '-Y' );

				// all four corners are the same -Z pole
				assert.closeAbs( unpackOctahedralToNormal( vec2( 1, 1 ) ), vec3( 0, 0, - 1 ), 1e-6, 'corner ( 1, 1 )' );
				assert.closeAbs( unpackOctahedralToNormal( vec2( - 1, 1 ) ), vec3( 0, 0, - 1 ), 1e-6, 'corner ( -1, 1 )' );
				assert.closeAbs( unpackOctahedralToNormal( vec2( 1, - 1 ) ), vec3( 0, 0, - 1 ), 1e-6, 'corner ( 1, -1 )' );
				assert.closeAbs( unpackOctahedralToNormal( vec2( - 1, - 1 ) ), vec3( 0, 0, - 1 ), 1e-6, 'corner ( -1, -1 )' );

				// the diamond |x| + |y| == 1 is the equator
				assert.closeAbs( unpackOctahedralToNormal( vec2( 3 / 7, - 4 / 7 ) ), vec3( 0.6, - 0.8, 0 ), 1e-6, 'equator' );

				// outer edge of the square, lower hemisphere
				assert.closeAbs( unpackOctahedralToNormal( vec2( 4 / 7, 1 ) ), vec3( 0, 0.6, - 0.8 ), 1e-6, 'edge x >= 0' );
				assert.closeAbs( unpackOctahedralToNormal( vec2( - 1, 4 / 7 ) ), vec3( - 0.6, 0, - 0.8 ), 1e-6, 'edge x < 0' );

				assert.closeAbs( length( unpackOctahedralToNormal( vec2( 0.3, - 0.9 ) ) ), float( 1 ), 1e-6, 'result is unit length' );

			} );

			gpuTest( 'round trip at poles, axes and fold edges', ( { assert } ) => {

				const roundTrip = ( x, y, z, message ) => {

					const n = normalize( vec3( x, y, z ) );
					assert.closeAbs( unpackOctahedralToNormal( packNormalToOctahedral( n ) ), n, 1e-6, message );

				};

				roundTrip( 0, 0, 1, '+Z' );
				roundTrip( 0, 0, - 1, '-Z' );
				roundTrip( 1, 0, 0, '+X' );
				roundTrip( - 1, 0, 0, '-X' );
				roundTrip( 0, 1, 0, '+Y' );
				roundTrip( 0, - 1, 0, '-Y' );
				roundTrip( 1, - 1, 0, 'equator diagonal' );
				roundTrip( 0, 1, - 1, 'x == 0, z < 0' );
				roundTrip( - 1, 0, - 1, 'y == 0, z < 0' );
				roundTrip( 1, 1, - 1e-4, 'just below the equator' );
				roundTrip( 1e-4, - 1e-4, - 1, 'next to the -Z pole' );
				roundTrip( - 1, - 1, - 1, 'lower octant diagonal' );

			} );

			gpuFuzzTest( 'round trip recovers unnormalized directions', 1024, ( { instanceIndex, assert } ) => {

				const seed = instanceIndex.toFloat();
				const direction = vec3( hash( seed.add( 1 ) ), hash( seed.add( 5000 ) ), hash( seed.add( 9000 ) ) ).sub( 0.5 );
				const n = unpackOctahedralToNormal( packNormalToOctahedral( direction.mul( 10 ) ) );

				assert.closeAbs( n, normalize( direction ), 1e-5, 'decoded direction matches the normalized input' );
				assert.closeAbs( length( n ), float( 1 ), 1e-5, 'decoded direction is unit length' );

			} );

		} );

	} );

} );
