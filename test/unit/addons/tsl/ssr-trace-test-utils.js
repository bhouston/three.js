import { DataTexture, FloatType, NearestFilter, RedFormat } from 'three/webgpu';
import { Fn, float, instanceIndex, instancedArray, texture, vec4 } from 'three/tsl';
import { hiZ } from '../../../../examples/jsm/tsl/display/HiZNode.js';
import { traceHiZ } from '../../../../examples/jsm/tsl/utils/SSRTrace.js';

export function createTraceFixture( width, height, data, reversed = false ) {

	const depth = new DataTexture( new Float32Array( data ), width, height, RedFormat, FloatType );
	depth.minFilter = depth.magFilter = NearestFilter;
	depth.needsUpdate = true;
	const hierarchy = hiZ( texture( depth ), { reversedDepth: reversed } );
	return { width, height, data: depth.image.data, depth, hierarchy, reversed, dispose() {

		hierarchy.dispose(); depth.dispose();

	} };

}

// Exhaustive CPU oracle: intersect every original texel rectangle independently,
// then choose the earliest valid intersection. No hierarchy or marching schedule.
export function referenceTrace( fixture, origin, end, { thickness = 0.000001, rejectedDepth = - 1 } = {} ) {

	const { width, height, data, reversed } = fixture;
	const start = origin.map( Math.fround );
	const finish = end.map( Math.fround );
	const direction = finish.map( ( value, i ) => value - start[ i ] );
	const z0 = reversed ? 1 - start[ 2 ] : start[ 2 ];
	const dz = reversed ? - direction[ 2 ] : direction[ 2 ];
	let first = Infinity;

	for ( let y = 0; y < height; y ++ ) {

		for ( let x = 0; x < width; x ++ ) {

			const raw = data[ y * width + x ];
			const depth = reversed ? 1 - raw : raw;
			if ( depth === 1 || raw === rejectedDepth ) continue;
			let entry = 0, exit = 1;
			for ( let axis = 0; axis < 2; axis ++ ) {

				const size = axis === 0 ? width : height;
				const index = axis === 0 ? x : y;
				const a = index / size, b = ( index + 1 ) / size;
				if ( direction[ axis ] === 0 ) {

					if ( start[ axis ] < a || start[ axis ] >= b ) {

						exit = - 1; break;

					}

				} else {

					const t0 = ( a - start[ axis ] ) / direction[ axis ];
					const t1 = ( b - start[ axis ] ) / direction[ axis ];
					entry = Math.max( entry, Math.min( t0, t1 ) );
					exit = Math.min( exit, Math.max( t0, t1 ) );

				}

			}

			if ( entry >= exit || entry >= first ) continue;
			const entryDepth = z0 + dz * entry;
			if ( entryDepth >= depth && entryDepth <= depth + thickness ) {

				first = entry;

			} else if ( dz !== 0 ) {

				const plane = ( depth - z0 ) / dz;
				if ( plane >= entry && plane < exit && plane < first ) first = plane;

			}

		}

	}

	return Number.isFinite( first ) ? { hit: true, parameter: first } : { hit: false, parameter: 0 };

}

export function buildTraceKernel( fixture, rays, { maxSteps = 2048, thickness = 0.000001, rejectedDepth = - 1, maxCandidates = 4096, minLevel = 0, leafSample = false, refineSteps = 0 } = {} ) {

	// vec4 inputs avoid storage-buffer vec3 alignment differences.
	const origins = instancedArray( new Float32Array( rays.flatMap( ray => [ ...ray[ 0 ], 0 ] ) ), 'vec4' );
	const ends = instancedArray( new Float32Array( rays.flatMap( ray => [ ...ray[ 1 ], 0 ] ) ), 'vec4' );
	const output = instancedArray( rays.length, 'vec4' );
	const metadata = instancedArray( rays.length, 'vec4' );
	const kernel = Fn( () => {

		const origin = origins.element( instanceIndex ).xyz.toConst();
		const end = ends.element( instanceIndex ).xyz.toConst();
		const result = traceHiZ( fixture.hierarchy, origin, end, {
			reversedDepth: fixture.reversed, maxSteps, maxCandidates, minLevel, leafSample, refineSteps,
			acceptHit: ( uv, s, raw ) => {

				const z = origin.z.add( end.z.sub( origin.z ).mul( s ) );
				return z.sub( raw ).abs().lessThanEqual( thickness ).and( raw.notEqual( rejectedDepth ) );

			}
		} );
		output.element( instanceIndex ).assign( vec4( result.uv, result.parameter, float( result.hit ) ) );
		metadata.element( instanceIndex ).assign( vec4( float( result.iterations ), float( result.exhausted ), result.depth, float( result.candidates ) ) );

	} )().compute( rays.length );
	return { kernel, output, metadata, origins, ends };

}

export async function traceRays( renderer, fixture, rays, options = {} ) {

	fixture.hierarchy.updateBefore( { renderer } );
	const { kernel, output, metadata } = buildTraceKernel( fixture, rays, options );
	try {

		await renderer.computeAsync( kernel );
		return {
			output: new Float32Array( await renderer.getArrayBufferAsync( output.value ) ),
			metadata: new Float32Array( await renderer.getArrayBufferAsync( metadata.value ) )
		};

	} finally {

		kernel.dispose();

	}

}

export function randomRays( count, seed = 12345 ) {

	const random = () => {

		seed ^= seed << 13;
		seed ^= seed >>> 17;
		seed ^= seed << 5;
		return ( seed >>> 0 ) / 4294967296;

	};

	return Array.from( { length: count }, () => [[ 0.001 + random() * 0.998, 0.001 + random() * 0.998, 0.05 + random() * 0.9 ], [ 0.001 + random() * 0.998, 0.001 + random() * 0.998, 0.05 + random() * 0.9 ]] );

}
