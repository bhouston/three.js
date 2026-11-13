import { Fn, bool, cross, distance, float, instanceIndex, instancedArray, int, max, texture, vec2, vec3, vec4 } from 'three/tsl';
import { traceHiZ } from '../../../../examples/jsm/tsl/utils/SSRTrace.js';
import { traceLinearSSR } from '../../../../examples/jsm/tsl/utils/SSRLinearTrace.js';
import { rawComputeTest } from '../../src/nodes/tsl/gpu-raw-test-utils.js';
import { createTraceFixture, randomRays, referenceTrace, traceRays } from './ssr-trace-test-utils.js';

// This deliberately retains the production one-texel stride floor. A very
// large maxSteps value cannot supersample that schedule. Its discrepancies
// against the exact original-texel oracle are reported, not hidden by changing
// the adaptive test's source buffers, endpoints, or acceptance tolerance.
async function traceLinearRays( renderer, fixture, rays, { stochastic = true, maxSteps = 4096, thickness = 0.000001, rejectedDepth = - 1 } = {} ) {

	const origins = instancedArray( new Float32Array( rays.flatMap( ray => [ ...ray[ 0 ], 0 ] ) ), 'vec4' );
	const ends = instancedArray( new Float32Array( rays.flatMap( ray => [ ...ray[ 1 ], 0 ] ) ), 'vec4' );
	const output = instancedArray( rays.length, 'vec4' );
	const sampleDepth = texture( fixture.depth );
	const kernel = Fn( () => {

		const origin = origins.element( instanceIndex ).xyz.toConst();
		const end = ends.element( instanceIndex ).xyz.toConst();
		const delta = end.sub( origin ).toConst();
		const pixels = delta.xy.abs().mul( vec2( fixture.width, fixture.height ) );
		const rayLen = max( pixels.x, pixels.y ).max( 1 ).toConst();
		const totalStep = stochastic ? int( maxSteps ) : int( rayLen.floor() ).max( 1 );
		const uvAt = s => origin.xy.add( delta.xy.mul( s ) );
		const rayDepthAt = s => origin.z.add( delta.z.mul( s ) );
		const result = traceLinearSSR( {
			totalStep,
			sampleFraction: stochastic ? index => max( index.div( totalStep ).pow( 2 ), index.div( rayLen ) ) : index => index.div( totalStep ),
			uvAt,
			sampleDepth: uv => sampleDepth.sample( uv ).r,
			projectDepth: raw => raw,
			sceneZ: raw => fixture.reversed ? raw : raw.negate(),
			rayZAt: s => fixture.reversed ? rayDepthAt( s ) : rayDepthAt( s ).negate(),
			isBackground: raw => fixture.reversed ? raw.lessThanEqual( 0 ) : raw.greaterThanEqual( 1 ),
			binaryRefine: true,
			frontFacing: () => bool( true ),
			acceptHit: ( uv, s, lastS, raw ) => rayDepthAt( s ).sub( raw ).abs().lessThanEqual( max( thickness, delta.z.mul( s.sub( lastS ) ).abs() ) ).and( raw.notEqual( rejectedDepth ) )
		} );
		// This is the fixture's common strict final acceptance rule. The actual
		// SSRNode separately retains its view-space thickness/backface validation.
		const valid = result.hit.and( rayDepthAt( result.parameter ).sub( result.depth ).abs().lessThanEqual( thickness ) );
		output.element( instanceIndex ).assign( vec4( result.uv, result.parameter, float( valid ) ) );

	} )().compute( rays.length );
	try {

		await renderer.computeAsync( kernel );
		return new Float32Array( await renderer.getArrayBufferAsync( output.value ) );

	} finally {

		kernel.dispose();

	}

}

// Orthographic view-space adapter using SSRNode's actual final distance metric
// and its three-source-pixel thickness floor. Both algorithms see the same
// source depth, normals (all front-facing), rays and self-hit exclusion. This
// separates sampling differences from the much stricter surface-oracle test.
async function traceGeometricRays( renderer, fixture, rays ) {

	fixture.hierarchy.updateBefore( { renderer } );
	const origins = instancedArray( new Float32Array( rays.flatMap( ray => [ ...ray[ 0 ], 0 ] ) ), 'vec4' );
	const ends = instancedArray( new Float32Array( rays.flatMap( ray => [ ...ray[ 1 ], 0 ] ) ), 'vec4' );
	const linearOutput = instancedArray( rays.length, 'vec4' );
	const adaptiveOutput = instancedArray( rays.length, 'vec4' );
	const depthTexture = texture( fixture.depth );
	const kernel = Fn( () => {

		const origin = origins.element( instanceIndex ).xyz.toConst();
		const end = ends.element( instanceIndex ).xyz.toConst();
		const delta = end.sub( origin ).toConst();
		const pixels = delta.xy.abs().mul( vec2( fixture.width, fixture.height ) );
		const rayLen = max( pixels.x, pixels.y ).max( 1 ).toConst();
		const sceneZ = raw => fixture.reversed ? raw.mul( 10 ).sub( 10.1 ) : raw.mul( - 10 ).sub( 0.1 );
		const point = ( uv, depth ) => vec3( uv.x.mul( 2 ).sub( 1 ), uv.y.mul( - 2 ).add( 1 ), sceneZ( depth ) );
		const a = point( origin.xy, origin.z ).toConst();
		const b = point( end.xy, end.z ).toConst();
		const lineLength = distance( a, b ).max( 1e-20 ).toConst();
		const uvAt = s => origin.xy.add( delta.xy.mul( s ) );
		const rayZAt = s => sceneZ( origin.z.add( delta.z.mul( s ) ) );
		const away = ( uv, depth ) => {

			const p = point( uv, depth );
			return cross( p.sub( a ), p.sub( b ) ).length().div( lineLength );

		};

		const thickness = float( 6 / fixture.width );
		const linear = traceLinearSSR( {
			totalStep: int( 4096 ),
			sampleFraction: index => max( index.div( 4096 ).pow( 2 ), index.div( rayLen ) ),
			uvAt,
			sampleDepth: uv => depthTexture.sample( uv ).r,
			projectDepth: raw => raw,
			sceneZ,
			rayZAt,
			isBackground: raw => fixture.reversed ? raw.lessThanEqual( 0 ) : raw.greaterThanEqual( 1 ),
			binaryRefine: true,
			frontFacing: () => bool( true ),
			acceptHit: ( uv, s, lastS, raw ) => away( uv, raw ).lessThanEqual( max( thickness, lineLength.mul( s.sub( lastS ) ) ) )
		} );
		const linearValid = linear.hit.and( away( linear.uv, linear.depth ).lessThanEqual( thickness ) );
		const adaptive = traceHiZ( fixture.hierarchy, origin, end, {
			reversedDepth: fixture.reversed,
			start: rayLen.reciprocal(),
			maxSteps: 2048,
			acceptHit: ( uv, s, raw ) => away( uv, raw ).lessThanEqual( thickness )
		} );
		linearOutput.element( instanceIndex ).assign( vec4( linear.uv, linear.parameter, float( linearValid ) ) );
		adaptiveOutput.element( instanceIndex ).assign( vec4( adaptive.uv, adaptive.parameter, float( adaptive.hit ) ) );

	} )().compute( rays.length );
	try {

		await renderer.computeAsync( kernel );
		return {
			linear: new Float32Array( await renderer.getArrayBufferAsync( linearOutput.value ) ),
			adaptive: new Float32Array( await renderer.getArrayBufferAsync( adaptiveOutput.value ) )
		};

	} finally {

		kernel.dispose();

	}

}

function differentialTest( name, run ) {

	rawComputeTest( name, {}, async ( { assert, renderer } ) => {

		renderer.backend.device.pushErrorScope( 'validation' );
		try {

			await run( { assert, renderer } );

		} finally {

			const error = await renderer.backend.device.popErrorScope();
			assert.strictEqual( error, null, error ? error.message : 'no WebGPU validation errors' );

		}

	} );

}

function reportComparison( name, fixture, rays, linear, adaptive, options = {} ) {

	const summary = { name, rays: rays.length, oracleHits: 0, linearHits: 0, adaptiveHits: 0, linearOracleDisagreements: 0, adaptiveOracleDisagreements: 0, linearAdaptiveDisagreements: 0, matchingHitParameters: 0, nonFinite: 0, firstDifferences: [] };
	for ( let i = 0; i < rays.length; i ++ ) {

		const expected = referenceTrace( fixture, ...rays[ i ], options );
		const a = linear[ i * 4 + 3 ] === 1;
		const b = adaptive[ i * 4 + 3 ] === 1;
		if ( expected.hit ) summary.oracleHits ++;
		if ( a ) summary.linearHits ++;
		if ( b ) summary.adaptiveHits ++;
		if ( a !== expected.hit ) summary.linearOracleDisagreements ++;
		if ( b !== expected.hit ) summary.adaptiveOracleDisagreements ++;
		if ( a !== b ) summary.linearAdaptiveDisagreements ++;
		if ( a && b && Math.abs( linear[ i * 4 + 2 ] - adaptive[ i * 4 + 2 ] ) <= 0.0001 ) summary.matchingHitParameters ++;
		if ( a !== b && summary.firstDifferences.length < 3 ) summary.firstDifferences.push( { origin: rays[ i ][ 0 ], end: rays[ i ][ 1 ], oracle: expected, linear: Array.from( linear.slice( i * 4, i * 4 + 4 ) ), adaptive: Array.from( adaptive.slice( i * 4, i * 4 + 4 ) ) } );
		for ( let c = 0; c < 4; c ++ ) if ( ! Number.isFinite( linear[ i * 4 + c ] ) || ! Number.isFinite( adaptive[ i * 4 + c ] ) ) summary.nonFinite ++;

	}

	globalThis._SSRLinearComparisons ??= [];
	globalThis._SSRLinearComparisons.push( summary );
	console.warn( `SSR linear comparison: ${ JSON.stringify( summary ) }` );
	return summary;

}

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'TSL', () => {

		QUnit.module( 'SSRLinearTrace', () => {

			for ( const reversed of [ false, true ] ) {

				for ( const [ width, height ] of [[ 1, 128 ], [ 128, 1 ], [ 1, 127 ], [ 127, 1 ]] ) {

					differentialTest( `same strip rays at ${ width }x${ height }, reversed=${ reversed }`, async ( { assert, renderer } ) => {

						const data = new Float32Array( width * height ).fill( reversed ? 0 : 1 );
						data[ 31 ] = data[ 95 ] = 0.5;
						const fixture = createTraceFixture( width, height, data, reversed );
						const point = ( p, z ) => width === 1 ? [ 0.5, p, z ] : [ p, 0.5, z ];
						const rays = [
							[ point( 0.001, 0.5 ), point( 0.999, 0.5 ) ],
							[ point( 0.999, 0.5 ), point( 0.001, 0.5 ) ],
							[ point( 0.001, reversed ? 0.95 : 0.05 ), point( 0.999, reversed ? 0.05 : 0.95 ) ],
							[ point( 0.999, reversed ? 0.05 : 0.95 ), point( 0.001, reversed ? 0.95 : 0.05 ) ],
							[ point( 31.5 / data.length, 0.1 ), point( 31.5 / data.length, 0.9 ) ],
							[ point( 31.5 / data.length, 0.9 ), point( 31.5 / data.length, 0.1 ) ]
						];
						try {

							const adaptive = await traceRays( renderer, fixture, rays );
							const linear = await traceLinearRays( renderer, fixture, rays );
							const summary = reportComparison( `${ width }x${ height }, reversed=${ reversed }`, fixture, rays, linear, adaptive.output );
							assert.strictEqual( summary.nonFinite, 0 );
							assert.strictEqual( summary.adaptiveOracleDisagreements, 0, 'adaptive agrees with the exhaustive oracle on the identical rays' );
							assert.strictEqual( linear[ 3 ], 1, 'linear detects the forward flat-depth strip hit' );
							assert.strictEqual( linear[ 7 ], 1, 'linear detects the backward flat-depth strip hit' );

						} finally {

							fixture.dispose();

						}

					} );

				}

				for ( const [ width, height ] of [[ 1, 1 ], [ 3, 5 ], [ 19, 11 ], [ 32, 32 ], [ 129, 127 ]] ) {

					differentialTest( `same seeded rays at ${ width }x${ height }, reversed=${ reversed }`, async ( { assert, renderer } ) => {

						const data = Float32Array.from( { length: width * height }, ( _, i ) => {

							const depth = i % 7 === 0 ? 1 : 0.2 + ( ( i * 73 ) % 101 ) / 160;
							return reversed ? 1 - depth : depth;

						} );
						const fixture = createTraceFixture( width, height, data, reversed );
						const rays = randomRays( 1024 );
						try {

							const adaptive = await traceRays( renderer, fixture, rays );
							const linear = await traceLinearRays( renderer, fixture, rays );
							const summary = reportComparison( `${ width }x${ height }, reversed=${ reversed }`, fixture, rays, linear, adaptive.output );
							assert.strictEqual( summary.nonFinite, 0, 'both production traversals return finite results' );
							assert.strictEqual( summary.adaptiveOracleDisagreements, 0, 'adaptive agrees with the exhaustive original-depth oracle' );
							assert.strictEqual( summary.rays, 1024, 'runs the identical seeded ray set used by adaptive tests' );
							const geometric = await traceGeometricRays( renderer, fixture, rays );
							const geometricSummary = reportComparison( `SSR view-space metric: ${ width }x${ height }, reversed=${ reversed }`, fixture, rays, geometric.linear, geometric.adaptive );
							assert.strictEqual( geometricSummary.nonFinite, 0, 'both traversals remain finite under the production thickness metric' );

						} finally {

							fixture.dispose();

						}

					} );

				}

			}

		} );

	} );

} );
