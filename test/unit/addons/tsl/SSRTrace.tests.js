import { DataTexture, FloatType, NearestFilter, NodeMaterial, OrthographicCamera, PerspectiveCamera, QuadMesh, RedFormat, RGBAFormat, RenderTarget, Vector2, Vector3, WebGPUCoordinateSystem } from 'three/webgpu';
import { float, texture } from 'three/tsl';
import { ENV_RAY_LENGTH_THRESHOLD } from '../../../../examples/jsm/tsl/utils/SpecularHelpers.js';
import { ssr } from '../../../../examples/jsm/tsl/display/SSRNode.js';
import { rawComputeTest } from '../../src/nodes/tsl/gpu-raw-test-utils.js';
import { createTraceFixture, randomRays, referenceTrace, traceRays } from './ssr-trace-test-utils.js';

function traceTest( name, run ) {

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

async function compareRays( assert, renderer, fixture, rays, options = {} ) {

	const { output, metadata } = await traceRays( renderer, fixture, rays, options );
	const failures = [];
	let hits = 0, maxIterations = 0;
	for ( let i = 0; i < rays.length; i ++ ) {

		const reference = referenceTrace( fixture, ...rays[ i ], options );
		const hit = output[ i * 4 + 3 ] === 1;
		if ( hit ) hits ++;
		maxIterations = Math.max( maxIterations, metadata[ i * 4 ] );
		if ( hit !== reference.hit || metadata[ i * 4 + 1 ] !== 0 || ( hit && Math.abs( output[ i * 4 + 2 ] - reference.parameter ) > 0.0001 ) ) {

			if ( failures.length < 12 ) failures.push( { ray: rays[ i ], reference, actual: Array.from( output.slice( i * 4, i * 4 + 4 ) ), metadata: Array.from( metadata.slice( i * 4, i * 4 + 4 ) ) } );

		}

	}

	assert.deepEqual( failures, [], `${ rays.length } identical rays match exhaustive source-depth oracle (${ hits } hits, max ${ maxIterations } iterations)` );
	return { output, metadata };

}

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'TSL', () => {

		QUnit.module( 'SSRTrace', () => {

			for ( const reversed of [ false, true ] ) {

				for ( const [ width, height ] of [[ 1, 128 ], [ 128, 1 ], [ 1, 127 ], [ 127, 1 ]] ) {

					traceTest( `back and forth in ${ width }x${ height }, reversed=${ reversed }`, async ( { assert, renderer } ) => {

						const data = new Float32Array( width * height ).fill( reversed ? 0 : 1 );
						data[ 31 ] = data[ 95 ] = 0.5;
						const fixture = createTraceFixture( width, height, data, reversed );
						const ray = ( p, z ) => width === 1 ? [ 0.5, p, z ] : [ p, 0.5, z ];
						const rays = [
							[ ray( 0.001, 0.5 ), ray( 0.999, 0.5 ) ],
							[ ray( 0.999, 0.5 ), ray( 0.001, 0.5 ) ],
							[ ray( 0.001, reversed ? 0.95 : 0.05 ), ray( 0.999, reversed ? 0.05 : 0.95 ) ],
							[ ray( 0.999, reversed ? 0.05 : 0.95 ), ray( 0.001, reversed ? 0.95 : 0.05 ) ],
							// Pure depth rays have no XY motion; both directions must work.
							[ ray( 31.5 / data.length, 0.1 ), ray( 31.5 / data.length, 0.9 ) ],
							[ ray( 31.5 / data.length, 0.9 ), ray( 31.5 / data.length, 0.1 ) ]
						];
						try {

							await compareRays( assert, renderer, fixture, rays );

						} finally {

							fixture.dispose();

						}

					} );

				}

				for ( const [ width, height ] of [[ 1, 1 ], [ 3, 5 ], [ 19, 11 ], [ 32, 32 ], [ 129, 127 ]] ) {

					traceTest( `seeded rays across ${ width }x${ height }, reversed=${ reversed }`, async ( { assert, renderer } ) => {

						const data = Float32Array.from( { length: width * height }, ( _, i ) => {

							const depth = i % 7 === 0 ? 1 : 0.2 + ( ( i * 73 ) % 101 ) / 160;
							return reversed ? 1 - depth : depth;

						} );
						const fixture = createTraceFixture( width, height, data, reversed );
						try {

							await compareRays( assert, renderer, fixture, randomRays( 1024 ) );

						} finally {

							fixture.dispose();

						}

					} );

				}

			}

			traceTest( 'rejected first leaf continues to a later valid surface', async ( { assert, renderer } ) => {

				const data = new Float32Array( 128 ).fill( 1 );
				data[ 31 ] = 0.5;
				data[ 95 ] = 0.5005;
				const fixture = createTraceFixture( 128, 1, data );
				try {

					const result = await compareRays( assert, renderer, fixture, [[[ 0.001, 0.5, 0.5005 ], [ 0.999, 0.5, 0.5005 ]]], { rejectedDepth: 0.5, thickness: 0.001 } );
					assert.ok( result.output[ 0 ] > 0.7, 'skips rejected surface and reaches second surface' );
					const stopped = await traceRays( renderer, fixture, [[[ 0.001, 0.5, 0.5005 ], [ 0.999, 0.5, 0.5005 ]]], { rejectedDepth: 0.5, thickness: 0.001, maxCandidates: 1 } );
					assert.strictEqual( stopped.output[ 3 ], 0, 'first-candidate mode rejects the first surface and stops' );
					assert.strictEqual( stopped.metadata[ 1 ], 0, 'rejected first candidate is a completed miss, not budget exhaustion' );

				} finally {

					fixture.dispose();

				}

			} );

			for ( const perspective of [ false, true ] ) {

				for ( const stochastic of [ false, true ] ) {

					traceTest( `SSRNode identical rays: perspective=${ perspective }, stochastic=${ stochastic }`, async ( { assert, renderer } ) => {

						const width = 128, height = 1;
						const camera = perspective ? new PerspectiveCamera( 50, 1, 0.1, 10.1 ) : new OrthographicCamera( - 1, 1, 1, - 1, 0.1, 10.1 );
						camera.coordinateSystem = WebGPUCoordinateSystem;
						camera.updateProjectionMatrix();
						camera.updateMatrixWorld();
						const depth = new DataTexture( new Float32Array( width ).fill( 0.5 ), width, height, RedFormat, FloatType );
						const normalData = new Float32Array( width * 4 );
						const colors = new Float32Array( width * 4 );
						for ( let x = 0; x < width; x ++ ) {

							const incident = perspective ? new Vector3( ( x + 0.5 ) / width * 2 - 1, 0, 0.5 ).applyMatrix4( camera.projectionMatrixInverse ).normalize() : new Vector3( 0, 0, - 1 );
							const receiverNormal = new Vector3( 1, 0, 0 ).sub( incident ).normalize();
							normalData.set( x < 96 ? [ receiverNormal.x, receiverNormal.y, receiverNormal.z, 1 ] : [ - 1, 0, 0, 1 ], x * 4 );
							colors.set( [ 0.8, 0.2, 0.1, 1 ], x * 4 );

						}

						const normal = new DataTexture( normalData, width, height, RGBAFormat, FloatType );
						const color = new DataTexture( colors, width, height, RGBAFormat, FloatType );
						for ( const tex of [ depth, normal, color ] ) {

							tex.minFilter = tex.magFilter = NearestFilter;
							tex.needsUpdate = true;

						}

						const nodes = [ 'linear', 'adaptive' ].map( tracing => {

							const node = ssr( texture( color ), texture( depth ), texture( normal ), { camera, stochastic, maxCandidates: 4096, metalnessNode: float( 1 ), roughnessNode: float( 0 ), tracing, binaryRefine: true } );
							node.quality.value = 1;
							node.maxSteps.value = 4096;
							node.adaptiveMaxSteps.value = 2048;
							node.maxDistance.value = 20;
							node.thickness.value = 0.0001;
							return node;

						} );
						const material = new NodeMaterial();
						const quad = new QuadMesh( material );
						const target = new RenderTarget( width, height, { type: FloatType } );
						const oldSize = renderer.getSize( new Vector2() );
						const oldTarget = renderer.getRenderTarget();
						try {

							renderer.setSize( width, height );
							const pixels = [];
							for ( const node of nodes ) {

								material.fragmentNode = node;
								material.needsUpdate = true;
								renderer.setRenderTarget( target );
								quad.render( renderer );
								pixels.push( await renderer.readRenderTargetPixelsAsync( target, 0, 0, width, height ) );

							}

							let hits = 0, maxDistanceError = 0, maxColorError = 0;
							const mismatches = [];
							for ( let x = 0; x < width; x ++ ) {

								const offset = x * 4;
								const a = pixels[ 0 ][ offset + 3 ], b = pixels[ 1 ][ offset + 3 ];
								if ( ( a > 0 && a < ENV_RAY_LENGTH_THRESHOLD ) !== ( b > 0 && b < ENV_RAY_LENGTH_THRESHOLD ) ) mismatches.push( { x, linear: a, adaptive: b } );
								if ( a > 0 && a < ENV_RAY_LENGTH_THRESHOLD && b > 0 && b < ENV_RAY_LENGTH_THRESHOLD ) {

									hits ++;
									maxDistanceError = Math.max( maxDistanceError, Math.abs( a - b ) );
									for ( let c = 0; c < 3; c ++ ) maxColorError = Math.max( maxColorError, Math.abs( pixels[ 0 ][ offset + c ] - pixels[ 1 ][ offset + c ] ) );

								}

							}

							if ( perspective && ! stochastic ) {

								// Nominally coplanar rays expose legacy projection rounding misses.
								// Accept additional adaptive hits only after checking their distance
								// against the analytic intersection with the x=96 texel boundary.
								const invalid = [];
								const boundary = new Vector3( 96 / width * 2 - 1, 0, 0.5 ).applyMatrix4( camera.projectionMatrixInverse );
								for ( const mismatch of mismatches ) {

									const start = new Vector3( ( mismatch.x + 0.5 ) / width * 2 - 1, 0, 0.5 ).applyMatrix4( camera.projectionMatrixInverse );
									const pixelStep = new Vector3( ( mismatch.x + 1.5 ) / width * 2 - 1, 0, 0.5 ).applyMatrix4( camera.projectionMatrixInverse ).distanceTo( start );
									if ( mismatch.linear > 0 || mismatch.x >= 96 || Math.abs( mismatch.adaptive - start.distanceTo( boundary ) ) > pixelStep ) invalid.push( mismatch );

								}

								assert.deepEqual( invalid, [], `${ mismatches.length } extra adaptive hits agree with analytic surface geometry` );

							} else {

								assert.deepEqual( mismatches, [], 'identical hit masks' );

							}

							assert.ok( hits > ( stochastic ? 32 : 80 ) && hits < 100, `${ hits } nontrivial matching receiver rays` );
							assert.ok( maxDistanceError < 0.02, `hit distance differs by less than one source pixel: ${ maxDistanceError }` );
							assert.ok( maxColorError < 0.002, `same radiance and shading: ${ maxColorError }` );

						} finally {

							renderer.setRenderTarget( oldTarget );
							renderer.setSize( oldSize.x, oldSize.y );
							for ( const node of nodes ) node.dispose();
							for ( const tex of [ depth, normal, color ] ) tex.dispose();
							material.dispose();
							target.dispose();

						}

					} );


				}

			}

			traceTest( 'candidate budgets stop after exactly 1, 2, 3 or 4 surfaces', async ( { assert, renderer } ) => {

				const data = new Float32Array( 128 ).fill( 1 );
				for ( let i = 0; i < 4; i ++ ) data[ 16 + i * 32 ] = 0.5001 + i * 0.0001;
				const fixture = createTraceFixture( 128, 1, data );
				const rays = [[[ 0.001, 0.5, data[ 112 ] ], [ 0.999, 0.5, data[ 112 ] ]]];
				try {

					for ( const maxCandidates of [ 1, 2, 3, 4 ] ) {

						const result = await traceRays( renderer, fixture, rays, { maxCandidates } );
						assert.strictEqual( result.metadata[ 3 ], maxCandidates, `visits ${ maxCandidates } candidates` );
						assert.strictEqual( result.metadata[ 1 ], 0, 'candidate limit is not iteration exhaustion' );
						assert.strictEqual( result.output[ 3 ], maxCandidates === 4 ? 1 : 0, 'only budget four reaches the valid surface' );

					}

				} finally {

					fixture.dispose();

				}

			} );

			for ( const reversed of [ false, true ] ) {

				traceTest( `Unity slab intersections in both depth directions, reversed=${ reversed }`, async ( { assert, renderer } ) => {

					const fixture = createTraceFixture( 1, 1, [ 0.5 ], reversed );
					const ray = z => [ 0.5, 0.5, reversed ? 1 - z : z ];
					const cases = [
						{ a: 0.2, b: 0.8, hit: true, parameter: 0.5 },
						{ a: 0.8, b: 0.2, hit: true, parameter: 0.5 },
						{ a: 0.2, b: 0.3, hit: false },
						{ a: 0.8, b: 0.9, hit: false },
						{ a: 0.52, b: 0.9, hit: true, parameter: 0 },
						{ a: 0.5, b: 0.5, hit: true, parameter: 0 }
					];
					try {

						const result = await traceRays( renderer, fixture, cases.map( c => [ ray( c.a ), ray( c.b ) ] ), { strategy: 'unity', maxCandidates: 1, thickness: 0.1, slabThickness: 0.05 } );
						for ( let i = 0; i < cases.length; i ++ ) {

							assert.strictEqual( result.output[ i * 4 + 3 ], cases[ i ].hit ? 1 : 0, `case ${ i } hit` );
							assert.strictEqual( result.metadata[ i * 4 + 1 ], 0, `case ${ i } completes` );
							if ( cases[ i ].hit ) assert.ok( Math.abs( result.output[ i * 4 + 2 ] - cases[ i ].parameter ) < 1e-5, `case ${ i } exact plane/slab entry` );

						}

					} finally {

						fixture.dispose();

					}

				} );

				for ( const length of [ 4, 5 ] ) {

					for ( const vertical of [ false, true ] ) {

						traceTest( `Unity rejects re-entry behind an occluder: length=${ length }, vertical=${ vertical }, reversed=${ reversed }`, async ( { assert, renderer } ) => {

							const data = new Float32Array( length ).fill( reversed ? 0 : 1 );
							data[ 0 ] = reversed ? 0.8 : 0.2;
							data[ 1 ] = reversed ? 0.52 : 0.48;
							const fixture = createTraceFixture( vertical ? 1 : length, vertical ? length : 1, data, reversed );
							const point = p => vertical ? [ 0.5, p, 0.5 ] : [ p, 0.5, 0.5 ];
							const rays = [[ point( 0.5 / length ), point( 0.999 ) ]];
							try {

								const unity = await traceRays( renderer, fixture, rays, { strategy: 'unity', maxCandidates: 4, thickness: 0.1, slabThickness: 0.05 } );
								assert.strictEqual( unity.output[ 3 ], 0, 'slab re-entry is occluded' );
								assert.strictEqual( unity.metadata[ 1 ], 0, 'occlusion is a completed miss' );
								const amd = await traceRays( renderer, fixture, rays, { strategy: 'amd', maxCandidates: 4, thickness: 0.1 } );
								assert.strictEqual( amd.output[ 3 ], 1, 'continued surface search can reach that later surface' );

							} finally {

								fixture.dispose();

							}

						} );

					}

				}

			}

			traceTest( 'budget exhaustion is distinct from a completed miss', async ( { assert, renderer } ) => {

				const fixture = createTraceFixture( 128, 1, new Float32Array( 128 ).fill( 1 ) );
				const rays = [[[ 0.001, 0.5, 0.2 ], [ 0.999, 0.5, 0.8 ]]];
				try {

					const limited = await traceRays( renderer, fixture, rays, { maxSteps: 1 } );
					assert.strictEqual( limited.metadata[ 1 ], 1, 'one iteration is explicitly exhausted' );
					const completed = await compareRays( assert, renderer, fixture, rays );
					assert.strictEqual( completed.output[ 3 ], 0, 'empty buffer is a miss' );
					assert.ok( completed.metadata[ 0 ] < 16, 'empty-space skipping takes fewer than 16 lookups across 128 pixels' );

				} finally {

					fixture.dispose();

				}

			} );

		} );

	} );

} );
