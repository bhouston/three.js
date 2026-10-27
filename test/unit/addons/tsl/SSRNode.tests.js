import { DataTexture, DataUtils, FloatType, NodeMaterial, OrthographicCamera, QuadMesh, RendererUtils, Vector2, Vector3 } from 'three/webgpu';
import { bool, context, float, texture, uniform, vec3, vec4 } from 'three/tsl';
import { ssr } from '../../../../examples/jsm/tsl/display/SSRNode.js';
import { getSharedRenderer } from '../../src/nodes/tsl/gpu-test-utils.js';

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'TSL', () => {

		QUnit.module( 'SSRNode', () => {

			QUnit.test( 'a trace mask requires coverage and external fallback', assert => {

				assert.throws( () => ssr( vec4( 1 ), float( 1 ), vec3( 0, 0, 1 ), { traceMaskNode: bool( false ) } ), /requires outputCoverage/, 'A mask cannot silently remove internal environment fallback' );

			} );

			QUnit.test( 'short rays respect their endpoint and refinement requires a depth bracket', async assert => {

				const renderer = await getSharedRenderer( 'webgpu' );
				if ( ! renderer ) {

					assert.ok( true, 'SKIPPED: WebGPU unavailable' );
					return;

				}

				const state = RendererUtils.saveRendererState( renderer );
				const previousContext = renderer.contextNode;
				const previousSize = renderer.getSize( new Vector2() );
				const previousRatio = renderer.getPixelRatio();
				const width = 64, height = 32, originX = 17;
				const camera = new OrthographicCamera( - 4, 4, 2, - 2, 0.1, 10 );
				camera.coordinateSystem = renderer.coordinateSystem;
				camera.updateProjectionMatrix();
				camera.updateMatrixWorld();
				const projectDepth = z => new Vector3( 0, 0, z ).applyMatrix4( camera.projectionMatrix ).z;
				const makeTexture = callback => {

					const data = new Float32Array( width * height * 4 );
					for ( let y = 0; y < height; y ++ ) {

						for ( let x = 0; x < width; x ++ ) data.set( callback( x, y ), 4 * ( x + width * y ) );

					}

					const result = new DataTexture( data, width, height );
					result.type = FloatType;
					result.needsUpdate = true;
					return result;

				};

				const target = x => x >= 22 && x < 26;
				const source = makeTexture( x => [ target( x ) ? 1 : 0, 0, 0, 1 ] );
				// Reflect (0, 0, -1) about this normal to obtain (1, 0, 0).
				const normal = makeTexture( x => x === originX ? [ Math.SQRT1_2, 0, Math.SQRT1_2, 1 ] : [ - 1, 0, 0, 1 ] );
				// An unrelated foreground sheet separates the origin from the red stripe.
				// Its one-unit separation exceeds both the configured and pixel thickness.
				// The stripe is slightly nearer than the origin to accommodate the
				// sampler's minimum GGX alpha, which perturbs a roughness-zero ray.
				const depth = makeTexture( x => [ projectDepth( x === originX ? - 2 : target( x ) ? - 1.9 : - 1 ), 0, 0, 1 ] );
				const eligible = uniform( true );
				const node = ssr( texture( source ), texture( depth ), texture( normal ), {
					camera, stochastic: true, outputCoverage: true, traceMaskNode: eligible,
					metalnessNode: float( 1 ), roughnessNode: float( 0 ),
					noiseNode: vec4( 0.5 )
				} );
				node.quality.value = 1;
				node.thickness.value = 0.01;
				node.screenEdgeFade.value = 0;
				node.mirrorBias.value = 0;
				const material = new NodeMaterial();
				material.fragmentNode = node.getTextureNode();
				const quad = new QuadMesh( material );
				const renderOrigin = async () => {

					await new Promise( resolve => requestAnimationFrame( resolve ) );
					renderer.setRenderTarget( null );
					quad.render( renderer );
					const result = node.getRenderTarget();
					const x = Math.floor( originX * node.resolutionScale ), y = Math.floor( height * node.resolutionScale / 2 );
					// radiance and coverage, then the hit distance from the hit texture
					const pixels = await renderer.readRenderTargetPixelsAsync( result, x, y, 1, 1 );
					const hit = await renderer.readRenderTargetPixelsAsync( result, x, y, 1, 1, 1 );
					return [ ...Array.from( pixels, DataUtils.fromHalfFloat ), DataUtils.fromHalfFloat( hit[ 0 ] ) ];

				};

				try {

					renderer.setPixelRatio( 1 );
					renderer.setSize( width, height );
					renderer.contextNode = context();
					// This endpoint is only 2.83 texels away. The old one-texel stride
					// floor continued past it and incorrectly reached the red stripe.
					node.maxDistance.value = 0.25;
					assert.strictEqual( ( await renderOrigin() )[ 0 ], 0, 'A target beyond the endpoint contributes no reflection' );
					node.binaryRefine = true;
					assert.strictEqual( ( await renderOrigin() )[ 0 ], 0, 'Refinement also respects the endpoint' );
					node.maxDistance.value = 2;
					node.binaryRefine = false;
					const hit = await renderOrigin();
					assert.ok( hit[ 0 ] > 0.9, `Extending the same ray reaches the red target (actual ${ hit[ 0 ] })` );
					assert.ok( hit.every( Number.isFinite ), 'Reflection radiance and hit distance stay finite' );
					// The previous sample is already behind the foreground sheet, but
					// rejected by thickness. It cannot bracket this stripe's crossing:
					// unconditional bisection incorrectly moves the hit onto that sheet.
					node.binaryRefine = true;
					const refinedHit = await renderOrigin();
					assert.ok( refinedHit[ 0 ] > 0.9, `Refinement preserves a hit following a rejected depth gap (actual ${ refinedHit[ 0 ] })` );
					eligible.value = false;
					const masked = await renderOrigin();
					assert.deepEqual( masked, [ 0, 0, 0, 0, 0 ], 'Excluding a previously lit texel clears stale radiance, coverage and distance' );
					node.useTraceMask.value = false;
					assert.deepEqual( await renderOrigin(), refinedHit, 'Disabling classification retains the original hit, including distance' );
					eligible.value = true;
					node.useTraceMask.value = true;
					assert.deepEqual( await renderOrigin(), refinedHit, 'Eligible texels preserve radiance and distance' );
					// A half-resolution texel lands between full-resolution depth centers.
					// Both passes must launch from the selected source texel's position.
					const originalDepth = depth.image.data.slice();
					for ( let y = 0; y < height; y ++ ) {

						for ( let x = 0; x < width; x ++ ) depth.image.data[ 4 * ( x + y * width ) ] = projectDepth( x === originX ? - 2 : target( x ) ? - 1.9 : - 4 );

					}

					depth.needsUpdate = true;
					const fullResolutionHit = await renderOrigin();
					node.resolutionScale = 0.5;
					const halfResolutionHit = await renderOrigin();
					assert.ok( halfResolutionHit[ 0 ] > 0.9, 'The half-resolution ray reaches the same target' );
					assert.ok( Math.abs( halfResolutionHit[ 4 ] - fullResolutionHit[ 4 ] ) < 0.01, `Full and half resolution reconstruct the same ray origin: ${ fullResolutionHit[ 4 ] }, ${ halfResolutionHit[ 4 ] }` );
					node.resolutionScale = 1;
					depth.image.data.set( originalDepth );
					depth.needsUpdate = true;
					// Black radiance and partial source validity are still real hits.
					for ( let i = 0; i < source.image.data.length; i += 4 ) source.image.data.set( [ 0, 0, 0, 0.25 ], i );
					source.needsUpdate = true;
					const black = await renderOrigin();
					assert.deepEqual( black.slice( 0, 3 ), [ 0, 0, 0 ], 'A valid black hit keeps black radiance' );
					assert.strictEqual( black[ 4 ], refinedHit[ 4 ], 'Black partial-coverage hits retain their geometry distance' );
					assert.strictEqual( black[ 3 ], 0.25, 'Black hits retain partial validity independently of RGB' );



				} finally {

					node.dispose();
					material.dispose();
					for ( const value of [ source, depth, normal ] ) value.dispose();
					renderer.contextNode = previousContext;
					renderer.setPixelRatio( previousRatio );
					renderer.setSize( previousSize.x, previousSize.y );
					RendererUtils.restoreRendererState( renderer, state );

				}

			} );

		} );

	} );

} );
