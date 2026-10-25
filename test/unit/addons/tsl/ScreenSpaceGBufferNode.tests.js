import { DataTexture, FloatType, NodeMaterial, QuadMesh, RendererUtils, Vector2 } from 'three/webgpu';
import { context, texture } from 'three/tsl';
import { screenSpaceGBuffer } from '../../../../examples/jsm/tsl/display/ScreenSpaceGBufferNode.js';
import { getSharedRenderer } from '../../src/nodes/tsl/gpu-test-utils.js';

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'TSL', () => {

		QUnit.module( 'ScreenSpaceGBufferNode', () => {

			for ( const reversedDepth of [ false, true ] ) {

				QUnit.test( `aligned nearest-surface reduction, reversed depth: ${ reversedDepth }`, async assert => {

					const renderer = await getSharedRenderer( 'webgpu' );
					if ( ! renderer ) {

						assert.ok( true, 'SKIPPED: WebGPU unavailable' );
						return;

					}

					const state = RendererUtils.saveRendererState( renderer );
					const previousContext = renderer.contextNode;
					const previousSize = renderer.getSize( new Vector2() );
					const width = 7, height = 5;
					const depths = new Float32Array( width * height * 4 );
					const normals = new Uint8Array( width * height * 4 );
					const materials = new Uint8Array( width * height * 4 );
					for ( let i = 0; i < width * height; i ++ ) {

						depths[ 4 * i ] = reversedDepth ? 0.1 + i * 0.02 : 0.9 - i * 0.02;
						normals.set( [ i, 255 - i, 128, i * 3 ], 4 * i );
						materials.set( [ 200 - i, i * 2, 0, i ], 4 * i );

					}

					const depth = new DataTexture( depths, width, height );
					depth.type = FloatType;
					const normal = new DataTexture( normals, width, height );
					const material = new DataTexture( materials, width, height );
					for ( const value of [ depth, normal, material ] ) value.needsUpdate = true;
					const guides = screenSpaceGBuffer( texture( depth ), texture( normal ), texture( material ), { reversedDepth } );
					const output = new NodeMaterial();
					output.fragmentNode = guides.getTextureNode();
					const quad = new QuadMesh( output );
					try {

						renderer.setPixelRatio( 1 );
						renderer.setSize( width, height );
						renderer.contextNode = context();
						// Switch sizes on the same node to exercise resource resizing too.
						for ( const scale of [ 0.5, 0.25, 1, 0.5 ] ) {

							guides.resolutionScale = scale;
							await new Promise( resolve => requestAnimationFrame( resolve ) );
							renderer.setRenderTarget( null );
							quad.render( renderer );
							const target = guides.getRenderTarget();
							const w = Math.round( width * scale ), h = Math.round( height * scale );
							assert.deepEqual( [ target.width, target.height ], [ w, h ], 'Guides match the effect grid at odd sizes' );
							const geometry = await renderer.readRenderTargetPixelsAsync( target, 0, 0, w, h );
							const normalResult = await renderer.readRenderTargetPixelsAsync( target, 0, 0, w, h, 1 );
							const materialResult = await renderer.readRenderTargetPixelsAsync( target, 0, 0, w, h, 2 );
							for ( let y = 0; y < h; y ++ ) {

								for ( let x = 0; x < w; x ++ ) {

									// The independent fixture orders depths so the last pixel
									// in each footprint is nearest, including trailing edges.
									const sx = Math.floor( ( x + 1 ) * width / w ) - 1;
									const sy = Math.floor( ( y + 1 ) * height / h ) - 1;
									const source = sx + width * sy;
									const offset = 4 * ( x + w * y );
									assert.strictEqual( geometry[ offset ], depths[ 4 * source ], 'Nearest depth is preserved without averaging' );
									assert.ok( Math.abs( geometry[ offset + 1 ] - ( sx + 0.5 ) / width ) < 1e-6 && Math.abs( geometry[ offset + 2 ] - ( sy + 0.5 ) / height ) < 1e-6, 'Radiance lookup identifies the selected source pixel center' );
									assert.deepEqual( normalResult.slice( offset, offset + 4 ), normals.slice( 4 * source, 4 * source + 4 ), 'Normal and roughness come from the selected surface' );
									assert.deepEqual( materialResult.slice( offset, offset + 4 ), materials.slice( 4 * source, 4 * source + 4 ), 'Albedo and metalness come from the same surface' );

								}

							}

						}

					} finally {

						guides.dispose();
						output.dispose();
						for ( const value of [ depth, normal, material ] ) value.dispose();
						renderer.contextNode = previousContext;
						renderer.setPixelRatio( state.pixelRatio );
						renderer.setSize( previousSize.x, previousSize.y );
						RendererUtils.restoreRendererState( renderer, state );

					}

				} );

			}

		} );

	} );

} );
