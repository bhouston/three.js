import { mrt, normalView, vec4 } from 'three/tsl';
import { PerspectiveCamera, RenderTarget, Scene, UnsignedByteType } from 'three/webgpu';
import { getSharedRenderer } from '../../nodes/tsl/gpu-test-utils.js';

export default QUnit.module( 'Renderers', () => {

	QUnit.module( 'Common', () => {

		QUnit.module( 'Background', () => {

			for ( const backend of [ 'webgpu', 'webgl' ] ) {

				QUnit.test( `backgrounds do not write surface data to MRT (${ backend })`, async ( assert ) => {

					const renderer = await getSharedRenderer( backend );

					if ( renderer === null ) {

						assert.ok( true, 'SKIPPED: The rendering backend is not available.' );
						return;

					}

					const scene = new Scene();
					scene.backgroundNode = vec4( 0.25, 0.5, 0.75, 1 );

					const camera = new PerspectiveCamera();
					const target = new RenderTarget( 4, 4, { count: 3, type: UnsignedByteType } );
					target.textures[ 0 ].name = 'output';
					target.textures[ 1 ].name = 'normal';
					target.textures[ 2 ].name = 'custom';

					// These would otherwise evaluate on the background sphere.
					const outputs = mrt( { normal: vec4( normalView, 1 ), custom: vec4( 1 ) } );
					outputs.setClearColor( 'custom', 0, 0.25 );

					const previousTarget = renderer.getRenderTarget();
					const previousMRT = renderer.getMRT();

					try {

						renderer.setRenderTarget( target );
						renderer.setMRT( outputs );
						renderer.render( scene, camera );

						const expected = [[ 64, 128, 191, 255 ], [ 0, 0, 0, 255 ], [ 0, 0, 0, 64 ]];

						for ( let attachment = 0; attachment < expected.length; attachment ++ ) {

							const pixels = await renderer.readRenderTargetPixelsAsync( target, 0, 0, 4, 4, attachment );
							const matches = pixels.every( ( value, index ) => Math.abs( value - expected[ attachment ][ index % 4 ] ) <= 1 );
							assert.ok( matches, `${ target.textures[ attachment ].name } contains only background color or its clear value.` );

						}

					} finally {

						renderer.setRenderTarget( previousTarget );
						renderer.setMRT( previousMRT );
						target.dispose();
						scene.backgroundNode.dispose();

					}

				} );

			}

		} );

	} );

} );
