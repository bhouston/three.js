import { Mesh, MeshBasicNodeMaterial, OrthographicCamera, PerspectiveCamera, RenderPipeline, Scene, SphereGeometry, WebGPURenderer } from 'three/webgpu';
import { float, mrt, output, pass, vec3, vec4 } from 'three/tsl';
import { ssr } from '../../../../examples/jsm/tsl/display/SSRNode.js';

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'TSL', () => {

		QUnit.module( 'SSRNode', () => {

			QUnit.test( 'back-facing shading normals do not contribute reflections -webonly', async ( assert ) => {

				const adapter = await navigator.gpu?.requestAdapter();
				const info = adapter?.info;

				if ( adapter === null || adapter === undefined || info?.isFallbackAdapter === true || /swiftshader|llvmpipe|lavapipe|software/i.test( info?.description || '' ) ) {

					assert.ok( true, 'SKIPPED: a hardware WebGPU adapter is required.' );
					return;

				}

				const renderer = new WebGPURenderer();
				renderer.setSize( 320, 320 );
				await renderer.init();

				const scene = new Scene();
				const geometry = new SphereGeometry( 1, 64, 32 );
				const material = new MeshBasicNodeMaterial( { color: 0xff0000 } );
				scene.add( new Mesh( geometry, material ) );

				try {

					for ( const orthographic of [ false, true ] ) {

						const camera = orthographic ? new OrthographicCamera( - 1.5, 1.5, 1.5, - 1.5, 0.1, 20 ) : new PerspectiveCamera( 45, 1, 0.1, 20 );
						camera.position.set( 0, 0, 4 );
						camera.lookAt( 0, 0, 0 );

						for ( const binaryRefine of [ false, true ] ) {

							const scenePass = pass( scene, camera );
							scenePass.setMRT( mrt( { output, normal: vec3( 0, 0, - 1 ) } ) );
							const reflections = ssr( scenePass.getTextureNode( 'output' ), scenePass.getTextureNode( 'depth' ), scenePass.getTextureNode( 'normal' ), {
								camera, stochastic: true, metalnessNode: float( 1 ), roughnessNode: float( 0.3 ), binaryRefine
							} );
							reflections.mirrorBias.value = 0;
							reflections.maxDistance.value = 10;
							reflections.quality.value = 1;
							reflections.screenEdgeFade.value = 0;

							const pipeline = new RenderPipeline( renderer );
							pipeline.outputNode = vec4( reflections.rgb, 1 );

							try {

								let selfHits = 0;

								for ( let frame = 0; frame < 8; frame ++ ) {

									pipeline.render();
									const pixels = await renderer.readRenderTargetPixelsAsync( reflections._ssrRenderTarget, 0, 0, 320, 320 );

									// Force back-facing shading normals on every visible point. These must
									// be rejected; any red reflection pixel is an invalid contribution.
									for ( let i = 0; i < pixels.length; i += 4 ) {

										if ( pixels[ i ] > 0 ) selfHits ++;

									}

								}

								assert.strictEqual( selfHits, 0, `No self-reflections: orthographic=${ orthographic }, binaryRefine=${ binaryRefine }` );

							} finally {

								pipeline.dispose();
								reflections.dispose();
								scenePass.dispose();

							}

						}

					}

				} finally {

					geometry.dispose();
					material.dispose();
					renderer.dispose();

				}

			} );

		} );

	} );

} );
