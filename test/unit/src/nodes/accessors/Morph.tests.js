import { WebGPURenderer, Scene, OrthographicCamera, PlaneGeometry, BufferAttribute, MeshBasicNodeMaterial, Mesh, InstancedMesh, Matrix4, RenderTarget, FloatType } from 'three/webgpu';
import { mrt, output, velocity } from 'three/tsl';

export default QUnit.module( 'Nodes', () => {

	QUnit.module( 'Accessors', () => {

		QUnit.module( 'Morph', () => {

			QUnit.test( 'motion vectors use previous morph weights', async ( assert ) => {

				const renderer = new WebGPURenderer();
				await renderer.init();

				try {

					for ( const relative of [ false, true ] ) {

						for ( const instanced of [ false, true ] ) {

							const scene = new Scene();
							const camera = new OrthographicCamera( - 1, 1, 1, - 1, 0.1, 10 );
							camera.position.z = 2;
							const geometry = new PlaneGeometry( 1, 1 );
							const positions = geometry.attributes.position;
							const target = new Float32Array( positions.array.length );

							for ( let i = 0; i < positions.count; i ++ ) {

								target[ i * 3 ] = relative ? 0.2 : positions.getX( i ) + 0.2;
								target[ i * 3 + 1 ] = relative ? 0 : positions.getY( i );

							}

							geometry.morphAttributes.position = [ new BufferAttribute( target, 3 ) ];
							geometry.morphTargetsRelative = relative;
							const material = new MeshBasicNodeMaterial();
							const mesh = instanced ? new InstancedMesh( geometry, material, 1 ) : new Mesh( geometry, material );
							const proxy = new Mesh( geometry, material );
							if ( instanced ) mesh.setMatrixAt( 0, new Matrix4() );
							scene.add( mesh );

							const targets = [ 0, 1 ].map( () => {

								const rt = new RenderTarget( 16, 16, { type: FloatType, count: 2 } );
								rt.textures[ 0 ].name = 'output';
								rt.textures[ 1 ].name = 'velocity';
								return rt;

							} );

							const weights = [ 0.5, 1, 0, 0 ];
							let frame = 0;
							let busy = false;
							renderer.setMRT( mrt( { output, velocity } ) );

							try {

								await new Promise( ( resolve, reject ) => renderer.setAnimationLoop( async () => {

									if ( busy ) return;
									busy = true;

									try {

										if ( instanced ) {

											proxy.morphTargetInfluences[ 0 ] = weights[ frame ];
											mesh.setMorphAt( 0, proxy );
											mesh.morphTexture.needsUpdate = true;

										} else {

											mesh.morphTargetInfluences[ 0 ] = weights[ frame ];

										}

										for ( const rt of targets ) {

											renderer.setRenderTarget( rt );
											renderer.render( scene, camera );

										}

										const expected = frame === 0 ? 0 : ( weights[ frame ] - weights[ frame - 1 ] ) * 0.2;

										for ( const rt of targets ) {

											const values = await renderer.readRenderTargetPixelsAsync( rt, 8, 8, 1, 1, 1 );
											assert.true( Math.abs( values[ 0 ] - expected ) < 1e-5, `relative=${relative}, instanced=${instanced}, frame=${frame}: velocity ${values[ 0 ]} matches ${expected}` );

										}

										if ( ++ frame === weights.length ) {

											renderer.setAnimationLoop( null );
											resolve();

										}

									} catch ( error ) {

										renderer.setAnimationLoop( null );
										reject( error );

									} finally {

										busy = false;

									}

								} ) );

							} finally {

								renderer.setRenderTarget( null );
								renderer.setMRT( null );
								targets.forEach( rt => rt.dispose() );
								geometry.dispose();
								material.dispose();
								if ( instanced ) mesh.dispose();

							}

						}

					}

				} finally {

					renderer.setAnimationLoop( null );
					renderer.dispose();

				}

			} );

		} );

	} );

} );
