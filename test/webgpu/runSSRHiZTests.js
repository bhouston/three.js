import * as THREE from 'three/webgpu';
import { pass, mrt, output, normalView, metalness, roughness, vec2, sample, packNormalToRGB, unpackRGBToNormal } from 'three/tsl';
import { ssr } from 'three/addons/tsl/display/SSRNode.js';
import { hiZ } from 'three/addons/tsl/display/HiZNode.js';

function assert( condition, message ) {

	if ( ! condition ) throw new Error( message );

}

// These are actual SSR shaders over procedural geometry, not a CPU approximation
// of ray traversal. No assets, network requests, environment maps or temporal history.
export async function runSSRHiZTests() {

	assert( navigator.gpu, 'Hardware WebGPU is required.' );
	const adapter = await navigator.gpu.requestAdapter( { powerPreference: 'high-performance' } );
	assert( adapter, 'No hardware adapter.' );
	const description = Object.values( adapter.info.toJSON ? adapter.info.toJSON() : { vendor: adapter.info.vendor, device: adapter.info.device, description: adapter.info.description } ).join( ' ' );
	assert( description.trim() && ! adapter.isFallbackAdapter && ! adapter.info.isFallbackAdapter && ! /swiftshader|llvmpipe|lavapipe|software|microsoft basic render/i.test( description ), `Hardware adapter required: ${description}` );
	let renderer = new THREE.WebGPURenderer( { device: await adapter.requestDevice() } );
	await renderer.init();
	const errors = [];
	const onError = event => errors.push( event.error.message );
	renderer.backend.device.addEventListener( 'uncapturederror', onError );
	const results = [];
	try {

		for ( const { reversedDepth, logarithmicDepthBuffer } of [
			{ reversedDepth: false, logarithmicDepthBuffer: false },
			{ reversedDepth: true, logarithmicDepthBuffer: false },
			{ reversedDepth: false, logarithmicDepthBuffer: true }
		] ) {

			renderer.backend.device.removeEventListener( 'uncapturederror', onError );
			renderer.dispose();
			renderer = new THREE.WebGPURenderer( { device: await ( await navigator.gpu.requestAdapter( { powerPreference: 'high-performance' } ) ).requestDevice(), reversedDepthBuffer: reversedDepth, logarithmicDepthBuffer } );
			await renderer.init();
			renderer.backend.device.addEventListener( 'uncapturederror', onError );
			for ( const orthographic of [ false, true ] ) {

				const width = 65, height = 49;
				renderer.setSize( width, height );
				const camera = orthographic ? new THREE.OrthographicCamera( - 4, 4, 3, - 3, 0.1, 40 ) : new THREE.PerspectiveCamera( 55, width / height, 0.1, 40 );
				camera.position.set( 4, 3, 6 );
				camera.lookAt( 0, 0.6, 0 );
				const scene = new THREE.Scene();
				scene.background = new THREE.Color( 0.05, 0.1, 0.2 );
				scene.add( new THREE.HemisphereLight( 0xffffff, 0x303040, 3 ) );
				const floor = new THREE.Mesh( new THREE.PlaneGeometry( 16, 16 ), new THREE.MeshStandardNodeMaterial( { color: 0xb0b0b0, metalness: 1, roughness: 0.12 } ) );
				floor.rotation.x = - Math.PI / 2;
				scene.add( floor );
				for ( let i = 0; i < 5; i ++ ) {

					const box = new THREE.Mesh( new THREE.BoxGeometry( 0.45 + i * 0.12, 0.7 + i * 0.35, 0.7 ), new THREE.MeshStandardNodeMaterial( { color: new THREE.Color().setHSL( i / 5, 0.8, 0.5 ), metalness: 0.7, roughness: 0.22 } ) );
					box.position.set( ( i - 2 ) * 1.2, ( 0.7 + i * 0.35 ) / 2, ( i % 2 ) * - 1.4 );
					scene.add( box );

				}

				const scenePass = pass( scene, camera );
				scenePass.setMRT( mrt( { output, normal: packNormalToRGB( normalView ), metalrough: vec2( metalness, roughness ) } ) );
				const depth = scenePass.getTextureNode( 'depth' );
				const pyramid = hiZ( depth );
				const normals = sample( uv => unpackRGBToNormal( scenePass.getTextureNode( 'normal' ).sample( uv ) ) );
				const target = new THREE.RenderTarget( width, height, { type: THREE.FloatType, depthBuffer: false } );
				try {

					for ( const stochastic of [ false, true ] ) {

						for ( const binaryRefine of [ false, true ] ) {

							const reflection = ssr( scenePass.getTextureNode( 'output' ), depth, normals, { hiZ: pyramid, stochastic, binaryRefine, camera, metalnessNode: scenePass.getTextureNode( 'metalrough' ).r, roughnessNode: scenePass.getTextureNode( 'metalrough' ).g } );
							reflection.maxDistance.value = 20;
							reflection.thickness.value = 0.15;
							// Keep reflection data in float all the way through the readback.
							reflection._ssrRenderTarget.texture.type = THREE.FloatType;
							const pipeline = new THREE.RenderPipeline( renderer, reflection );
							pipeline.outputColorTransform = false;
							try {

								for ( const quality of [ 0.25, 1 ] ) {

									reflection.quality.value = quality;
									const read = async enabled => {

										reflection.hiZEnabled = enabled;
										reflection._noiseIndex.value = 0;
										renderer.setRenderTarget( target );
										// Standalone draws must advance FRAME nodes just as Animation does.
										renderer._nodes.nodeFrame.update();
										pipeline.render();
										assert( reflection._noiseIndex.value === 1, 'SSR did not execute for this frame.' );
										await renderer.backend.device.queue.onSubmittedWorkDone();
										return renderer.readRenderTargetPixelsAsync( target, 0, 0, width, height );

									};

									await read( false ); await read( false ); await read( false ); // Populate camera matrices, targets and shader resources.
									const linear = await read( false );
									const hierarchical = await read( true );
									let maxError = 0, hits = 0, changedHits = 0;
									for ( let i = 0; i < linear.length; i ++ ) {

										assert( Number.isFinite( linear[ i ] ) && Number.isFinite( hierarchical[ i ] ), `Nonfinite SSR output ${i}` );
										maxError = Math.max( maxError, Math.abs( linear[ i ] - hierarchical[ i ] ) );
										if ( i % 4 === 3 ) {

											if ( linear[ i ] > 0 ) hits ++;
											if ( ( linear[ i ] > 0 ) !== ( hierarchical[ i ] > 0 ) ) changedHits ++;

										}

									}

									const label = JSON.stringify( { reversedDepth, logarithmicDepthBuffer, orthographic, stochastic, binaryRefine, quality } );
									assert( hits > 0, `Scene produced no SSR hits: ${label}` );
									assert( changedHits === 0 && maxError <= 0.002, `SSR mismatch ${label}: changed hits=${changedHits}, max error=${maxError}` );
									results.push( { reversedDepth, logarithmicDepthBuffer, orthographic, stochastic, binaryRefine, quality, hits, changedHits, maxError } );

								}

							} finally {

								pipeline.dispose();
								reflection.dispose();

							}

						}

					}

				} finally {

					renderer.setRenderTarget( null );
					target.dispose();
					pyramid.dispose();
					scenePass.dispose();
					scene.traverse( object => {

						object.geometry?.dispose(); object.material?.dispose();

					} );

				}

			}

		}

		assert( errors.length === 0, errors.join( '\n' ) );
		return { adapter: description, cases: results.length, results };

	} finally {

		renderer.backend.device.removeEventListener( 'uncapturederror', onError );
		renderer.dispose();

	}

}



