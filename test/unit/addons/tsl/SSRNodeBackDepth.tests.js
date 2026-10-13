import { BackSide, BoxGeometry, Mesh, MeshBasicNodeMaterial, PerspectiveCamera, PlaneGeometry, RenderPipeline, Scene, WebGPURenderer } from 'three/webgpu';
import { float, mrt, normalView, output, pass, vec4 } from 'three/tsl';
import { ssr } from '../../../../examples/jsm/tsl/display/SSRNode.js';

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'TSL', () => {

		QUnit.test( 'SSR back-face depth recovers a solid crossing -webonly', async ( assert ) => {

			const adapter = await navigator.gpu?.requestAdapter();
			if ( ! adapter || adapter.info.isFallbackAdapter || /swiftshader|llvmpipe|lavapipe|software/i.test( adapter.info.description ) ) {

				assert.ok( true, 'SKIPPED: hardware WebGPU required.' );
				return;

			}

			const renderer = new WebGPURenderer();
			renderer.setSize( 320, 320 );
			await renderer.init();
			const scene = new Scene();
			const camera = new PerspectiveCamera( 45, 1, 0.1, 20 );
			camera.position.set( 0, 0, 3 );
			camera.lookAt( 0, 0, - 1 );
			const receiver = new Mesh( new PlaneGeometry( 4, 4 ), new MeshBasicNodeMaterial( { color: 0 } ) );
			receiver.position.z = - 1;
			const emitter = new Mesh( new BoxGeometry( 0.6, 0.6, 0.3 ), new MeshBasicNodeMaterial( { color: 0xff0000 } ) );
			emitter.position.set( 0.7, 0, 1 );
			scene.add( receiver, emitter );

			let baselinePixels = 0;
			try {

				for ( const enabled of [ false, true ] ) {

					const backPass = pass( scene, camera );
					backPass.overrideMaterial = new MeshBasicNodeMaterial( { side: BackSide } );
					const scenePass = pass( scene, camera );
					scenePass.setMRT( mrt( { output, normal: normalView } ) );
					const reflections = ssr( scenePass.getTextureNode( 'output' ), scenePass.getTextureNode( 'depth' ), scenePass.getTextureNode( 'normal' ), {
						camera, metalnessNode: float( 1 ), roughnessNode: float( 0 ), backDepthNode: enabled ? backPass.getTextureNode( 'depth' ) : null
					} );
					reflections.maxDistance.value = 20;
					reflections.quality.value = 1;
					reflections.thickness.value = 0.001;
					const pipeline = new RenderPipeline( renderer );
					pipeline.outputNode = vec4( reflections.rgb, 1 );

					try {

						pipeline.render();
						const pixels = await renderer.readRenderTargetPixelsAsync( reflections._ssrRenderTarget, 0, 0, 320, 320 );
						let reflectedPixels = 0;
						// This region is on the black receiver, outside the directly visible red box.
						for ( let y = 110; y < 210; y ++ ) {

							for ( let x = 160; x < 230; x ++ ) {

								if ( pixels[ ( y * 320 + x ) * 4 ] > 0 ) reflectedPixels ++;

							}

						}

						if ( enabled ) assert.ok( reflectedPixels > baselinePixels, `Solid crossing recovers pixels: ${reflectedPixels} > ${baselinePixels}.` );
						else baselinePixels = reflectedPixels;

					} finally {

						pipeline.dispose();
						reflections.dispose();
						scenePass.dispose();
						backPass.dispose();
						backPass.overrideMaterial.dispose();

					}

				}

			} finally {

				for ( const mesh of [ receiver, emitter ] ) {

					mesh.geometry.dispose();
					mesh.material.dispose();

				}

				renderer.dispose();

			}

		} );

	} );

} );
