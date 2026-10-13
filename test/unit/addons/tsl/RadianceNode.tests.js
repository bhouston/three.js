import { DataUtils, Mesh, MeshStandardNodeMaterial, PerspectiveCamera, PlaneGeometry, RenderPipeline, Scene, WebGPURenderer } from 'three/webgpu';
import { context, pass, vec3 } from 'three/tsl';

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'TSL', () => {

		QUnit.test( 'Indirect specular radiance replaces environment specular -webonly', async ( assert ) => {

			const adapter = await navigator.gpu?.requestAdapter();
			if ( ! adapter || adapter.isFallbackAdapter || /swiftshader|llvmpipe|lavapipe|software/i.test( JSON.stringify( adapter.info ) ) ) {

				assert.ok( true, 'SKIPPED: hardware WebGPU required.' );
				return;

			}

			const renderer = new WebGPURenderer();
			renderer.setSize( 32, 32 );
			await renderer.init();
			const camera = new PerspectiveCamera( 45, 1, 0.1, 10 );
			camera.position.z = 3;
			let baselineRed = 0;
			try {

				for ( const mode of [ 'default', 'null', 'replace', 'dielectric', 'no-environment' ] ) {

					const scene = new Scene();
					const material = new MeshStandardNodeMaterial( { metalness: mode === 'dielectric' ? 0 : 1, roughness: 0.3 } );
					if ( mode !== 'no-environment' ) material.envNode = vec3( 1, 0, 0 );
					const mesh = new Mesh( new PlaneGeometry( 4, 4 ), material );
					scene.add( mesh );
					const scenePass = pass( scene, camera );
					if ( mode !== 'default' ) scenePass.contextNode = context( null, { getRadiance: () => mode === 'null' ? null : vec3( 0, 1, 0 ) } );
					const pipeline = new RenderPipeline( renderer );
					pipeline.outputNode = scenePass;
					try {

						pipeline.render();
						const buffer = await renderer.readRenderTargetPixelsAsync( scenePass.renderTarget, 16, 16, 1, 1 );
						const pixel = buffer instanceof Uint16Array ? Array.from( buffer, DataUtils.fromHalfFloat ) : buffer;
						if ( mode === 'default' ) baselineRed = pixel[ 0 ];
						if ( mode === 'no-environment' ) {

							assert.strictEqual( pixel[ 0 ], 0, 'No environment contributes red.' );
							assert.ok( pixel[ 1 ] > 0, 'Radiance hook also works without an environment.' );

						} else if ( mode === 'replace' ) {

							assert.ok( pixel[ 0 ] < baselineRed * 0.1, 'Red single-scattering environment specular is replaced; its small multiscattering term remains.' );
							assert.ok( pixel[ 1 ] > 0, 'Green incoming radiance is shaded by the material.' );

						} else if ( mode === 'dielectric' ) {

							assert.ok( pixel[ 0 ] > 0, 'Red environment diffuse is retained.' );
							assert.ok( pixel[ 1 ] > 0, 'Dielectric specular receives the green radiance.' );

						} else {

							assert.ok( pixel[ 0 ] > 0, 'Default and null hook preserve red environment specular.' );
							assert.strictEqual( pixel[ 1 ], 0, 'No replacement was applied.' );

						}

					} finally {

						pipeline.dispose(); scenePass.dispose(); mesh.geometry.dispose(); material.dispose();

					}

				}

			} finally {

				renderer.dispose();

			}

		} );

	} );

} );
