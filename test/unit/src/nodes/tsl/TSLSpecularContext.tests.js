import { WebGPURenderer, Scene, OrthographicCamera, Mesh, PlaneGeometry, MeshStandardNodeMaterial, RenderTarget, UnsignedByteType, NoColorSpace, DirectionalLight } from 'three/webgpu';
import { builtinSpecularContext, context, vec3 } from 'three/tsl';

QUnit.module( 'Nodes', () => {

	QUnit.test( 'builtinSpecularContext replaces resolved specular and preserves other lighting', async ( assert ) => {

		const renderer = new WebGPURenderer();
		await renderer.init();
		const target = new RenderTarget( 1, 1, { type: UnsignedByteType, depthBuffer: false } );
		target.texture.colorSpace = NoColorSpace;
		const scene = new Scene();
		const camera = new OrthographicCamera( - 1, 1, 1, - 1, 0.1, 10 );
		camera.position.z = 1;
		const material = new MeshStandardNodeMaterial( { metalness: 1, roughness: 0.5 } );
		material.envNode = vec3( 1 );
		const geometry = new PlaneGeometry( 2, 2 );
		scene.add( new Mesh( geometry, material ) );
		renderer.setRenderTarget( target );
		renderer.setSize( 1, 1 );

		async function render( radiance ) {

			renderer.contextNode = radiance === null ? context() : builtinSpecularContext( radiance );
			material.needsUpdate = true;
			renderer.render( scene, camera );
			return renderer.readRenderTargetPixelsAsync( target, 0, 0, 1, 1 );

		}

		try {

			const environment = await render( null );
			assert.true( environment[ 0 ] > 100, 'Environment lights the metal without a context' );
			const black = await render( vec3( 0 ) );
			assert.equal( black[ 0 ], 0, 'Zero replacement removes environment specular, including multiscattering' );
			const red = await render( vec3( 0.25, 0, 0 ) );
			assert.true( Math.abs( red[ 0 ] - 64 ) <= 1 && red[ 1 ] === 0, 'Resolved radiance is used once without BRDF weighting or environment addition' );
			material.envNode = null;
			const withoutEnvironment = await render( vec3( 0.25, 0, 0 ) );
			assert.true( Math.abs( withoutEnvironment[ 0 ] - 64 ) <= 1, 'Replacement works without an environment' );
			material.envNode = vec3( 1 );
			material.metalness = 0;
			const diffuse = await render( vec3( 0 ) );
			assert.true( diffuse[ 0 ] > 0, 'Environment diffuse is preserved' );
			const light = new DirectionalLight( 0xffffff, 1 );
			light.position.z = 1;
			scene.add( light );
			const direct = await render( vec3( 0 ) );
			assert.true( direct[ 0 ] > diffuse[ 0 ], 'Direct lighting is preserved' );
			scene.remove( light );
			material.metalness = 1;
			material.transparent = true;
			const transparent = await render( vec3( 0 ) );
			assert.true( transparent[ 0 ] > 100, 'Transparent materials retain environment reflections' );

		} finally {

			geometry.dispose();
			material.dispose();
			target.dispose();
			renderer.dispose();

		}

	} );

} );
