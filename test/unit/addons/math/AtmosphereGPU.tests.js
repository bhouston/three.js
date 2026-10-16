import * as THREE from 'three/webgpu';

QUnit.module( 'Maths', () => {

	QUnit.module( 'Atmosphere lighting units', () => {

		const gpuTest = navigator.gpu ? QUnit.test : QUnit.skip;

		gpuTest( 'physical radiance conventions -webonly', async assert => {

			const adapter = await navigator.gpu.requestAdapter();
			assert.ok( adapter && adapter.info.isFallbackAdapter !== true, 'requires a hardware WebGPU adapter' );
			if ( ! adapter || adapter.info.isFallbackAdapter ) return;

			const requiredFeatures = adapter.features.has( 'float32-filterable' ) ? [ 'float32-filterable' ] : [];
			const device = await adapter.requestDevice( { requiredFeatures } );
			const renderer = new THREE.WebGPURenderer( { device } );
			const target = new THREE.RenderTarget( 4, 4, { type: THREE.FloatType } );
			const generator = new THREE.PMREMGenerator( renderer );
			const geometry = new THREE.PlaneGeometry( 2, 2 );
			const material = new THREE.MeshPhysicalNodeMaterial( { color: 0xffffff, roughness: 1, metalness: 0, specularIntensity: 0 } );
			const data = new Uint16Array( 32 * 16 * 4 ).fill( THREE.DataUtils.toHalfFloat( 0.5 ) );
			for ( let i = 3; i < data.length; i += 4 ) data[ i ] = THREE.DataUtils.toHalfFloat( 1 );
			const texture = new THREE.DataTexture( data, 32, 16, THREE.RGBAFormat, THREE.HalfFloatType );
			texture.mapping = THREE.EquirectangularReflectionMapping;
			texture.needsUpdate = true;
			let environment;

			try {

				await renderer.init();
				renderer.toneMapping = THREE.NoToneMapping;
				const scene = new THREE.Scene();
				const camera = new THREE.OrthographicCamera( - 1, 1, 1, - 1, 0.1, 10 );
				camera.position.z = 2;
				scene.add( new THREE.Mesh( geometry, material ) );
				const sun = new THREE.DirectionalLight( 0xffffff, 2 );
				sun.position.z = 1;
				sun.position.y = 0;
				scene.add( sun );
				renderer.setRenderTarget( target );
				renderer.render( scene, camera );
				const direct = await renderer.readRenderTargetPixelsAsync( target, 1, 1, 1, 1 );
				for ( let c = 0; c < 3; c ++ ) assert.ok( Math.abs( direct[ c ] - 2 / Math.PI ) < 1e-5, 'E lux gives E / PI on a white diffuse surface' );

				scene.remove( sun );
				environment = generator.fromEquirectangular( texture );
				scene.environment = environment.texture;
				renderer.render( scene, camera );
				const indirect = await renderer.readRenderTargetPixelsAsync( target, 1, 1, 1, 1 );
				for ( let c = 0; c < 3; c ++ ) assert.ok( Math.abs( indirect[ c ] - 0.5 ) < 1e-3, 'uniform environment L gives outgoing diffuse radiance L' );

			} finally {

				renderer.setRenderTarget( null );
				if ( environment ) environment.dispose();
				generator.dispose();
				target.dispose();
				texture.dispose();
				material.dispose();
				geometry.dispose();
				renderer.dispose();
				device.destroy();

			}

		} );

	} );

} );
