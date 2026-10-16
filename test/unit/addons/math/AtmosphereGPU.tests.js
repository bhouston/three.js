import * as THREE from '../../../../src/Three.WebGPU.js';
import { vec4, equirectDirection, uv } from '../../../../src/Three.TSL.js';
import { atmosphere, transmittance, skyRadiance, sunDirection } from '../../../../examples/jsm/math/Atmosphere.js';
import { AtmosphereNode } from '../../../../examples/jsm/tsl/display/AtmosphereNode.js';

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

		gpuTest( 'transmittance LUT and sky match CPU reference -webonly', async assert => {

			const adapter = await navigator.gpu.requestAdapter();
			assert.ok( adapter && adapter.info.isFallbackAdapter !== true, 'requires a hardware WebGPU adapter' );
			if ( ! adapter || adapter.info.isFallbackAdapter ) return;
			const device = await adapter.requestDevice();
			const renderer = new THREE.WebGPURenderer( { device } );
			const lut = new THREE.RenderTarget( 256, 128, { type: THREE.HalfFloatType, depthBuffer: false } );
			const sky = new THREE.RenderTarget( 512, 256, { type: THREE.HalfFloatType, depthBuffer: false } );
			const node = new AtmosphereNode( lut.texture, equirectDirection( uv() ) );
			const material = new THREE.NodeMaterial();
			const quad = new THREE.QuadMesh( material );
			try {

				await renderer.init();
				material.fragmentNode = vec4( node.getTransmittanceLUTNode(), 1 );
				renderer.setRenderTarget( lut );
				quad.render( renderer );
				const pixels = await renderer.readRenderTargetPixelsAsync( lut, 0, 0, 256, 128 );
				const sample = ( u, v, channel ) => {

					const px = Math.max( 0, Math.min( 255, u * 256 - 0.5 ) );
					const py = Math.max( 0, Math.min( 127, v * 128 - 0.5 ) );
					const x = Math.floor( px ), y = Math.floor( py ), fx = px - x, fy = py - y;
					const get = ( dx, dy ) => THREE.DataUtils.fromHalfFloat( pixels[ ( Math.min( y + dy, 127 ) * 256 + Math.min( x + dx, 255 ) ) * 4 + channel ] );
					return ( get( 0, 0 ) * ( 1 - fx ) + get( 1, 0 ) * fx ) * ( 1 - fy ) + ( get( 0, 1 ) * ( 1 - fx ) + get( 1, 1 ) * fx ) * fy;

				};

				for ( const altitude of [ 0, 1, 2, 5, 10, 20, 30, 45, 60, 90 ] ) {

					const mu = Math.sin( altitude * Math.PI / 180 );
					const reference = transmittance( atmosphere.planetRadius + atmosphere.observerAltitude, mu );
					for ( let c = 0; c < 3; c ++ ) {

						const value = sample( 0.5 + Math.sqrt( mu ) * 0.5, Math.sqrt( atmosphere.observerAltitude / ( atmosphere.atmosphereRadius - atmosphere.planetRadius ) ), c );
						assert.ok( Math.abs( value / reference[ c ] - 1 ) < 0.01, 'LUT within 1% at altitude ' + altitude + ', channel ' + c );

					}

				}

				node.preExposure.value = 1 / 65536;
				const sun = sunDirection( 32, 235 );
				node.sunDirection.value.fromArray( sun );
				material.fragmentNode = vec4( node, 1 );
				material.needsUpdate = true;
				renderer.setRenderTarget( sky );
				quad.render( renderer );
				for ( const [ x, y ] of [[ 256, 192 ], [ 64, 180 ], [ 400, 150 ], [ 256, 255 ]] ) {

					const h = ( ( y + 0.5 ) / 256 - 0.5 ) * Math.PI;
					const a = ( ( x + 0.5 ) / 512 - 0.5 ) * 2 * Math.PI;
					const reference = skyRadiance( [ Math.cos( h ) * Math.cos( a ), Math.sin( h ), Math.cos( h ) * Math.sin( a ) ], sun );
					const result = await renderer.readRenderTargetPixelsAsync( sky, x, y, 1, 1 );
					for ( let c = 0; c < 3; c ++ ) assert.ok( Math.abs( THREE.DataUtils.fromHalfFloat( result[ c ] ) / node.preExposure.value / reference[ c ] - 1 ) < 0.02, 'sky radiance within 2% of CPU reference' );

				}

			} finally {

				renderer.setRenderTarget( null );
				material.dispose();
				lut.dispose();
				sky.dispose();
				renderer.dispose();
				device.destroy();

			}

		} );

	} );

} );
