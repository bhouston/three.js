import { AmbientLight, DataUtils, Group, Mesh, MeshBasicMaterial, PlaneGeometry } from 'three/webgpu';
import { indirectDiffuse, indirectLightingContext, vec3, vec4 } from 'three/tsl';
import { LightMapper } from '../../../../examples/jsm/lighting/LightMapper.js';
import { getSharedRenderer } from '../../src/nodes/tsl/gpu-test-utils.js';

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'Lighting', () => {

		QUnit.module( 'LightMapper', () => {

			QUnit.test( 'Bakes a diffuse chain with geometry normals and progressive sample indices', async assert => {

				const adapter = await navigator.gpu?.requestAdapter();
				const info = adapter?.info;
				if ( ! info || info.isFallbackAdapter || /swiftshader|llvmpipe|lavapipe|software/i.test( `${ info.vendor } ${ info.device } ${ info.description }` ) ) {

					assert.ok( true, 'SKIPPED: a hardware WebGPU adapter is required.' );
					return;

				}

				const renderer = await getSharedRenderer( 'webgpu' );
				if ( ! renderer ) return;
				const geometry = new PlaneGeometry( 2, 2 );
				geometry.setAttribute( 'uv1', geometry.getAttribute( 'uv' ).clone() );
				const material = new MeshBasicMaterial();
				const mesh = new Mesh( geometry, material );
				const diffuse = indirectDiffuse( [ q => vec4( q.sampleIndex.add( 1 ), 0, q.normal.z.abs().mul( Math.PI ), 1 ) ] );
				const mappers = [
					new LightMapper( renderer, 16, diffuse ),
					new LightMapper( renderer, 16, diffuse, { lights: [ new AmbientLight( 0xffffff, 0 ) ] } )
				];
				const previousContext = renderer.contextNode;
				renderer.contextNode = indirectLightingContext( indirectDiffuse( [ () => vec4( 100, 100, 100, 1 ) ] ) );
				const previousTarget = renderer.getRenderTarget();

				try {

					for ( const mapper of mappers ) {

						mapper.addObjects( [ mesh ] );
						mapper.update( 4 );
						const pixel = await renderer.readRenderTargetPixelsAsync( mapper._renderTarget, 8, 8, 1, 1 );
						const value = Array.from( pixel, DataUtils.fromHalfFloat );
						assert.ok( Math.abs( value[ 0 ] - 2.5 ) < 0.005, 'The chain receives each progressive sample index' );
						assert.ok( Math.abs( value[ 2 ] - Math.PI ) < 0.005, 'The chain receives world-space geometry normals and stores irradiance without dividing by pi' );

					}

				} finally {

					renderer.setRenderTarget( previousTarget );
					renderer.contextNode = previousContext;
					for ( const mapper of mappers ) mapper.dispose();
					geometry.dispose();
					material.dispose();

				}

			} );

			QUnit.test( 'Uses current world transforms without rendering the source scene first', assert => {

				const parent = new Group();
				parent.position.set( 2, 3, 4 );
				const geometry = new PlaneGeometry();
				geometry.setAttribute( 'uv1', geometry.getAttribute( 'uv' ).clone() );
				const material = new MeshBasicMaterial();
				const mesh = new Mesh( geometry, material );
				mesh.position.set( 5, 6, 7 );
				parent.add( mesh );
				let bakedPosition;
				const renderer = {
					autoClear: true,
					getRenderTarget: () => null,
					getActiveCubeFace: () => 0,
					getActiveMipmapLevel: () => 0,
					getRenderObjectFunction: () => null,
					getPixelRatio: () => 1,
					getMRT: () => null,
					getClearColor: value => value.set( 0 ),
					getClearAlpha: () => 1,
					getScissorTest: () => false,
					setRenderTarget: () => {},
					setRenderObjectFunction: () => {},
					setPixelRatio: () => {},
					setMRT: () => {},
					setClearColor: () => {},
					setScissorTest: () => {},
					clear: () => {},
					render: scene => {

						if ( scene.isScene ) bakedPosition = scene.children[ 0 ].matrix.elements.slice( 12, 15 );

					}
				};
				const mapper = new LightMapper( renderer, 32, () => vec3( 1 ) );
				mapper.addObjects( [ mesh ] );
				mapper.update();
				assert.deepEqual( bakedPosition, [ 7, 9, 11 ] );
				mapper.dispose();
				geometry.dispose();
				material.dispose();

			} );

		} );

	} );

} );
