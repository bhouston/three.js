import { DataTexture, FloatType, Mesh, MeshStandardNodeMaterial, NodeMaterial, OrthographicCamera, PlaneGeometry, RenderTarget, Scene } from 'three/webgpu';
import { context, float, indirectDiffuse, indirectSpecular, indirectLightingContext, lightMapDiffuse, texture, uv, vec3, vec4 } from 'three/tsl';
import { getSharedRenderer } from '../tsl/gpu-test-utils.js';

// These tests render and read back actual shader results. Do not accept a
// software adapter as evidence for lighting correctness.
async function hardwareRenderer( assert ) {

	const adapter = await navigator.gpu?.requestAdapter();
	const info = adapter?.info;
	if ( ! info || info.isFallbackAdapter || /swiftshader|llvmpipe|lavapipe|software/i.test( `${ info.vendor } ${ info.device } ${ info.description }` ) ) {

		assert.ok( true, 'SKIPPED: a hardware WebGPU adapter is required.' );
		return null;

	}

	return getSharedRenderer( 'webgpu' );

}

function fixture( renderer ) {

	const previousContext = renderer.contextNode;
	const previousTarget = renderer.getRenderTarget();
	const scene = new Scene();
	const camera = new OrthographicCamera( - 1, 1, 1, - 1, 0.1, 10 );
	camera.position.z = 2;
	const geometry = new PlaneGeometry( 2, 2 );
	geometry.setAttribute( 'uv1', geometry.getAttribute( 'uv' ).clone() );
	const mesh = new Mesh( geometry );
	scene.add( mesh );
	const target = new RenderTarget( 16, 16, { type: FloatType } );

	return {
		async draw( material, lighting = context() ) {

			mesh.material = material;
			renderer.contextNode = lighting;
			renderer.setRenderTarget( target );
			renderer.render( scene, camera );
			return Array.from( await renderer.readRenderTargetPixelsAsync( target, 8, 8, 1, 1 ) );

		},
		dispose() {

			renderer.contextNode = previousContext;
			renderer.setRenderTarget( previousTarget );
			geometry.dispose();
			target.dispose();

		}
	};

}

function close( assert, actual, expected, message ) {

	assert.ok( expected.every( ( value, i ) => Math.abs( value - actual[ i ] ) < 1e-4 ), `${ message }: ${ actual }` );

}

export default QUnit.module( 'Nodes', () => {

	QUnit.module( 'Lighting', () => {

		QUnit.module( 'IndirectLightingNode', () => {

			QUnit.test( 'Rejects mismatched query domains and invalid providers', assert => {

				assert.throws( () => indirectDiffuse( [ vec4( 1 ) ] ), TypeError );
				assert.throws( () => indirectLightingContext( indirectSpecular( [] ) ), TypeError );
				assert.throws( () => indirectLightingContext( null, indirectDiffuse( [] ) ), TypeError );

			} );

			QUnit.test( 'Coverage composition, darkness, missing data, nesting and specular queries', async assert => {

				const renderer = await hardwareRenderer( assert );
				if ( ! renderer ) return;
				const view = fixture( renderer );
				const material = new NodeMaterial();
				const query = { position: vec3( 0 ), normal: vec3( 0, 1, 0 ) };
				const cases = [
					[ 'Remaining coverage', indirectDiffuse( [ () => vec4( 1, 0, 0, 0.8 ), () => vec4( 0, 1, 0, 0.5 ), () => vec4( 0, 0, 1, 1 ) ] ), [ 0.8, 0.1, 0.1, 1 ]],
					[ 'Light map coverage and intensity', indirectDiffuse( [ lightMapDiffuse( vec3( 2, 4, 6 ), 0.25, 0.5 ), () => vec4( 0, 0, 1, 1 ) ] ), [ 0.25, 0.5, 1.5, 1 ]],
					[ 'Black light maps block environment fallback', indirectDiffuse( [ lightMapDiffuse( vec3( 0 ) ), () => vec4( 100, 100, 100, 1 ) ] ), [ 0, 0, 0, 1 ]],
					[ 'Valid black consumes coverage', indirectDiffuse( [ () => vec4( 0, 0, 0, 1 ), () => vec4( 100, 100, 100, 1 ) ] ), [ 0, 0, 0, 1 ]],
					[ 'Missing data permits fallback', indirectDiffuse( [ () => vec4( 100, 0, 0, 0 ), () => vec4( 0, 1, 0, 1 ) ] ), [ 0, 1, 0, 1 ]],
					[ 'Unresolved coverage remains black', indirectDiffuse( [] ), [ 0, 0, 0, 0 ]],
					[ 'Coverage is clamped', indirectDiffuse( [ () => vec4( 100, 0, 0, - 1 ), () => vec4( 0, 1, 0, 2 ) ] ), [ 0, 1, 0, 1 ]],
					[ 'Nested coverage is applied once', indirectDiffuse( [ indirectDiffuse( [ () => vec4( 1, 0, 0, 0.5 ) ] ).asProvider(), () => vec4( 0, 0, 1, 1 ) ] ), [ 0.5, 0, 0.5, 1 ]]
				];

				try {

					for ( const [ name, chain, expected ] of cases ) {

						material.fragmentNode = chain.sample( query );
						material.needsUpdate = true;
						close( assert, await view.draw( material ), expected, name );

					}

					const specular = indirectSpecular( [ q => vec4( q.direction.mul( q.roughness ), 1 ) ] );
					material.fragmentNode = specular.sample( { ...query, viewDirection: vec3( 0, 1, 0 ), roughness: float( 0.5 ) } );
					material.needsUpdate = true;
					close( assert, await view.draw( material ), [ 0, 0.5, 0, 1 ], 'Specular query uses view direction and roughness' );

				} finally {

					material.dispose();
					view.dispose();

				}

			} );

			QUnit.test( 'Material light maps stay additive and specular multiple scattering is preserved', async assert => {

				const renderer = await hardwareRenderer( assert );
				if ( ! renderer ) return;
				const view = fixture( renderer );
				const material = new MeshStandardNodeMaterial( { roughness: 0.5 } );
				const diffuse = indirectDiffuse( [ () => vec4( Math.PI, 0, 0, 1 ) ] );
				const lighting = indirectLightingContext( diffuse, indirectSpecular( [] ) );
				const lightMap = new DataTexture( new Float32Array( [ 0, Math.PI, 0, 1 ] ), 1, 1 );
				lightMap.type = FloatType;
				lightMap.channel = 1;
				lightMap.needsUpdate = true;

				try {

					const unbaked = await view.draw( material, lighting );
					material.lightMap = lightMap;
					material.needsUpdate = true;
					const baked = await view.draw( material, lighting );
					assert.ok( baked[ 1 ] > 0.8, 'The material light map still adds green irradiance' );
					assert.ok( Math.abs( baked[ 0 ] - unbaked[ 0 ] ) < 1e-4, 'The provider chain still supplies the same red irradiance' );

					const bakedOnly = await view.draw( material, indirectLightingContext( indirectDiffuse( [] ), indirectSpecular( [] ) ) );
					assert.ok( bakedOnly[ 0 ] < 1e-4 && Math.abs( bakedOnly[ 1 ] - baked[ 1 ] ) < 1e-4, 'The baked view disables live diffuse GI while keeping the material light map' );

					material.lightMap = null;
					material.needsUpdate = true;
					const atlasChain = indirectDiffuse( [ lightMapDiffuse( texture( lightMap, uv( 1 ) ) ), () => vec4( 100, 0, 0, 1 ) ] );
					const atlasOnly = await view.draw( material, indirectLightingContext( atlasChain, indirectSpecular( [] ) ) );
					close( assert, atlasOnly, bakedOnly, 'An explicit light-map provider matches material irradiance without adding environment twice' );

					material.envNode = vec3( 1 );
					material.needsUpdate = true;
					const legacy = await view.draw( material );
					const resolved = await view.draw( material, indirectLightingContext( indirectDiffuse( [] ) ) );
					assert.ok( resolved[ 0 ] > 0.01 && resolved[ 0 ] < legacy[ 0 ] * 0.2, 'The explicit diffuse chain suppresses environment diffuse while keeping specular' );

					material.metalness = 1;
					material.roughness = 1;
					material.needsUpdate = true;
					const legacyMetal = await view.draw( material );
					const resolvedMetal = await view.draw( material, indirectLightingContext( indirectDiffuse( [] ) ) );
					close( assert, resolvedMetal, legacyMetal, 'Metallic multiple scattering is unchanged' );

				} finally {

					material.dispose();
					lightMap.dispose();
					view.dispose();

				}

			} );

		} );

	} );

} );
