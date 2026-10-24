import { BoxGeometry, DataTexture, EquirectangularReflectionMapping, FloatType, Mesh, MeshStandardNodeMaterial, NodeMaterial, OrthographicCamera, PlaneGeometry, RenderTarget, RGBAFormat, Scene, Vector3 } from 'three/webgpu';
import { Fn, context, float, indirectDiffuse, indirectSpecular, vec3, vec4 } from 'three/tsl';
import { ReflectionProbe, ReflectionProbes } from '../../../../examples/jsm/lighting/ReflectionProbes.js';
import { reflectionProbesSpecular } from '../../../../examples/jsm/tsl/lighting/ReflectionProbesNode.js';
import { LightProbeGrid } from '../../../../examples/jsm/lighting/LightProbeGrid.js';
import { lightProbeGridDiffuse } from '../../../../examples/jsm/tsl/lighting/LightProbeGridNode.js';
import { VXGIVolume } from '../../../../examples/jsm/lighting/vxgi/VXGIVolume.js';
import { gather } from '../../../../examples/jsm/lighting/vxgi/VXGIConeTracer.js';
import { vxgiDiffuse } from '../../../../examples/jsm/lighting/vxgi/VXGIIrradiance.js';
import { getSharedRenderer } from '../../src/nodes/tsl/gpu-test-utils.js';

QUnit.module( 'Addons / Lighting / Indirect transport', () => {

	QUnit.test( 'Sky is a visible transport boundary, darkness is covered, and rebakes update existing shaders', async assert => {

		const info = ( await navigator.gpu?.requestAdapter() )?.info;
		if ( ! info || info.isFallbackAdapter || /swiftshader|llvmpipe|lavapipe|software/i.test( `${ info.vendor } ${ info.device } ${ info.description }` ) ) {

			assert.ok( true, 'SKIPPED: a hardware WebGPU adapter is required.' );
			return;

		}

		const renderer = await getSharedRenderer( 'webgpu' );
		if ( ! renderer ) return;
		const previousContext = renderer.contextNode;
		const previousTarget = renderer.getRenderTarget();
		const cameraContext = context( {
			getAO: () => {

				throw new Error( 'Camera GTAO must not run during a bake' );

			},
			getSpecularOcclusion: () => {

				throw new Error( 'Camera GTSO must not run during a bake' );

			}
		} );
		const scene = new Scene();
		const sky = new DataTexture( new Float32Array( [ 1, 1, 1, 1 ] ), 1, 1, RGBAFormat, FloatType );
		sky.mapping = EquirectangularReflectionMapping;
		sky.needsUpdate = true;
		scene.environment = sky;
		const previousBackgroundNode = scene.backgroundNode;

		const geometry = new BoxGeometry( 4.4, 4.4, 0.2 );
		const wallMaterial = new MeshStandardNodeMaterial( { color: 0xffffff, roughness: 1 } );
		const walls = [];
		for ( let axis = 0; axis < 3; axis ++ ) for ( const sign of [ - 1, 1 ] ) {

			const wall = new Mesh( geometry, wallMaterial );
			if ( axis === 0 ) wall.rotation.y = Math.PI / 2;
			if ( axis === 1 ) wall.rotation.x = Math.PI / 2;
			wall.position.setComponent( axis, sign * 2.1 );
			scene.add( wall );
			walls.push( wall );

		}

		const grid = new LightProbeGrid( 3, 3, 3, 3, 3, 3 );
		grid.visibility = true;
		grid.multiBounce = true;
		scene.add( grid );
		const reflections = new ReflectionProbes();
		reflections.add( new ReflectionProbe( 3, 3, 3 ) );
		const volume = new VXGIVolume( 32 );
		volume.bounds.set( new Vector3( - 2.2, - 2.2, - 2.2 ), new Vector3( 2.2, 2.2, 2.2 ) );
		const output = new Scene();
		const material = new NodeMaterial();
		const quadGeometry = new PlaneGeometry( 2, 2 );
		output.add( new Mesh( quadGeometry, material ) );
		const camera = new OrthographicCamera( - 1, 1, 1, - 1, 0.1, 10 );
		camera.position.z = 2;
		const target = new RenderTarget( 8, 8, { type: FloatType } );
		const query = { position: vec3( 0 ), normal: vec3( 0, 1, 0 ) };
		const draw = async node => {

			if ( material.fragmentNode !== node ) {

				material.fragmentNode = node;
				material.needsUpdate = true;

			}

			renderer.contextNode = context();
			renderer.setRenderTarget( target );
			renderer.render( output, camera );
			return Array.from( await renderer.readRenderTargetPixelsAsync( target, 4, 4, 1, 1 ) );

		};

		try {

			const directional = await draw( Fn( () => {

				const trace = ( origin, direction ) => ( { color: vec3( 0 ), alpha: direction.x.greaterThan( 0 ).select( 1, 0 ) } );
				const environment = direction => direction.x.greaterThan( 0 ).select( vec3( 10, 0, 0 ), vec3( 0, 0, 1 ) );
				const { color } = gather( trace, vec3( 0 ), vec3( 0, 1, 0 ), 128, float( 0 ), float( 1 ), 0, 0.5, null, environment );
				return vec4( color, 1 );

			} )() );
			assert.ok( Math.abs( directional[ 0 ] ) < 0.001 && Math.abs( directional[ 2 ] - 0.5 ) < 0.02, 'A blocked red sky contributes nothing; only the visible blue half is integrated' );

			renderer.contextNode = cameraContext;
			grid.bake( renderer, scene, { cubemapSize: 16, near: 0.05, far: 20, bounces: 2 } );
			assert.strictEqual( renderer.contextNode, cameraContext, 'Diffuse probe baking restores the camera occlusion context' );
			const probe = await draw( lightProbeGridDiffuse( grid )( query ) );
			assert.ok( probe.slice( 0, 3 ).every( x => Math.abs( x ) < 0.001 ), 'A sealed unlit white room receives no sky irradiance, including cached bounces' );
			assert.ok( probe[ 3 ] > 0.99, 'Valid dark probe data remains covered' );
			assert.strictEqual( scene.background, null, 'The display background is restored' );
			assert.strictEqual( scene.backgroundNode, previousBackgroundNode, 'The display background node is restored' );

			renderer.contextNode = cameraContext;
			reflections.bake( renderer, scene );
			assert.strictEqual( renderer.contextNode, cameraContext, 'Reflection baking excludes camera occlusion and restores the caller context' );

			// Seed completed reflection probes, then remove emission. With metallic
			// walls, the diffuse grid can now receive light only through specular feedback.
			wallMaterial.metalness = 1;
			wallMaterial.roughness = 0.2;
			wallMaterial.emissive.setScalar( 1 );
			renderer.contextNode = cameraContext;
			reflections.bake( renderer, scene );
			wallMaterial.emissive.setScalar( 0 );
			grid.bake( renderer, scene, { cubemapSize: 16, near: 0.05, far: 20 } );
			const withoutReflections = await draw( lightProbeGridDiffuse( grid )( query ) );
			assert.ok( withoutReflections.slice( 0, 3 ).every( x => Math.abs( x ) < 0.001 ), 'Metallic walls stay dark when diffuse capture excludes specular feedback' );
			renderer.contextNode = cameraContext;
			grid.bake( renderer, scene, {
				cubemapSize: 16, near: 0.05, far: 20,
				indirectSpecularNode: indirectSpecular( [ reflectionProbesSpecular( reflections ) ] )
			} );
			assert.strictEqual( renderer.contextNode, cameraContext, 'Specular feedback restores the caller context' );
			const withReflections = await draw( lightProbeGridDiffuse( grid )( query ) );
			assert.ok( withReflections.slice( 0, 3 ).every( x => x > 0.1 ), 'Diffuse probes capture radiance reflected by metallic surfaces from completed reflection probes' );
			wallMaterial.metalness = 0;
			wallMaterial.roughness = 1;

			scene.remove( ...walls );
			grid.visible = false;
			renderer.contextNode = cameraContext;
			volume.update( renderer, scene );
			assert.strictEqual( renderer.contextNode, cameraContext, 'Voxel preparation restores the camera occlusion context' );
			const collectLights = volume._collectLights;
			try {

				volume._collectLights = () => {

					throw new Error( 'Voxel preparation failed' );

				};

				renderer.setRenderTarget( target );
				assert.throws( () => volume.update( renderer, scene ), /Voxel preparation failed/ );
				assert.strictEqual( renderer.contextNode, cameraContext, 'A failed voxel preparation restores the camera occlusion context' );
				assert.strictEqual( renderer.getRenderTarget(), target, 'A failed voxel preparation restores the render target' );

			} finally {

				volume._collectLights = collectLights;

			}

			const source = vxgiDiffuse( volume, { coneCount: 16 } );
			const node = indirectDiffuse( [ source, () => vec4( 100, 0, 0, 1 ) ] ).sample( query );
			const empty = await draw( node );
			assert.ok( empty.slice( 0, 3 ).every( x => Math.abs( x - Math.PI ) < 0.01 ), 'Empty space under a unit-white sky receives pi irradiance and blocks extra environment fallback' );

			scene.environmentIntensity = 0.25;
			volume.update( renderer, scene );
			const dimmed = await draw( node );
			assert.ok( dimmed.slice( 0, 3 ).every( x => Math.abs( x - Math.PI / 4 ) < 0.01 ), 'Changing sky intensity updates an existing compiled provider' );

			scene.environment = null;
			volume.update( renderer, scene );
			const dark = await draw( node );
			assert.ok( dark.slice( 0, 3 ).every( x => Math.abs( x ) < 0.001 ) && dark[ 3 ] > 0.99, 'Disabling the sky returns valid black, rather than enabling the bright fallback' );
			const outside = await draw( indirectDiffuse( [ source, () => vec4( 0, 1, 0, 1 ) ] ).sample( { ...query, position: vec3( 20 ) } ) );
			assert.ok( Math.abs( outside[ 1 ] - 1 ) < 0.001, 'Queries outside the voxel region fall back to the next provider' );

			const floor = new Mesh( quadGeometry, wallMaterial );
			floor.rotation.x = - Math.PI / 2;
			floor.scale.setScalar( 4 );
			scene.add( floor );
			scene.environment = sky;
			scene.environmentIntensity = 1;
			volume.needsUpdate = true;
			volume.bounces = 0;
			volume.update( renderer, scene );
			const bounceNode = source( { position: vec3( 0, 0.5, 0 ), normal: vec3( 0, - 1, 0 ) } );
			const unreflected = await draw( bounceNode );
			volume.bounces = 1;
			volume.update( renderer, scene );
			const reflected = await draw( bounceNode );
			assert.ok( reflected[ 0 ] > unreflected[ 0 ] + 0.2, 'A cached bounce includes sky reflected by the floor, without any explicit light' );

		} finally {

			renderer.contextNode = previousContext;
			renderer.setRenderTarget( previousTarget );
			reflections.dispose();
			volume.dispose();
			grid.dispose();
			geometry.dispose();
			wallMaterial.dispose();
			quadGeometry.dispose();
			material.dispose();
			target.dispose();
			sky.dispose();

		}

	} );

} );
