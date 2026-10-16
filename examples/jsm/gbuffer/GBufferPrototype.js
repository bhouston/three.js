import * as THREE from 'three/webgpu';
import { pass, mrt, output, normalView, materialRoughness, materialMetalness, diffuseColor, velocity, octEncode, octDecode, sample, vec2, vec3, vec4, float, roughness, specularColorBlended, specularF90, positionViewDirection, screenUV, reflect, mix, builtinAOContext, uv, equirectDirection } from 'three/tsl';
import EnvironmentBRDF from '../../../src/nodes/functions/BSDF/EnvironmentBRDF.js';
import { OrbitControls } from '../controls/OrbitControls.js';
import { GUI } from '../libs/lil-gui.module.min.js';
import { LightProbeGrid } from '../lighting/LightProbeGrid.js';
import { SpecularProbe } from '../lighting/SpecularProbe.js';
import { SpecularProbes } from '../lighting/SpecularProbes.js';
import { specularProbes } from '../tsl/lighting/SpecularProbesNode.js';
import { AtmosphereNode } from '../tsl/display/AtmosphereNode.js';
import { sunDirection, sunLightState } from '../math/Atmosphere.js';
import { ao, specularOcclusion } from '../tsl/display/GTAONode.js';
import { hiZ } from '../tsl/display/HiZNode.js';
import { ssr } from '../tsl/display/SSRNode.js';
import { temporalReproject } from '../tsl/display/TemporalReprojectNode.js';
import { recurrentDenoise } from '../tsl/display/RecurrentDenoiseNode.js';
import { traa } from '../tsl/display/TRAANode.js';
import { createGBufferScene } from './GBufferScenes.js';
import { enableDepthPrepass } from './DepthPrepass.js';

const status = document.getElementById( 'status' );
const query = new URLSearchParams( location.search );
const params = {
	scene: query.get( 'scene' ) || 'Cornell', preset: query.get( 'preset' ) || 'High',
	depthPrepass: true, hiZ: true, gtao: true, ssr: true, denoise: true, traa: true,
	gi: true, visibility: true, probes: true, view: 'Beauty', hiZLevel: 3,
	exposure: 1, quality: 0.25, aoRadius: 0.6
};
const renderer = new THREE.WebGPURenderer();
renderer.setPixelRatio( Math.min( devicePixelRatio, 1.5 ) );
renderer.setSize( innerWidth, innerHeight );
renderer.shadowMap.enabled = true;
renderer.toneMapping = THREE.AgXToneMapping;
document.body.appendChild( renderer.domElement );
const camera = new THREE.PerspectiveCamera( 55, innerWidth / innerHeight, 0.1, 80 );
const controls = new OrbitControls( camera, renderer.domElement );
controls.enableDamping = true;
const owned = [];
let scene, pipeline, gbuffer, grids, reflectionSystem, skyTarget, environment;
let busy = false;

init().catch( error => {

	status.textContent = error.message;
	console.error( error );

} );

async function init() {

	const adapter = await navigator.gpu?.requestAdapter( { powerPreference: 'high-performance' } );
	if ( ! adapter ) throw new Error( 'No hardware WebGPU adapter available.' );
	const limit = Math.min( 32, adapter.limits.maxSampledTexturesPerShaderStage );
	renderer.backend.parameters.requiredLimits.maxSampledTexturesPerShaderStage = limit;
	if ( limit < 24 ) params.preset = 'Low';
	await renderer.init();
	if ( ! renderer.backend.isWebGPUBackend ) throw new Error( 'This prototype requires hardware WebGPU.' );
	const info = renderer.backend.device.adapterInfo;
	const adapterName = [ info?.vendor, info?.architecture, info?.device, info?.description ].filter( Boolean ).join( ' ' );
	if ( /swiftshader|llvmpipe|lavapipe|software|basic render/i.test( adapterName ) ) throw new Error( 'Hardware GPU required: ' + adapterName );
	if ( ! adapterName.trim() ) throw new Error( 'Cannot confirm hardware GPU identity.' );
	window.gbufferAdapter = adapterName;
	await createSky();
	await loadScene();
	const gui = new GUI( { title: 'G-buffer prototype' } );
	gui.add( params, 'scene', [ 'Cornell', 'Two rooms', 'GTAO' ] ).onChange( () => {

		const url = new URL( location.href );
		url.searchParams.set( 'scene', params.scene );
		url.searchParams.set( 'preset', params.preset );
		location.href = url.href;

	} );
	gui.add( params, 'preset', [ 'High', 'Low' ] ).onChange( buildPipeline );
	for ( const name of [ 'depthPrepass', 'hiZ', 'gtao', 'ssr', 'denoise', 'traa', 'probes' ] ) gui.add( params, name ).onChange( buildPipeline );
	gui.add( params, 'gi' ).onChange( value => {

		for ( const grid of grids ) grid.visible = value;
		buildPipeline();

	} );
	gui.add( params, 'visibility' ).onChange( value => {

		for ( const grid of grids ) grid.visibility = value;
		buildPipeline();

	} );
	gui.add( params, 'view', [ 'Beauty', 'Depth', 'Normal', 'Albedo', 'Roughness', 'Velocity', 'Hi-Z', 'AO', 'Bent normals', 'SSR' ] ).onChange( buildPipeline );
	gui.add( params, 'hiZLevel', 0, 10, 1 ).onChange( buildPipeline );
	gui.add( params, 'quality', 0.05, 1 ).onFinishChange( buildPipeline );
	gui.add( params, 'aoRadius', 0.1, 2 ).onFinishChange( buildPipeline );
	gui.add( params, 'exposure', 0.1, 4 ).onChange( value => renderer.toneMappingExposure = value );
	window.addEventListener( 'resize', () => {

		camera.aspect = innerWidth / innerHeight;
		camera.updateProjectionMatrix();
		renderer.setSize( innerWidth, innerHeight );

	} );
	renderer.setAnimationLoop( () => {

		if ( busy ) return;
		controls.update();
		pipeline.render();
		window.gbufferFrames = ( window.gbufferFrames || 0 ) + 1;

	} );
	window.gbufferSet = values => {

		Object.assign( params, values );
		buildPipeline();

	};

}

async function createSky() {

	const lut = new THREE.RenderTarget( 256, 128, { type: THREE.HalfFloatType, depthBuffer: false } );
	skyTarget = new THREE.RenderTarget( 512, 256, { type: THREE.HalfFloatType, depthBuffer: false } );
	skyTarget.texture.mapping = THREE.EquirectangularReflectionMapping;
	skyTarget.texture.wrapS = THREE.RepeatWrapping;
	const atmosphere = new AtmosphereNode( lut.texture, equirectDirection( uv() ) );
	atmosphere.sunDirection.value.fromArray( sunDirection( 35, 140 ) );
	// Store sky and sun in the same scene-referred scale, preserving local lights.
	atmosphere.preExposure.value = 0.0001;
	const material = new THREE.NodeMaterial();
	const quad = new THREE.QuadMesh( material );
	material.fragmentNode = vec4( atmosphere.getTransmittanceLUTNode(), 1 );
	renderer.setRenderTarget( lut );
	quad.render( renderer );
	const skyMaterial = new THREE.NodeMaterial();
	skyMaterial.fragmentNode = vec4( atmosphere, 1 );
	quad.material = skyMaterial;
	renderer.setRenderTarget( skyTarget );
	quad.render( renderer );
	renderer.setRenderTarget( null );
	const pmrem = new THREE.PMREMGenerator( renderer );
	environment = pmrem.fromEquirectangular( skyTarget.texture );
	pmrem.dispose();
	lut.dispose();
	material.dispose();
	skyMaterial.dispose();

}

async function loadScene() {

	busy = true;
	const data = await createGBufferScene( params.scene );
	scene = data.scene;
	scene.environment = environment.texture;
	scene.background = skyTarget.texture;
	camera.position.copy( data.cameraPosition );
	controls.target.copy( data.target );
	controls.update();
	const sunState = sunLightState( 35 );
	const sun = new THREE.DirectionalLight();
	sun.color.setRGB( ...sunState.color );
	sun.intensity = sunState.illuminance * 0.0001;
	sun.position.fromArray( sunDirection( 35, 140 ) ).multiplyScalar( 25 );
	sun.castShadow = true;
	sun.shadow.mapSize.setScalar( 2048 );
	Object.assign( sun.shadow.camera, { left: - 16, right: 16, top: 16, bottom: - 16, near: 0.1, far: 70 } );
	sun.shadow.normalBias = 0.02;
	scene.add( sun );
	grids = [];
	status.textContent = 'Baking visibility-aware diffuse GI…';
	for ( const volume of data.probeVolumes ) {

		const size = volume.box.getSize( new THREE.Vector3() );
		const grid = new LightProbeGrid( size.x, size.y, size.z, 4, 4, 4 );
		grid.position.copy( volume.position );
		grid.visibility = true;
		await grid.bake( renderer, scene, { cubemapSize: 16, near: 0.05, far: 40 } );
		grids.push( grid );

	}

	scene.add( ...grids );
	status.textContent = 'Capturing and filtering specular probes…';
	reflectionSystem = new SpecularProbes( data.probeVolumes.map( volume => new SpecularProbe( {
		position: volume.position, influenceBox: volume.box.clone().expandByScalar( 0.4 ),
		projectionBox: volume.box, blendDistance: 0.5
	} ) ), skyTarget.texture );
	await reflectionSystem.capture( renderer, scene, { resolution: 64, filter: 'PMREM' } );
	buildPipeline();
	busy = false;

}

function buildPipeline() {

	if ( ! scene ) return;
	if ( pipeline ) pipeline.dispose();
	for ( const node of owned ) node.dispose();
	owned.length = 0;
	const keep = node => {

		owned.push( node );
		return node;

	};

	const high = params.preset === 'High';
	pipeline = new THREE.RenderPipeline( renderer );
	gbuffer = keep( enableDepthPrepass( pass( scene, camera ) ) );
	gbuffer.depthPrepass = params.depthPrepass;
	gbuffer.name = 'G-buffer (depth pre-pass + equal)';
	const attachments = { output };
	if ( high || params.traa || params.view === 'Velocity' ) attachments.velocity = velocity;
	if ( high ) Object.assign( attachments, {
		normal: octEncode( normalView ), albedo: vec4( diffuseColor.rgb, materialMetalness ), roughness: materialRoughness
	} );
	gbuffer.setMRT( mrt( attachments ) );
	const depth = gbuffer.getTextureNode( 'depth' );
	const motion = attachments.velocity ? gbuffer.getTextureNode( 'velocity' ) : vec2( 0 );
	const color = gbuffer.getTextureNode();
	if ( attachments.velocity ) motion.value.format = THREE.RGFormat;
	depth.value.type = THREE.FloatType;
	const probeRadiance = specularProbes( reflectionSystem, roughness );
	const probeSpecular = () => probeRadiance.mul( EnvironmentBRDF( {
		dotNV: normalView.dot( positionViewDirection ).clamp(), specularColor: specularColorBlended, specularF90, roughness
	} ) );
	if ( params.probes ) gbuffer.contextNode = builtinAOContext( float( 1 ) );
	if ( params.probes ) gbuffer.contextNode.value.getIndirectSpecular = () => probeSpecular();
	let result = color;
	let normal, albedo, rough, pyramid, gtao, reflections;
	if ( high ) {

		const packed = gbuffer.getTextureNode( 'normal' );
		packed.value.format = THREE.RGFormat;
		packed.value.minFilter = packed.value.magFilter = THREE.NearestFilter;
		normal = sample( coord => octDecode( packed.sample( coord ).rg ) );
		albedo = gbuffer.getTextureNode( 'albedo' );
		albedo.value.type = THREE.UnsignedByteType;
		rough = gbuffer.getTextureNode( 'roughness' );
		rough.value.format = THREE.RedFormat;
		rough.value.type = THREE.UnsignedByteType;
		pyramid = keep( hiZ( depth ) );
		pyramid.level.value = params.hiZLevel;
		gtao = keep( ao( depth, normal, camera ) );
		gtao.resolutionScale = 0.5;
		gtao.radius.value = params.aoRadius;
		let filtered;
		if ( params.ssr ) {

			reflections = keep( ssr( color, depth, normal, {
				stochastic: true, hiZ: params.hiZ ? pyramid : null,
				diffuseNode: albedo, metalnessNode: albedo.a, roughnessNode: rough.r
			} ) );
			reflections.quality.value = params.quality;
			reflections.maxDistance.value = 20;
			reflections.thickness.value = 0.15;
			filtered = reflections;
			if ( params.denoise ) {

				const temporal = keep( temporalReproject( reflections, depth, packed, motion, camera, { mode: 'specular', normalEncoding: 'octahedral', accumulate: false } ) );
				filtered = keep( recurrentDenoise( temporal, camera, {
					depth, normal: packed, raw: reflections,
					metalRoughness: sample( coord => vec2( albedo.sample( coord ).a, rough.sample( coord ).r ) ),
					mode: 'specular', normalEncoding: 'octahedral', accumulate: true
				} ) );
				filtered.alphaSource = 'raylength';
				temporal.setHistoryTexture( filtered );

			}

		}

		// Separate source and resolve draws avoid a dependency cycle through SSR.
		// SSR returns weighted outgoing light; the probe fallback is weighted once here.
		const beauty = keep( pass( scene, camera ) );
		const visibility = params.gtao ? gtao.getTextureNode().sample( screenUV ).r : float( 1 );
		beauty.contextNode = builtinAOContext( visibility );
		const context = beauty.contextNode.value;
		context.getIndirectSpecular = original => {

			const fallback = params.probes ? probeSpecular() : original;
			return filtered ? mix( fallback, filtered.rgb, float( reflections.a.greaterThan( 0 ) ) ) : fallback;

		};

		if ( params.gtao ) context.getSpecularOcclusion = ( original, n, r, v ) => specularOcclusion( visibility, gtao.getBentNormalNode(), reflect( v.negate(), n ), r );
		result = beauty;

	}

	if ( params.traa ) result = keep( traa( result, depth, motion, camera ) );
	const views = {
		Depth: vec4( vec3( gbuffer.getLinearDepthNode() ), 1 ),
		Velocity: vec4( motion.rg.mul( 10 ).add( 0.5 ), 0, 1 )
	};
	if ( high ) Object.assign( views, {
		Normal: vec4( normal.mul( 0.5 ).add( 0.5 ), 1 ), Albedo: albedo, Roughness: vec4( vec3( rough.r ), 1 ),
		'Hi-Z': vec4( vec3( pyramid ), 1 ),
		AO: vec4( vec3( gtao.r ), 1 ), 'Bent normals': vec4( gtao.getBentNormalNode().mul( 0.5 ).add( 0.5 ), 1 ),
		SSR: reflections ? vec4( reflections.rgb, 1 ) : color
	} );
	pipeline.outputNode = views[ params.view ] || result;
	status.textContent = `${ params.scene } · ${ params.preset } · ${ window.gbufferAdapter }\n${ high ? 'Depth → MRT → Hi-Z / GTAO / SSR → material resolve → TRAA' : 'Depth → beauty + velocity → TRAA' }\nPrototype: synchronous probe bakes; approximate probe BRDF; opaque geometry only. No SSR demodulation.`;
	window.gbufferReady = true;

}


