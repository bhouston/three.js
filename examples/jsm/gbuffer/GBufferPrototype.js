import * as THREE from 'three/webgpu';
import { pass, mrt, output, normalView, materialRoughness, materialMetalness, diffuseColor, velocity, octEncode, octDecode, sample, vec2, vec3, vec4, float, roughness, specularColorBlended, specularF90, positionViewDirection, screenUV, reflect, mix, builtinAOContext, builtinGIContext, uv, equirectDirection, perspectiveDepthToViewZ } from 'three/tsl';
import EnvironmentBRDF from '../../../src/nodes/functions/BSDF/EnvironmentBRDF.js';
import { OrbitControls } from '../controls/OrbitControls.js';
import { Inspector } from '../inspector/Inspector.js';
import { ssgi } from '../tsl/display/SSGINode.js';
import { LightProbeGrid } from '../lighting/LightProbeGrid.js';
import { SpecularProbe } from '../lighting/SpecularProbe.js';
import { SpecularProbes } from '../lighting/SpecularProbes.js';
import { specularProbes } from '../tsl/lighting/SpecularProbesNode.js';
import { AtmosphereNode } from '../tsl/display/AtmosphereNode.js';
import { sunDirection, sunLightState } from '../math/Atmosphere.js';
import { ao, specularOcclusion } from '../tsl/display/GTAONode.js';
import { hiZ } from '../tsl/display/HiZNode.js';
import { ssr } from '../tsl/display/SSRNode.js';
import { ENV_RAY_LENGTH_THRESHOLD } from '../tsl/utils/SpecularHelpers.js';
import { temporalReproject } from '../tsl/display/TemporalReprojectNode.js';
import { recurrentDenoise } from '../tsl/display/RecurrentDenoiseNode.js';
import { traa } from '../tsl/display/TRAANode.js';
import { createGBufferScene } from './GBufferScenes.js';
import { enableDepthPrepass } from './DepthPrepass.js';

const status = document.getElementById( 'status' );
const query = new URLSearchParams( location.search );
const params = {
	scene: query.get( 'scene' ) || 'Cornell', preset: query.get( 'preset' ) || 'High',
	depthMode: 'Off', ssrTracing: 'Hi-Z', aoMethod: 'GTAO', specularMethod: 'SSR',
	diffuseMethod: 'Diffuse probes', reflectionDenoise: 'Temporal + spatial', antialiasing: 'TRAA', visibility: true, view: 'Beauty', hiZLevel: 3,
	giIntensity: 1, exposure: 1, quality: 0.25, aoRadius: 0.6
};
const renderer = new THREE.WebGPURenderer();
renderer.inspector = new Inspector();
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
const editors = new Map();
let refreshQueued = false;
let busy = false;
let highSupported = true;

init().catch( error => {

	status.textContent = error.message;
	console.error( error );

} );

async function init() {

	const adapter = await navigator.gpu?.requestAdapter( { powerPreference: 'high-performance' } );
	if ( ! adapter ) throw new Error( 'No hardware WebGPU adapter available.' );
	const limit = Math.min( 32, adapter.limits.maxSampledTexturesPerShaderStage );
	renderer.backend.parameters.requiredLimits.maxSampledTexturesPerShaderStage = limit;
	highSupported = limit >= 24;
	if ( ! highSupported ) params.preset = 'Low';
	await renderer.init();
	if ( ! renderer.backend.isWebGPUBackend ) throw new Error( 'This prototype requires hardware WebGPU.' );
	const info = renderer.backend.device.adapterInfo;
	const adapterName = [ info?.vendor, info?.architecture, info?.device, info?.description ].filter( Boolean ).join( ' ' );
	if ( /swiftshader|llvmpipe|lavapipe|software|basic render/i.test( adapterName ) ) throw new Error( 'Hardware GPU required: ' + adapterName );
	if ( ! adapterName.trim() ) throw new Error( 'Cannot confirm hardware GPU identity.' );
	window.gbufferAdapter = adapterName;
	await createSky();
	applyPreset( params.preset, false );
	await loadScene();
	const gui = renderer.inspector.createParameters( 'G-buffer prototype' );
	const presets = { High: () => applyPreset( 'High' ), Low: () => applyPreset( 'Low' ) };
	gui.add( presets, 'High' ).name( 'High preset' );
	gui.add( presets, 'Low' ).name( 'Low preset' );
	gui.add( params, 'scene', [ 'Cornell', 'Two rooms', 'GTAO' ] ).onChange( () => {

		const url = new URL( location.href );
		url.searchParams.set( 'scene', params.scene );
		url.searchParams.set( 'preset', params.preset );
		location.href = url.href;

	} );
	const choices = {
		diffuseMethod: [ 'None', 'Sky IBL', 'Diffuse probes', 'SSGI' ],
		specularMethod: { None: 'None', 'Specular probes': 'Specular probes', 'SSR (probe fallback)': 'SSR' },
		depthMode: [ 'Off', 'Conservative' ], aoMethod: [ 'Off', 'GTAO' ],
		ssrTracing: [ 'Linear', 'Hi-Z' ], reflectionDenoise: [ 'Off', 'Temporal + spatial' ],
		antialiasing: [ 'Off', 'TRAA' ]
	};
	const labels = { diffuseMethod: 'Diffuse GI', specularMethod: 'Specular reflections', depthMode: 'Depth pre-pass', aoMethod: 'Ambient occlusion', ssrTracing: 'SSR traversal', reflectionDenoise: 'SSR denoising', antialiasing: 'Antialiasing' };
	for ( const [ key, values ] of Object.entries( choices ) ) {

		editors.set( key, gui.add( params, key, values ).name( labels[ key ] ).onChange( changed ) );

	}

	editors.set( 'visibility', gui.add( params, 'visibility' ).name( 'Probe visibility' ).onChange( changed ) );
	gui.add( params, 'view', [ 'Beauty', 'Depth', 'Normal', 'Albedo', 'Roughness', 'Velocity', 'Hi-Z', 'AO', 'Bent normals', 'SSR', 'SSGI' ] ).onChange( changed );
	gui.add( params, 'hiZLevel', 0, 10, 1 ).name( 'Hi-Z debug level' ).onChange( changed );
	gui.add( params, 'quality', 0.05, 1 ).name( 'SSR quality' ).onChange( changed ).debounce( 200 );
	gui.add( params, 'aoRadius', 0.1, 2 ).name( 'AO radius' ).onChange( changed ).debounce( 200 );
	gui.add( params, 'giIntensity', 0, 3, 0.05 ).name( 'Diffuse GI strength' ).onChange( changed ).debounce( 200 );
	gui.add( params, 'exposure', 0.1, 4 ).onChange( value => renderer.toneMappingExposure = value );
	for ( const { key, editor, object } of gui.objects ) {

		if ( object === params ) editors.set( key, editor );

	}

	buildPipeline();
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
	window.gbufferApplyPreset = applyPreset;
	window.gbufferReadBuffer = async name => {

		await renderer.backend.device.queue.onSubmittedWorkDone();
		return renderer.readRenderTargetPixelsAsync( gbuffer.renderTarget, 0, 0, gbuffer.renderTarget.width, gbuffer.renderTarget.height, gbuffer.renderTarget.textures.indexOf( gbuffer.getTexture( name ) ) );

	};

	window.gbufferGetLightingState = () => ( { skyDiffuse: scene.environment !== null, probeGrids: grids.filter( grid => grid.visible ).length } );
	window.gbufferGetControlValues = () => Object.fromEntries( [ ...editors ].map( ( [ key, editor ] ) => [ key, editor.getValue() ] ) );
	window.gbufferGetSettings = () => ( { ...params } );
	window.gbufferWaitForGPU = () => renderer.backend.device.queue.onSubmittedWorkDone();
	window.gbufferSet = values => {

		Object.assign( params, values );
		buildPipeline();
		syncEditors();

	};

}

function syncEditors() {

	for ( const [ key, editor ] of editors ) {

		if ( editor.getValue() !== params[ key ] ) {

			// Update Inspector values without firing each control's rebuild handler.
			const dispatch = editor.dispatchChange;
			editor.dispatchChange = () => {};

			try {

				editor.setValue( params[ key ] );

			} finally {

				editor.dispatchChange = dispatch;

			}

		}

	}

}

function changed() {

	if ( refreshQueued ) return;
	refreshQueued = true;
	requestAnimationFrame( () => {

		refreshQueued = false;
		params.preset = 'Custom';
		buildPipeline();

	} );

}

function applyPreset( name, rebuild = true ) {

	const high = name !== 'Low' && highSupported;
	Object.assign( params, {
		preset: high ? 'High' : 'Low', depthMode: 'Off', diffuseMethod: 'Diffuse probes',
		specularMethod: high ? 'SSR' : 'Specular probes', aoMethod: high ? 'GTAO' : 'Off',
		ssrTracing: high ? 'Hi-Z' : 'Linear', reflectionDenoise: high ? 'Temporal + spatial' : 'Off',
		antialiasing: high ? 'TRAA' : 'Off', visibility: true, view: 'Beauty',
		hiZLevel: 3, quality: high ? 0.25 : 0.1, aoRadius: 0.6, giIntensity: 1, exposure: 1
	} );
	renderer.toneMappingExposure = params.exposure;
	syncEditors();
	if ( rebuild ) buildPipeline();

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
	// A single visibility grid spans both rooms. Multiple grids plus shadow,
	// SSR and AO samplers exceed WebGPU's portable 16-sampler stage limit.
	const giBounds = data.probeVolumes.reduce( ( box, volume ) => box.union( volume.box ), new THREE.Box3() );
	for ( const volume of [ { box: giBounds, position: giBounds.getCenter( new THREE.Vector3() ) } ] ) {

		const size = volume.box.getSize( new THREE.Vector3() );
		const grid = new LightProbeGrid( size.x, size.y, size.z, data.probeVolumes.length > 1 ? 8 : 4, 4, 4 );
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
	if ( ! highSupported ) {

		params.specularMethod = params.specularMethod === 'SSR' ? 'Specular probes' : params.specularMethod;
		params.diffuseMethod = params.diffuseMethod === 'SSGI' ? 'Diffuse probes' : params.diffuseMethod;

	}

	for ( const grid of grids ) {

		grid.visible = params.diffuseMethod === 'Diffuse probes';
		grid.visibility = params.visibility;
		grid.intensity = params.giIntensity;

	}

	if ( pipeline ) pipeline.dispose();
	for ( const node of owned ) node.dispose();
	owned.length = 0;
	const keep = node => {

		owned.push( node );
		return node;

	};

	// Buffer requirements follow the visible controls, never a hidden preset flag.
	const high = params.specularMethod === 'SSR' || params.diffuseMethod === 'SSGI' || params.aoMethod === 'GTAO' || ! [ 'Beauty', 'Depth', 'Velocity' ].includes( params.view );
	// Probes include captured sky diffuse. Select it OR live sky diffuse, never both.
	scene.environment = params.diffuseMethod === 'Sky IBL' ? environment.texture : null;
	scene.environmentIntensity = params.giIntensity;
	pipeline = new THREE.RenderPipeline( renderer );
	gbuffer = keep( enableDepthPrepass( pass( scene, camera ) ) );
	gbuffer.depthPrepass = params.depthMode === 'Conservative';
	gbuffer.name = 'G-buffer (optional conservative depth pre-pass)';
	const attachments = { output };
	if ( high || ( params.antialiasing === 'TRAA' ) || params.view === 'Velocity' ) attachments.velocity = velocity;
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
	gbuffer.contextNode = builtinAOContext( float( 1 ) );
	gbuffer.contextNode.value.getIndirectSpecular = () => params.specularMethod === 'None' || params.diffuseMethod === 'SSGI' ? vec3( 0 ) : probeSpecular();
	let result = color;
	let normal, albedo, rough, pyramid, gtao, reflections, diffuseGI;
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
		if ( params.specularMethod === 'SSR' ) {

			reflections = keep( ssr( color, depth, normal, {
				stochastic: true, hiZ: params.ssrTracing === 'Hi-Z' ? pyramid : null,
				diffuseNode: albedo, metalnessNode: albedo.a, roughnessNode: rough.r
			} ) );
			reflections.quality.value = params.quality;
			// Misses are resolved against local probes below, not a black SSR environment.
			reflections.screenEdgeFade.value = 0;
			reflections.maxDistance.value = 20;
			reflections.thickness.value = 0.15;
			filtered = reflections;
			if ( params.reflectionDenoise !== 'Off' ) {

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
		const visibility = ( params.aoMethod === 'GTAO' ) ? gtao.getTextureNode().sample( screenUV ).r : float( 1 );
		if ( params.diffuseMethod === 'SSGI' ) {

			diffuseGI = keep( ssgi( color, depth, normal, camera ) );
			diffuseGI.sliceCount.value = 2;
			diffuseGI.stepCount.value = 8;
			diffuseGI.giIntensity.value = params.giIntensity;
			// SSGI's RGB is radiance for an albedo multiply; material GI expects irradiance.
			beauty.contextNode = builtinGIContext( visibility, diffuseGI.getGINode().rgb.mul( Math.PI ) );

		} else {

			beauty.contextNode = builtinAOContext( visibility );

		}

		const context = beauty.contextNode.value;
		context.getIndirectSpecular = () => {

			const fallback = params.specularMethod === 'None' ? vec3( 0 ) : probeSpecular();
			return filtered ? mix( fallback, filtered.rgb, float( reflections.a.greaterThan( 0 ).and( reflections.a.lessThan( ENV_RAY_LENGTH_THRESHOLD ) ) ) ) : fallback;

		};

		if ( params.aoMethod === 'GTAO' ) context.getSpecularOcclusion = ( original, n, r, v ) => specularOcclusion( visibility, gtao.getBentNormalNode(), reflect( v.negate(), n ), r );
		result = beauty;

	}

	if ( params.antialiasing === 'TRAA' ) result = keep( traa( result, depth, motion, camera ) );
	const views = {
		Depth: vec4( vec3( gbuffer.getLinearDepthNode() ), 1 ),
		Velocity: vec4( motion.rg.mul( 10 ).add( 0.5 ), 0, 1 )
	};
	if ( high ) Object.assign( views, {
		Normal: vec4( normal.mul( 0.5 ).add( 0.5 ), 1 ), Albedo: albedo, Roughness: vec4( vec3( rough.r ), 1 ),
		'Hi-Z': vec4( vec3( perspectiveDepthToViewZ( pyramid, camera.near, camera.far ).negate().div( 20 ) ), 1 ),
		AO: vec4( vec3( gtao.r ), 1 ), 'Bent normals': vec4( gtao.getBentNormalNode().mul( 0.5 ).add( 0.5 ), 1 ),
		SSR: reflections ? vec4( reflections.rgb, 1 ) : color,
		SSGI: diffuseGI ? vec4( diffuseGI.getGINode().rgb, 1 ) : color
	} );
	pipeline.outputNode = views[ params.view ] || result;
	pipeline.outputColorTransform = ! views[ params.view ];
	status.textContent = `${ params.scene } · ${ params.preset } · ${ window.gbufferAdapter }\n${ high ? 'MRT → selected effects → material resolve' : 'Minimal MRT → beauty' }\n${ params.diffuseMethod } · ${ params.specularMethod } · Depth: ${ params.depthMode }\nPrototype: synchronous probe bakes; approximate probe BRDF; opaque geometry only. No SSR demodulation.`;
	syncEditors();
	for ( const [ key, active ] of Object.entries( {
		visibility: params.diffuseMethod === 'Diffuse probes', giIntensity: params.diffuseMethod !== 'None',
		ssrTracing: params.specularMethod === 'SSR', reflectionDenoise: params.specularMethod === 'SSR',
		quality: params.specularMethod === 'SSR', hiZLevel: params.view === 'Hi-Z', aoRadius: params.aoMethod === 'GTAO'
	} ) ) {

		if ( editors.has( key ) ) editors.get( key )[ active ? 'show' : 'hide' ]();

	}

	window.gbufferReady = true;

}
