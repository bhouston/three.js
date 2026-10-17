import * as THREE from 'three/webgpu';
import { Inspector } from 'three/addons/inspector/Inspector.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { KTX2Loader } from 'three/addons/loaders/KTX2Loader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { GBufferPipeline } from './GBufferPipeline.js';
import { plan, trackTextures } from './contracts.js';

const query = new URLSearchParams( location.search );
const status = document.getElementById( 'status' );
const params = {
	model: query.get( 'model' ) || 'examples/models/gltf/DamagedHelmet/glTF/DamagedHelmet.gltf',
	mode: query.get( 'mode' ) === 'forward' ? 'forward' : 'deferred',
	mobile: query.get( 'mobile' ) === '1',
	passMode: query.get( 'passMode' ) === 'materialized' ? 'materialized' : 'bypass',
	resolution: Math.min( 2, Math.max( 0.25, Number( query.get( 'resolution' ) ) || 1 ) ),
	prepass: query.get( 'prepass' ) === '1',
	shadows: query.get( 'shadows' ) !== '0',
	ao: query.get( 'ao' ) === '1',
	traa: query.get( 'traa' ) !== '0',
	sharpen: query.get( 'sharpen' ) === '1',
	tonemap: query.get( 'tonemap' ) !== '0',
	output: query.get( 'output' ) !== '0',
	playing: false,
	animation: '',
	scene: '0',
	search: ''
};
let renderer, scene, camera, controls, loader, pipeline, model, mixer, gltf, allocationTracker, adapterInfo, modelStats;
let selection = 0, loading = false, rebuilding = false, lastTime = performance.now(), lastStatus = 0;
let rebuildQueue = Promise.resolve();
const previousCameraPosition = new THREE.Vector3();
const previousCameraRotation = new THREE.Quaternion();
let manifest, modelControl, sceneControl, animationControl;
const lights = [], errors = [], warnings = [];
const readouts = { backend: '', dimensions: '', passes: '', traversals: '', textures: '', benchmark: 'Not measured (display FPS is not throughput)' };
const editors = [];
const api = { ready: null, configure, renderFrame, drain: () => renderer.backend.device.queue.onSubmittedWorkDone(), snapshot, moveCamera, moveObject, params,
	readBuffer: async ( name, index = 0 ) => {

		const target = name === 'surface' ? pipeline.surface : name === 'opaque' ? pipeline.opaque : pipeline.renderTarget;
		return Array.from( await renderer.readRenderTargetPixelsAsync( target, 0, 0, target.width, target.height, index ) );

	},
	showBuffer: name => {

		const entry = [ ...renderer.inspector.viewer.canvasNodes.values() ].find( entry => entry.name === name );
		if ( entry ) renderer.inspector.viewer.showNodeView( entry.id );

	},
	openBuffers: () => {

		const inspector = renderer.inspector;
		inspector.viewer.show();
		inspector.setActiveTab( inspector.viewer );
		inspector.profiler.setPosition( 'bottom' );
		if ( ! inspector.profiler.panel.classList.contains( 'visible' ) ) inspector.profiler.togglePanel();
		inspector.profiler.show( inspector.parameters );

	}
};
window.gbuffer = api;
api.ready = init().catch( error => {

	showError( error );
	throw error;

} );

function showError( error ) {

	const message = error?.message || String( error );
	errors.push( message );
	status.textContent = message + '\nChoose another model in Inspector Parameters to recover.';
	console.error( error );

}

async function init() {

	if ( ! navigator.gpu ) throw new Error( 'Hardware WebGPU unavailable. Software fallback is disabled.' );
	const adapter = await navigator.gpu.requestAdapter( { powerPreference: 'high-performance', forceFallbackAdapter: false } );
	if ( ! adapter ) throw new Error( 'No hardware WebGPU adapter in this browser session.' );
	const info = adapter.info;
	adapterInfo = { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description, isFallbackAdapter: info.isFallbackAdapter };
	if ( info.isFallbackAdapter || /swiftshader|llvmpipe|lavapipe|software|microsoft basic/i.test( Object.values( adapterInfo ).join( ' ' ) ) ) throw new Error( 'A physical GPU is required.' );
	const device = await adapter.requestDevice( { requiredFeatures: [ ...adapter.features ] } );
	allocationTracker = trackTextures( device );
	device.addEventListener( 'uncapturederror', event => showError( event.error ) );
	renderer = new THREE.WebGPURenderer( { device, antialias: false } );
	await renderer.init();
	if ( ! renderer.backend.isWebGPUBackend ) throw new Error( 'Expected WebGPU backend.' );
	renderer.setPixelRatio( 1 ); // Fixed DPR; resolution scale is independent of the mode.
	renderer.shadowMap.enabled = true;
	renderer.shadowMap.type = THREE.PCFShadowMap;
	renderer.inspector = new Inspector();
	renderer.inspector.enabled = ! query.has( 'benchmark' );
	document.body.appendChild( renderer.domElement );

	scene = new THREE.Scene();
	scene.background = new THREE.Color( 0x293441 );
	const pmrem = new THREE.PMREMGenerator( renderer );
	const room = new RoomEnvironment();
	scene.environment = pmrem.fromScene( room, 0.04 ).texture;
	room.dispose();
	pmrem.dispose();
	camera = new THREE.PerspectiveCamera( 45, innerWidth / innerHeight, 0.02, 100 );
	camera.position.set( 3, 2, 4 );
	controls = new OrbitControls( camera, renderer.domElement );
	controls.enableDamping = false;
	controls.addEventListener( 'start', () => {

		lastTime = performance.now();

	} );
	const sun = new THREE.DirectionalLight( 0xffe5ca, 3 );
	sun.name = 'key';
	sun.position.set( 3, 6, 4 );
	sun.layers.enable( 1 );
	sun.layers.enable( 2 );
	sun.shadow.mapSize.set( 1024, 1024 );
	Object.assign( sun.shadow.camera, { left: - 4, right: 4, top: 4, bottom: - 4, near: 0.1, far: 20 } );
	sun.shadow.camera.layers.enable( 1 );
	sun.shadow.normalBias = 0.015;
	sun.shadow.bias = - 0.0003;
	scene.add( sun, sun.target );
	lights.push( sun );
	const ground = new THREE.Mesh( new THREE.PlaneGeometry( 200, 200 ), new THREE.MeshStandardMaterial( { color: 0x747d88, roughness: 0.8 } ) );
	ground.name = 'ground';
	ground.rotation.x = - Math.PI / 2;
	ground.position.y = - 1.01;
	ground.receiveShadow = true;
	scene.add( ground );
	const draco = new DRACOLoader().setDecoderPath( 'jsm/libs/draco/gltf/' );
	const ktx = new KTX2Loader().setTranscoderPath( 'jsm/libs/basis/' ).detectSupport( renderer );
	loader = new GLTFLoader().setDRACOLoader( draco ).setKTX2Loader( ktx ).setMeshoptDecoder( MeshoptDecoder );
	manifest = await fetch( 'models/gltf/gbuffer-model-manifest.json' ).then( response => response.json() );
	createControls();
	await loadModel( params.model );
	renderFrame();
	renderer.inspector.profiler.show( renderer.inspector.parameters );
	if ( query.has( 'benchmark' ) ) renderer._animation.stop();
	window.addEventListener( 'resize', () => rebuild().catch( showError ) );
	if ( ! query.has( 'benchmark' ) ) renderer.setAnimationLoop( () => {

		if ( loading || rebuilding || ! pipeline ) return;
		renderFrame();
		if ( performance.now() - lastStatus > 1000 ) updateStatus();

	} );
	updateStatus();
	return true;

}

function createControls() {

	const gui = renderer.inspector.createParameters( 'G-buffer throughput investigation' );
	gui.add( params, 'mobile' ).name( 'Mobile mode' ).onChange( () => rebuild().catch( showError ) );
	gui.add( params, 'mode', [ 'deferred', 'forward' ] ).name( 'Desktop / matched forward baseline' ).onChange( () => rebuild().catch( showError ) );
	gui.add( params, 'passMode', [ 'bypass', 'materialized' ] ).name( 'Neutral pass behavior' ).onChange( () => rebuild().catch( showError ) );
	for ( const name of [ 'prepass', 'shadows', 'ao', 'traa', 'sharpen', 'tonemap', 'output' ] ) gui.add( params, name ).onChange( () => rebuild().catch( showError ) );
	gui.add( params, 'resolution', 0.25, 2, 0.05 ).name( 'Resolution scale (DPR 1)' ).onChange( () => rebuild().catch( showError ) );
	const models = gui.addFolder( 'Repository models (all ' + manifest.length + ' tracked assets)' );
	modelControl = models.add( params, 'model', manifest ).name( 'Full repository path' ).onChange( value => loadModel( value ).catch( showError ) );
	models.add( params, 'search' ).name( 'Filter model paths' ).onChange( value => {

		for ( const option of modelControl.select.options ) option.hidden = ! option.value.toLowerCase().includes( value.toLowerCase() );

	} );
	sceneControl = models.add( params, 'scene', [ '0' ] ).name( 'glTF scene' ).onChange( () => activateScene().catch( showError ) );
	animationControl = models.add( params, 'animation', [ '' ] ).onChange( playAnimation );
	models.add( params, 'playing' ).name( 'Play animation' );
	const metrics = gui.addFolder( 'Active pipeline / resource inventory' );
	for ( const name of Object.keys( readouts ) ) editors.push( [ name, metrics.add( readouts, name ) ] );
	gui.add( { benchmark: benchmarkHere }, 'benchmark' ).name( 'Measure sustained GPU throughput (5s)' );
	renderer.inspector.setActiveTab( renderer.inspector.parameters );

}

function selectOptions( control, values ) {

	control.options = values;
	control.select.replaceChildren( ...values.map( value => new Option( value, value ) ) );

}

function fallbackReasons( material ) {

	const reasons = [];
	if ( ! material.isMeshStandardMaterial ) reasons.push( material.type );
	for ( const key of [ 'clearcoat', 'sheen', 'anisotropy', 'iridescence', 'transmission', 'dispersion', 'diffuseRoughness', 'retroreflectivity' ] ) if ( material[ key ] > 0 ) reasons.push( key );
	if ( material.specularIntensity !== undefined && material.specularIntensity !== 1 ) reasons.push( 'specularIntensity' );
	if ( material.ior !== undefined && material.ior !== 1.5 ) reasons.push( 'ior' );
	if ( material.specularColor && material.specularColor.getHex() !== 0xffffff ) reasons.push( 'specularColor' );
	if ( material.specularColorMap || material.specularIntensityMap ) reasons.push( 'specular maps' );
	if ( material.lightMap ) reasons.push( 'lightmap UV contribution (forward)' );
	if ( material.depthWrite === false || material.depthTest === false || material.depthFunc !== THREE.LessEqualDepth ) reasons.push( 'custom depth state' );
	return reasons;

}

async function loadModel( path ) {

	const ticket = ++ selection;
	loading = true;
	status.textContent = 'Loading ' + path;
	try {

		if ( ! manifest.includes( path ) ) throw new Error( 'Model is not in the tracked asset manifest: ' + path );
		const loaded = await loader.loadAsync( '../' + path );
		if ( ticket !== selection ) {

			disposeGLTF( loaded );
			return;

		}

		await api.drain();
		if ( ticket !== selection ) {

			disposeGLTF( loaded );
			return;

		}

		if ( gltf ) {

			scene.remove( model );
			mixer?.stopAllAction();
			disposeGLTF( gltf );

		}

		gltf = loaded;
		params.model = path;
		params.scene = '0';
		selectOptions( sceneControl, loaded.scenes.map( ( root, index ) => String( index ) ) );
		selectOptions( animationControl, loaded.animations.map( ( clip, index ) => index + ': ' + clip.name ) );
		params.animation = loaded.animations.length ? '0: ' + loaded.animations[ 0 ].name : '';
		await activateScene();

	} catch ( error ) {

		if ( ticket === selection ) showError( error );

	} finally {

		if ( ticket === selection ) loading = false;

	}

}

async function activateScene() {

	if ( model ) scene.remove( model );
	model = gltf.scenes[ Number( params.scene ) ] || gltf.scene;
	model.position.set( 0, 0, 0 );
	model.scale.setScalar( 1 );
	model.updateMatrixWorld( true );
	const box = new THREE.Box3().setFromObject( model );
	const size = box.getSize( new THREE.Vector3() );
	const center = box.getCenter( new THREE.Vector3() );
	const scale = 2 / ( Math.max( size.x, size.y, size.z ) || 1 );
	model.scale.setScalar( scale );
	model.position.copy( center ).multiplyScalar( - scale );
	scene.add( model );
	controls.target.set( 0, 0, 0 );
	camera.position.set( 2.8, 1.8, 3.2 );
	controls.update();
	modelStats = { meshes: 0, triangles: 0, materials: 0, deferredMeshes: 0, fallbackMeshes: 0, transparentMeshes: 0, fallbackReasons: [], animations: gltf.animations.map( clip => clip.name ), hasMorph: false, skinnedMeshes: 0 };
	const materials = new Set(), reasons = new Set();
	model.traverse( object => {

		if ( ! object.isMesh ) return;
		modelStats.meshes ++;
		modelStats.skinnedMeshes += object.isSkinnedMesh ? 1 : 0;
		modelStats.hasMorph ||= !! object.morphTargetInfluences;
		modelStats.triangles += ( object.geometry.index?.count || object.geometry.attributes.position.count ) / 3 * ( object.isInstancedMesh ? object.count : 1 );
		const list = Array.isArray( object.material ) ? object.material : [ object.material ];
		list.forEach( material => materials.add( material ) );
		const unsupported = list.flatMap( fallbackReasons );
		unsupported.forEach( reason => reasons.add( reason ) );
		const transparent = list.some( material => material.transparent || material.transmission > 0 );
		object.layers.set( unsupported.length || transparent ? 1 : 0 );
		if ( transparent ) modelStats.transparentMeshes ++;
		else if ( unsupported.length ) modelStats.fallbackMeshes ++;
		else modelStats.deferredMeshes ++;
		object.castShadow = ! transparent;
		object.receiveShadow = true;

	} );
	modelStats.materials = materials.size;
	modelStats.fallbackReasons = [ ...reasons ];
	mixer?.stopAllAction();
	mixer = new THREE.AnimationMixer( model );
	playAnimation();
	warnings.length = 0;
	if ( reasons.size ) warnings.push( 'Explicit forward fallback: ' + [ ...reasons ].join( ', ' ) );
	if ( modelStats.hasMorph ) warnings.push( 'Morph motion is not supported by upstream velocity; TRAA restarts while morph animation plays.' );
	await rebuild();

}

function playAnimation() {

	if ( ! mixer ) return;
	mixer.stopAllAction();
	const clip = gltf.animations[ Number.parseInt( params.animation ) ];
	if ( clip ) mixer.clipAction( clip ).play();
	lastTime = performance.now();

}

function disposeGLTF( asset ) {

	const resources = new Set(), bitmaps = new Set();
	for ( const root of asset.scenes ) root.traverse( object => {

		if ( object.geometry ) resources.add( object.geometry );
		if ( object.skeleton ) resources.add( object.skeleton );
		for ( const material of Array.isArray( object.material ) ? object.material : [ object.material ] ) if ( material ) {

			resources.add( material );
			for ( const value of Object.values( material ) ) if ( value?.isTexture ) {

				resources.add( value );
				if ( value.source?.data?.close ) bitmaps.add( value.source.data );

			}

		}

	} );
	resources.forEach( resource => resource.dispose() );
	bitmaps.forEach( bitmap => bitmap.close() );

}

function rebuild() {

	const task = rebuildQueue.catch( () => {} ).then( performRebuild );
	rebuildQueue = task;
	return task;

}

async function performRebuild() {

	rebuilding = true;
	try {

		await api.drain();
		if ( pipeline && ! query.has( 'benchmark' ) ) {

			// Inspector viewers retain preview materials. Dispose them before retiring
			// their textures, then recreate the same controls against the new pipeline.
			renderer.inspector.dispose();
			renderer.inspector = new Inspector();
			editors.length = 0;
			createControls();
			renderer.inspector.profiler.show( renderer.inspector.parameters );

		}

		pipeline?.dispose();
		renderer.setSize( Math.max( 1, Math.floor( innerWidth * params.resolution ) ), Math.max( 1, Math.floor( innerHeight * params.resolution ) ), false );
		renderer.domElement.style.width = '100vw';
		renderer.domElement.style.height = '100vh';
		camera.aspect = innerWidth / innerHeight;
		camera.updateProjectionMatrix();
		renderer.toneMapping = params.tonemap ? THREE.ACESFilmicToneMapping : THREE.NoToneMapping;
		for ( const light of lights ) {

			if ( light.shadow.shadowNode ) {

				light.dispose();
				light.shadow.shadowNode.dispose();
				delete light.shadow.shadowNode;

			}

		}

		pipeline = new GBufferPipeline( renderer, scene, camera, { ...params }, lights );
		await pipeline.compile();
		previousCameraPosition.copy( camera.position );
		previousCameraRotation.copy( camera.quaternion );
		persist();
		allocationTracker.resetPeak();
		updateStatus();

	} finally {

		rebuilding = false;

	}

}

async function configure( next ) {

	await api.ready;
	await api.drain();
	if ( next.animationPlaying !== undefined ) next = { ...next, playing: next.animationPlaying };
	const changeModel = next.model !== undefined && next.model !== params.model;
	Object.assign( params, next );
	if ( changeModel ) await loadModel( params.model );
	else await rebuild();
	return snapshot();

}

function persist() {

	const url = new URL( location.href );
	for ( const name of [ 'model', 'mode', 'mobile', 'passMode', 'resolution', 'prepass', 'shadows', 'ao', 'traa', 'sharpen', 'tonemap', 'output' ] ) url.searchParams.set( name, typeof params[ name ] === 'boolean' ? Number( params[ name ] ) : params[ name ] );
	history.replaceState( null, '', url );

}

function renderFrame() {

	// Renderer frame-scoped nodes normally advance in its rAF Animation module.
	// Explicitly advance once per WHOLE frame so an uncapped batch never reuses
	// surface / shadow / temporal work from an earlier render in the same rAF.
	renderer._nodes.nodeFrame.update();
	renderer.info.frame = renderer._nodes.nodeFrame.frameId;
	const now = performance.now();
	if ( params.playing ) mixer?.update( Math.min( ( now - lastTime ) / 1000, 0.1 ) );
	lastTime = now;
	controls.update();
	if ( camera.position.distanceTo( previousCameraPosition ) > 1 || Math.abs( camera.quaternion.dot( previousCameraRotation ) ) < 0.8 ) {

		pipeline.history.reset( 'camera cut' );
		if ( pipeline.temporal ) pipeline.temporal._historyRenderTarget.setSize( 1, 1 );

	}

	previousCameraPosition.copy( camera.position );
	previousCameraRotation.copy( camera.quaternion );
	// Upstream morph velocity lacks previous weights. Avoid silently accumulating wrong history.
	if ( params.playing && modelStats.hasMorph && pipeline.temporal ) pipeline.temporal._historyRenderTarget.setSize( 1, 1 );
	renderer.info.reset();
	pipeline.post.render();

}

function moveCamera() {

	camera.position.x += 0.25;
	controls.update();
	return rebuild(); // explicit camera-cut API; ordinary orbit motion keeps temporal history

}

function moveObject() {

	model.rotation.y += 0.15;

}

function snapshot() {

	const enabled = plan( params );
	const resources = allocationTracker.snapshot();
	return {
		backend: 'WebGPU', adapter: adapterInfo, width: renderer.domElement.width, height: renderer.domElement.height, dpr: 1,
		config: { ...params }, model: modelStats, passes: pipeline?.executed || [],
		geometryTraversals: { shadow: params.shadows ? lights.length : 0, surface: enabled.deferred ? 1 : 0, prepass: enabled.prepass ? 1 : 0, forwardOpaque: enabled.deferred ? 0 : 1, restrictedFallback: enabled.deferred && modelStats.fallbackMeshes ? 1 : 0, transparent: modelStats.transparentMeshes ? 1 : 0 },
		resources: { ...resources, surfaceBytes: enabled.deferred ? renderer.domElement.width * renderer.domElement.height * pipeline.surfaceBytesPerPixel : 0, excludes: 'Driver alignment/heaps and presentation swap chain. Buffer descriptors and immutable geometry CPU bytes reported separately.' },
		history: { valid: pipeline.history.valid, previous: pipeline.previousMetadata, current: pipeline.history.metadata[ pipeline.history.read ], geometryCPUBytes: pipeline.geometryHistory.reduce( ( total, entries ) => total + ( entries || [] ).reduce( ( sum, entry ) => sum + entry.world.byteLength + ( entry.bones?.byteLength || 0 ) + ( entry.instances?.byteLength || 0 ) + ( entry.morph?.byteLength || 0 ), 0 ), 0 ) },
		renderInfo: { calls: renderer.info.render.frameCalls, draws: renderer.info.render.drawCalls, triangles: renderer.info.render.triangles }, errors: [ ...errors ], warnings: [ ...warnings ]
	};

}

function updateStatus() {

	if ( ! pipeline ) return;
	lastStatus = performance.now();
	const current = snapshot();
	readouts.backend = 'WebGPU / ' + adapterInfo.vendor + ' ' + adapterInfo.architecture;
	readouts.dimensions = current.width + ' × ' + current.height + ' / DPR 1';
	readouts.passes = current.passes.length + ': ' + current.passes.join( ' → ' );
	readouts.traversals = JSON.stringify( current.geometryTraversals );
	readouts.textures = ( current.resources.liveBytes / 1048576 ).toFixed( 2 ) + ' MiB live / ' + ( current.resources.peakBytes / 1048576 ).toFixed( 2 ) + ' MiB peak';
	for ( const [ name, editor ] of editors ) editor.setValue( readouts[ name ] );
	status.textContent = [ readouts.backend, readouts.dimensions + ' | ' + readouts.textures, modelStats.triangles.toLocaleString() + ' model triangles / ' + modelStats.materials + ' materials', readouts.benchmark, ...warnings, ...errors.slice( - 1 ) ].join( '\n' );

}

async function benchmarkHere() {

	if ( loading || rebuilding ) return;
	renderer.setAnimationLoop( null );
	renderer._animation.stop();
	const inspectorEnabled = renderer.inspector.enabled;
	renderer.inspector.enabled = false;
	const queue = renderer.backend.device.queue;
	const run = async duration => {

		await queue.onSubmittedWorkDone();
		let completed = 0;
		const start = performance.now();
		while ( performance.now() - start < duration ) {

			for ( let i = 0; i < 8; i ++ ) renderFrame();
			await queue.onSubmittedWorkDone();
			completed += 8;
			await new Promise( resolve => setTimeout( resolve, 0 ) );

		}

		await queue.onSubmittedWorkDone();
		return { completed, seconds: ( performance.now() - start ) / 1000 };

	};

	await run( 2000 );
	const result = await run( 5000 );
	readouts.benchmark = ( result.completed / result.seconds ).toFixed( 1 ) + ' GPU-completed frames/s (' + result.completed + ' / ' + result.seconds.toFixed( 3 ) + ' s). Use uncapped launch flags; see report.';
	renderer.inspector.enabled = inspectorEnabled;
	api.lastBenchmark = { ...result, framesPerSecond: result.completed / result.seconds, configuration: { ...params } };
	updateStatus();
	if ( ! query.has( 'benchmark' ) ) {

		renderer._animation.start();
		renderer.setAnimationLoop( () => {

			if ( ! loading && ! rebuilding ) renderFrame();

		} );

	}

}
