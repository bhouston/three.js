import * as THREE from '/build/three.webgpu.js';
import { inspectQuery } from './diagnostics.js';
import { LightProbeGrid } from '/examples/jsm/lighting/LightProbeGrid.js';

export const cases = [
	... [ 0.05, 0.15, 0.3, 0.6, 1.2 ].map( thickness => ( { name: `wall-${thickness}`, thickness } ) ),
	{ name: 'doorway', thickness: 0.3, doorway: true },
	{ name: 'corner', thickness: 0.3, corner: true },
	{ name: 'two-walls', thickness: 0.15, double: true },
	{ name: 'angled-wall', thickness: 0.3, angle: Math.PI / 6 },
	{ name: 'oblique-view', thickness: 0.3, oblique: true }
];

export async function runCase( renderer, config, variant ) {

	const scene = new THREE.Scene();
	scene.background = new THREE.Color( 0 );
	const black = new THREE.MeshBasicNodeMaterial( { color: 0 } );
	const shellMaterial = new THREE.MeshBasicNodeMaterial( { color: 0, side: THREE.BackSide } );
	const shell = new THREE.Mesh( new THREE.BoxGeometry( 8, 6, 8 ), shellMaterial );
	scene.add( shell );
	const emitter = new THREE.Mesh( new THREE.PlaneGeometry( 7.8, 5.8 ), new THREE.MeshBasicNodeMaterial( { color: 0xffffff, side: THREE.DoubleSide } ) );
	emitter.position.x = 3.9;
	emitter.rotation.y = - Math.PI / 2;
	scene.add( emitter );
	const blockers = [];
	const wall = ( center, size, angle = 0 ) => {

		const mesh = new THREE.Mesh( new THREE.BoxGeometry( ...size ), black );
		mesh.position.fromArray( center );
		mesh.rotation.y = angle;
		scene.add( mesh );
		blockers.push( { center, half: size.map( x => x / 2 ), angle } );

	};

	const t = config.thickness;
	const angle = config.angle || 0;
	if ( config.doorway ) {

		wall( [ 0, 0, - 2.625 ], [ t, 6, 2.75 ] );
		wall( [ 0, 0, 2.625 ], [ t, 6, 2.75 ] );
		wall( [ 0, 2.45, 0 ], [ t, 1.1, 2.5 ] );
		wall( [ 0, - 2.45, 0 ], [ t, 1.1, 2.5 ] );

	} else if ( config.double ) {

		wall( [ - 0.25, 0, 0 ], [ t, 6, 8 ] );
		wall( [ 0.25, 0, 0 ], [ t, 6, 8 ] );

	} else {

		wall( [ 0, 0, 0 ], [ t, 6, config.angle ? 12 : 8 ], angle );
		if ( config.corner ) wall( [ - 2, 0, 0 ], [ 4, 6, t ] );

	}

	const probes = new LightProbeGrid( 7.2, 4.8, 6.4, 8, 5, 7 );
	probes.visibility = variant !== 'visibility-off';
	probes._testBlockers = blockers;
	if ( variant.startsWith( 'normal-' ) ) probes.normalBias = Number( variant.slice( 7 ) );
	if ( variant.startsWith( 'view-' ) ) probes.viewBias = Number( variant.slice( 5 ) );
	probes.bake( renderer, scene, { cubemapSize: 32, near: 0.02, far: 20 } );
	await renderer.backend.device.queue.onSubmittedWorkDone();
	scene.add( probes );
	// Receivers are added after baking: the black surfaces absorb all light,
	// so the dark-facing side has exactly zero reference irradiance.
	const material = new THREE.MeshStandardNodeMaterial( { color: 0xffffff, roughness: 1, metalness: 0 } );
	const receiver = new THREE.Mesh( new THREE.PlaneGeometry( config.corner ? 0.5 : 3.6, config.doorway ? 0.7 : 3.6 ), material );
	const normal = new THREE.Vector3( - Math.cos( angle ), 0, Math.sin( angle ) );
	const position = normal.clone().multiplyScalar( config.double ? 0.25 + t / 2 + 0.002 : t / 2 + 0.002 );
	if ( config.doorway ) position.y = 2.35;
	if ( config.corner ) position.z = - 0.5;
	receiver.position.copy( position );
	receiver.quaternion.setFromUnitVectors( new THREE.Vector3( 0, 0, 1 ), normal );
	scene.add( receiver );
	const camera = new THREE.PerspectiveCamera( 50, 1, 0.01, 30 );
	const cameraDistance = config.doorway ? 0.6 : config.corner ? 0.45 : 1.4;
	camera.position.copy( position ).addScaledVector( normal, cameraDistance );
	if ( config.oblique ) camera.position.z += 1.4;
	camera.lookAt( position );
	const target = new THREE.RenderTarget( 64, 64, { type: THREE.FloatType } );
	const render = () => {

		renderer.setRenderTarget( target );
		renderer.render( scene, camera );

	};

	const measure = async () => {

		render();
		const data = await renderer.readRenderTargetPixelsAsync( target, 8, 8, 48, 48 );
		const luminance = [];
		for ( let i = 0; i < data.length; i += 4 ) luminance.push( 0.2126 * data[ i ] + 0.7152 * data[ i + 1 ] + 0.0722 * data[ i + 2 ] );
		luminance.sort( ( a, b ) => a - b );
		const canvas = document.createElement( 'canvas' );
		canvas.width = 48; canvas.height = 48;
		const context = canvas.getContext( '2d' );
		const image = context.createImageData( 48, 48 );
		for ( let i = 0; i < data.length; i += 4 ) {

			for ( let c = 0; c < 3; c ++ ) image.data[ i + c ] = Math.round( Math.pow( Math.min( 1, Math.max( 0, data[ i + c ] * 8 ) ), 1 / 2.2 ) * 255 );
			image.data[ i + 3 ] = 255;

		}

		context.putImageData( image, 0, 0 );
		return { png: canvas.toDataURL(), mean: luminance.reduce( ( a, b ) => a + b, 0 ) / luminance.length, p95: luminance[ Math.floor( luminance.length * 0.95 ) ], max: luminance.at( - 1 ) };

	};

	const views = {};
	const saveView = name => {

		views[ name ] = { position: receiver.position.clone(), quaternion: receiver.quaternion.clone(), camera: camera.position.clone() };

	};

	saveView( 'leak' );
	const leak = await measure();
	const visible = probes.visible;
	probes.visible = false;
	const directOnly = await measure();
	probes.visible = visible;
	if ( directOnly.max > 1e-8 ) throw new Error( `${config.name}: leakage crop includes direct illumination or uncovered pixels (${directOnly.max})` );
	const diagnostics = {};
	let distancePixels;
	if ( variant === 'baseline' ) {

		distancePixels = await renderer.readRenderTargetPixelsAsync( probes._distanceTarget, 0, 0, probes._distanceTarget.width, probes._distanceTarget.height );
		diagnostics.wall = inspectQuery( probes, distancePixels, receiver.position, normal, camera.position, blockers );

	}

	// Visible side of the same wall: same local surface, opposite face.
	receiver.position.copy( position ).negate();
	if ( config.doorway ) receiver.position.y = position.y;
	if ( config.corner ) receiver.position.z = position.z;
	receiver.quaternion.setFromUnitVectors( new THREE.Vector3( 0, 0, 1 ), normal.clone().negate() );
	camera.position.copy( receiver.position ).addScaledVector( normal, - cameraDistance );
	camera.lookAt( receiver.position );
	saveView( 'lit' );
	const lit = await measure();
	// A separate surface faces the light source from the dark room. With a
	// closed partition its reference is zero; through the doorway it is positive.
	receiver.position.set( - 0.8, 0, config.corner ? - 0.5 : 0 );
	receiver.quaternion.setFromUnitVectors( new THREE.Vector3( 0, 0, 1 ), new THREE.Vector3( 1, 0, 0 ) );
	camera.position.copy( receiver.position ).add( new THREE.Vector3( 0.5, 0, 0 ) );
	camera.lookAt( receiver.position );
	saveView( 'interior' );
	const interior = await measure();
	if ( distancePixels ) diagnostics.interior = inspectQuery( probes, distancePixels, receiver.position, new THREE.Vector3( 1, 0, 0 ), camera.position, blockers );
	const doorway = config.doorway ? interior : null;

	const memory = probes._distanceTarget.width * probes._distanceTarget.height * 4;
	return { name: config.name, leak, lit, interior, doorway, diagnostics, distanceBytes: memory, render, renderer, scene, camera, target, probes,
		showView( name ) {

			const view = views[ name ];
			receiver.position.copy( view.position );
			receiver.quaternion.copy( view.quaternion );
			camera.position.copy( view.camera );
			camera.lookAt( receiver.position );
			renderer.setRenderTarget( null );
			renderer.render( scene, camera );

		}, dispose() {

			renderer.setRenderTarget( null );
			probes.dispose();
			target.dispose();
			scene.traverse( o => {

				if ( o.isMesh ) o.geometry.dispose();

			} );
			black.dispose(); shellMaterial.dispose(); material.dispose(); emitter.material.dispose();

		} };

}
