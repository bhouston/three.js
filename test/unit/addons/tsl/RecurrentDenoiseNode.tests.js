import { DataTexture, DataUtils, FloatType, NodeMaterial, PerspectiveCamera, QuadMesh, RendererUtils, Vector2 } from 'three/webgpu';
import { context, texture, vec3, vec4 } from 'three/tsl';
import { recurrentDenoise, demodulate, remodulate } from '../../../../examples/jsm/tsl/display/RecurrentDenoiseNode.js';
import { getSharedRenderer, gpuTest } from '../../src/nodes/tsl/gpu-test-utils.js';

async function fixture( assert, run ) {

	const renderer = await getSharedRenderer( 'webgpu' );
	if ( ! renderer ) {

		assert.ok( true, 'SKIPPED: WebGPU unavailable' ); return;

	}

	const state = RendererUtils.saveRendererState( renderer );
	const ratio = renderer.getPixelRatio(), size = renderer.getSize( new Vector2() ), oldContext = renderer.contextNode;
	const textures = [];
	const make = callback => {

		const values = new Float32Array( 16 * 16 * 4 );
		for ( let y = 0; y < 16; y ++ ) for ( let x = 0; x < 16; x ++ ) values.set( callback( x, y ), ( x + 16 * y ) * 4 );
		const value = new DataTexture( values, 16, 16 );
		value.type = FloatType; value.needsUpdate = true; textures.push( value );
		return value;

	};

	const input = make( () => [ 4, 2, 1, 0.25 ] );
	const raw = make( () => [ 4, 2, 1, 0.3 ] );
	const depth = make( () => [ 100 / 99.9 - 10 / 99.9, 0, 0, 0 ] );
	const normal = make( () => [ 0.5, 0.5, 1, 0 ] );
	const metalRoughness = make( () => [ 0, 0.3, 0, 0 ] );
	const camera = new PerspectiveCamera( 50, 1, 0.1, 100 );
	camera.coordinateSystem = renderer.coordinateSystem; camera.updateProjectionMatrix(); camera.updateMatrixWorld();
	const node = recurrentDenoise( texture( input ), camera, { depth: texture( depth ), normal: texture( normal ), metalRoughness: texture( metalRoughness ), raw: texture( raw ), mode: 'specular' } );
	const material = new NodeMaterial(); material.fragmentNode = node;
	const quad = new QuadMesh( material );
	const render = async samples => {

		node.samples.value = samples;
		await new Promise( resolve => requestAnimationFrame( resolve ) );
		renderer.setRenderTarget( null ); quad.render( renderer );
		const target = node.getRenderTarget();
		return Array.from( await renderer.readRenderTargetPixelsAsync( target, 0, 0, 16, 16 ), DataUtils.fromHalfFloat );

	};

	try {

		renderer.setPixelRatio( 1 ); renderer.setSize( 16, 16 ); renderer.contextNode = context();
		await run( { input, raw, normal, render } );

	} finally {

		node.dispose(); material.dispose(); for ( const value of textures ) value.dispose();
		renderer.contextNode = oldContext; renderer.setPixelRatio( ratio ); renderer.setSize( size.x, size.y ); RendererUtils.restoreRendererState( renderer, state );

	}

}

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'TSL', () => {

		QUnit.module( 'RecurrentDenoiseNode', () => {

			gpuTest( 'demodulation preserves HDR lighting and ray metadata', ( { assert } ) => {

				const signal = vec4( 2, 6, 20, 37 );
				const modulation = vec3( 0.25, 0.5, 2 );
				const lighting = demodulate( signal, modulation );
				assert.closeAbs( lighting, vec4( 8, 12, 10, 37 ), 1e-5, 'only RGB is divided by the material response' );
				assert.closeAbs( remodulate( lighting, modulation ), signal, 1e-5, 'lighting resolve restores the HDR signal' );

			} );

			gpuTest( 'black modulation stays finite and reversible', ( { assert } ) => {

				const modulation = vec3( 0, 1e-8, 0.5 );
				const signal = vec4( 0.125, 0, 16, 0 );
				const lighting = demodulate( signal, modulation );
				assert.closeAbs( lighting, vec4( 1250, 0, 32, 0 ), 1e-3, 'near-black BRDF channels use the floor' );
				assert.closeAbs( remodulate( lighting, modulation ), signal, 1e-5, 'the same floor preserves the round trip' );

			} );

			gpuTest( 'filtering lighting preserves surface detail', ( { assert } ) => {

				const red = vec3( 1, 0.125, 0.25 );
				const blue = vec3( 0.125, 0.25, 1 );
				const redSignal = vec4( 4, 0.25, 0.25, 3 );
				const blueSignal = vec4( 0.5, 0.5, 1, 3 );
				const filtered = demodulate( redSignal, red ).add( demodulate( blueSignal, blue ) ).mul( 0.5 );
				assert.closeAbs( filtered, vec4( 4, 2, 1, 3 ), 1e-6, 'filter averages illumination' );
				assert.closeAbs( remodulate( filtered, red ), redSignal, 1e-6, 'red surface detail survives filtering' );
				assert.closeAbs( remodulate( filtered, blue ), blueSignal, 1e-6, 'blue surface detail survives filtering' );

			} );


			QUnit.test( 'spatial budgets preserve constant HDR energy and remain bounded', async assert => {

				await fixture( assert, async ( { render } ) => {

					for ( const samples of [ 8, 4, 0, 99 ] ) {

						const pixels = await render( samples );
						assert.ok( pixels.every( Number.isFinite ), `${ samples } sample setting remains finite` );
						assert.ok( pixels.every( ( value, index ) => index % 4 === 3 ? value > 0 && value <= 1 : Math.abs( value - [ 4, 2, 1 ][ index % 4 ] ) < 0.005 ), `${ samples } sample setting preserves HDR energy and accumulation metadata` );

					}

				} );

			} );

			QUnit.test( 'reduced kernel retains normal edge stopping', async assert => {

				await fixture( assert, async ( { input, raw, normal, render } ) => {

					for ( let i = 0; i < 16 * 16; i ++ ) {

						const right = i % 16 >= 8;
						input.image.data.set( right ? [ 0, 0, 4, 0.25 ] : [ 4, 0, 0, 0.25 ], i * 4 );
						raw.image.data.set( right ? [ 0, 0, 4, 0.3 ] : [ 4, 0, 0, 0.3 ], i * 4 );
						normal.image.data.set( right ? [ 0.5, 1, 0.5, 0 ] : [ 0.5, 0.5, 1, 0 ], i * 4 );

					}

					input.needsUpdate = raw.needsUpdate = normal.needsUpdate = true;
					for ( const samples of [ 8, 4 ] ) {

						const pixels = await render( samples );
						assert.ok( pixels.every( Number.isFinite ), 'All edge pixels remain finite' );
						assert.ok( pixels[ ( 8 * 16 + 7 ) * 4 + 2 ] < 0.005, 'Blue does not leak across the normal boundary' );
						assert.ok( pixels[ ( 8 * 16 + 8 ) * 4 ] < 0.005, 'Red does not leak across the normal boundary' );

					}

				} );

			} );

		} );

	} );

} );
