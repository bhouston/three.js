import { DataTexture, DataUtils, FloatType, HalfFloatType, LinearFilter, NodeMaterial, PerspectiveCamera, QuadMesh, RenderTarget, RendererUtils } from 'three/webgpu';
import { context, texture, uniform, uv } from 'three/tsl';
import { reprojectedRadiance } from '../../../../examples/jsm/tsl/display/ReprojectedRadianceNode.js';
import { getSharedRenderer } from '../../src/nodes/tsl/gpu-test-utils.js';

const size = 16;
const deviceDepth = distance => 100 / 99.9 - 10 / ( 99.9 * distance );

async function fixture( assert, run ) {

	const renderer = await getSharedRenderer( 'webgpu' );
	if ( ! renderer ) {

		assert.ok( true, 'SKIPPED: WebGPU unavailable' );
		return;

	}

	const state = RendererUtils.saveRendererState( renderer );
	const oldContext = renderer.contextNode;
	const textures = [];
	const make = ( callback, half = false ) => {

		const data = half ? new Uint16Array( size * size * 4 ) : new Float32Array( size * size * 4 );
		for ( let y = 0; y < size; y ++ ) {

			for ( let x = 0; x < size; x ++ ) data.set( callback( x, y ).map( value => half ? DataUtils.toHalfFloat( value ) : value ), 4 * ( x + size * y ) );

		}

		const value = new DataTexture( data, size, size );
		value.type = half ? HalfFloatType : FloatType;
		if ( half ) value.minFilter = value.magFilter = LinearFilter;
		value.needsUpdate = true;
		textures.push( value );
		return value;

	};

	const color = make( x => x < size / 2 ? [ 4, 2, 1, 1 ] : [ 100, 50, 25, 1 ], true );
	const depth = make( () => [ deviceDepth( 1 ), 0, 0, 0 ] );
	const historyDepth = make( x => [ deviceDepth( x < size / 2 ? 1 : 2 ), 0, 0, 0 ] );
	const normal = make( () => [ 0, 0, 1, 0 ] );
	const camera = new PerspectiveCamera( 50, 1, 0.1, 100 );
	camera.coordinateSystem = renderer.coordinateSystem;
	camera.updateProjectionMatrix();
	camera.updateMatrixWorld();
	const previousProjection = uniform( camera.projectionMatrix.clone() );
	// Move the reprojected center 3/4 of a pixel left: nearest depth is valid,
	// but hardware-linear radiance would mix 25% of the rejected bright surface.
	previousProjection.value.elements[ 8 ] = 1.5 / size;
	const options = {
		previousView: uniform( camera.matrixWorldInverse.clone() ), previousProjection,
		previousProjectionInverse: uniform( previousProjection.value.clone().invert() ),
		valid: uniform( true ), tolerancePixels: uniform( 0.25 ), toleranceWorld: uniform( 0 )
	};
	const source = reprojectedRadiance( texture( color ), texture( historyDepth ), texture( depth ), texture( normal ), camera, options );
	const material = new NodeMaterial();
	material.fragmentNode = source.sample( uv() );
	const target = new RenderTarget( size, size, { type: FloatType, depthBuffer: false } );
	const quad = new QuadMesh( material );
	const render = async () => {

		renderer.setRenderTarget( target );
		quad.render( renderer );
		return Array.from( await renderer.readRenderTargetPixelsAsync( target, 8, 8, 1, 1 ) );

	};

	try {

		renderer.contextNode = context();
		await run( { color, depth, historyDepth, normal, options, render } );

	} finally {

		material.dispose();
		target.dispose();
		for ( const value of textures ) value.dispose();
		renderer.contextNode = oldContext;
		RendererUtils.restoreRendererState( renderer, state );

	}

}

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'TSL', () => {

		QUnit.module( 'ReprojectedRadianceNode', () => {

			QUnit.test( 'historical depth rejection precedes bilinear radiance interpolation', async assert => {

				await fixture( assert, async ( { render } ) => {

					const value = await render();
					assert.ok( value.every( ( channel, i ) => Math.abs( channel - [ 4, 2, 1, 0.75 ][ i ] ) < 0.002 ), `Rejected bright history never enters the valid 75% footprint; actual=${ value }` );

				} );

			} );

			QUnit.test( 'coplanar history survives depth variation within its pixel footprint', async assert => {

				await fixture( assert, async ( { color, depth, historyDepth, normal, options, render } ) => {

					const tangent = Math.tan( 25 * Math.PI / 180 );
					const shift = options.previousProjection.value.elements[ 8 ];
					for ( let y = 0; y < size; y ++ ) {

						for ( let x = 0; x < size; x ++ ) {

							const i = 4 * ( x + size * y );
							const rayX = ( 2 * ( x + 0.5 ) / size - 1 ) * tangent;
							depth.image.data[ i ] = deviceDepth( 0.6 / ( 0.6 - 0.8 * rayX ) );
							historyDepth.image.data[ i ] = deviceDepth( 0.6 / ( 0.6 - 0.8 * ( rayX + shift * tangent ) ) );
							normal.image.data.set( [ 0.8, 0, 0.6, 0 ], i );
							color.image.data.set( [ 4, 2, 1, 1 ].map( DataUtils.toHalfFloat ), i );

						}

					}

					color.needsUpdate = depth.needsUpdate = historyDepth.needsUpdate = normal.needsUpdate = true;
					options.tolerancePixels.value = 0;
					options.toleranceWorld.value = 0.0001;
					const value = await render();
					assert.ok( value.every( ( channel, i ) => Math.abs( channel - [ 4, 2, 1, 1 ][ i ] ) < 0.002 ), `Plane-aware validation accepts a sloped surface even when neighboring view depths differ; actual=${ value }` );

				} );

			} );

			QUnit.test( 'pixel uncertainty and world floor control history independently of trace thickness', async assert => {

				await fixture( assert, async ( { historyDepth, options, render } ) => {

					for ( let i = 0; i < historyDepth.image.data.length; i += 4 ) historyDepth.image.data[ i ] = deviceDepth( 1.01 );
					historyDepth.needsUpdate = true;
					options.tolerancePixels.value = 0;
					options.toleranceWorld.value = 0.001;
					assert.deepEqual( await render(), [ 0, 0, 0, 0 ], 'A surface beyond the world tolerance is rejected' );
					options.tolerancePixels.value = 1;
					assert.ok( ( await render() )[ 3 ] > 0.99, 'One previous pixel footprint supplies geometric uncertainty' );
					options.tolerancePixels.value = 0;
					options.toleranceWorld.value = 0.02;
					assert.ok( ( await render() )[ 3 ] > 0.99, 'The independent world floor can retain a nearby surface' );
					options.valid.value = false;
					assert.deepEqual( await render(), [ 0, 0, 0, 0 ], 'No initialized history means zero support regardless of tolerance' );

				} );

			} );

		} );

	} );

} );
