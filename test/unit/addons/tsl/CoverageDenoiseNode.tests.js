import { DataTexture, DataUtils, FloatType, NodeMaterial, PerspectiveCamera, QuadMesh, RendererUtils, Vector2 } from 'three/webgpu';
import { context, texture } from 'three/tsl';
import { coverageDenoise } from '../../../../examples/jsm/tsl/display/CoverageDenoiseNode.js';
import { getSharedRenderer } from '../../src/nodes/tsl/gpu-test-utils.js';

const size = 16;

async function fixture( assert, run ) {

	const renderer = await getSharedRenderer( 'webgpu' );
	if ( ! renderer ) {

		assert.ok( true, 'SKIPPED: WebGPU unavailable' );
		return;

	}

	const state = RendererUtils.saveRendererState( renderer );
	const oldContext = renderer.contextNode;
	const oldRatio = renderer.getPixelRatio();
	const oldSize = renderer.getSize( new Vector2() );
	const textures = [];
	const makeTexture = callback => {

		const data = new Float32Array( size * size * 4 );
		for ( let y = 0; y < size; y ++ ) {

			for ( let x = 0; x < size; x ++ ) data.set( callback( x, y ), 4 * ( x + size * y ) );

		}

		const result = new DataTexture( data, size, size );
		result.type = FloatType;
		result.needsUpdate = true;
		textures.push( result );
		return result;

	};

	const input = makeTexture( () => [ 0, 0, 0, 0 ] );
	const depth = makeTexture( () => [ 0.5, 0, 0, 0 ] );
	const normal = makeTexture( () => [ 0.5, 0.5, 1, 1 ] );
	const velocity = makeTexture( () => [ 0, 0, 0, 0 ] );
	const camera = new PerspectiveCamera( 50, 1, 0.1, 100 );
	camera.updateMatrixWorld();
	const node = coverageDenoise( texture( input ), camera, { depth: texture( depth ), normal: texture( normal ), velocity: texture( velocity ) } );
	node.radius.value = 1;
	const material = new NodeMaterial();
	material.fragmentNode = node;
	const quad = new QuadMesh( material );
	const update = callback => {

		for ( let y = 0; y < size; y ++ ) {

			for ( let x = 0; x < size; x ++ ) input.image.data.set( callback( x, y ), 4 * ( x + size * y ) );

		}

		input.needsUpdate = true;

	};

	const render = async () => {

		// Frame-updated nodes must run on distinct animation frames.
		await new Promise( resolve => requestAnimationFrame( resolve ) );
		renderer.setRenderTarget( null );
		quad.render( renderer );
		const target = node.getRenderTarget();
		const data = await renderer.readRenderTargetPixelsAsync( target, 0, 0, target.width, target.height );
		return Array.from( data, DataUtils.fromHalfFloat );

	};

	try {

		renderer.setPixelRatio( 1 );
		renderer.setSize( size, size );
		renderer.contextNode = context();
		await run( { node, renderer, input, depth, normal, velocity, update, render } );

	} finally {

		node.dispose();
		material.dispose();
		for ( const value of textures ) value.dispose();
		renderer.contextNode = oldContext;
		renderer.setPixelRatio( oldRatio );
		renderer.setSize( oldSize.x, oldSize.y );
		RendererUtils.restoreRendererState( renderer, state );

	}

}

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'TSL', () => {

		QUnit.module( 'CoverageDenoiseNode', () => {

			QUnit.test( 'sparse HDR hits retain radiance; invalid colored samples are ignored', async assert => {

				await fixture( assert, async ( { update, render } ) => {

					update( ( x, y ) => ( x + y ) % 2 === 0 ? [ 4, 2, 1, 1 ] : [ 100, 50, 25, 0 ] );
					const data = await render();
					let valid = true;
					let partial = 0;
					for ( let i = 0; i < data.length; i += 4 ) {

						valid &&= Math.abs( data[ i ] - 4 ) < 0.01 && Math.abs( data[ i + 1 ] - 2 ) < 0.01 && Math.abs( data[ i + 2 ] - 1 ) < 0.01;
						if ( data[ i + 3 ] > 0 && data[ i + 3 ] < 1 ) partial ++;

					}

					assert.ok( valid, 'Missing samples neither darken nor contaminate valid HDR radiance' );
					assert.strictEqual( partial, size * size, 'Coverage is reconstructed independently of brightness' );

				} );

			} );

			QUnit.test( 'valid black and fractional coverage survive normalization', async assert => {

				await fixture( assert, async ( { update, render } ) => {

					update( () => [ 0, 0, 0, 0.25 ] );
					const black = await render();
					assert.ok( black.every( ( v, i ) => Math.abs( v - ( i % 4 === 3 ? 0.25 : 0 ) ) < 0.001 ), 'Black reflections consume their coverage' );
					update( () => [ 8, 4, 2, 0.25 ] );
					const hdr = await render();
					const expected = [ 8, 4, 2, 0.25 ];
					assert.ok( hdr.every( ( v, i ) => Math.abs( v - expected[ i % 4 ] ) < 0.01 ), 'Fractional coverage does not attenuate unweighted radiance' );

				} );

			} );

			QUnit.test( 'all misses stay finite and reset removes old reflections', async assert => {

				await fixture( assert, async ( { node, update, render } ) => {

					update( () => [ 4, 2, 1, 1 ] );
					await render();
					update( () => [ 100, 50, 25, 0 ] );
					node.reset();
					const data = await render();
					assert.ok( data.every( v => v === 0 ), 'No hit means zero radiance and coverage, including after a bright history' );

				} );

			} );

			QUnit.test( 'temporal filtering uses matching weights for radiance and coverage', async assert => {

				await fixture( assert, async ( { node, update, render } ) => {

					node.radius.value = 0;
					update( x => [ 4, 2, 1, x % 2 ] );
					await render();
					update( x => [ 4, 2, 1, 1 - x % 2 ] );
					const data = await render();
					assert.ok( data.every( ( v, i ) => Math.abs( v - [ 4, 2, 1, 0.5 ][ i % 4 ] ) < 0.01 ), 'Alternating hits converge to half coverage with unchanged lighting' );

				} );

			} );

			for ( const guide of [ 'depth', 'normal', 'velocity' ] ) {

				QUnit.test( `${ guide } rejects disoccluded or off-screen history`, async assert => {

					await fixture( assert, async ( { node, depth, normal, velocity, update, render } ) => {

						node.radius.value = 0;
						update( x => [ 4, 2, 1, x % 2 ] );
						await render();
						const buffer = { depth, normal, velocity }[ guide ];
						const value = { depth: [ 0.8, 0, 0, 0 ], normal: [ 0.5, 0.5, 0, 1 ], velocity: [ 2, 0, 0, 0 ] }[ guide ];
						for ( let i = 0; i < buffer.image.data.length; i += 4 ) buffer.image.data.set( value, i );
						buffer.needsUpdate = true;
						update( x => [ 4, 2, 1, 1 - x % 2 ] );
						const data = await render();
						assert.ok( data.every( ( v, i ) => {

							const coverage = 1 - Math.floor( i / 4 ) % size % 2;
							return Math.abs( v - ( i % 4 === 3 ? coverage : coverage * [ 4, 2, 1 ][ i % 4 ] ) ) < 0.01;

						} ), 'Rejected history does not retain old coverage or radiance' );

					} );

				} );

			}

			QUnit.test( 'resize restarts temporal accumulation', async assert => {

				await fixture( assert, async ( { node, renderer, update, render } ) => {

					update( () => [ 8, 4, 2, 1 ] );
					await render();
					renderer.setSize( 8, 8 );
					update( () => [ 0, 0, 0, 0 ] );
					const data = await render();
					assert.strictEqual( node.getRenderTarget().width, 8, 'Output follows the drawing buffer' );
					assert.ok( data.every( v => v === 0 ), 'Resized buffers do not reuse stale reflections' );

				} );

			} );

			QUnit.test( 'normal boundaries prevent spatial color leaks', async assert => {

				await fixture( assert, async ( { node, normal, update, render } ) => {

					for ( let y = 0; y < size; y ++ ) {

						for ( let x = size / 2; x < size; x ++ ) normal.image.data.set( [ 1, 0.5, 0.5, 1 ], 4 * ( x + size * y ) );

					}

					normal.needsUpdate = true;
					update( x => x < size / 2 ? [ 4, 0, 0, 1 ] : [ 0, 0, 4, 1 ] );
					const data = await render();
					assert.ok( data.every( ( v, i ) => Math.abs( v - ( ( Math.floor( i / 4 ) % size < size / 2 ? [ 4, 0, 0, 1 ] : [ 0, 0, 4, 1 ] )[ i % 4 ] ) ) < 0.01 ), 'Perpendicular surfaces retain separate radiance' );
					node.reset();

				} );

			} );

		} );

	} );

} );
