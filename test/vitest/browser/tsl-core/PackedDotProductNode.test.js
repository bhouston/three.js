import { afterEach, describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { dot4I8Packed, dot4U8Packed, float, uint, vec4 } from 'three/tsl';

// Core TSL node test for dot4I8Packed/dot4U8Packed (src/nodes/math/
// PackedDotProductNode.js) - not addon-specific (examples/jsm/ntc's own
// NTCPackedDotProduct.js now just re-exports/consumes these directly,
// see that file's own tests for the higher-level MLP-evaluator usage).
// Lives under this vitest project (real browser, real WebGPU launch args -
// see vitest.config.js) since that's needed for the WebGPU/native half of
// this test; the WebGL/polyfill half only needs an ordinary WebGL2 context,
// which the same Chromium instance already provides with no special flags.
//
// Runs the exact same assertions against two renderer configurations -
// `forceWebGL: false` (native WGSL dot4I8Packed/dot4U8Packed on WebGPU) and
// `forceWebGL: true` (the GLSL polyfill on the WebGL fallback backend,
// registered in GLSLNodeBuilder.js's glslPolyfills/glslMethods) - so a
// regression in either implementation shows up here, and so the two are
// cross-checked against exactly the same expected values, not just each
// checked in isolation.

/**
 * Renders `fragmentNode` to a 32x32 half-float target (32 texels avoids the
 * RGBA16F readback row-alignment case entirely - not what this file is
 * testing) via a fullscreen quad, and reads pixel (0,0) back as
 * `[r,g,b,a]` floats.
 */
async function bakeAndReadPixel( renderer, fragmentNode ) {

	const size = 32;

	const scene = new THREE.Scene();
	const camera = new THREE.OrthographicCamera( - 1, 1, 1, - 1, 0, 4 );
	camera.position.set( 0, 0, 2 );

	const material = new THREE.NodeMaterial();
	material.lights = false;
	material.toneMapped = false;
	material.blending = THREE.NoBlending;
	material.fragmentNode = vec4( fragmentNode );

	const mesh = new THREE.Mesh( new THREE.PlaneGeometry( 2, 2 ), material );
	scene.add( mesh );

	const renderTarget = new THREE.RenderTarget( size, size, { type: THREE.HalfFloatType } );
	const previousTarget = renderer.getRenderTarget();
	renderer.setRenderTarget( renderTarget );
	renderer.render( scene, camera );
	renderer.setRenderTarget( previousTarget );

	const pixels = await renderer.readRenderTargetPixelsAsync( renderTarget, 0, 0, 1, 1 );

	renderTarget.dispose();
	material.dispose();
	mesh.geometry.dispose();

	return [ 0, 1, 2, 3 ].map( ( i ) => THREE.DataUtils.fromHalfFloat( pixels[ i ] ) );

}

function packInt8x4( values ) {

	let packed = 0;
	for ( let i = 0; i < 4; i ++ ) {

		const value = Math.max( - 128, Math.min( 127, Math.round( values[ i ] || 0 ) ) );
		packed |= ( value & 0xFF ) << ( 8 * i );

	}

	return packed >>> 0;

}

function packUint8x4( values ) {

	let packed = 0;
	for ( let i = 0; i < 4; i ++ ) {

		const value = Math.max( 0, Math.min( 255, Math.round( values[ i ] || 0 ) ) );
		packed |= ( value & 0xFF ) << ( 8 * i );

	}

	return packed >>> 0;

}

const backends = [
	{ name: 'WebGPU (native dot4I8Packed/dot4U8Packed)', forceWebGL: false },
	{ name: 'WebGL fallback (GLSL polyfill)', forceWebGL: true }
];

describe.each( backends )( 'PackedDotProductNode - $name', ( { forceWebGL } ) => {

	let renderer;

	afterEach( () => {

		renderer?.dispose();
		renderer = undefined;

	} );

	async function getRenderer() {

		if ( ! renderer ) {

			renderer = new THREE.WebGPURenderer( { forceWebGL } );
			await renderer.init();

		}

		return renderer;

	}

	it( 'dot4I8Packed matches a hand-computed signed int8 dot product', async () => {

		const r = await getRenderer();

		const a = [ 3, - 4, 5, - 6 ];
		const b = [ 1, 2, - 3, 4 ];
		const expectedDot = a.reduce( ( sum, value, i ) => sum + value * b[ i ], 0 );

		const node = float( dot4I8Packed( uint( packInt8x4( a ) ), uint( packInt8x4( b ) ) ) );
		const [ result ] = await bakeAndReadPixel( r, node );

		expect( result ).toBeCloseTo( expectedDot, 5 );

	} );

	it( 'dot4I8Packed handles the full signed int8 range (127, -128) without overflow', async () => {

		const r = await getRenderer();

		const a = [ 127, - 128, 127, - 128 ];
		const b = [ 127, 127, - 128, - 128 ];
		const expectedDot = a.reduce( ( sum, value, i ) => sum + value * b[ i ], 0 ); // 127*127 + -128*127 + 127*-128 + -128*-128 = 254

		const node = float( dot4I8Packed( uint( packInt8x4( a ) ), uint( packInt8x4( b ) ) ) );
		const [ result ] = await bakeAndReadPixel( r, node );

		expect( result ).toBeCloseTo( expectedDot, 5 );

	} );

	it( 'dot4U8Packed matches a hand-computed unsigned uint8 dot product', async () => {

		const r = await getRenderer();

		const a = [ 3, 200, 5, 250 ];
		const b = [ 1, 2, 100, 4 ];
		const expectedDot = a.reduce( ( sum, value, i ) => sum + value * b[ i ], 0 );

		const node = float( dot4U8Packed( uint( packUint8x4( a ) ), uint( packUint8x4( b ) ) ) );
		const [ result ] = await bakeAndReadPixel( r, node );

		expect( result ).toBeCloseTo( expectedDot, 5 );

	} );

	it( 'dot4I8Packed and dot4U8Packed diverge for a byte with the high bit set, confirming each reads its operands with the documented signedness', async () => {

		const r = await getRenderer();

		// 200 (as an unsigned byte) is -56 as a signed int8 (200 - 256).
		const packedA = packUint8x4( [ 200, 0, 0, 0 ] ); // same bit pattern packInt8x4 would produce for [-56, 0, 0, 0]
		const packedB = packUint8x4( [ 1, 0, 0, 0 ] );

		const unsignedNode = float( dot4U8Packed( uint( packedA ), uint( packedB ) ) );
		const signedNode = float( dot4I8Packed( uint( packedA ), uint( packedB ) ) );

		const [ unsignedResult ] = await bakeAndReadPixel( r, unsignedNode );
		const [ signedResult ] = await bakeAndReadPixel( r, signedNode );

		expect( unsignedResult ).toBeCloseTo( 200, 5 );
		expect( signedResult ).toBeCloseTo( - 56, 5 );

	} );

} );
