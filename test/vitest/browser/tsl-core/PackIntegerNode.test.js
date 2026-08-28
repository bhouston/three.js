import { afterEach, describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { bitAnd, float, ivec4, pack4xI8, pack4xU8, shiftRight, uint, unpack4xI8, unpack4xU8, uvec4, vec4 } from 'three/tsl';

// Core TSL node test for pack4xI8/pack4xU8/unpack4xI8/unpack4xU8 (src/nodes/
// math/PackIntegerNode.js, src/nodes/math/UnpackIntegerNode.js) - the rest
// of the "Packed 4x8 Integer Dot Product" WGSL language feature, alongside
// dot4I8Packed/dot4U8Packed (see PackedDotProductNode.test.js, the direct
// precedent this file follows).
//
// Runs the exact same assertions against two renderer configurations -
// `forceWebGL: false` (native WGSL pack4xI8/pack4xU8/unpack4xI8/unpack4xU8
// on WebGPU) and `forceWebGL: true` (the GLSL polyfill on the WebGL
// fallback backend, registered in GLSLNodeBuilder.js's glslPolyfills/
// glslMethods) - so a regression in either implementation shows up here.

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

/**
 * A packed uint can exceed half-float's ~65504 max representable value (a
 * packed byte in bits [24, 32) alone can contribute up to ~4.28 billion) -
 * and even within range, half-float's 11-bit mantissa can't exactly
 * represent every integer above 2048, so a coarser split than "two 16-bit
 * halves" is needed. Splits the packed value into its 4 individual bytes
 * (each 0-255, trivially exact in half-float) before baking, and
 * reconstructs the full uint value on the JS side.
 */
async function readPackedUint( renderer, packedNode ) {

	const byte = ( shiftAmount ) => float( bitAnd( shiftRight( packedNode, uint( shiftAmount ) ), uint( 0xFF ) ) );

	const bytes = await bakeAndReadPixel( renderer, vec4( byte( 0 ), byte( 8 ), byte( 16 ), byte( 24 ) ) );

	return bytes.reduce( ( sum, value, i ) => sum + Math.round( value ) * ( 256 ** i ), 0 );

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
	{ name: 'WebGPU (native pack4xI8/pack4xU8/unpack4xI8/unpack4xU8)', forceWebGL: false },
	{ name: 'WebGL fallback (GLSL polyfill)', forceWebGL: true }
];

describe.each( backends )( 'PackIntegerNode/UnpackIntegerNode - $name', ( { forceWebGL } ) => {

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

	it( 'pack4xI8 matches the hand-computed little-endian signed packing', async () => {

		const r = await getRenderer();

		const values = [ 3, - 4, 127, - 128 ];
		const expected = packInt8x4( values );

		const result = await readPackedUint( r, pack4xI8( ivec4( ...values ) ) );

		expect( result ).toBe( expected );

	} );

	it( 'pack4xU8 matches the hand-computed little-endian unsigned packing', async () => {

		const r = await getRenderer();

		const values = [ 3, 200, 5, 255 ];
		const expected = packUint8x4( values );

		const result = await readPackedUint( r, pack4xU8( uvec4( ...values ) ) );

		expect( result ).toBe( expected );

	} );

	it( 'unpack4xI8 is the exact inverse of pack4xI8 across the full signed int8 range', async () => {

		const r = await getRenderer();

		const values = [ 127, - 128, - 1, 42 ];
		const packed = packInt8x4( values );

		const unpacked = unpack4xI8( uint( packed ) );
		const node = vec4( float( unpacked.x ), float( unpacked.y ), float( unpacked.z ), float( unpacked.w ) );
		const result = await bakeAndReadPixel( r, node );

		expect( result ).toEqual( values.map( ( v ) => expect.closeTo( v, 5 ) ) );

	} );

	it( 'unpack4xU8 is the exact inverse of pack4xU8 across the full unsigned uint8 range', async () => {

		const r = await getRenderer();

		const values = [ 0, 128, 255, 42 ];
		const packed = packUint8x4( values );

		const unpacked = unpack4xU8( uint( packed ) );
		const node = vec4( float( unpacked.x ), float( unpacked.y ), float( unpacked.z ), float( unpacked.w ) );
		const result = await bakeAndReadPixel( r, node );

		expect( result ).toEqual( values.map( ( v ) => expect.closeTo( v, 5 ) ) );

	} );

	it( 'unpack4xI8(pack4xI8(x)) round-trips for a value with the high bit set (confirms sign extension, not just unsigned truncation)', async () => {

		const r = await getRenderer();

		// 200 as a signed int8 wraps to -56 (200 - 256) - pack4xI8 should
		// store its low byte as 0xC8 and unpack4xI8 should sign-extend that
		// back to -56, not truncate it to 200 (that would indicate the GLSL
		// polyfill's shift-based sign extension idiom is broken).
		const packed = pack4xI8( ivec4( 200, 0, 0, 0 ) );
		const unpacked = unpack4xI8( packed );
		const node = float( unpacked.x );

		const [ result ] = await bakeAndReadPixel( r, node );

		expect( result ).toBeCloseTo( - 56, 5 );

	} );

} );
