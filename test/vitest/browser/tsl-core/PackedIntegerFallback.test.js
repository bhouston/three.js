import { afterEach, describe, expect, it } from 'vitest';
import * as THREE from 'three';
import {
	dot4I8Packed,
	dot4U8Packed,
	ivec4,
	pack4xI8,
	pack4xU8,
	setPackedIntegerDotProductFeatureOverrideForTesting,
	uint,
	unpack4xI8,
	unpack4xU8,
	uvec4,
	vec4
} from 'three/tsl';

// The "Packed 4x8 Integer Dot Product" WGSL builtins that PackedDotProductNode.js/
// PackIntegerNode.js/UnpackIntegerNode.js wrap are an *optional* WGSL language
// feature - not every WebGPU implementation is guaranteed to support it, even
// though every environment this repo's own tests actually run in does. Each of
// those three node classes therefore checks
// PackedIntegerFeatureDetection.js's supportsPackedIntegerDotProductFeature()
// on the WebGPU backend and falls back to a hand-written TSL polyfill (built
// from ordinary bitwise operators) when it's absent, rather than assuming
// WebGPU always implies support.
//
// There's no real environment available to this test suite where the feature
// is actually absent, so this file uses
// setPackedIntegerDotProductFeatureOverrideForTesting() (a deliberate,
// documented test-only hook - see PackedIntegerFeatureDetection.js) to force
// that fallback branch to run on a real WebGPU backend, and checks it against
// the same expected values PackIntegerNode.test.js/PackedDotProductNode.test.js
// already confirm for the native path - so a regression in the fallback
// specifically (not just the native call) would show up here.

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
 * See PackIntegerNode.test.js's `readPackedUint` for why: a raw packed uint
 * can't survive a half-float readback intact (out of range, and imprecise
 * even within range), so this splits it into its 4 individual bytes first.
 */
function bitByte( packedNode, shiftAmount ) {

	return packedNode.shiftRight( shiftAmount ).bitAnd( 0xFF ).toFloat();

}

async function readPackedUint( renderer, packedNode ) {

	const bytes = await bakeAndReadPixel( renderer, vec4(
		bitByte( packedNode, 0 ), bitByte( packedNode, 8 ), bitByte( packedNode, 16 ), bitByte( packedNode, 24 )
	) );

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

describe( 'PackedDotProductNode/PackIntegerNode/UnpackIntegerNode - WebGPU fallback (forced, no language feature)', () => {

	let renderer;

	afterEach( async () => {

		setPackedIntegerDotProductFeatureOverrideForTesting( undefined );
		renderer?.dispose();
		renderer = undefined;

	} );

	async function getRenderer() {

		if ( ! renderer ) {

			setPackedIntegerDotProductFeatureOverrideForTesting( false );
			renderer = new THREE.WebGPURenderer( { forceWebGL: false } );
			await renderer.init();

		}

		return renderer;

	}

	it( 'pack4xI8 fallback matches the hand-computed little-endian signed packing', async () => {

		const r = await getRenderer();

		const values = [ 3, - 4, 127, - 128 ];
		const expected = packInt8x4( values );

		const result = await readPackedUint( r, pack4xI8( ivec4( ...values ) ) );

		expect( result ).toBe( expected );

	} );

	it( 'pack4xU8 fallback matches the hand-computed little-endian unsigned packing', async () => {

		const r = await getRenderer();

		const values = [ 3, 200, 5, 255 ];
		const expected = packUint8x4( values );

		const result = await readPackedUint( r, pack4xU8( uvec4( ...values ) ) );

		expect( result ).toBe( expected );

	} );

	it( 'unpack4xI8 fallback is the exact inverse of pack4xI8 across the full signed int8 range', async () => {

		const r = await getRenderer();

		const values = [ 127, - 128, - 1, 42 ];
		const packed = packInt8x4( values );

		const unpacked = unpack4xI8( uint( packed ) );
		const node = vec4( unpacked.x.toFloat(), unpacked.y.toFloat(), unpacked.z.toFloat(), unpacked.w.toFloat() );
		const result = await bakeAndReadPixel( r, node );

		expect( result ).toEqual( values.map( ( v ) => expect.closeTo( v, 5 ) ) );

	} );

	it( 'unpack4xI8 fallback sign-extends a byte with the high bit set (confirms sign extension, not unsigned truncation)', async () => {

		const r = await getRenderer();

		// 200 (as an unsigned byte) is -56 as a signed int8 (200 - 256) -
		// packUint8x4 produces the exact bit pattern packInt8x4 would for
		// [-56, 0, 0, 0], letting this construct that bit pattern directly
		// without packInt8x4's own clamping getting in the way.
		const packed = packUint8x4( [ 200, 0, 0, 0 ] );

		const unpacked = unpack4xI8( uint( packed ) );
		const [ result ] = await bakeAndReadPixel( r, unpacked.x.toFloat() );

		expect( result ).toBeCloseTo( - 56, 5 );

	} );

	it( 'unpack4xU8 fallback is the exact inverse of pack4xU8', async () => {

		const r = await getRenderer();

		const values = [ 0, 128, 255, 42 ];
		const packed = packUint8x4( values );

		const unpacked = unpack4xU8( uint( packed ) );
		const node = vec4( unpacked.x.toFloat(), unpacked.y.toFloat(), unpacked.z.toFloat(), unpacked.w.toFloat() );
		const result = await bakeAndReadPixel( r, node );

		expect( result ).toEqual( values.map( ( v ) => expect.closeTo( v, 5 ) ) );

	} );

	it( 'dot4I8Packed fallback matches a hand-computed signed int8 dot product', async () => {

		const r = await getRenderer();

		const a = [ 3, - 4, 5, - 6 ];
		const b = [ 1, 2, - 3, 4 ];
		const expectedDot = a.reduce( ( sum, value, i ) => sum + value * b[ i ], 0 );

		const node = dot4I8Packed( uint( packInt8x4( a ) ), uint( packInt8x4( b ) ) ).toFloat();
		const [ result ] = await bakeAndReadPixel( r, node );

		expect( result ).toBeCloseTo( expectedDot, 5 );

	} );

	it( 'dot4U8Packed fallback matches a hand-computed unsigned uint8 dot product', async () => {

		const r = await getRenderer();

		const a = [ 3, 200, 5, 250 ];
		const b = [ 1, 2, 100, 4 ];
		const expectedDot = a.reduce( ( sum, value, i ) => sum + value * b[ i ], 0 );

		const node = dot4U8Packed( uint( packUint8x4( a ) ), uint( packUint8x4( b ) ) ).toFloat();
		const [ result ] = await bakeAndReadPixel( r, node );

		expect( result ).toBeCloseTo( expectedDot, 5 );

	} );

} );
