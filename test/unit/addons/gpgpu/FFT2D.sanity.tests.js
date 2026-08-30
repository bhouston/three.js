import { WebGPURenderer, DataTexture, StorageTexture, StorageBufferAttribute, FloatType, NearestFilter } from 'three/webgpu';
import { Fn, storage, texture, uint, int, ivec2, instanceIndex } from 'three/tsl';
import { FFT2D } from '../../../../examples/jsm/gpgpu/FFT2D.js';

// Throwaway correctness sanity checks (not wired into any npm script) confirming FFT2D's
// numerical output, both the plain complex path and the real (row-pair-packed) fast path.

async function makeRenderer() {

	if ( typeof navigator === 'undefined' || navigator.gpu === undefined || ! ( await navigator.gpu.requestAdapter() ) ) {

		return null;

	}

	const renderer = new WebGPURenderer();
	await renderer.init();
	return renderer;

}

function makeRealDataTexture( width, height, fn ) {

	const data = new Float32Array( width * height * 4 );

	for ( let i = 0; i < width * height; i ++ ) {

		data[ i * 4 ] = fn( i );
		data[ i * 4 + 1 ] = 0;
		data[ i * 4 + 3 ] = 1;

	}

	const tex = new DataTexture( data, width, height, undefined, FloatType );
	tex.magFilter = NearestFilter; tex.minFilter = NearestFilter; tex.generateMipmaps = false; tex.needsUpdate = true;

	return { texture: tex, data };

}

function makeComplexStorageTexture( width, height ) {

	const tex = new StorageTexture( width, height );
	tex.type = FloatType;
	return tex;

}

// Reads a StorageTexture's `.rg` back via a small probe compute kernel, forcing a full readback
// through a storage buffer.
async function readTextureRG( renderer, width, height, tex ) {

	const sourceNode = texture( tex );
	const probeAttribute = new StorageBufferAttribute( width * height, 2 );
	const probeWrite = storage( probeAttribute, 'vec2', width * height );

	const kernel = Fn( () => {

		const x = instanceIndex.mod( uint( width ) );
		const y = instanceIndex.div( uint( width ) );
		probeWrite.element( instanceIndex ).assign( sourceNode.load( ivec2( int( x ), int( y ) ) ).rg );

	} )().compute( width * height );

	renderer.compute( kernel );

	return new Float32Array( await renderer.getArrayBufferAsync( probeAttribute ) );

}

function maxDiff( a, b, stride, offset ) {

	let maxError = 0;

	for ( let i = 0; i < a.length / stride; i ++ ) {

		maxError = Math.max( maxError, Math.abs( a[ i * stride + offset ] - b[ i * stride + offset ] ) );

	}

	return maxError;

}

export default QUnit.module( 'Sanity', () => {

	QUnit.test( 'FFT2D forward+inverse round trip recovers input', async ( assert ) => {

		assert.timeout( 60000 );

		const renderer = await makeRenderer();

		if ( renderer === null ) {

			assert.ok( true, 'skipped: no webgpu' );
			return;

		}

		const size = 64;
		const { texture: src, data } = makeRealDataTexture( size, size, ( i ) => Math.sin( i * 0.123 ) * 0.5 + 0.5 );

		const spectrum = makeComplexStorageTexture( size, size );
		const recon = makeComplexStorageTexture( size, size );

		const fft = new FFT2D( size, size );
		fft.computeForward( renderer, src, spectrum );
		fft.computeInverse( renderer, spectrum, recon );

		const out = await readTextureRG( renderer, size, size, recon );

		let maxError = 0;
		for ( let i = 0; i < size * size; i ++ ) maxError = Math.max( maxError, Math.abs( out[ i * 2 ] - data[ i * 4 ] ) );

		assert.ok( maxError < 1e-3, `round trip max error ${ maxError } < 1e-3` );

		renderer.dispose();

	} );

	QUnit.test( 'FFT2D computeForwardReal matches computeForward on real input', async ( assert ) => {

		assert.timeout( 60000 );

		const renderer = await makeRenderer();

		if ( renderer === null ) {

			assert.ok( true, 'skipped: no webgpu' );
			return;

		}

		// Non-square (width != height) to exercise row-pair packing (height/2) independently of width.
		const width = 64, height = 32;
		const { texture: src } = makeRealDataTexture( width, height, ( i ) => Math.sin( i * 0.091 ) * 0.5 + Math.cos( i * 0.037 ) * 0.3 );

		const specRef = makeComplexStorageTexture( width, height );
		const specFast = makeComplexStorageTexture( width, height );

		const fftRef = new FFT2D( width, height );
		fftRef.computeForward( renderer, src, specRef );

		const fftFast = new FFT2D( width, height );
		fftFast.computeForwardReal( renderer, src, specFast );

		const outRef = await readTextureRG( renderer, width, height, specRef );
		const outFast = await readTextureRG( renderer, width, height, specFast );

		const err = Math.max( maxDiff( outRef, outFast, 2, 0 ), maxDiff( outRef, outFast, 2, 1 ) );

		assert.ok( err < 1e-2, `spectrum max error ${ err } < 1e-2` );

		renderer.dispose();

	} );

	QUnit.test( 'FFT2D computeForwardReal+computeInverseReal round trip recovers input', async ( assert ) => {

		assert.timeout( 60000 );

		const renderer = await makeRenderer();

		if ( renderer === null ) {

			assert.ok( true, 'skipped: no webgpu' );
			return;

		}

		const width = 64, height = 32;
		const { texture: src, data } = makeRealDataTexture( width, height, ( i ) => Math.sin( i * 0.091 ) * 0.5 + Math.cos( i * 0.037 ) * 0.3 );

		const spectrum = makeComplexStorageTexture( width, height );
		const recon = makeComplexStorageTexture( width, height );

		const fft = new FFT2D( width, height );
		fft.computeForwardReal( renderer, src, spectrum );
		fft.computeInverseReal( renderer, spectrum, recon );

		const out = await readTextureRG( renderer, width, height, recon );

		let maxError = 0;
		for ( let i = 0; i < width * height; i ++ ) maxError = Math.max( maxError, Math.abs( out[ i * 2 ] - data[ i * 4 ] ) );

		assert.ok( maxError < 1e-3, `round trip max error ${ maxError } < 1e-3` );

		renderer.dispose();

	} );

} );
