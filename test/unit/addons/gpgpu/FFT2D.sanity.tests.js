import { WebGPURenderer, DataTexture, StorageTexture, StorageBufferAttribute, FloatType, NearestFilter } from 'three/webgpu';
import { Fn, storage, texture, uint, int, ivec2, instanceIndex } from 'three/tsl';
import { FFT2D } from '../../../../examples/jsm/gpgpu/FFT2D.js';

// Throwaway correctness sanity checks (not wired into any npm script) confirming FFT2D's
// numerical output, both the plain complex path and the real2 "pack two reals into one complex
// FFT" path.

function makeRenderer() {

	return ( async () => {

		if ( typeof navigator === 'undefined' || navigator.gpu === undefined || ! ( await navigator.gpu.requestAdapter() ) ) {

			return null;

		}

		const renderer = new WebGPURenderer();
		await renderer.init();
		return renderer;

	} )();

}

function makeRealDataTexture( size, fn ) {

	const data = new Float32Array( size * size * 4 );

	for ( let i = 0; i < size * size; i ++ ) {

		data[ i * 4 ] = fn( i );
		data[ i * 4 + 1 ] = 0;
		data[ i * 4 + 3 ] = 1;

	}

	const tex = new DataTexture( data, size, size, undefined, FloatType );
	tex.magFilter = NearestFilter; tex.minFilter = NearestFilter; tex.generateMipmaps = false; tex.needsUpdate = true;

	return { texture: tex, data };

}

function makeComplexStorageTexture( size ) {

	const tex = new StorageTexture( size, size );
	tex.type = FloatType;
	return tex;

}

// Reads a StorageTexture's `.rg` (or just `.r`, for a real-only one) back via a small probe
// compute kernel, forcing a full readback through a storage buffer.
async function readTextureRG( renderer, size, tex ) {

	const sourceNode = texture( tex );
	const probeAttribute = new StorageBufferAttribute( size * size, 2 );
	const probeWrite = storage( probeAttribute, 'vec2', size * size );

	const kernel = Fn( () => {

		const x = instanceIndex.mod( uint( size ) );
		const y = instanceIndex.div( uint( size ) );
		probeWrite.element( instanceIndex ).assign( sourceNode.load( ivec2( int( x ), int( y ) ) ).rg );

	} )().compute( size * size );

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
		const { texture: src, data } = makeRealDataTexture( size, ( i ) => Math.sin( i * 0.123 ) * 0.5 + 0.5 );

		const spectrum = makeComplexStorageTexture( size );
		const recon = makeComplexStorageTexture( size );

		const fft = new FFT2D( size, size );
		fft.computeForward( renderer, src, spectrum );
		fft.computeInverse( renderer, spectrum, recon );

		const out = await readTextureRG( renderer, size, recon );

		let maxError = 0;
		for ( let i = 0; i < size * size; i ++ ) maxError = Math.max( maxError, Math.abs( out[ i * 2 ] - data[ i * 4 ] ) );

		assert.ok( maxError < 1e-3, `round trip max error ${ maxError } < 1e-3` );

		renderer.dispose();

	} );

	QUnit.test( 'FFT2D real2 forward matches two separate forward transforms', async ( assert ) => {

		assert.timeout( 60000 );

		const renderer = await makeRenderer();

		if ( renderer === null ) {

			assert.ok( true, 'skipped: no webgpu' );
			return;

		}

		const size = 64;
		const { texture: srcA } = makeRealDataTexture( size, ( i ) => Math.sin( i * 0.123 ) * 0.5 + 0.5 );
		const { texture: srcB } = makeRealDataTexture( size, ( i ) => Math.cos( i * 0.071 ) * 0.5 + 0.5 );

		const specA = makeComplexStorageTexture( size );
		const specB = makeComplexStorageTexture( size );
		const specA2 = makeComplexStorageTexture( size );
		const specB2 = makeComplexStorageTexture( size );

		// Reference: two separate full complex transforms (imag = 0 on input, via makeRealDataTexture).
		const fftRef = new FFT2D( size, size );
		fftRef.computeForward( renderer, srcA, specA );
		fftRef.computeForward( renderer, srcB, specB );

		// Real2: one packed transform.
		const fft = new FFT2D( size, size );
		fft.computeForwardReal2( renderer, srcA, srcB, specA2, specB2 );

		const outA = await readTextureRG( renderer, size, specA );
		const outB = await readTextureRG( renderer, size, specB );
		const outA2 = await readTextureRG( renderer, size, specA2 );
		const outB2 = await readTextureRG( renderer, size, specB2 );

		const errA = Math.max( maxDiff( outA, outA2, 2, 0 ), maxDiff( outA, outA2, 2, 1 ) );
		const errB = Math.max( maxDiff( outB, outB2, 2, 0 ), maxDiff( outB, outB2, 2, 1 ) );

		assert.ok( errA < 1e-2, `spectrum A max error ${ errA } < 1e-2` );
		assert.ok( errB < 1e-2, `spectrum B max error ${ errB } < 1e-2` );

		renderer.dispose();

	} );

	QUnit.test( 'FFT2D real2 forward+inverse round trip recovers both inputs', async ( assert ) => {

		assert.timeout( 60000 );

		const renderer = await makeRenderer();

		if ( renderer === null ) {

			assert.ok( true, 'skipped: no webgpu' );
			return;

		}

		const size = 64;
		const { texture: srcA, data: dataA } = makeRealDataTexture( size, ( i ) => Math.sin( i * 0.123 ) * 0.5 + 0.5 );
		const { texture: srcB, data: dataB } = makeRealDataTexture( size, ( i ) => Math.cos( i * 0.071 ) * 0.5 + 0.5 );

		const specA = makeComplexStorageTexture( size );
		const specB = makeComplexStorageTexture( size );
		const reconA = makeComplexStorageTexture( size );
		const reconB = makeComplexStorageTexture( size );

		const fft = new FFT2D( size, size );
		fft.computeForwardReal2( renderer, srcA, srcB, specA, specB );
		fft.computeInverseReal2( renderer, specA, specB, reconA, reconB );

		const outA = await readTextureRG( renderer, size, reconA );
		const outB = await readTextureRG( renderer, size, reconB );

		let maxErrorA = 0, maxErrorB = 0;
		for ( let i = 0; i < size * size; i ++ ) {

			maxErrorA = Math.max( maxErrorA, Math.abs( outA[ i * 2 ] - dataA[ i * 4 ] ) );
			maxErrorB = Math.max( maxErrorB, Math.abs( outB[ i * 2 ] - dataB[ i * 4 ] ) );

		}

		assert.ok( maxErrorA < 1e-3, `signal A round trip max error ${ maxErrorA } < 1e-3` );
		assert.ok( maxErrorB < 1e-3, `signal B round trip max error ${ maxErrorB } < 1e-3` );

		renderer.dispose();

	} );

} );
