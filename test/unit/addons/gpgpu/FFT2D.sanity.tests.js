import { WebGPURenderer, DataTexture, StorageTexture, StorageBufferAttribute, FloatType, NearestFilter } from 'three/webgpu';
import { Fn, storage, texture, uint, int, ivec2, instanceIndex } from 'three/tsl';
import { FFT2D } from '../../../../examples/jsm/gpgpu/FFT2D.js';

// A throwaway correctness sanity check (not wired into any npm script) confirming the
// adaptive-workgroup-sizing refactor of FFT2D.js didn't change its numerical output: a
// forward+inverse round trip should recover the original signal.
export default QUnit.module( 'Sanity', () => {

	QUnit.test( 'FFT2D forward+inverse round trip recovers input', async ( assert ) => {

		assert.timeout( 60000 );

		if ( typeof navigator === 'undefined' || navigator.gpu === undefined || ! ( await navigator.gpu.requestAdapter() ) ) {

			assert.ok( true, 'skipped: no webgpu' );
			return;

		}

		const renderer = new WebGPURenderer();
		await renderer.init();

		const size = 64;
		const data = new Float32Array( size * size * 4 );
		for ( let i = 0; i < size * size; i ++ ) {

			data[ i * 4 ] = Math.sin( i * 0.123 ) * 0.5 + 0.5;
			data[ i * 4 + 1 ] = 0;
			data[ i * 4 + 3 ] = 1;

		}

		const src = new DataTexture( data, size, size, undefined, FloatType );
		src.magFilter = NearestFilter; src.minFilter = NearestFilter; src.generateMipmaps = false; src.needsUpdate = true;

		const spectrum = new StorageTexture( size, size ); spectrum.type = FloatType;
		const recon = new StorageTexture( size, size ); recon.type = FloatType;

		const fft = new FFT2D( size, size );
		fft.computeForward( renderer, src, spectrum );
		fft.computeInverse( renderer, spectrum, recon );

		const sourceNode = texture( recon );
		const probeAttribute = new StorageBufferAttribute( size * size, 2 );
		const probeWrite = storage( probeAttribute, 'vec2', size * size );

		const fullCopy = Fn( () => {

			const x = instanceIndex.mod( uint( size ) );
			const y = instanceIndex.div( uint( size ) );
			probeWrite.element( instanceIndex ).assign( sourceNode.load( ivec2( int( x ), int( y ) ) ).rg );

		} )().compute( size * size );

		renderer.compute( fullCopy );
		const out = new Float32Array( await renderer.getArrayBufferAsync( probeAttribute ) );

		let maxError = 0;
		for ( let i = 0; i < size * size; i ++ ) {

			maxError = Math.max( maxError, Math.abs( out[ i * 2 ] - data[ i * 4 ] ) );

		}

		assert.ok( maxError < 1e-3, `round trip max error ${ maxError } < 1e-3` );

		renderer.dispose();

	} );

} );
