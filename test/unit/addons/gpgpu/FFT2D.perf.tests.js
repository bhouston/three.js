import { DataTexture, StorageTexture, StorageBufferAttribute, FloatType, NearestFilter } from 'three/webgpu';
import { Fn, storage, texture, uint, int, ivec2, uniform } from 'three/tsl';

import { FFT2D } from '../../../../examples/jsm/gpgpu/FFT2D.js';
import { isWebGPUAvailable, createRenderer, benchmark, report, seededFloat32Array } from './perf-utils.js';

// Sizes to benchmark, largest last. Must be powers of two - FFT2D's requirement. Edit this to try
// other scales.
const SIZES = [ 256, 512, 1024, 2048 ];

// A float source texture packed with deterministic pseudo-random (real, imag) pairs - FFT2D's
// complex-number format, `.rg` = `(real, imag)`.
function seededComplexTexture( width, height ) {

	const data = seededFloat32Array( width * height * 4 );

	// Zero out b/a so only .rg (the complex value FFT2D reads) carries data; keeps the source
	// texture's unused channels well-defined rather than more pseudo-random noise.
	for ( let i = 0; i < width * height; i ++ ) {

		data[ i * 4 + 2 ] = 0;
		data[ i * 4 + 3 ] = 1;

	}

	const tex = new DataTexture( data, width, height, undefined, FloatType );
	tex.magFilter = NearestFilter;
	tex.minFilter = NearestFilter;
	tex.generateMipmaps = false;
	tex.needsUpdate = true;

	return tex;

}

function createComplexStorageTexture( width, height ) {

	const tex = new StorageTexture( width, height );
	tex.type = FloatType;
	tex.magFilter = NearestFilter;
	tex.minFilter = NearestFilter;
	tex.generateMipmaps = false;

	return tex;

}

// A tiny read-back kernel: copies destinationTexture's bin 0 into a 2-element storage buffer, so
// `benchmark`'s `sync` can force the GPU to finish a `computeForward` call via
// `renderer.getArrayBufferAsync` without transferring the whole texture back.
function buildProbe( width, sourceTexture ) {

	const sourceNode = texture( sourceTexture );
	const probeAttribute = new StorageBufferAttribute( 1, 2 );
	const probeWrite = storage( probeAttribute, 'vec2', 1 );
	const indexUniform = uniform( 0, 'uint' );

	const kernel = Fn( () => {

		const x = indexUniform.mod( uint( width ) );
		const y = indexUniform.div( uint( width ) );

		probeWrite.element( 0 ).assign( sourceNode.load( ivec2( int( x ), int( y ) ) ).rg );

	} )().compute( 1 );

	return { kernel, probeAttribute };

}

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'GPGPU', () => {

		QUnit.module( 'FFT2D (perf)', () => {

			for ( const size of SIZES ) {

				QUnit.test( `computeForward() over ${ size }x${ size }`, async ( assert ) => {

					assert.timeout( 180000 );

					if ( ! ( await isWebGPUAvailable() ) ) {

						assert.ok( true, 'skipped: WebGPU is not available in this environment' );
						return;

					}

					const renderer = await createRenderer();

					const sourceTexture = seededComplexTexture( size, size );
					const destinationTexture = createComplexStorageTexture( size, size );

					const fft = new FFT2D( size, size );

					const { kernel: probeKernel, probeAttribute } = buildProbe( size, destinationTexture );

					const stats = await benchmark(
						() => fft.computeForward( renderer, sourceTexture, destinationTexture ),
						async () => {

							renderer.compute( probeKernel );
							await renderer.getArrayBufferAsync( probeAttribute );

						},
						{ runs: 50, warmup: 5 },
					);

					report( assert, 'FFT2D.computeForward()', size * size, stats );

					fft.dispose();
					sourceTexture.dispose();
					destinationTexture.dispose();
					renderer.dispose();

				} );

			}

		} );

	} );

} );
