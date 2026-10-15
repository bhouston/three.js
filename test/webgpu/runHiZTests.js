import { WebGPURenderer, DataTexture, FloatType, RedFormat, RenderTarget, NodeMaterial, QuadMesh } from 'three/webgpu';
import { texture, vec4 } from 'three/tsl';
import { hiZ } from 'three/addons/tsl/display/HiZNode.js';
import { reduceDepth, traceSamples } from './HiZReference.js';

function assert( condition, message ) {

	if ( ! condition ) throw new Error( message );

}

export async function runHiZTests() {

	assert( navigator.gpu, 'Hardware WebGPU is required.' );
	const adapter = await navigator.gpu.requestAdapter( { powerPreference: 'high-performance' } );
	assert( adapter, 'No WebGPU adapter.' );
	const info = adapter.info;
	const description = [ info.vendor, info.architecture, info.device, info.description ].join( ' ' );
	assert( info.isFallbackAdapter !== true && adapter.isFallbackAdapter !== true && ! /swiftshader|llvmpipe|lavapipe|software|microsoft basic render/i.test( description ), `Software adapter rejected: ${description}` );
	assert( description.trim().length > 0, 'Adapter identity unavailable; cannot confirm hardware rendering.' );

	const renderer = new WebGPURenderer( { device: await adapter.requestDevice() } );
	await renderer.init();
	const device = renderer.backend.device;
	const errors = [];
	const onError = event => errors.push( event.error.message );
	device.addEventListener( 'uncapturederror', onError );
	let texels = 0, rayTests = 0, cases = 0, sampledTexels = 0;
	const sizes = [[ 1, 1 ], [ 1, 7 ], [ 7, 1 ], [ 2, 2 ], [ 3, 5 ], [ 5, 3 ], [ 7, 9 ], [ 32, 32 ], [ 65, 33 ], [ 257, 129 ]];

	try {

		for ( const { reversedDepth, automatic } of [
			{ reversedDepth: false, automatic: false },
			{ reversedDepth: true, automatic: false },
			{ reversedDepth: false, automatic: true },
			{ reversedDepth: true, automatic: true }
		] ) {

			// Reuse the same node while replacing/resizing its source texture.
			const sourceNode = texture( new DataTexture( new Float32Array( [ 0 ] ), 1, 1, RedFormat, FloatType ) );
			renderer.reversedDepthBuffer = automatic ? reversedDepth : ! reversedDepth;
			const pyramid = hiZ( sourceNode, automatic ? {} : { reversedDepth } );
			try {

				for ( const [ width, height ] of sizes ) {

					for ( const pattern of [ 'constant', 'random', 'trailingCorner', 'trailingRow', 'trailingColumn' ] ) {

						const data = new Float32Array( width * height );
						let seed = 123456789;
						for ( let y = 0; y < height; y ++ ) {

							for ( let x = 0; x < width; x ++ ) {

								seed = ( Math.imul( seed, 1664525 ) + 1013904223 ) >>> 0;
								let value = pattern === 'random' ? ( seed >>> 8 ) / 0x1000000 : 0.25;
								if ( pattern === 'trailingCorner' && x === width - 1 && y === height - 1 || pattern === 'trailingRow' && y === height - 1 || pattern === 'trailingColumn' && x === width - 1 ) value = 0.875;
								data[ y * width + x ] = reversedDepth ? 1 - value : value;

							}

						}

						sourceNode.value.dispose();
						sourceNode.value = new DataTexture( data, width, height, RedFormat, FloatType );
						sourceNode.value.needsUpdate = true;
						await pyramid.updateBefore( { renderer } );
						await device.queue.onSubmittedWorkDone();
						const reference = reduceDepth( data, width, height, reversedDepth );
						const nearest = reduceDepth( data, width, height, reversedDepth, true );
						assert( pyramid.levels.length === reference.length, 'Mip count mismatch' );
						const atlas = pyramid.texture;
						// Raw texture readback preserves rg32float and avoids filtering/tone mapping.
						const actual = await renderer.backend.copyTextureToBuffer( atlas, 0, 0, atlas.image.width, atlas.image.height, 0 );
						const gpuLevels = [];
						for ( let level = 0; level < reference.length; level ++ ) {

							const expected = reference[ level ];
							const layout = pyramid.levels[ level ];
							assert( layout.width === expected.width && layout.height === expected.height, `Layout mismatch at ${level}` );
							const values = new Float32Array( expected.width * expected.height );
							for ( let y = 0; y < expected.height; y ++ ) {

								for ( let x = 0; x < expected.width; x ++ ) {

									const index = y * expected.width + x;
									values[ index ] = actual[ ( ( layout.offset + y ) * atlas.image.width + x ) * 2 ];
									assert( values[ index ] === expected.data[ index ], `${width}x${height} ${pattern} reverse=${reversedDepth} mip=${level} (${x},${y}): expected ${expected.data[ index ]}, got ${values[ index ]}` );
									assert( actual[ ( ( layout.offset + y ) * atlas.image.width + x ) * 2 + 1 ] === nearest[ level ].data[ index ], `Nearest depth mismatch: ${width}x${height} ${pattern} reverse=${reversedDepth} mip=${level} (${x},${y})` );
									texels += 2;

								}

							}

							gpuLevels.push( { width: expected.width, height: expected.height, data: values } );

						}

						// Tiny horizontal/vertical/diagonal rays, including explicit misses and
						// hits on the odd trailing edge. Compare GPU bounds to linear leaf tests.
						const rays = [
							Array.from( { length: width }, ( _, x ) => [ x, height - 1 ] ),
							Array.from( { length: height }, ( _, y ) => [ width - 1, y ] ),
							Array.from( { length: Math.max( width, height ) }, ( _, i ) => [ Math.min( i, width - 1 ), Math.min( i, height - 1 ) ] )
						];
						for ( const ray of rays ) {

							for ( const depth of [ 0, 0.5, 1, data[ data.length - 1 ] ] ) {

								const linear = traceSamples( reference, ray, depth, reversedDepth, false );
								const hierarchical = traceSamples( gpuLevels, ray, depth, reversedDepth, true );
								if ( depth === 0 || depth === 1 ) assert( linear === - 1, 'Expected an empty endpoint ray' );
								if ( pattern === 'trailingCorner' && depth === data[ data.length - 1 ] ) assert( linear === ray.length - 1, 'Expected the trailing-corner hit' );
								assert( linear === hierarchical, `Ray mismatch: ${linear} vs ${hierarchical}` );
								rayTests ++;

							}

						}

						cases ++;

					}

				}

			} finally {

				pyramid.dispose();
				sourceNode.value.dispose();

			}

		}

		// Exercise the public shader sampler and graph scheduling, not only the
		// atlas layout. Odd tails must retain their original-image UV footprint.
		const width = 7, height = 9;
		const data = Float32Array.from( { length: width * height }, ( _, i ) => i / ( width * height ) );
		const source = new DataTexture( data, width, height, RedFormat, FloatType );
		source.needsUpdate = true;
		const pyramid = hiZ( texture( source ), { reversedDepth: false } );
		const target = new RenderTarget( width, height, { type: FloatType, depthBuffer: false } );
		const material = new NodeMaterial();
		material.fragmentNode = vec4( pyramid, 0, 0, 1 );
		const quad = new QuadMesh( material );
		const reference = reduceDepth( data, width, height, false );
		try {

			for ( const level of [ 0, 1, 2, 3, 15 ] ) {

				pyramid.level.value = level;
				renderer.setRenderTarget( target );
				quad.render( renderer );
				const actual = await renderer.readRenderTargetPixelsAsync( target, 0, 0, width, height );
				const clampedLevel = Math.min( level, reference.length - 1 );
				const mip = reference[ clampedLevel ];
				for ( let y = 0; y < height; y ++ ) {

					for ( let x = 0; x < width; x ++ ) {

						const sx = Math.min( Math.floor( x / 2 ** clampedLevel ), mip.width - 1 );
						const sy = Math.min( Math.floor( y / 2 ** clampedLevel ), mip.height - 1 );
						assert( actual[ ( y * width + x ) * 4 ] === mip.data[ sy * mip.width + sx ], `Shader sample mismatch level=${level} (${x},${y})` );
						sampledTexels ++;

					}

				}

			}

		} finally {

			renderer.setRenderTarget( null );
			material.dispose();
			target.dispose();
			pyramid.dispose();
			source.dispose();

		}

		await device.queue.onSubmittedWorkDone();
		assert( errors.length === 0, errors.join( '\n' ) );
		return { adapter: description.trim(), cases, texels, rayTests, sampledTexels };

	} finally {

		device.removeEventListener( 'uncapturederror', onError );
		renderer.dispose();

	}

}
