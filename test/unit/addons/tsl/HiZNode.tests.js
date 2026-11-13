import { DataTexture, DepthTexture, FloatType, Mesh, MeshBasicNodeMaterial, NearestFilter, NodeMaterial, OrthographicCamera, PlaneGeometry, QuadMesh, RedFormat, RenderTarget, Scene, StorageTexture, Vector2 } from 'three/webgpu';
import { Fn, instanceIndex, instancedArray, ivec2, pass, texture, textureLoad, uint, vec2, vec4 } from 'three/tsl';
import HiZNode, { hiZ } from '../../../../examples/jsm/tsl/display/HiZNode.js';
import { rawComputeTest } from '../../src/nodes/tsl/gpu-raw-test-utils.js';

function hiZTest( name, run ) {

	rawComputeTest( name, {}, async ( { assert, renderer } ) => {

		const device = renderer.backend.device;
		device.pushErrorScope( 'validation' );

		try {

			await run( { assert, renderer } );

		} finally {

			const error = await device.popErrorScope();
			assert.strictEqual( error, null, error ? error.message : 'no WebGPU validation errors' );

		}

	} );

}

function createDepth( width, height, data ) {

	const depth = new DataTexture( data, width, height, RedFormat, FloatType );
	depth.minFilter = depth.magFilter = NearestFilter;
	depth.needsUpdate = true;
	return depth;

}

// Independent oracle: scan ORIGINAL pixels for each cell, never reduce the
// preceding mip or reproduce the GPU's workgroup/shared-memory algorithm.
function referenceMip( data, width, height, level, reversed ) {

	const stride = 2 ** level;
	const mipWidth = Math.max( 1, Math.floor( width / stride ) );
	const mipHeight = Math.max( 1, Math.floor( height / stride ) );
	const result = new Float32Array( mipWidth * mipHeight );

	for ( let y = 0; y < mipHeight; y ++ ) {

		for ( let x = 0; x < mipWidth; x ++ ) {

			const endX = x === mipWidth - 1 ? width : ( x + 1 ) * stride;
			const endY = y === mipHeight - 1 ? height : ( y + 1 ) * stride;
			let nearest = reversed ? 0 : 1;

			for ( let sy = y * stride; sy < endY; sy ++ ) {

				for ( let sx = x * stride; sx < endX; sx ++ ) {

					nearest = reversed ? Math.max( nearest, data[ sy * width + sx ] ) : Math.min( nearest, data[ sy * width + sx ] );

				}

			}

			result[ y * mipWidth + x ] = nearest;

		}

	}

	return result;

}

function randomDepth( width, height ) {

	const data = new Float32Array( width * height );
	let seed = 0x12345678;

	for ( let i = 0; i < data.length; i ++ ) {

		seed ^= seed << 13;
		seed ^= seed >>> 17;
		seed ^= seed << 5;
		data[ i ] = ( seed >>> 16 ) / 65536;

	}

	return data;

}

async function readMip( renderer, pyramid, level ) {

	const { width, height } = pyramid.levels[ level ];
	const output = instancedArray( width * height, 'float' );
	const read = Fn( () => {

		const coord = ivec2( instanceIndex.mod( width ), instanceIndex.div( width ) );
		// Read the actual native mip directly, independently of HiZNode's UV accessor.
		output.element( instanceIndex ).assign( textureLoad( pyramid.texture, coord, uint( level ) ).r );

	} )().compute( width * height );

	try {

		await renderer.computeAsync( read );
		return new Float32Array( await renderer.getArrayBufferAsync( output.value ) );

	} finally {

		read.dispose();

	}

}

function equalTexels( assert, actual, expected, width, label ) {

	assert.strictEqual( actual.length, expected.length, `${ label }: texel count` );
	const mismatches = [];
	let count = 0;

	for ( let i = 0; i < expected.length; i ++ ) {

		if ( actual[ i ] !== expected[ i ] ) {

			count ++;
			if ( mismatches.length < 8 ) mismatches.push( { x: i % width, y: Math.floor( i / width ), actual: actual[ i ], expected: expected[ i ] } );

		}

	}

	assert.deepEqual( mismatches, [], `${ label }: ${ count } mismatched texels (first eight shown)` );

}

async function checkPyramid( assert, renderer, pyramid, data, width, height, reversed ) {

	pyramid.updateBefore( { renderer } );
	const levelCount = Math.floor( Math.log2( Math.max( width, height ) ) ) + 1;
	assert.strictEqual( pyramid.levels.length, levelCount, 'complete chain through 1x1' );

	for ( let level = 0; level < levelCount; level ++ ) {

		const actual = await readMip( renderer, pyramid, level );
		const expected = referenceMip( data, width, height, level, reversed );
		equalTexels( assert, actual, expected, Math.max( 1, width >> level ), `mip ${ level }` );

	}

}

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'TSL', () => {

		QUnit.module( 'HiZNode', () => {

			QUnit.test( 'native nearest-depth texture and floor-halved layout', assert => {

				const depth = new DepthTexture( 5, 3 );
				const pyramid = hiZ( texture( depth ) );
				pyramid.setSize( 5, 3 );
				assert.ok( pyramid instanceof HiZNode );
				assert.ok( pyramid.texture instanceof StorageTexture );
				assert.strictEqual( pyramid.texture.format, RedFormat );
				assert.strictEqual( pyramid.texture.type, FloatType );
				assert.strictEqual( pyramid.texture.generateMipmaps, false, 'no averaging of depth extrema' );
				assert.strictEqual( pyramid.texture.mipmapsAutoUpdate, false, 'sampling must not overwrite the computed mips' );
				assert.strictEqual( pyramid.texture.minFilter, NearestFilter );
				assert.deepEqual( pyramid.levels, [ { width: 5, height: 3 }, { width: 2, height: 1 }, { width: 1, height: 1 } ] );
				assert.deepEqual( pyramid.texture.mipmaps, pyramid.levels, 'native mip descriptors' );
				pyramid.dispose();
				depth.dispose();

			} );

			QUnit.test( 'maxLevel caps the hierarchy', assert => {

				const depth = new DepthTexture( 9, 5 );
				const pyramid = hiZ( texture( depth ), { maxLevel: 1 } );
				pyramid.setSize( 9, 5 );
				assert.deepEqual( pyramid.levels, [ { width: 9, height: 5 }, { width: 4, height: 2 } ] );
				assert.strictEqual( pyramid.getLastLevelNode().value, 1 );
				pyramid.dispose();
				depth.dispose();

			} );

			QUnit.test( 'invalid inputs and unsupported backend fail explicitly', assert => {

				assert.throws( () => new HiZNode( {} ), /raw depth TextureNode/ );
				assert.throws( () => new HiZNode( null ), /raw depth TextureNode/ );
				const depth = new DepthTexture( 1, 1 );
				const pyramid = hiZ( texture( depth ) );
				for ( const size of [ 0, - 1, 1.5, NaN, Infinity ] ) {

					assert.throws( () => pyramid.setSize( size, 1 ), /positive integers/ );
					assert.throws( () => pyramid.setSize( 1, size ), /positive integers/ );

				}

				assert.throws( () => pyramid.updateBefore( { renderer: { backend: {} } } ), /WebGPU backend/ );
				assert.throws( () => pyramid.setSize( 65536, 1 ), /Too many hierarchy levels/ );
				pyramid.setSize( 1, 1 );
				assert.deepEqual( pyramid.levels, [ { width: 1, height: 1 } ], 'invalid resize does not corrupt the layout' );
				const renderer = { backend: { isWebGPUBackend: true, device: { limits: { maxTextureDimension2D: 0 } } } };
				assert.throws( () => pyramid.updateBefore( { renderer } ), /exceeds maxTextureDimension2D/ );
				pyramid.dispose();
				depth.dispose();

			} );

			for ( const reversed of [ false, true ] ) {

				const convention = reversed ? 'reversed' : 'standard';

				hiZTest( `${ convention }: clear depth, thin edges and isolated corner`, async ( { assert, renderer } ) => {

					const width = 35, height = 19;
					const clear = reversed ? 0 : 1;
					const nearest = reversed ? 1 : 0;
					const data = new Float32Array( width * height ).fill( clear );
					const depth = createDepth( width, height, data );
					const pyramid = hiZ( texture( depth ), { reversedDepth: reversed } );

					try {

						for ( const pattern of [ 'clear', 'row', 'column', 'corner', 'constant', 'precision' ] ) {

							data.fill( pattern === 'constant' ? 0.375 : clear );
							if ( pattern === 'row' ) data.fill( nearest, width * ( height - 1 ) );
							if ( pattern === 'column' ) for ( let y = 0; y < height; y ++ ) data[ y * width + width - 1 ] = nearest;
							if ( pattern === 'corner' ) data[ data.length - 1 ] = nearest;
							if ( pattern === 'precision' ) {

								data.fill( reversed ? 0.5 : 1 );
								data[ data.length - 1 ] = reversed ? 0.5 + 2 ** - 24 : 1 - 2 ** - 24;

							}

							depth.needsUpdate = true;
							await checkPyramid( assert, renderer, pyramid, data, width, height, reversed );

						}

					} finally {

						pyramid.dispose();
						depth.dispose();

					}

				} );

				hiZTest( `${ convention }: hand-written odd-size golden chain`, async ( { assert, renderer } ) => {

					// Final column/row contain the extrema; dropping either is detectable.
					const data = new Float32Array( [
						0.75, 0.5, 0.875, 0.625, 0.25,
						0.625, 0.75, 0.75, 0.5, 0.125,
						0.375, 0.25, 0.625, 0.875, 0
					] );
					const depth = createDepth( 5, 3, data );
					const pyramid = hiZ( texture( depth ), { reversedDepth: reversed } );

					try {

						pyramid.updateBefore( { renderer } );
						equalTexels( assert, await readMip( renderer, pyramid, 0 ), data, 5, 'base' );
						equalTexels( assert, await readMip( renderer, pyramid, 1 ), reversed ? [ 0.75, 0.875 ] : [ 0.25, 0 ], 2, 'first mip' );
						equalTexels( assert, await readMip( renderer, pyramid, 2 ), reversed ? [ 0.875 ] : [ 0 ], 1, 'root' );

					} finally {

						pyramid.dispose();
						depth.dispose();

					}

				} );

				for ( const [ width, height ] of [
					[ 1, 1 ], [ 1, 2 ], [ 2, 1 ], [ 1, 17 ], [ 17, 1 ],
					[ 2, 2 ], [ 3, 3 ], [ 3, 5 ], [ 5, 3 ], [ 4, 4 ], [ 7, 9 ], [ 8, 8 ],
					[ 15, 17 ], [ 16, 16 ], [ 17, 15 ], [ 31, 33 ],
					[ 63, 65 ], [ 64, 64 ], [ 65, 63 ], [ 127, 129 ], [ 129, 127 ],
					[ 257, 131 ], [ 1919, 1079 ], [ 1081, 1919 ]
				] ) {

					hiZTest( `${ convention }: all texels at ${ width }x${ height }`, async ( { assert, renderer } ) => {

						const data = randomDepth( width, height );
						const depth = createDepth( width, height, data );
						const pyramid = hiZ( texture( depth ), { reversedDepth: reversed } );

						try {

							await checkPyramid( assert, renderer, pyramid, data, width, height, reversed );

						} finally {

							pyramid.dispose();
							depth.dispose();

						}

					} );

				}

			}

			hiZTest( 'updates, freeze, depth convention changes and resize on the same node', async ( { assert, renderer } ) => {

				const depth = createDepth( 9, 7, randomDepth( 9, 7 ) );
				const pyramid = hiZ( texture( depth ) );

				try {

					await checkPyramid( assert, renderer, pyramid, depth.image.data, 9, 7, renderer.reversedDepthBuffer );
					const oldRoot = await readMip( renderer, pyramid, pyramid.levels.length - 1 );
					depth.image.data.fill( 0.5 );
					depth.needsUpdate = true;
					pyramid.enabled = false;
					pyramid.updateBefore( { renderer } );
					equalTexels( assert, await readMip( renderer, pyramid, pyramid.levels.length - 1 ), oldRoot, 1, 'frozen root' );
					pyramid.enabled = true;
					await checkPyramid( assert, renderer, pyramid, depth.image.data, 9, 7, renderer.reversedDepthBuffer );

					pyramid.reversedDepth = ! renderer.reversedDepthBuffer;
					depth.image.data.set( randomDepth( 9, 7 ) );
					depth.needsUpdate = true;
					await checkPyramid( assert, renderer, pyramid, depth.image.data, 9, 7, pyramid.reversedDepth );

					// Includes the one-level special case and transitions across dispatch groups.
					for ( const [ width, height ] of [[ 1, 1 ], [ 65, 33 ], [ 3, 19 ], [ 1, 1 ]] ) {

						depth.dispose();
						depth.image = { data: randomDepth( width, height ), width, height };
						depth.needsUpdate = true;
						await checkPyramid( assert, renderer, pyramid, depth.image.data, width, height, pyramid.reversedDepth );

					}

				} finally {

					pyramid.dispose();
					depth.dispose();

				}

			} );

			hiZTest( 'scene depth pass runs before reduction and refreshes on the next frame', async ( { assert, renderer } ) => {

				const width = 17, height = 9;
				const scene = new Scene();
				const camera = new OrthographicCamera( - 1, 1, 1, - 1, 0.25, 4.25 );
				camera.position.z = 2;
				const geometry = new PlaneGeometry( 1, 1 );
				const surface = new MeshBasicNodeMaterial();
				const mesh = new Mesh( geometry, surface );
				mesh.position.set( - 0.125, 0.125, 0 );
				scene.add( mesh );
				const scenePass = pass( scene, camera );
				const pyramid = hiZ( scenePass.getTextureNode( 'depth' ) );
				const material = new NodeMaterial();
				material.fragmentNode = vec4( pyramid, 0, 0, 1 );
				const directMaterial = new NodeMaterial();
				directMaterial.fragmentNode = vec4( scenePass.getTextureNode( 'depth' ).r, 0, 0, 1 );
				const quad = new QuadMesh( material );
				const target = new RenderTarget( width, height, { type: FloatType } );
				const previousTarget = renderer.getRenderTarget();
				const previousSize = renderer.getSize( new Vector2() );

				try {

					renderer.setSize( width, height );
					renderer.setRenderTarget( target );
					for ( const z of [ 0, 0.5 ] ) {

						// PassNode and HiZNode update once per animation frame, not per
						// draw call. GPU readback alone need not advance the frame.
						await new Promise( resolve => requestAnimationFrame( resolve ) );
						mesh.position.z = z;
						// No manual scenePass or pyramid update: exercise real node scheduling.
						quad.render( renderer );
						const pixels = await renderer.readRenderTargetPixelsAsync( target, 0, 0, width, height );
						const center = ( Math.floor( height / 2 ) * width + Math.floor( width / 2 ) ) * 4;
						const expected = renderer.reversedDepthBuffer ? 1 - ( 2 - z - 0.25 ) / 4 : ( 2 - z - 0.25 ) / 4;
						const clear = renderer.reversedDepthBuffer ? 0 : 1;
						assert.ok( Math.abs( pixels[ center ] - expected ) < 1e-6, `current-frame plane depth at z=${ z }` );
						assert.strictEqual( pixels[ 0 ], clear, 'uncovered background depth' );
						assert.deepEqual( pyramid.levels[ 0 ], { width, height }, 'size follows producing pass' );

						// Use rendered base depth as the input to the independent oracle.
						const base = await readMip( renderer, pyramid, 0 );
						quad.material = directMaterial;
						quad.render( renderer );
						const directPixels = await renderer.readRenderTargetPixelsAsync( target, 0, 0, width, height );
						const directDepth = new Float32Array( width * height );
						const pyramidDepth = new Float32Array( width * height );
						for ( let i = 0; i < directDepth.length; i ++ ) {

							directDepth[ i ] = directPixels[ i * 4 ];
							pyramidDepth[ i ] = pixels[ i * 4 ];

						}

						equalTexels( assert, pyramidDepth, directDepth, width, 'base output matches source pass at every pixel' );
						quad.material = material;
						for ( let level = 1; level < pyramid.levels.length; level ++ ) {

							equalTexels( assert, await readMip( renderer, pyramid, level ), referenceMip( base, width, height, level, renderer.reversedDepthBuffer ), pyramid.levels[ level ].width, `rendered depth mip ${ level }` );

						}

						pyramid.level.value = pyramid.levels.length - 1;
						quad.render( renderer );
						const rootPixels = await renderer.readRenderTargetPixelsAsync( target, 0, 0, width, height );
						assert.ok( Math.abs( rootPixels[ 0 ] - expected ) < 1e-6, 'selected debug mip displays the nearest surface over the full image' );
						pyramid.level.value = 0;

					}

				} finally {

					renderer.setRenderTarget( previousTarget );
					renderer.setSize( previousSize.x, previousSize.y );
					material.dispose();
					directMaterial.dispose();
					target.dispose();
					pyramid.dispose();
					scenePass.dispose();
					geometry.dispose();
					surface.dispose();

				}

			} );

			hiZTest( 'UV lookups and cell bounds agree with source footprints', async ( { assert, renderer } ) => {

				const width = 19, height = 11;
				const data = randomDepth( width, height );
				const depth = createDepth( width, height, data );
				const pyramid = hiZ( texture( depth ), { reversedDepth: false } );

				try {

					pyramid.updateBefore( { renderer } );
					// Include UVs at both edges and slightly outside the image.
					const count = ( width + 2 ) * ( height + 2 );
					for ( let level = 0; level <= pyramid.levels.length; level ++ ) {

						const samples = instancedArray( count, 'float' );
						const bounds = instancedArray( count, 'vec4' );
						const read = Fn( () => {

							const pixel = vec2( instanceIndex.mod( width + 2 ), instanceIndex.div( width + 2 ) ).sub( 0.5 );
							const coord = pixel.div( vec2( width, height ) );
							samples.element( instanceIndex ).assign( pyramid.getTextureNode( level, coord ) );
							bounds.element( instanceIndex ).assign( pyramid.getCellBoundsNode( level, coord ) );

						} )().compute( count );

						try {

							await renderer.computeAsync( read );
							const actual = new Float32Array( await renderer.getArrayBufferAsync( samples.value ) );
							const actualBounds = new Float32Array( await renderer.getArrayBufferAsync( bounds.value ) );
							const mip = Math.min( level, pyramid.levels.length - 1 );
							const stride = 2 ** mip;
							const mipWidth = Math.max( 1, Math.floor( width / stride ) );
							const mipHeight = Math.max( 1, Math.floor( height / stride ) );
							const expectedMip = referenceMip( data, width, height, mip, false );
							const expected = new Float32Array( count );
							let maxError = 0;

							for ( let i = 0; i < count; i ++ ) {

								const x = Math.max( 0, Math.min( mipWidth - 1, Math.floor( ( i % ( width + 2 ) - 0.5 ) / stride ) ) );
								const y = Math.max( 0, Math.min( mipHeight - 1, Math.floor( ( Math.floor( i / ( width + 2 ) ) - 0.5 ) / stride ) ) );
								expected[ i ] = expectedMip[ y * mipWidth + x ];
								const rectangle = [ x * stride / width, y * stride / height,
									( x === mipWidth - 1 ? width : ( x + 1 ) * stride ) / width,
									( y === mipHeight - 1 ? height : ( y + 1 ) * stride ) / height ];

								for ( let c = 0; c < 4; c ++ ) maxError = Math.max( maxError, Math.abs( actualBounds[ i * 4 + c ] - rectangle[ c ] ) );

							}

							equalTexels( assert, actual, expected, width + 2, `UV lookup at mip ${ level }` );
							assert.ok( Number.isFinite( maxError ) && maxError < 1e-6, `cell bounds at mip ${ level }: maximum error ${ maxError }` );

						} finally {

							read.dispose();

						}

					}

				} finally {

					pyramid.dispose();
					depth.dispose();

				}

			} );

		} );

	} );

} );
