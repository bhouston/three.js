import WebGPUTextureUtils from '../../../../../src/renderers/webgpu/utils/WebGPUTextureUtils.js';
import { CompressedCubeTexture } from '../../../../../src/textures/CompressedCubeTexture.js';
import { RGBAFormat, RGBA_S3TC_DXT1_Format } from '../../../../../src/constants.js';

export default QUnit.module( 'Renderers', () => {

	QUnit.module( 'WebGPUTextureUtils', () => {

		for ( const [ format, gpuFormat, bytesPerRow ] of [
			[ RGBAFormat, 'rgba8unorm', [ 16, 8 ]],
			[ RGBA_S3TC_DXT1_Format, 'bc1-rgba-unorm', [ 8, 8 ]]
		] ) {

			QUnit.test( 'updateTexture compressed cube ' + gpuFormat, ( assert ) => {

				const faces = Array.from( { length: 6 }, () => ( {
					width: 4,
					height: 4,
					mipmaps: [ 4, 2 ].map( ( size ) => ( {
						width: size,
						height: size,
						data: new Uint8Array( format === RGBAFormat ? size * size * 4 : 8 )
					} ) )
				} ) );
				const texture = new CompressedCubeTexture( faces, format );
				const gpuTexture = {};
				const writes = [];
				const textureData = {
					texture: gpuTexture,
					textureDescriptorGPU: { format: gpuFormat, size: { width: 4, height: 4, depthOrArrayLayers: 6 } }
				};
				const backend = {
					get: () => textureData,
					device: { queue: { writeTexture: ( destination, data, layout, size ) => {

						writes.push( {
							texture: destination.texture,
							face: destination.origin.z,
							mipLevel: destination.mipLevel,
							data,
							offset: layout.offset,
							bytesPerRow: layout.bytesPerRow,
							width: size.width,
							height: size.height,
							depth: size.depthOrArrayLayers
						} );

					} } }
				};

				new WebGPUTextureUtils( backend ).updateTexture( texture, {} );

				assert.strictEqual( writes.length, 12, 'Uploads each face and mip exactly once' );

				for ( let face = 0; face < 6; face ++ ) {

					for ( let mip = 0; mip < 2; mip ++ ) {

						const size = format === RGBAFormat ? 4 >> mip : 4;
						assert.strictEqual( writes[ face * 2 + mip ].data, faces[ face ].mipmaps[ mip ].data, 'Uploads the face mip typed array' );
						assert.deepEqual( writes[ face * 2 + mip ], {
							texture: gpuTexture,
							face,
							mipLevel: mip,
							data: faces[ face ].mipmaps[ mip ].data,
							offset: 0,
							bytesPerRow: bytesPerRow[ mip ],
							width: size,
							height: size,
							depth: 1
						}, 'Correct data and destination for face ' + face + ', mip ' + mip );

					}

				}

			} );

		}

	} );

} );
