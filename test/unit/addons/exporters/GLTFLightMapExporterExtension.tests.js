import { CompressedTexture, DataTexture, Mesh, MeshStandardMaterial, PlaneGeometry, Texture } from '../../../../src/Three.js';
import { GLTFExporter } from '../../../../examples/jsm/exporters/GLTFExporter.js';
import { GLTFLightMapExporterExtension } from '../../../../examples/jsm/exporters/GLTFLightMapExporterExtension.js';
import { GLTFLoader } from '../../../../examples/jsm/loaders/GLTFLoader.js';
import { GLTFLightMapLoaderExtension } from '../../../../examples/jsm/loaders/GLTFLightMapLoaderExtension.js';

function createMesh( map, intensity = 1 ) {

	map.channel = 1;
	const geometry = new PlaneGeometry();
	geometry.setAttribute( 'uv1', geometry.attributes.uv.clone() );
	return new Mesh( geometry, new MeshStandardMaterial( { lightMap: map, lightMapIntensity: intensity } ) );

}

function createExporter() {

	const exporter = new GLTFExporter();
	exporter.register( writer => new GLTFLightMapExporterExtension( writer ) );
	return exporter;

}

function readPixels( image ) {

	const canvas = document.createElement( 'canvas' );
	canvas.width = image.width;
	canvas.height = image.height;
	const context = canvas.getContext( '2d' );
	context.drawImage( image, 0, 0 );
	return context.getImageData( 0, 0, image.width, image.height ).data;

}

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'Exporters', () => {

		QUnit.module( 'GLTFLightMapExporterExtension', () => {

			QUnit.test( 'GLB round trip preserves light map and intensity', async assert => {

				const data = new Uint8Array( [ 64, 128, 255, 255, 32, 16, 8, 255 ] );
				const map = new DataTexture( data, 1, 2 );
				const mesh = createMesh( map, 2 );
				const buffer = await createExporter().parseAsync( mesh, { binary: true } );
				const loader = new GLTFLoader();
				loader.register( parser => new GLTFLightMapLoaderExtension( parser ) );
				const gltf = await loader.parseAsync( buffer, '' );
				const material = gltf.scene.children[ 0 ].material;
				assert.strictEqual( material.lightMapIntensity, 2 );
				assert.strictEqual( material.lightMap.channel, 1 );
				assert.deepEqual( Array.from( readPixels( material.lightMap.image ) ), Array.from( data ) );
				assert.deepEqual( Array.from( map.image.data ), Array.from( data ), 'Source pixels unchanged' );
				assert.strictEqual( mesh.material.lightMapIntensity, 2, 'Source intensity unchanged' );

			} );

			QUnit.test( 'Shared atlas, transforms and zero intensity', async assert => {

				const map = new DataTexture( new Uint8Array( [ 64, 128, 255, 255 ] ), 1, 1 );
				map.offset.set( 0.25, 0.5 );
				map.repeat.set( 0.5, 0.75 );
				map.rotation = 0.2;
				const mesh = createMesh( map, 0 );
				mesh.material.emissiveMap = map;
				const result = await createExporter().parseAsync( [ mesh, createMesh( map, 3 ) ] );
				assert.strictEqual( result.textures.length, 1, 'Shared atlas is embedded once' );
				assert.strictEqual( result.images.length, 1 );
				assert.ok( result.extensionsUsed.includes( 'MOZ_lightmap' ) );
				assert.ok( result.extensionsUsed.includes( 'KHR_texture_transform' ) );
				assert.notOk( result.extensionsRequired?.includes( 'MOZ_lightmap' ), 'Ordinary PBR remains a fallback' );
				const extension = result.materials[ 0 ].extensions.MOZ_lightmap;
				assert.strictEqual( extension.index, result.materials[ 0 ].emissiveTexture.index, 'Light map shares the emissive texture' );
				assert.strictEqual( extension.intensity, 0 );
				assert.strictEqual( extension.texCoord, 1 );
				assert.deepEqual( extension.extensions.KHR_texture_transform, { offset: [ 0.25, 0.5 ], scale: [ 0.5, 0.75 ], rotation: 0.2 } );
				assert.strictEqual( result.materials[ 1 ].extensions.MOZ_lightmap.index, extension.index );

			} );

			QUnit.test( 'Texture orientation', async assert => {

				const canvas = document.createElement( 'canvas' );
				canvas.width = 1;
				canvas.height = 2;
				canvas.getContext( '2d' ).putImageData( new ImageData( new Uint8ClampedArray( [ 255, 0, 0, 255, 0, 255, 0, 255 ] ), 1, 2 ), 0, 0 );
				const map = new Texture( canvas );
				map.flipY = true;
				const result = await createExporter().parseAsync( createMesh( map ) );
				const image = await createImageBitmap( await ( await fetch( result.images[ 0 ].uri ) ).blob() );
				assert.deepEqual( Array.from( readPixels( image ) ), [ 0, 255, 0, 255, 255, 0, 0, 255 ] );
				image.close();

			} );

			QUnit.test( 'Compressed light maps use exporter texture utils', async assert => {

				const map = new CompressedTexture( [ { data: new Uint8Array( [ 0 ] ), width: 1, height: 1 } ], 1, 1 );
				const readable = new DataTexture( new Uint8Array( [ 64, 128, 255, 255 ] ), 1, 1 );
				const exporter = createExporter();
				exporter.setTextureUtils( {
					async decompress( texture, maxTextureSize ) {

						assert.strictEqual( texture, map );
						assert.strictEqual( maxTextureSize, 16 );
						return readable;

					}
				} );
				const result = await exporter.parseAsync( createMesh( map, 2 ), { maxTextureSize: 16 } );
				assert.strictEqual( result.materials[ 0 ].extensions.MOZ_lightmap.intensity, 2 );
				const image = await createImageBitmap( await ( await fetch( result.images[ 0 ].uri ) ).blob() );
				assert.deepEqual( Array.from( readPixels( image ) ), [ 64, 128, 255, 255 ] );
				image.close();

			} );

			QUnit.test( 'Materials without a light map', async assert => {

				const mesh = new Mesh( new PlaneGeometry(), new MeshStandardMaterial() );
				const result = await createExporter().parseAsync( mesh );
				assert.strictEqual( result.textures, undefined );
				assert.notOk( result.extensionsUsed?.includes( 'MOZ_lightmap' ) );

			} );

		} );

	} );

} );
