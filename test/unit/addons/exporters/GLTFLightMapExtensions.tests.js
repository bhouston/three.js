import { BoxGeometry, DataTexture, LinearSRGBColorSpace, Mesh, MeshBasicMaterial, MeshStandardMaterial, SRGBColorSpace, Texture } from '../../../../src/Three.js';
import { GLTFExporter } from '../../../../examples/jsm/exporters/GLTFExporter.js';
import { GLTFLoader } from '../../../../examples/jsm/loaders/GLTFLoader.js';
import { GLTFLightMapExporterExtension } from '../../../../examples/jsm/exporters/GLTFLightMapExporterExtension.js';
import { GLTFLightMapLoaderExtension } from '../../../../examples/jsm/loaders/GLTFLightMapLoaderExtension.js';

QUnit.module( 'Addons', () => {

	QUnit.module( 'GLTFLightMapExtensions', () => {

		QUnit.test( 'Linear conversion preserves source data, alpha and texture state', async assert => {

			const map = new DataTexture( new Uint8Array( [ 128, 64, 0, 71 ] ), 1, 1 );
			map.colorSpace = LinearSRGBColorSpace;
			map.channel = 1;
			map.offset.set( 0.2, 0.3 );
			const plugin = new GLTFLightMapExporterExtension( {} );
			const encoded = await plugin.getSRGBTextureAsync( map );

			assert.deepEqual( Array.from( encoded.image.data ), [ 188, 137, 0, 71 ], 'RGB encoded, alpha unchanged' );
			assert.deepEqual( Array.from( map.image.data ), [ 128, 64, 0, 71 ], 'Original pixels unchanged' );
			assert.strictEqual( map.colorSpace, LinearSRGBColorSpace, 'Original color space unchanged' );
			assert.notStrictEqual( encoded.source, map.source, 'Conversion owns its image source' );
			assert.strictEqual( encoded.colorSpace, SRGBColorSpace, 'Converted texture marked sRGB' );
			assert.strictEqual( encoded.channel, 1, 'UV set preserved' );
			assert.deepEqual( encoded.offset.toArray(), [ 0.2, 0.3 ], 'Transform preserved' );
			assert.strictEqual( await plugin.getSRGBTextureAsync( map ), encoded, 'Conversion reused across materials' );
			assert.strictEqual( await plugin.getSRGBTextureAsync( encoded ), encoded, 'sRGB pixels are not encoded twice' );

		} );

		QUnit.test( 'Linear canvas image is encoded without mutation', async assert => {

			const canvas = document.createElement( 'canvas' );
			canvas.width = canvas.height = 1;
			const context = canvas.getContext( '2d' );
			context.putImageData( new ImageData( new Uint8ClampedArray( [ 128, 64, 0, 255 ] ), 1, 1 ), 0, 0 );
			const map = new Texture( canvas );
			map.colorSpace = LinearSRGBColorSpace;
			const encoded = await new GLTFLightMapExporterExtension( {} ).getSRGBTextureAsync( map );

			assert.deepEqual( Array.from( encoded.image.getContext( '2d' ).getImageData( 0, 0, 1, 1 ).data ), [ 188, 137, 0, 255 ], 'Canvas RGB encoded' );
			assert.deepEqual( Array.from( context.getImageData( 0, 0, 1, 1 ).data ), [ 128, 64, 0, 255 ], 'Original canvas unchanged' );

		} );

		QUnit.test( 'PBR and unlit export/import preserve sRGB pixels and intensity', async assert => {

			for ( const MaterialType of [ MeshStandardMaterial, MeshBasicMaterial ] ) {

				const map = new DataTexture( new Uint8Array( [ 128, 64, 0, 255 ] ), 1, 1 );
				map.colorSpace = LinearSRGBColorSpace;
				map.channel = 1;
				const material = new MaterialType();
				material.lightMap = map;
				material.lightMapIntensity = material.isMeshBasicMaterial ? 2 * Math.PI : 2;
				const geometry = new BoxGeometry();
				geometry.setAttribute( 'uv1', geometry.getAttribute( 'uv' ).clone() );
				const exporter = new GLTFExporter();
				exporter.register( writer => new GLTFLightMapExporterExtension( writer ) );
				const json = await exporter.parseAsync( new Mesh( geometry, material ) );
				const extension = json.materials[ 0 ].extensions.MOZ_lightmap;

				assert.strictEqual( extension.intensity, 2, 'Serialized intensity excludes renderer PI conversion' );
				assert.strictEqual( extension.texCoord, 1, 'Serialized UV set preserved' );

				const image = new Image();
				image.src = json.images[ 0 ].uri;
				await image.decode();
				const canvas = document.createElement( 'canvas' );
				canvas.width = canvas.height = 1;
				const context = canvas.getContext( '2d' );
				context.drawImage( image, 0, 0 );
				assert.deepEqual( Array.from( context.getImageData( 0, 0, 1, 1 ).data ), [ 188, 137, 0, 255 ], 'Exported PNG contains sRGB RGB' );

				const loader = new GLTFLoader();
				loader.register( parser => new GLTFLightMapLoaderExtension( parser ) );
				const gltf = await loader.parseAsync( JSON.stringify( json ), '' );
				const loadedMaterial = gltf.scene.children[ 0 ].material;
				assert.strictEqual( loadedMaterial.lightMap.colorSpace, SRGBColorSpace, 'Loader requests sRGB decoding' );
				assert.strictEqual( loadedMaterial.lightMapIntensity, material.lightMapIntensity, 'Runtime intensity round-trips' );
				assert.deepEqual( Array.from( map.image.data ), [ 128, 64, 0, 255 ], 'Export leaves source image unchanged' );

				// Export the loaded asset again: its sRGB pixels must not be re-encoded.
				const secondExport = await exporter.parseAsync( gltf.scene );
				const secondImage = new Image();
				secondImage.src = secondExport.images[ 0 ].uri;
				await secondImage.decode();
				context.drawImage( secondImage, 0, 0 );
				assert.deepEqual( Array.from( context.getImageData( 0, 0, 1, 1 ).data ), [ 188, 137, 0, 255 ], 'Second export preserves encoded pixels' );

			}

		} );

	} );

} );
