import { GLTFLoader } from '../../../../examples/jsm/loaders/GLTFLoader.js';
import { UltraHDRLoader } from '../../../../examples/jsm/loaders/UltraHDRLoader.js';
import { DataTexture, DataUtils, HalfFloatType, LinearSRGBColorSpace, SRGBColorSpace, NearestFilter, ClampToEdgeWrapping, Texture } from 'three';

QUnit.module( 'EXT_ultrajpeg', hooks => {

	let originalLoad;
	hooks.beforeEach( () => {

		originalLoad = UltraHDRLoader.prototype.load;

	} );
	hooks.afterEach( () => {

		UltraHDRLoader.prototype.load = originalLoad;

	} );

	async function makeParser( { embedded = false, required = false, fallback = false, defaultSampler = false } = {} ) {

		const json = {
			asset: { version: '2.0' },
			extensionsUsed: [ 'EXT_ultrajpeg', 'KHR_texture_transform' ],
			extensionsRequired: required ? [ 'EXT_ultrajpeg' ] : [],
			images: [ embedded ? { bufferView: 0, mimeType: 'image/jpeg' } : { uri: 'linear.jpg' }, { uri: 'fallback.png' } ],
			textures: [ { sampler: 0, extensions: { EXT_ultrajpeg: { source: 0 } }, ...( fallback ? { source: 1 } : {} ) } ],
			samplers: [ defaultSampler ? {} : { magFilter: 9728, minFilter: 9728, wrapS: 33071 } ],
			materials: [ {
				emissiveFactor: [ 1, 1, 1 ], emissiveTexture: { index: 0 },
				pbrMetallicRoughness: { metallicRoughnessTexture: { index: 0 } }
			} ],
			buffers: [ { uri: 'data:application/octet-stream;base64,AAAAAA==', byteLength: 4 } ],
			bufferViews: [ { buffer: 0, byteLength: 4 } ]
		};
		return ( await new GLTFLoader().setRequestHeader( { 'X-Test': 'HDR' } ).setCrossOrigin( 'use-credentials' ).parseAsync( JSON.stringify( json ), '/assets/' ) ).parser;

	}

	for ( const embedded of [ false, true ] ) {

		QUnit.test( embedded ? 'bufferView HDR texture' : 'external HDR texture', async assert => {

			let loadedURL;
			UltraHDRLoader.prototype.load = function ( url, onLoad ) {

				loadedURL = url;
				assert.true( this.fullHDR, 'automatic full HDR mode' );
				assert.deepEqual( this.requestHeader, { 'X-Test': 'HDR' }, 'request headers forwarded' );
				assert.true( this.withCredentials, 'credentials forwarded' );
				const texture = new DataTexture( new Uint16Array( [ 0x4000, 0, 0, 0x3c00 ] ), 1, 1 );
				texture.type = HalfFloatType;
				onLoad( texture );

			};

			const parser = await makeParser( { embedded } );
			const params = {};
			const texture = await parser.assignTexture( params, 'map', {
				index: 0, texCoord: 1, extensions: { KHR_texture_transform: { offset: [ 0.25, 0.5 ] } }
			}, SRGBColorSpace );
			assert.ok( embedded ? loadedURL.startsWith( 'blob:' ) : loadedURL === '/assets/linear.jpg', 'resolves image source' );
			assert.strictEqual( texture.colorSpace, LinearSRGBColorSpace, 'color slot keeps linear encoding' );
			assert.strictEqual( texture.image.data[ 0 ], 0x4000, 'HDR value 2 remains unchanged' );
			assert.strictEqual( texture.flipY, false, 'glTF orientation' );
			assert.strictEqual( texture.magFilter, NearestFilter, 'sampler filter' );
			assert.strictEqual( texture.wrapS, ClampToEdgeWrapping, 'sampler wrapping' );
			assert.strictEqual( texture.channel, 1, 'texture coordinates' );
			assert.deepEqual( texture.offset.toArray(), [ 0.25, 0.5 ], 'texture transform' );

		} );

	}

	QUnit.test( 'optional decode failure uses core fallback', async assert => {

		UltraHDRLoader.prototype.load = ( url, onLoad, onProgress, onError ) => onError( new Error( 'unsupported' ) );
		const parser = await makeParser( { fallback: true } );
		parser.textureLoader.load = ( url, onLoad ) => {

			assert.strictEqual( url, '/assets/fallback.png', 'loads core image' );
			onLoad( new Texture() );

		};

		const texture = await parser.assignTexture( {}, 'map', { index: 0 }, SRGBColorSpace );
		assert.strictEqual( texture.colorSpace, SRGBColorSpace, 'fallback uses core color semantics' );

	} );

	QUnit.test( 'material shares linear HDR color and non-color texture with default mipmaps', async assert => {

		UltraHDRLoader.prototype.load = ( url, onLoad ) => onLoad( new DataTexture( new Uint16Array( [ 0x4000, 0, 0, 0x3c00 ] ), 1, 1 ) );
		const parser = await makeParser( { defaultSampler: true } );
		const material = await parser.getDependency( 'material', 0 );
		assert.strictEqual( material.emissiveMap.colorSpace, LinearSRGBColorSpace, 'emissive slot stays linear' );
		assert.strictEqual( material.roughnessMap, material.emissiveMap, 'non-color slot can share image' );
		assert.true( material.emissiveMap.generateMipmaps, 'default sampler generates mipmaps' );
		assert.strictEqual( material.emissiveMap.image.data[ 0 ], 0x4000, 'material retains HDR sample' );

	} );

	for ( const invalid of [ 0x7c00, 0x7e00, 0xbc00, 0x4000 ] ) {

		QUnit.test( `invalid half-float sample ${ invalid } rejects`, async assert => {

			UltraHDRLoader.prototype.load = ( url, onLoad ) => {

				const data = new Uint16Array( [ 0, 0, 0, 0x3c00 ] );
				data[ invalid === 0x4000 ? 3 : 0 ] = invalid;
				onLoad( new DataTexture( data, 1, 1 ) );

			};

			const parser = await makeParser( { required: true } );
			await assert.rejects( parser.getDependency( 'texture', 0 ), /EXT_ultrajpeg/, 'invalid output fails instead of clipping' );

		} );

	}

	for ( const required of [ false, true ] ) {

		QUnit.test( required ? 'required decode failure rejects despite fallback' : 'missing fallback rejects', async assert => {

			UltraHDRLoader.prototype.load = ( url, onLoad, onProgress, onError ) => onError( new Error( 'unsupported' ) );
			const parser = await makeParser( { required, fallback: required } );
			await assert.rejects( parser.getDependency( 'texture', 0 ), /EXT_ultrajpeg/, 'failure is reported' );

		} );

	}

} );


QUnit.module( 'EXT_ultrajpeg real decoding', () => {

	for ( const embedded of [ false, true ] ) {

		QUnit.test( embedded ? 'real bufferView HDR material' : 'real external HDR material', async assert => {

			const base = '/test/unit/addons/loaders/fixtures/ultrajpeg/';
			const bytes = await ( await fetch( base + 'linear-hdr.jpg' ) ).arrayBuffer();
			const json = {
				asset: { version: '2.0' },
				extensionsUsed: [ 'EXT_ultrajpeg' ],
				extensionsRequired: [ 'EXT_ultrajpeg' ],
				images: [ embedded ? { bufferView: 0, mimeType: 'image/jpeg' } : { uri: 'linear-hdr.jpg', mimeType: 'image/jpeg' } ],
				textures: [ { extensions: { EXT_ultrajpeg: { source: 0 } } } ],
				materials: [ { emissiveFactor: [ 1, 1, 1 ], emissiveTexture: { index: 0 } } ]
			};
			if ( embedded ) {

				let binary = '';
				for ( const value of new Uint8Array( bytes ) ) binary += String.fromCharCode( value );
				json.buffers = [ { uri: 'data:application/octet-stream;base64,' + btoa( binary ), byteLength: bytes.byteLength } ];
				json.bufferViews = [ { buffer: 0, byteLength: bytes.byteLength } ];

			}

			const { parser } = await new GLTFLoader().parseAsync( JSON.stringify( json ), base );
			const material = await parser.getDependency( 'material', 0 );
			const texture = material.emissiveMap;
			assert.strictEqual( texture.colorSpace, LinearSRGBColorSpace, 'emissive texture is linear' );
			assert.strictEqual( texture.flipY, false, 'glTF orientation' );
			const values = Array.from( texture.image.data, value => DataUtils.fromHalfFloat( value ) );
			assert.ok( values.every( ( value, index ) => index % 4 === 3 ? value === 1 : Math.abs( value - 4 ) < 0.005 ), 'full gain-map reconstruction yields linear RGB 4, alpha 1' );
			texture.dispose();
			material.dispose();

		} );

	}

	QUnit.test( 'same-file SDR fallback has a separate decoder cache', async assert => {

		const originalLoad = UltraHDRLoader.prototype.load;
		UltraHDRLoader.prototype.load = ( url, onLoad, onProgress, onError ) => onError( new Error( 'unsupported' ) );
		try {

			const json = {
				asset: { version: '2.0' }, extensionsUsed: [ 'EXT_ultrajpeg' ],
				images: [ { uri: 'linear-hdr.jpg' } ],
				textures: [ { source: 0, extensions: { EXT_ultrajpeg: { source: 0 } } } ]
			};
			const { parser } = await new GLTFLoader().parseAsync( JSON.stringify( json ), '/textures/' );
			parser.textureLoader.load = ( url, onLoad ) => {

				assert.strictEqual( url, '/textures/linear-hdr.jpg', 'core decoder loads same URI' );
				onLoad( new Texture() );

			};

			const texture = await parser.assignTexture( {}, 'map', { index: 0 }, SRGBColorSpace );
			assert.strictEqual( texture.colorSpace, SRGBColorSpace, 'fallback has core color semantics' );

		} finally {

			UltraHDRLoader.prototype.load = originalLoad;

		}

	} );

} );

