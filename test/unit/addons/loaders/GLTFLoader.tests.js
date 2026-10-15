import { GLTFLoader } from '../../../../examples/jsm/loaders/GLTFLoader.js';
import { JXLLoader } from '../../../../examples/jsm/loaders/JXLLoader.js';
import { DataTexture, DataUtils, HalfFloatType, LinearSRGBColorSpace, SRGBColorSpace, NearestFilter, ClampToEdgeWrapping, Texture } from 'three';
import { GLTFGaussianSplatLoaderExtension } from '../../../../examples/jsm/loaders/GLTFGaussianSplatLoaderExtension.js';
import { makeClipAdditive } from '../../../../src/animation/AnimationUtils.js';
import { unpackSphericalHarmonicsBand } from '../utils/GaussianSplatTestUtils.js';

const EPS = 1e-6;
const FLOAT = 5126;
const UNSIGNED_BYTE = 5121;

let hasNativeHDR = false;

try {

	const hdrContext = document.createElement( 'canvas' ).getContext( '2d', { colorSpace: 'srgb-linear', colorType: 'float16' } );
	hasNativeHDR = hdrContext !== null && hdrContext.getContextAttributes().colorType === 'float16' && 'globalHDRHeadroom' in hdrContext;

} catch {

	// Browsers may reject unsupported color-space enums instead of ignoring them.

}

QUnit.module( 'KHR_jpegxl_linear native decoding', () => {

	for ( const filename of [ 'linear-hdr.jxl', 'linear-hdr-1000.jxl' ] ) {

		( hasNativeHDR ? QUnit.test : QUnit.skip )( filename, async assert => {

			const json = {
				asset: { version: '2.0' },
				extensionsUsed: [ 'KHR_jpegxl_linear' ],
				extensionsRequired: [ 'KHR_jpegxl_linear' ],
				images: [ { uri: filename, mimeType: 'image/jxl' } ],
				textures: [ { extensions: { KHR_jpegxl_linear: { source: 0 } } } ],
				materials: [ { emissiveFactor: [ 1, 1, 1 ], emissiveTexture: { index: 0 } } ]
			};
			const { parser } = await new GLTFLoader().parseAsync( JSON.stringify( json ), '/test/unit/addons/loaders/fixtures/jpegxl-linear/' );
			const material = await parser.getDependency( 'material', 0 );
			const texture = material.emissiveMap;
			const values = Array.from( texture.image.data, value => DataUtils.fromHalfFloat( value ) );
			assert.deepEqual( values, [ 0, 0.5, 1, 1, 2, 4, 8, 1, 1, 1, 1, 1, 0.25, 0.5, 0.75, 1 ], 'preserves linear source samples, HDR range, alpha, orientation, and luminance scale' );
			assert.strictEqual( texture.colorSpace, LinearSRGBColorSpace, 'material sampling stays linear' );
			assert.strictEqual( texture.flipY, false, 'glTF texture orientation' );
			texture.dispose();
			material.dispose();

		} );

	}

} );

QUnit.module( 'KHR_jpegxl_linear', hooks => {

	let originalLoad;
	hooks.beforeEach( () => {

		originalLoad = JXLLoader.prototype.load;

	} );
	hooks.afterEach( () => {

		JXLLoader.prototype.load = originalLoad;

	} );

	async function makeParser( { embedded = false, required = false, fallback = false, defaultSampler = false } = {} ) {

		const json = {
			asset: { version: '2.0' },
			extensionsUsed: [ 'KHR_jpegxl_linear', 'KHR_texture_transform' ],
			extensionsRequired: required ? [ 'KHR_jpegxl_linear' ] : [],
			images: [ embedded ? { bufferView: 0, mimeType: 'image/jxl' } : { uri: 'linear.jxl' }, { uri: 'fallback.png' } ],
			textures: [ { sampler: 0, extensions: { KHR_jpegxl_linear: { source: 0 } }, ...( fallback ? { source: 1 } : {} ) } ],
			samplers: [ defaultSampler ? {} : { magFilter: 9728, minFilter: 9728, wrapS: 33071 } ],
			materials: [ {
				emissiveFactor: [ 1, 1, 1 ], emissiveTexture: { index: 0 },
				pbrMetallicRoughness: { metallicRoughnessTexture: { index: 0 } }
			} ],
			buffers: [ { uri: 'data:application/octet-stream;base64,AAAAAA==', byteLength: 4 } ],
			bufferViews: [ { buffer: 0, byteLength: 4 } ]
		};
		return ( await new GLTFLoader().parseAsync( JSON.stringify( json ), '/assets/' ) ).parser;

	}

	for ( const embedded of [ false, true ] ) {

		QUnit.test( embedded ? 'bufferView HDR texture' : 'external HDR texture', async assert => {

			let loadedURL;
			JXLLoader.prototype.load = function ( url, onLoad ) {

				loadedURL = url;
				const texture = new DataTexture( new Uint16Array( [ 0x4000, 0, 0, 0x3c00 ] ), 1, 1 );
				texture.type = HalfFloatType;
				onLoad( texture );

			};

			const parser = await makeParser( { embedded } );
			const params = {};
			const texture = await parser.assignTexture( params, 'map', {
				index: 0, texCoord: 1, extensions: { KHR_texture_transform: { offset: [ 0.25, 0.5 ] } }
			}, SRGBColorSpace );
			assert.ok( embedded ? loadedURL.startsWith( 'blob:' ) : loadedURL === '/assets/linear.jxl', 'resolves image source' );
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

		JXLLoader.prototype.load = ( url, onLoad, onProgress, onError ) => onError( new Error( 'unsupported' ) );
		const parser = await makeParser( { fallback: true } );
		parser.textureLoader.load = ( url, onLoad ) => {

			assert.strictEqual( url, '/assets/fallback.png', 'loads core image' );
			onLoad( new Texture() );

		};

		const texture = await parser.assignTexture( {}, 'map', { index: 0 }, SRGBColorSpace );
		assert.strictEqual( texture.colorSpace, SRGBColorSpace, 'fallback uses core color semantics' );

	} );

	QUnit.test( 'material shares linear HDR color and non-color texture with default mipmaps', async assert => {

		JXLLoader.prototype.load = ( url, onLoad ) => onLoad( new DataTexture( new Uint16Array( [ 0x4000, 0, 0, 0x3c00 ] ), 1, 1 ) );
		const parser = await makeParser( { defaultSampler: true } );
		const material = await parser.getDependency( 'material', 0 );
		assert.strictEqual( material.emissiveMap.colorSpace, LinearSRGBColorSpace, 'emissive slot stays linear' );
		assert.strictEqual( material.roughnessMap, material.emissiveMap, 'non-color slot can share image' );
		assert.true( material.emissiveMap.generateMipmaps, 'default sampler generates mipmaps' );
		assert.strictEqual( material.emissiveMap.image.data[ 0 ], 0x4000, 'material retains HDR sample' );

	} );

	for ( const invalid of [ 0x7c00, 0x7e00, 0xbc00, 0x4000 ] ) {

		QUnit.test( `invalid half-float sample ${ invalid } rejects`, async assert => {

			JXLLoader.prototype.load = ( url, onLoad ) => {

				const data = new Uint16Array( [ 0, 0, 0, 0x3c00 ] );
				data[ invalid === 0x4000 ? 3 : 0 ] = invalid;
				onLoad( new DataTexture( data, 1, 1 ) );

			};

			const parser = await makeParser( { required: true } );
			await assert.rejects( parser.getDependency( 'texture', 0 ), /KHR_jpegxl_linear/, 'invalid output fails instead of clipping' );

		} );

	}

	for ( const required of [ false, true ] ) {

		QUnit.test( required ? 'required decode failure rejects despite fallback' : 'missing fallback rejects', async assert => {

			JXLLoader.prototype.load = ( url, onLoad, onProgress, onError ) => onError( new Error( 'unsupported' ) );
			const parser = await makeParser( { required, fallback: required } );
			await assert.rejects( parser.getDependency( 'texture', 0 ), /KHR_jpegxl_linear/, 'failure is reported' );

		} );

	}

} );

function closeTo( assert, actual, expected, message ) {

	assert.ok( Math.abs( actual - expected ) < EPS, `${ message }: ${ actual } ~= ${ expected }` );

}

function arrayBufferToBase64( buffer ) {

	let binary = '';
	const bytes = new Uint8Array( buffer );

	for ( let i = 0; i < bytes.length; i ++ ) {

		binary += String.fromCharCode( bytes[ i ] );

	}

	return btoa( binary );

}

function createGaussianSplatGLTF( { sphericalHarmonicsDegree = 0 } = {} ) {

	const chunks = [];
	const bufferViews = [];
	const accessors = [];
	const attributes = {};
	let byteOffset = 0;

	function addAccessor( array, type, componentType, normalized = false, min = undefined, max = undefined ) {

		while ( byteOffset % 4 !== 0 ) {

			chunks.push( new Uint8Array( [ 0 ] ) );
			byteOffset ++;

		}

		const bytes = new Uint8Array( array.buffer, array.byteOffset, array.byteLength );
		const bufferView = bufferViews.push( {
			buffer: 0,
			byteOffset,
			byteLength: bytes.byteLength
		} ) - 1;
		const accessor = {
			bufferView,
			componentType,
			count: 1,
			type
		};

		if ( normalized === true ) accessor.normalized = true;
		if ( min !== undefined ) accessor.min = min;
		if ( max !== undefined ) accessor.max = max;

		chunks.push( bytes );
		byteOffset += bytes.byteLength;

		return accessors.push( accessor ) - 1;

	}

	const position = addAccessor( new Float32Array( [ 1, 2, 3 ] ), 'VEC3', FLOAT, false, [ 1, 2, 3 ], [ 1, 2, 3 ] );
	const scale = addAccessor( new Float32Array( [ 2, 3, 4 ] ), 'VEC3', FLOAT );
	const rotation = addAccessor( new Int8Array( [ 0, 0, 0, 127 ] ), 'VEC4', 5120, true );
	const opacity = addAccessor( new Uint8Array( [ 128 ] ), 'SCALAR', UNSIGNED_BYTE, true );
	const sh0 = addAccessor( new Float32Array( [ 0, 0, 0 ] ), 'VEC3', FLOAT );

	attributes.POSITION = position;
	attributes[ 'KHR_gaussian_splatting:SCALE' ] = scale;
	attributes[ 'KHR_gaussian_splatting:ROTATION' ] = rotation;
	attributes[ 'KHR_gaussian_splatting:OPACITY' ] = opacity;
	attributes[ 'KHR_gaussian_splatting:SH_DEGREE_0_COEF_0' ] = sh0;

	for ( let degree = 1; degree <= sphericalHarmonicsDegree; degree ++ ) {

		const coefficientCount = degree * 2 + 1;

		for ( let coefficient = 0; coefficient < coefficientCount; coefficient ++ ) {

			attributes[ `KHR_gaussian_splatting:SH_DEGREE_${ degree }_COEF_${ coefficient }` ] = addAccessor(
				new Float32Array( [ ( coefficient + 1 ) / 128, ( coefficient + 2 ) / 128, ( coefficient + 3 ) / 128 ] ),
				'VEC3',
				FLOAT
			);

		}

	}

	const buffer = new Uint8Array( byteOffset );
	let offset = 0;

	for ( const chunk of chunks ) {

		buffer.set( chunk, offset );
		offset += chunk.byteLength;

	}

	return {
		asset: { version: '2.0' },
		scene: 0,
		scenes: [ { nodes: [ 0 ] } ],
		nodes: [ { mesh: 0 } ],
		meshes: [ {
			primitives: [ {
				mode: 0,
				attributes,
				extensions: {
					KHR_gaussian_splatting: {
						kernel: 'ellipse',
						colorSpace: 'srgb_rec709_display'
					}
				}
			} ]
		} ],
		accessors,
		bufferViews,
		buffers: [ {
			byteLength: buffer.byteLength,
			uri: 'data:application/octet-stream;base64,' + arrayBufferToBase64( buffer.buffer )
		} ],
		extensionsUsed: [ 'KHR_gaussian_splatting' ]
	};

}

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'Loaders', () => {

		QUnit.module( 'GLTFLoader', () => {

			QUnit.test( 'CUBICSPLINE animations can be made additive at an interpolated frame', async ( assert ) => {

				const buffer = new Float32Array( [
					0, 1,
					0, 0, 0, 1, 2, 3, 0, 0, 0,
					0, 0, 0, 3, 4, 5, 0, 0, 0
				] );
				const json = {
					asset: { version: '2.0' },
					scene: 0,
					scenes: [ { nodes: [ 0 ] } ],
					nodes: [ {} ],
					buffers: [ { byteLength: buffer.byteLength, uri: 'data:application/octet-stream;base64,' + arrayBufferToBase64( buffer.buffer ) } ],
					bufferViews: [ { buffer: 0, byteLength: 8 }, { buffer: 0, byteOffset: 8, byteLength: 72 } ],
					accessors: [
						{ bufferView: 0, componentType: FLOAT, count: 2, type: 'SCALAR', min: [ 0 ], max: [ 1 ] },
						{ bufferView: 1, componentType: FLOAT, count: 6, type: 'VEC3' }
					],
					animations: [ {
						samplers: [ { input: 0, output: 1, interpolation: 'CUBICSPLINE' } ],
						channels: [ { sampler: 0, target: { node: 0, path: 'translation' } } ]
					} ]
				};
				const gltf = await new GLTFLoader().parseAsync( JSON.stringify( json ), '' );
				const clip = gltf.animations[ 0 ];

				makeClipAdditive( clip, 15, clip, 30 );

				assert.deepEqual( Array.from( clip.tracks[ 0 ].values ), [
					0, 0, 0, - 1, - 1, - 1, 0, 0, 0,
					0, 0, 0, 1, 1, 1, 0, 0, 0
				], 'Subtracts the interpolated reference value and preserves tangents' );

			} );

			QUnit.test( 'loads KHR_gaussian_splatting primitives as GaussianSplat', async ( assert ) => {

				const loader = new GLTFLoader();
				loader.register( function ( parser ) {

					return new GLTFGaussianSplatLoaderExtension( parser );

				} );

				const gltf = await loader.parseAsync( JSON.stringify( createGaussianSplatGLTF() ), '' );
				const mesh = gltf.scene.children[ 0 ];
				const covariances = mesh.splatGeometry.getAttribute( 'covariance' ).array;

				assert.ok( mesh.isGaussianSplat, 'creates GaussianSplat' );
				assert.deepEqual( Array.from( mesh.splatGeometry.getAttribute( 'position' ).array ), [ 1, 2, 3 ], 'loads centers' );
				closeTo( assert, covariances[ 0 ], 4, 'covariance xx' );
				closeTo( assert, covariances[ 3 ], 9, 'covariance yy' );
				closeTo( assert, covariances[ 5 ], 16, 'covariance zz' );
				assert.deepEqual( Array.from( mesh.splatGeometry.getAttribute( 'color' ).array ), [ 128, 128, 128, 128 ], 'loads degree-0 color and opacity' );

			} );

			QUnit.test( 'loads KHR_gaussian_splatting spherical harmonics attributes', async ( assert ) => {

				const loader = new GLTFLoader();
				loader.register( function ( parser ) {

					return new GLTFGaussianSplatLoaderExtension( parser );

				} );

				const gltf = await loader.parseAsync( JSON.stringify( createGaussianSplatGLTF( { sphericalHarmonicsDegree: 1 } ) ), '' );
				const mesh = gltf.scene.children[ 0 ];

				assert.deepEqual( Array.from( unpackSphericalHarmonicsBand( mesh.splatGeometry.getAttribute( 'sphericalHarmonics1' ).array, 1, 1 ) ), [
					129, 130, 131,
					130, 131, 132,
					131, 132, 133
				], 'loads SH1 coefficients' );

			} );

		} );

	} );

} );
