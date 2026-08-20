import { describe, test, expect } from 'vitest';
import { GLTFLoader } from '../../../../examples/jsm/loaders/GLTFLoader.js';
import { GLTFGaussianSplatLoaderExtension } from '../../../../examples/jsm/loaders/GLTFGaussianSplatLoaderExtension.js';
import { unpackSphericalHarmonicsBand } from '../utils/GaussianSplatTestUtils.js';

const EPS = 1e-6;
const FLOAT = 5126;
const UNSIGNED_BYTE = 5121;

function closeTo( actual, expected ) {

	expect( Math.abs( actual - expected ) < EPS ).toBeTruthy();

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

describe( 'Addons', () => {

	describe( 'Loaders', () => {

		describe( 'GLTFLoader', () => {

			test( 'loads KHR_gaussian_splatting primitives as GaussianSplat', async () => {

				const loader = new GLTFLoader();
				loader.register( function ( parser ) {

					return new GLTFGaussianSplatLoaderExtension( parser );

				} );

				const gltf = await loader.parseAsync( JSON.stringify( createGaussianSplatGLTF() ), '' );
				const mesh = gltf.scene.children[ 0 ];
				const covariances = mesh.splatGeometry.getAttribute( 'covariance' ).array;

				expect( mesh.isGaussianSplat ).toBeTruthy();
				expect( Array.from( mesh.splatGeometry.getAttribute( 'position' ).array ) ).toEqual( [ 1, 2, 3 ] );
				closeTo( covariances[ 0 ], 4 );
				closeTo( covariances[ 3 ], 9 );
				closeTo( covariances[ 5 ], 16 );
				expect( Array.from( mesh.splatGeometry.getAttribute( 'color' ).array ) ).toEqual( [ 128, 128, 128, 128 ] );

			} );

			test( 'loads KHR_gaussian_splatting spherical harmonics attributes', async () => {

				const loader = new GLTFLoader();
				loader.register( function ( parser ) {

					return new GLTFGaussianSplatLoaderExtension( parser );

				} );

				const gltf = await loader.parseAsync( JSON.stringify( createGaussianSplatGLTF( { sphericalHarmonicsDegree: 1 } ) ), '' );
				const mesh = gltf.scene.children[ 0 ];

				expect( Array.from( unpackSphericalHarmonicsBand( mesh.splatGeometry.getAttribute( 'sphericalHarmonics1' ).array, 1, 1 ) ) ).toEqual( [
					129, 130, 131,
					130, 131, 132,
					131, 132, 133
				] );

			} );

		} );

	} );

} );
