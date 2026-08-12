import {
	MathUtils,
	Quaternion
} from 'three';

import { ZSTDEncoder } from '../libs/zstdenc.module.js';
import {
	SH_BAND_COMPONENTS,
	SH_BAND_WORDS,
	SH_C0,
	decomposeCovariance,
	getSphericalHarmonicsDegree
} from '../utils/GaussianSplatUtils.js';

const SPZ_MAGIC = 0x5053474e;
const SPZ_VERSION = 4;
const HEADER_SIZE_BYTES = 32;
const TOC_ENTRY_SIZE_BYTES = 16;
const SPZ_COLOR_SCALE = SH_C0 / 0.15;
const SH_DEGREE_TO_VECTORS = [ 0, 3, 8, 15 ];
const MAX_SUPPORTED_SH_DEGREE = 3;
const FLAG_ANTIALIASED = 0x01;
const INT24_MIN = - 0x7fffff;
const INT24_MAX = 0x7fffff;

let _zstd;

const _quaternion = new Quaternion();
const _quaternionArray = [ 0, 0, 0, 1 ];

/**
 * An exporter for compressed Gaussian splat `.spz` files.
 *
 * This exporter writes SPZ version 4 files only. Input must be the Gaussian
 * splat `BufferGeometry` layout used by `GaussianSplatMesh`: `position`,
 * `covariance`, `color`, and optional `sphericalHarmonics1` through
 * `sphericalHarmonics3` attributes.
 *
 * ```js
 * const exporter = new SPZExporter();
 * const arrayBuffer = await exporter.parseAsync( splatGeometry );
 * ```
 *
 * @three_import import { SPZExporter } from 'three/addons/exporters/SPZExporter.js';
 */
class SPZExporter {

	/**
	 * Parses the given Gaussian splat geometry and generates SPZ v4 output.
	 *
	 * @param {BufferGeometry|GaussianSplatMesh} input - The splat geometry, or an object with a `splatGeometry` property.
	 * @param {SPZExporter~OnDone} onDone - A callback function that is executed when the export has finished.
	 * @param {SPZExporter~OnError} onError - A callback function that is executed when an error happens.
	 * @param {SPZExporter~Options} options - The export options.
	 * @return {Promise<ArrayBuffer>|undefined} A promise when no `onDone` callback is provided.
	 */
	parse( input, onDone, onError, options = {} ) {

		const promise = this.parseAsync( input, options );

		if ( typeof onDone === 'function' ) {

			promise.then( onDone ).catch( onError );
			return;

		}

		return promise;

	}

	/**
	 * Async version of {@link SPZExporter#parse}.
	 *
	 * @async
	 * @param {BufferGeometry|GaussianSplatMesh} input - The splat geometry, or an object with a `splatGeometry` property.
	 * @param {SPZExporter~Options} options - The export options.
	 * @return {Promise<ArrayBuffer>} A Promise that resolves with the exported SPZ v4 data.
	 */
	async parseAsync( input, options = {} ) {

		options = Object.assign( {
			fractionalBits: 12,
			compressionLevel: 3,
			antialiased: false,
			sh1Bits: 5,
			shRestBits: 4,
			maxSphericalHarmonicsDegree: MAX_SUPPORTED_SH_DEGREE
		}, options );

		validateOptions( options );

		const geometry = input && input.isBufferGeometry === true ? input : input && input.splatGeometry;

		const zstd = await getZSTDEncoder();
		const position = geometry.getAttribute( 'position' );
		const covariance = geometry.getAttribute( 'covariance' );
		const color = geometry.getAttribute( 'color' );
		const count = position.count;
		const shDegree = Math.min( getSphericalHarmonicsDegree( geometry ), options.maxSphericalHarmonicsDegree );
		const streams = createSPZStreams( geometry, position, covariance, color, count, shDegree, options );
		const compressedStreams = streams.map( ( stream ) => zstd.encode( stream, options.compressionLevel ) );

		return buildSPZBuffer( compressedStreams, streams, count, shDegree, options ).buffer;

	}

}

function getZSTDEncoder() {

	if ( _zstd === undefined ) {

		const encoder = new ZSTDEncoder();
		_zstd = encoder.init().then( () => encoder ).catch( ( e ) => {

			_zstd = undefined;
			throw e;

		} );

	}

	return _zstd;

}

function validateOptions( options ) {

	if ( options.fractionalBits < 0 || options.fractionalBits > 23 || Number.isInteger( options.fractionalBits ) === false ) {

		throw new Error( 'THREE.SPZExporter: fractionalBits must be an integer in the range [0, 23].' );

	}

	if ( options.compressionLevel < 0 || options.compressionLevel > 9 || Number.isInteger( options.compressionLevel ) === false ) {

		throw new Error( 'THREE.SPZExporter: compressionLevel must be an integer in the range [0, 9].' );

	}

	if ( options.sh1Bits < 1 || options.sh1Bits > 8 || Number.isInteger( options.sh1Bits ) === false ) {

		throw new Error( 'THREE.SPZExporter: sh1Bits must be an integer in the range [1, 8].' );

	}

	if ( options.shRestBits < 1 || options.shRestBits > 8 || Number.isInteger( options.shRestBits ) === false ) {

		throw new Error( 'THREE.SPZExporter: shRestBits must be an integer in the range [1, 8].' );

	}

	if ( options.maxSphericalHarmonicsDegree < 0 || options.maxSphericalHarmonicsDegree > MAX_SUPPORTED_SH_DEGREE || Number.isInteger( options.maxSphericalHarmonicsDegree ) === false ) {

		throw new Error( 'THREE.SPZExporter: maxSphericalHarmonicsDegree must be an integer in the range [0, 3].' );

	}

}

function createSPZStreams( geometry, position, covariance, color, count, shDegree, options ) {

	const positions = new Uint8Array( count * 9 );
	const alphas = new Uint8Array( count );
	const colors = new Uint8Array( count * 3 );
	const scales = new Uint8Array( count * 3 );
	const rotations = new Uint8Array( count * 4 );
	const decomposed = [ 0, 0, 0, 0, 0, 0, 1 ];
	const fixedScale = 1 << options.fractionalBits;

	for ( let i = 0; i < count; i ++ ) {

		const i3 = i * 3;
		const i4 = i * 4;
		const positionOffset = i * 9;

		writeInt24( positions, positionOffset, MathUtils.clamp( Math.round( position.array[ i3 ] * fixedScale ), INT24_MIN, INT24_MAX ) );
		writeInt24( positions, positionOffset + 3, MathUtils.clamp( Math.round( position.array[ i3 + 1 ] * fixedScale ), INT24_MIN, INT24_MAX ) );
		writeInt24( positions, positionOffset + 6, MathUtils.clamp( Math.round( position.array[ i3 + 2 ] * fixedScale ), INT24_MIN, INT24_MAX ) );

		alphas[ i ] = color.array[ i4 + 3 ];
		colors[ i3 ] = encodeColorByte( color.array[ i4 ] );
		colors[ i3 + 1 ] = encodeColorByte( color.array[ i4 + 1 ] );
		colors[ i3 + 2 ] = encodeColorByte( color.array[ i4 + 2 ] );

		decomposeCovariance( covariance.array, i * 6, decomposed );
		scales[ i3 ] = encodeScaleByte( decomposed[ 0 ] );
		scales[ i3 + 1 ] = encodeScaleByte( decomposed[ 1 ] );
		scales[ i3 + 2 ] = encodeScaleByte( decomposed[ 2 ] );
		writeSmallestThreeQuaternion( rotations, i * 4, decomposed[ 3 ], decomposed[ 4 ], decomposed[ 5 ], decomposed[ 6 ] );

	}

	const streams = [ positions, alphas, colors, scales, rotations ];

	if ( shDegree > 0 ) {

		streams.push( createSphericalHarmonicsStream( geometry, count, shDegree, options ) );

	}

	return streams;

}

function writeInt24( target, offset, value ) {

	target[ offset ] = value & 0xff;
	target[ offset + 1 ] = ( value >> 8 ) & 0xff;
	target[ offset + 2 ] = ( value >> 16 ) & 0xff;

}

function encodeColorByte( value ) {

	return MathUtils.clamp( Math.round( ( ( value / 255 - 0.5 ) / SPZ_COLOR_SCALE + 0.5 ) * 255 ), 0, 255 );

}

function encodeScaleByte( value ) {

	return MathUtils.clamp( Math.round( ( Math.log( Math.max( value, 1e-20 ) ) + 10 ) * 16 ), 0, 255 );

}

function writeSmallestThreeQuaternion( target, offset, qx, qy, qz, qw ) {

	const quaternion = _quaternion.set( qx, qy, qz, qw ).normalize().toArray( _quaternionArray );

	let largestIndex = 0;

	for ( let i = 1; i < 4; i ++ ) {

		if ( Math.abs( quaternion[ i ] ) > Math.abs( quaternion[ largestIndex ] ) ) {

			largestIndex = i;

		}

	}

	const sign = quaternion[ largestIndex ] < 0 ? - 1 : 1;
	let packed = largestIndex;

	for ( let i = 0; i < 4; i ++ ) {

		if ( i === largestIndex ) continue;

		const value = quaternion[ i ] * sign;
		const magnitude = MathUtils.clamp( Math.round( Math.abs( value ) / Math.SQRT1_2 * 511 ), 0, 511 );
		const signBit = value < 0 ? 512 : 0;

		packed = ( packed << 10 ) | signBit | magnitude;

	}

	target[ offset ] = packed & 0xff;
	target[ offset + 1 ] = ( packed >> 8 ) & 0xff;
	target[ offset + 2 ] = ( packed >> 16 ) & 0xff;
	target[ offset + 3 ] = ( packed >>> 24 ) & 0xff;

}

function createSphericalHarmonicsStream( geometry, count, shDegree, options ) {

	const sphericalHarmonics = new Uint8Array( count * SH_DEGREE_TO_VECTORS[ shDegree ] * 3 );
	const bandBytes = {};

	for ( let degree = 1; degree <= shDegree; degree ++ ) {

		const attribute = geometry.getAttribute( `sphericalHarmonics${ degree }` );
		bandBytes[ degree ] = new Uint8Array( attribute.array.buffer, attribute.array.byteOffset, attribute.array.byteLength );

	}

	for ( let i = 0; i < count; i ++ ) {

		const targetOffset = i * SH_DEGREE_TO_VECTORS[ shDegree ] * 3;
		let bandOffset = targetOffset;

		for ( let degree = 1; degree <= shDegree; degree ++ ) {

			const bytes = bandBytes[ degree ];
			const sourceOffset = i * SH_BAND_WORDS[ degree ] * 4;
			const componentCount = SH_BAND_COMPONENTS[ degree ];
			const bits = degree === 1 ? options.sh1Bits : options.shRestBits;

			for ( let j = 0; j < componentCount; j ++ ) {

				sphericalHarmonics[ bandOffset ++ ] = quantizeSphericalHarmonicsByte( bytes[ sourceOffset + j ], bits );

			}

		}

	}

	return sphericalHarmonics;

}

function quantizeSphericalHarmonicsByte( value, bits ) {

	const bucketSize = 1 << ( 8 - bits );
	return MathUtils.clamp( Math.floor( ( value + bucketSize / 2 ) / bucketSize ) * bucketSize, 0, 255 );

}

function buildSPZBuffer( compressedStreams, streams, count, shDegree, options ) {

	const tocSize = compressedStreams.length * TOC_ENTRY_SIZE_BYTES;
	const compressedSize = compressedStreams.reduce( ( sum, stream ) => sum + stream.byteLength, 0 );
	const bytes = new Uint8Array( HEADER_SIZE_BYTES + tocSize + compressedSize );
	const view = new DataView( bytes.buffer );
	let offset = HEADER_SIZE_BYTES + tocSize;

	view.setUint32( 0, SPZ_MAGIC, true );
	view.setUint32( 4, SPZ_VERSION, true );
	view.setUint32( 8, count, true );
	view.setUint8( 12, shDegree );
	view.setUint8( 13, options.fractionalBits );
	view.setUint8( 14, options.antialiased ? FLAG_ANTIALIASED : 0 );
	view.setUint8( 15, compressedStreams.length );
	view.setUint32( 16, HEADER_SIZE_BYTES, true );

	for ( let i = 0; i < compressedStreams.length; i ++ ) {

		const compressed = compressedStreams[ i ];
		const tocOffset = HEADER_SIZE_BYTES + i * TOC_ENTRY_SIZE_BYTES;

		view.setBigUint64( tocOffset, BigInt( compressed.byteLength ), true );
		view.setBigUint64( tocOffset + 8, BigInt( streams[ i ].byteLength ), true );
		bytes.set( compressed, offset );
		offset += compressed.byteLength;

	}

	return bytes;

}

/**
 * Export options of `SPZExporter`.
 *
 * @typedef {Object} SPZExporter~Options
 * @property {number} [fractionalBits=12] - Number of fractional bits used for fixed-point centers.
 * @property {number} [compressionLevel=3] - ZSTD compression level.
 * @property {boolean} [antialiased=false] - Whether to set the SPZ antialiasing flag.
 * @property {number} [sh1Bits=5] - Quantization bits for degree 1 spherical harmonics coefficients.
 * @property {number} [shRestBits=4] - Quantization bits for degree 2 and 3 spherical harmonics coefficients.
 * @property {number} [maxSphericalHarmonicsDegree=3] - Maximum spherical harmonics degree to export.
 **/

/**
 * onDone callback of `SPZExporter`.
 *
 * @callback SPZExporter~OnDone
 * @param {ArrayBuffer} result - The generated SPZ v4 data.
 */

/**
 * onError callback of `SPZExporter`.
 *
 * @callback SPZExporter~OnError
 * @param {Error} error - The error that happened during export.
 */

export { SPZExporter };
