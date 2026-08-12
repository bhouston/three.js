import {
	BufferAttribute,
	BufferGeometry,
	Matrix3,
	Matrix4,
	Quaternion,
	Vector3
} from 'three';

const SH_C0 = 0.2820947917738781;
const SH_DEGREE_TO_COMPONENTS = [ 0, 9, 24, 45 ];
const SH_BAND_COMPONENTS = [ 0, 9, 15, 21 ];
// GPU upload packs four clamped-byte coefficients per uint32 word.
const SH_BAND_WORDS = [ 0, 3, 4, 6 ];
const GAUSSIAN_SPLAT_PLY_PROPERTY_MAPPING = {
	scale: [ 'scale_0', 'scale_1', 'scale_2' ],
	rotation: [ 'rot_0', 'rot_1', 'rot_2', 'rot_3' ],
	f_dc: [ 'f_dc_0', 'f_dc_1', 'f_dc_2' ],
	opacity: [ 'opacity' ]
};

const _covarianceMatrix = new Matrix3();
const _covarianceMatrixTranspose = new Matrix3();
const _eigenvectorMatrix = new Matrix3();
const _eigenvectorMatrix4 = new Matrix4();
const _rotationScaleMatrix = new Matrix4();
const _quaternion = new Quaternion();
const _scale = new Vector3();
const _zero = new Vector3();

function sigmoid( value ) {

	return 1 / ( 1 + Math.exp( - value ) );

}

// The target is expected to be a Uint8ClampedArray, which clamps and rounds
// assigned values natively.
function writeColorBytes( target, offset, r, g, b, a ) {

	target[ offset ] = r;
	target[ offset + 1 ] = g;
	target[ offset + 2 ] = b;
	target[ offset + 3 ] = a;

}

function sh0ToLinear( coefficient ) {

	return coefficient * SH_C0 + 0.5;

}

function linearToSH0( color ) {

	return ( color - 0.5 ) / SH_C0;

}

function writeColorBytesFromSH0( target, offset, r, g, b, a ) {

	writeColorBytes(
		target,
		offset,
		sh0ToLinear( r ) * 255,
		sh0ToLinear( g ) * 255,
		sh0ToLinear( b ) * 255,
		a * 255
	);

}

function writeCovariance( target, offset, sx, sy, sz, qx, qy, qz, qw ) {

	_quaternion.set( qx, qy, qz, qw ).normalize();
	_scale.set( sx, sy, sz );
	_rotationScaleMatrix.compose( _zero, _quaternion, _scale );

	_covarianceMatrix.setFromMatrix4( _rotationScaleMatrix );
	_covarianceMatrixTranspose.copy( _covarianceMatrix ).transpose();
	_covarianceMatrix.multiply( _covarianceMatrixTranspose );

	const elements = _covarianceMatrix.elements;

	target[ offset ] = elements[ 0 ];
	target[ offset + 1 ] = elements[ 3 ];
	target[ offset + 2 ] = elements[ 6 ];
	target[ offset + 3 ] = elements[ 4 ];
	target[ offset + 4 ] = elements[ 7 ];
	target[ offset + 5 ] = elements[ 8 ];

}

function decomposeCovariance( source, offset, target ) {

	let m00 = source[ offset ];
	let m01 = source[ offset + 1 ];
	let m02 = source[ offset + 2 ];
	let m11 = source[ offset + 3 ];
	let m12 = source[ offset + 4 ];
	let m22 = source[ offset + 5 ];

	let v00 = 1, v01 = 0, v02 = 0;
	let v10 = 0, v11 = 1, v12 = 0;
	let v20 = 0, v21 = 0, v22 = 1;

	for ( let i = 0; i < 10; i ++ ) {

		let p = 0;
		let q = 1;
		let value = Math.abs( m01 );
		const abs02 = Math.abs( m02 );
		const abs12 = Math.abs( m12 );

		if ( abs02 > value ) {

			p = 0;
			q = 2;
			value = abs02;

		}

		if ( abs12 > value ) {

			p = 1;
			q = 2;
			value = abs12;

		}

		if ( value <= 1e-10 ) break;

		let app, apq, aqq;

		if ( p === 0 && q === 1 ) {

			app = m00; apq = m01; aqq = m11;

		} else if ( p === 0 && q === 2 ) {

			app = m00; apq = m02; aqq = m22;

		} else {

			app = m11; apq = m12; aqq = m22;

		}

		const tau = ( aqq - app ) / ( 2 * apq );
		const t = Math.sign( tau ) / ( Math.abs( tau ) + Math.sqrt( 1 + tau * tau ) ) || 1;
		const c = 1 / Math.sqrt( 1 + t * t );
		const s = t * c;

		if ( p === 0 && q === 1 ) {

			const m02n = c * m02 - s * m12;
			const m12n = s * m02 + c * m12;

			m00 = app - t * apq;
			m11 = aqq + t * apq;
			m01 = 0;
			m02 = m02n;
			m12 = m12n;

			const v00n = c * v00 - s * v01;
			const v01n = s * v00 + c * v01;
			const v10n = c * v10 - s * v11;
			const v11n = s * v10 + c * v11;
			const v20n = c * v20 - s * v21;
			const v21n = s * v20 + c * v21;

			v00 = v00n; v01 = v01n;
			v10 = v10n; v11 = v11n;
			v20 = v20n; v21 = v21n;

		} else if ( p === 0 && q === 2 ) {

			const m01n = c * m01 - s * m12;
			const m12n = s * m01 + c * m12;

			m00 = app - t * apq;
			m22 = aqq + t * apq;
			m02 = 0;
			m01 = m01n;
			m12 = m12n;

			const v00n = c * v00 - s * v02;
			const v02n = s * v00 + c * v02;
			const v10n = c * v10 - s * v12;
			const v12n = s * v10 + c * v12;
			const v20n = c * v20 - s * v22;
			const v22n = s * v20 + c * v22;

			v00 = v00n; v02 = v02n;
			v10 = v10n; v12 = v12n;
			v20 = v20n; v22 = v22n;

		} else {

			const m01n = c * m01 - s * m02;
			const m02n = s * m01 + c * m02;

			m11 = app - t * apq;
			m22 = aqq + t * apq;
			m12 = 0;
			m01 = m01n;
			m02 = m02n;

			const v01n = c * v01 - s * v02;
			const v02n = s * v01 + c * v02;
			const v11n = c * v11 - s * v12;
			const v12n = s * v11 + c * v12;
			const v21n = c * v21 - s * v22;
			const v22n = s * v21 + c * v22;

			v01 = v01n; v02 = v02n;
			v11 = v11n; v12 = v12n;
			v21 = v21n; v22 = v22n;

		}

	}

	_eigenvectorMatrix.set(
		v00, v01, v02,
		v10, v11, v12,
		v20, v21, v22
	);

	if ( _eigenvectorMatrix.determinant() < 0 ) {

		v02 = - v02;
		v12 = - v12;
		v22 = - v22;

		_eigenvectorMatrix.set(
			v00, v01, v02,
			v10, v11, v12,
			v20, v21, v22
		);

	}

	target[ 0 ] = Math.sqrt( Math.max( 0, m00 ) );
	target[ 1 ] = Math.sqrt( Math.max( 0, m11 ) );
	target[ 2 ] = Math.sqrt( Math.max( 0, m22 ) );
	_eigenvectorMatrix4.setFromMatrix3( _eigenvectorMatrix );
	_quaternion.setFromRotationMatrix( _eigenvectorMatrix4 ).normalize().toArray( target, 3 );

}

function getGaussianSplatPLYPropertyMapping( sphericalHarmonicsDegree = 0 ) {

	const restComponentCount = SH_DEGREE_TO_COMPONENTS[ sphericalHarmonicsDegree ];

	if ( restComponentCount === undefined ) {

		throw new Error( `THREE.getGaussianSplatPLYPropertyMapping: Unsupported spherical harmonics degree ${ sphericalHarmonicsDegree }.` );

	}

	const mapping = {
		scale: GAUSSIAN_SPLAT_PLY_PROPERTY_MAPPING.scale,
		rotation: GAUSSIAN_SPLAT_PLY_PROPERTY_MAPPING.rotation,
		f_dc: GAUSSIAN_SPLAT_PLY_PROPERTY_MAPPING.f_dc,
		opacity: GAUSSIAN_SPLAT_PLY_PROPERTY_MAPPING.opacity
	};

	if ( restComponentCount > 0 ) {

		mapping.f_rest = Array.from( { length: restComponentCount }, ( _, i ) => `f_rest_${ i }` );

	}

	return mapping;

}

function createPackedSphericalHarmonicsBand( count, degree ) {

	const packed = new Uint32Array( count * SH_BAND_WORDS[ degree ] );
	packed.fill( 0x80808080 );

	return {
		packed,
		bytes: new Uint8ClampedArray( packed.buffer )
	};

}

function createSphericalHarmonicsAttribute( values, count, degree ) {

	const words = SH_BAND_WORDS[ degree ];

	if ( values instanceof Uint32Array === false ) {

		throw new Error( `THREE.createGaussianSplatGeometry: sphericalHarmonics${ degree } must use packed uint32 words.` );

	}

	if ( values.length !== count * words ) {

		throw new Error( `THREE.createGaussianSplatGeometry: Invalid sphericalHarmonics${ degree } packed length.` );

	}

	return new BufferAttribute( values, words );

}

function getSphericalHarmonicsDegree( geometry ) {

	if ( geometry === undefined || geometry.isBufferGeometry !== true ) return 0;

	let degree = 0;

	for ( let i = 1; i <= 3; i ++ ) {

		const attribute = geometry.getAttribute( `sphericalHarmonics${ i }` );

		if ( attribute === undefined ) break;

		if ( attribute.itemSize !== SH_BAND_WORDS[ i ] ) {

			throw new Error( `THREE.getSphericalHarmonicsDegree: Invalid sphericalHarmonics${ i } itemSize.` );

		}

		if ( attribute.array instanceof Uint32Array === false ) {

			throw new Error( `THREE.getSphericalHarmonicsDegree: sphericalHarmonics${ i } must use packed uint32 words.` );

		}

		degree = i;

	}

	for ( let i = degree + 1; i <= 3; i ++ ) {

		if ( geometry.getAttribute( `sphericalHarmonics${ i }` ) !== undefined ) {

			throw new Error( 'THREE.getSphericalHarmonicsDegree: Spherical harmonics attributes must be contiguous.' );

		}

	}

	const position = geometry.getAttribute( 'position' );

	if ( position !== undefined ) {

		for ( let i = 1; i <= degree; i ++ ) {

			if ( geometry.getAttribute( `sphericalHarmonics${ i }` ).count !== position.count ) {

				throw new Error( 'THREE.getSphericalHarmonicsDegree: Spherical harmonics attribute counts must match position.' );

			}

		}

	}

	return degree;

}

/**
 * Creates Gaussian splat geometry from packed attribute arrays. Higher-order
 * spherical harmonics must be supplied as packed `Uint32Array` words
 * (`SH_BAND_WORDS[ degree ]` words per splat, four clamped-byte coefficients
 * per word using `( value - 128 ) / 128`).
 *
 * @param {Float32Array} centers - Splat centers.
 * @param {Float32Array} covariances - Splat covariance matrices.
 * @param {Uint8Array|Uint8ClampedArray} colors - RGBA colors.
 * @param {Object} [sphericalHarmonics={}] - Optional packed SH band arrays.
 * @return {BufferGeometry} The Gaussian splat geometry.
 */
function createGaussianSplatGeometry( centers, covariances, colors, sphericalHarmonics = {} ) {

	const geometry = new BufferGeometry();
	geometry.setAttribute( 'position', new BufferAttribute( centers, 3 ) );
	geometry.setAttribute( 'covariance', new BufferAttribute( covariances, 6 ) );
	geometry.setAttribute( 'color', new BufferAttribute( colors, 4, true ) );

	const count = centers.length / 3;

	for ( let i = 1; i <= 3; i ++ ) {

		const values = sphericalHarmonics[ `sh${ i }` ] || sphericalHarmonics[ `sphericalHarmonics${ i }` ];

		if ( values !== undefined ) {

			geometry.setAttribute( `sphericalHarmonics${ i }`, createSphericalHarmonicsAttribute( values, count, i ) );

		}

	}

	getSphericalHarmonicsDegree( geometry );
	geometry.computeBoundingBox();
	geometry.computeBoundingSphere();

	return geometry;

}

function createGaussianSplatGeometryFromPLYGeometry( geometry, {
	scaleAttribute = 'scale',
	rotationAttribute = 'rotation',
	sh0Attribute = 'f_dc',
	shRestAttribute = 'f_rest',
	opacityAttribute = 'opacity'
} = {} ) {

	if ( geometry === undefined || geometry.isBufferGeometry !== true ) {

		throw new Error( 'THREE.createGaussianSplatGeometryFromPLYGeometry: PLY geometry must be a BufferGeometry.' );

	}

	const position = geometry.getAttribute( 'position' );
	const scale = geometry.getAttribute( scaleAttribute );
	const rotation = geometry.getAttribute( rotationAttribute );
	const sh0 = geometry.getAttribute( sh0Attribute );
	const shRest = geometry.getAttribute( shRestAttribute );
	const opacity = geometry.getAttribute( opacityAttribute );

	if ( position === undefined || scale === undefined || rotation === undefined || sh0 === undefined || opacity === undefined ) {

		throw new Error( 'THREE.createGaussianSplatGeometryFromPLYGeometry: PLY geometry requires position, scale, rotation, f_dc and opacity attributes.' );

	}

	const count = position.count;

	if ( position.itemSize !== 3 || scale.itemSize !== 3 || rotation.itemSize !== 4 || sh0.itemSize !== 3 || opacity.itemSize !== 1 ) {

		throw new Error( 'THREE.createGaussianSplatGeometryFromPLYGeometry: Invalid Gaussian splat PLY attribute itemSize.' );

	}

	if ( scale.count !== count || rotation.count !== count || sh0.count !== count || opacity.count !== count ) {

		throw new Error( 'THREE.createGaussianSplatGeometryFromPLYGeometry: Gaussian splat PLY attribute counts must match position.' );

	}

	const centers = new Float32Array( count * 3 );
	const covariances = new Float32Array( count * 6 );
	const colors = new Uint8ClampedArray( count * 4 );
	const sphericalHarmonicsDegree = getPLYRestSphericalHarmonicsDegree( shRest );
	const sphericalHarmonics = {};
	const sphericalHarmonicsBytes = {};

	for ( let degree = 1; degree <= sphericalHarmonicsDegree; degree ++ ) {

		const band = createPackedSphericalHarmonicsBand( count, degree );
		sphericalHarmonics[ `sh${ degree }` ] = band.packed;
		sphericalHarmonicsBytes[ `sh${ degree }` ] = band.bytes;

	}

	for ( let i = 0; i < count; i ++ ) {

		const i3 = i * 3;
		centers[ i3 ] = position.getX( i );
		centers[ i3 + 1 ] = position.getY( i );
		centers[ i3 + 2 ] = position.getZ( i );

		const sx = Math.exp( scale.getX( i ) );
		const sy = Math.exp( scale.getY( i ) );
		const sz = Math.exp( scale.getZ( i ) );

		// GraphDECO/INRIA PLY stores quaternions as rot_0=w, rot_1=x, rot_2=y, rot_3=z.
		const qw = rotation.getX( i );
		const qx = rotation.getY( i );
		const qy = rotation.getZ( i );
		const qz = rotation.getW( i );

		writeCovariance( covariances, i * 6, sx, sy, sz, qx, qy, qz, qw );
		writeColorBytesFromSH0(
			colors,
			i * 4,
			sh0.getX( i ),
			sh0.getY( i ),
			sh0.getZ( i ),
			sigmoid( opacity.getX( i ) )
		);

		if ( sphericalHarmonicsDegree > 0 ) {

			writeSphericalHarmonicsFromPLYRest( sphericalHarmonicsBytes, i, shRest );

		}

	}

	return createGaussianSplatGeometry( centers, covariances, colors, sphericalHarmonics );

}

function getPLYRestSphericalHarmonicsDegree( shRest ) {

	if ( shRest === undefined ) return 0;

	const degree = SH_DEGREE_TO_COMPONENTS.indexOf( shRest.itemSize );

	if ( degree === - 1 ) {

		throw new Error( 'THREE.createGaussianSplatGeometryFromPLYGeometry: Unsupported number of f_rest spherical harmonics coefficients.' );

	}

	return degree;

}

function writeSphericalHarmonicsFromPLYRest( sphericalHarmonicsBytes, index, shRest ) {

	const stride = shRest.itemSize / 3;
	const source = shRest.array;
	const sourceOffset = index * shRest.itemSize;

	for ( let degree = 1; degree <= 3; degree ++ ) {

		const target = sphericalHarmonicsBytes[ `sh${ degree }` ];

		if ( target === undefined ) break;

		const bandOffset = degree === 1 ? 0 : degree === 2 ? 3 : 8;
		const byteStride = SH_BAND_WORDS[ degree ] * 4;
		const targetOffset = index * byteStride;

		for ( let j = 0; j < SH_BAND_COMPONENTS[ degree ]; j ++ ) {

			const coefficient = Math.floor( j / 3 );
			const channel = j % 3;
			target[ targetOffset + j ] = source[ sourceOffset + bandOffset + coefficient + channel * stride ] * 128 + 128;

		}

	}

}

export {
	GAUSSIAN_SPLAT_PLY_PROPERTY_MAPPING,
	SH_BAND_COMPONENTS,
	SH_BAND_WORDS,
	SH_C0,
	SH_DEGREE_TO_COMPONENTS,
	createGaussianSplatGeometry,
	createGaussianSplatGeometryFromPLYGeometry,
	createPackedSphericalHarmonicsBand,
	decomposeCovariance,
	getGaussianSplatPLYPropertyMapping,
	getSphericalHarmonicsDegree,
	linearToSH0,
	sh0ToLinear,
	sigmoid,
	writeColorBytes,
	writeColorBytesFromSH0,
	writeCovariance
};
