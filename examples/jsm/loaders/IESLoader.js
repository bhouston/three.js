import {
	DataTexture,
	FileLoader,
	FloatType,
	RedFormat,
	MathUtils,
	Loader,
	RepeatWrapping,
	UnsignedByteType,
	LinearFilter,
	HalfFloatType,
	DataUtils
} from 'three';

/**
 * A loader for Type C photometry in the IES LM-63-1986, 1991, 1995, 2002
 * and 2019 formats. Type A/B photometry and external TILT files are not supported.
 * Embedded TILT data is read, but lamp tilt correction is not applied.
 *
 * The loaded texture should be assigned to {@link IESSpotLight#map}.
 *
 * ```js
 * const loader = new IESLoader();
 * const texture = await loader.loadAsync( 'ies/007cfb11e343e2f42e3b476be4ab684e.ies' );
 *
 * const spotLight = new THREE.IESSpotLight( 0xff0000, 500 );
 * spotLight.iesMap = texture;
 * ```
 *
 * @augments Loader
 * @three_import import { IESLoader } from 'three/addons/loaders/IESLoader.js';
 */
class IESLoader extends Loader {

	/**
	 * Constructs a new IES loader.
	 *
	 * @param {LoadingManager} [manager] - The loading manager.
	 */
	constructor( manager ) {

		super( manager );

		/**
		 * The texture type.
		 *
		 * @type {(HalfFloatType|FloatType)}
		 * @default HalfFloatType
		 */
		this.type = HalfFloatType;

	}

	_getIESValues( iesLamp, type ) {

		const width = iesLamp.numHorAngles === 1 ? 1 : 360;
		const height = 180;
		const size = width * height;

		const data = new Array( size );

		function interpolateCandelaValues( phi, theta ) {

			// A missing hemisphere has no measured light; do not extrapolate its edge.
			if ( phi < iesLamp.verAngles[ 0 ] || phi > iesLamp.verAngles[ iesLamp.numVerAngles - 1 ] ) return 0;

			let phiIndex = 0, thetaIndex = 0;
			let startTheta = 0, endTheta = 0, startPhi = 0, endPhi = 0;

			for ( let i = 0; i < iesLamp.numHorAngles - 1; ++ i ) {

				if ( theta < iesLamp.horAngles[ i + 1 ] || i == iesLamp.numHorAngles - 2 ) {

					thetaIndex = i;
					startTheta = iesLamp.horAngles[ i ];
					endTheta = iesLamp.horAngles[ i + 1 ];

					break;

				}

			}

			for ( let i = 0; i < iesLamp.numVerAngles - 1; ++ i ) {

				if ( phi < iesLamp.verAngles[ i + 1 ] || i == iesLamp.numVerAngles - 2 ) {

					phiIndex = i;
					startPhi = iesLamp.verAngles[ i ];
					endPhi = iesLamp.verAngles[ i + 1 ];

					break;

				}

			}

			const deltaTheta = endTheta - startTheta;
			const deltaPhi = endPhi - startPhi;

			if ( deltaPhi === 0 ) // Outside range
				return 0;

			const t1 = deltaTheta === 0 ? 0 : ( theta - startTheta ) / deltaTheta;
			const t2 = ( phi - startPhi ) / deltaPhi;

			const nextThetaIndex = deltaTheta === 0 ? thetaIndex : thetaIndex + 1;

			const v1 = MathUtils.lerp( iesLamp.candelaValues[ thetaIndex ][ phiIndex ], iesLamp.candelaValues[ nextThetaIndex ][ phiIndex ], t1 );
			const v2 = MathUtils.lerp( iesLamp.candelaValues[ thetaIndex ][ phiIndex + 1 ], iesLamp.candelaValues[ nextThetaIndex ][ phiIndex + 1 ], t1 );
			const v = MathUtils.lerp( v1, v2, t2 );

			return v;

		}

		const endTheta = iesLamp.horAngles[ iesLamp.numHorAngles - 1 ];

		for ( let i = 0; i < size; ++ i ) {

			// the horizontal angles cover 0 to 359 degrees
			const theta = i % width;
			const phi = Math.floor( i / width );

			let sampleTheta = theta;

 			// mirror the measured range around the cycle
 			if ( endTheta !== 0 && sampleTheta > endTheta ) {

				sampleTheta %= endTheta * 2;

				if ( sampleTheta > endTheta ) {

					sampleTheta = endTheta * 2 - sampleTheta;

				}

			}

			// the vertical angles span [0, 180]
			const samplePhi = phi * 180 / ( height - 1 );
			data[ phi + theta * height ] = interpolateCandelaValues( samplePhi, sampleTheta );

		}

		let result = null;

		if ( type === UnsignedByteType ) result = Uint8Array.from( data.map( v => Math.min( v * 0xFF, 0xFF ) ) );
		else if ( type === HalfFloatType ) result = Uint16Array.from( data.map( v => DataUtils.toHalfFloat( v ) ) );
		else if ( type === FloatType ) result = Float32Array.from( data );
		else console.error( 'IESLoader: Unsupported type:', type );

		return result;

	}

	/**
	 * Starts loading from the given URL and passes the loaded IES texture
	 * to the `onLoad()` callback.
	 *
	 * @param {string} url - The path/URL of the file to be loaded. This can also be a data URI.
	 * @param {function(DataTexture)} onLoad - Executed when the loading process has been finished.
	 * @param {onProgressCallback} onProgress - Executed while the loading is in progress.
	 * @param {onErrorCallback} onError - Executed when errors occur.
	 */
	load( url, onLoad, onProgress, onError ) {

		const loader = new FileLoader( this.manager );
		loader.setPriority( this.priority );
		loader.setCache( this.cache );
		loader.setResponseType( 'text' );
		loader.setCrossOrigin( this.crossOrigin );
		loader.setWithCredentials( this.withCredentials );
		loader.setPath( this.path );
		loader.setRequestHeader( this.requestHeader );

		loader.load( url, text => {

			try {

				onLoad( this.parse( text ) );

			} catch ( error ) {

				if ( onError ) {

					onError( error );

				} else {

					console.error( error );

				}

				this.manager.itemError( url );

			}

		}, onProgress, onError );

	}

	/**
	 * Parses the given IES data.
	 *
	 * @param {string} text - The raw IES data.
	 * @return {DataTexture} THE IES data as a texture.
	 */
	parse( text ) {

		const type = this.type;

		const iesLamp = new IESLamp( text );
		const data = this._getIESValues( iesLamp, type );

		// X holds the vertical angle from 0 to 180 degrees inclusively, Y holds the horizontal
		// angle from 0 to 359 degrees and wraps around, or a single row for symmetric profiles
		const texture = new DataTexture( data, 180, data.length / 180, RedFormat, type );
		texture.minFilter = LinearFilter;
		texture.magFilter = LinearFilter;
		texture.wrapT = RepeatWrapping;
		texture.needsUpdate = true;

		return texture;

	}

}


function IESLamp( text ) {

	const lines = text.replace( /^\uFEFF/, '' ).trim().split( /\r\n?|\n/ );
	const header = lines[ 0 ].trim();
	let version = 1986;

	if ( /^IESNA91$/i.test( header ) ) {

		version = 1991;

	} else if ( /^IES(?:NA)?:LM-63-(1991|1995|2002|2019)$/i.test( header ) ) {

		version = Number( header.slice( - 4 ) );

	} else if ( /^IES(?:NA)?[:\d]/i.test( header ) ) {

		throw new Error( 'IESLoader: Unsupported LM-63 version.' );

	}

	const tiltIndex = lines.findIndex( line => /^\s*TILT\s*=/i.test( line ) );
	if ( tiltIndex === - 1 ) throw new Error( 'IESLoader: Missing TILT declaration.' );

	const tiltMode = lines[ tiltIndex ].split( '=' )[ 1 ].trim().toUpperCase();
	if ( tiltMode !== 'NONE' && tiltMode !== 'INCLUDE' ) throw new Error( 'IESLoader: External TILT files are not supported.' );

	// Numeric tables can wrap anywhere, including across logical array boundaries.
	const tokens = lines.slice( tiltIndex + 1 ).join( ' ' ).match( /[^\s,]+/g ) || [];
	let cursor = 0;

	function readNumber( label ) {

		if ( cursor >= tokens.length ) throw new Error( `IESLoader: Unexpected end of data while reading ${ label }.` );

		const token = tokens[ cursor ++ ];
		if ( ! /^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test( token ) ) throw new Error( `IESLoader: Invalid number for ${ label }.` );

		const value = Number( token );
		if ( ! Number.isFinite( value ) ) throw new Error( `IESLoader: Non-finite number for ${ label }.` );

		return value;

	}

	function readCount( label, minimum = 1 ) {

		const value = readNumber( label );
		if ( ! Number.isSafeInteger( value ) || value < minimum ) throw new Error( `IESLoader: Invalid ${ label }.` );

		return value;

	}

	function readArray( count, label ) {

		if ( count > tokens.length - cursor ) throw new Error( `IESLoader: Unexpected end of data while reading ${ label }.` );

		const values = [];
		for ( let i = 0; i < count; i ++ ) values.push( readNumber( label ) );
		return values;

	}

	function validateAngles( values, minimum, maximum, label ) {

		for ( let i = 0; i < values.length; i ++ ) {

			if ( values[ i ] < minimum || values[ i ] > maximum || ( i > 0 && values[ i ] <= values[ i - 1 ] ) ) {

				throw new Error( `IESLoader: ${ label } must be strictly increasing in [${ minimum }, ${ maximum }].` );

			}

		}

	}

	this.tiltData = { angles: [], mulFactors: [] };
	if ( tiltMode === 'INCLUDE' ) {

		this.tiltData.lampToLumGeometry = readCount( 'tilt geometry' );
		if ( this.tiltData.lampToLumGeometry > 3 ) throw new Error( 'IESLoader: Invalid tilt geometry.' );
		this.tiltData.numAngles = readCount( 'tilt angle count' );
		this.tiltData.angles = readArray( this.tiltData.numAngles, 'tilt angles' );
		this.tiltData.mulFactors = readArray( this.tiltData.numAngles, 'tilt factors' );
		validateAngles( this.tiltData.angles, 0, 180, 'Tilt angles' );
		if ( this.tiltData.mulFactors.some( value => value < 0 ) ) throw new Error( 'IESLoader: Negative tilt factor.' );

	}

	this.count = readCount( 'lamp count' );
	this.lumens = readNumber( 'lumens per lamp' );
	this.multiplier = readNumber( 'candela multiplier' );
	this.numVerAngles = readCount( 'vertical angle count', 2 );
	this.numHorAngles = readCount( 'horizontal angle count' );
	this.gonioType = readNumber( 'photometric type' );
	if ( this.gonioType !== 1 ) throw new Error( 'IESLoader: Only Type C photometry (type 1) is supported.' );

	this.units = readNumber( 'units' );
	if ( this.units !== 1 && this.units !== 2 ) throw new Error( 'IESLoader: Invalid units.' );
	this.width = readNumber( 'width' );
	this.length = readNumber( 'length' );
	this.height = readNumber( 'height' );
	this.ballFactor = readNumber( 'ballast factor' );

	const secondFactor = readNumber( 'second factor' );
	// LM-63-1995/2002 reserve this field. In 2019 it is a file generation type,
	// not a multiplier (its decimal digits describe how the file was generated).
	this.blpFactor = version < 1995 ? secondFactor : 1;
	this.inputWatts = readNumber( 'input watts' );
	if ( this.multiplier < 0 || this.ballFactor < 0 || secondFactor < 0 || this.inputWatts < 0 ) throw new Error( 'IESLoader: Negative photometric factor or input watts.' );

	const sampleCount = this.numVerAngles * this.numHorAngles;
	if ( ! Number.isSafeInteger( sampleCount ) || sampleCount + this.numVerAngles + this.numHorAngles > tokens.length - cursor ) {

		throw new Error( 'IESLoader: Unexpected end of photometric table.' );

	}

	this.verAngles = readArray( this.numVerAngles, 'vertical angles' );
	this.horAngles = readArray( this.numHorAngles, 'horizontal angles' );
	validateAngles( this.verAngles, 0, 180, 'Vertical angles' );
	validateAngles( this.horAngles, 0, 360, 'Horizontal angles' );

	const span = this.horAngles[ this.numHorAngles - 1 ] - this.horAngles[ 0 ];
	if ( this.numHorAngles === 1 ? this.horAngles[ 0 ] !== 0 : span !== 90 && span !== 180 && ! ( span > 180 && span <= 360 ) ) {

		throw new Error( 'IESLoader: Unsupported Type C horizontal coverage.' );

	}

	this.candelaValues = [];
	let maxVal = 0;
	for ( let h = 0; h < this.numHorAngles; h ++ ) {

		const values = readArray( this.numVerAngles, 'candela values' );
		for ( let v = 0; v < values.length; v ++ ) {

			if ( values[ v ] < 0 ) throw new Error( 'IESLoader: Negative candela value.' );
			values[ v ] *= this.multiplier * this.ballFactor * this.blpFactor;
			if ( ! Number.isFinite( values[ v ] ) ) throw new Error( 'IESLoader: Non-finite scaled candela value.' );
			maxVal = Math.max( maxVal, values[ v ] );

		}

		this.candelaValues.push( values );

	}

	if ( cursor !== tokens.length ) throw new Error( 'IESLoader: Unexpected trailing photometric data.' );

	const baseAngle = this.horAngles[ 0 ];
	for ( let h = 0; h < this.numHorAngles; h ++ ) this.horAngles[ h ] -= baseAngle;

	// A full asymmetric cycle may omit the duplicate closing plane.
	const endAngle = this.horAngles[ this.numHorAngles - 1 ];
	if ( endAngle > 180 && endAngle < 360 ) {

		this.horAngles.push( 360 );
		this.candelaValues.push( this.candelaValues[ 0 ].slice() );
		this.numHorAngles = this.horAngles.length;

	}

	if ( maxVal > 0 ) {

		for ( const values of this.candelaValues ) {

			for ( let v = 0; v < values.length; v ++ ) values[ v ] /= maxVal;

		}

	}

}


export { IESLoader };
