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

			// Zero samples outside the measured vertical range.

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

			onLoad( this.parse( text ) );

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

	}

	const tiltIndex = lines.findIndex( line => /^\s*TILT\s*=/i.test( line ) );
	const tiltMode = lines[ tiltIndex ].split( '=' )[ 1 ].trim().toUpperCase();

	// Numeric tables can wrap anywhere, including across logical array boundaries.
	const tokens = lines.slice( tiltIndex + 1 ).join( ' ' ).match( /[^\s,]+/g ) || [];
	let cursor = 0;

	function readNumber() {

		return Number( tokens[ cursor ++ ] );

	}

	function readArray( count ) {

		const values = tokens.slice( cursor, cursor + count ).map( Number );
		cursor += count;
		return values;

	}

	this.tiltData = { angles: [], mulFactors: [] };
	if ( tiltMode === 'INCLUDE' ) {

		this.tiltData.lampToLumGeometry = readNumber();
		this.tiltData.numAngles = readNumber();
		this.tiltData.angles = readArray( this.tiltData.numAngles );
		this.tiltData.mulFactors = readArray( this.tiltData.numAngles );

	}

	this.count = readNumber();
	this.lumens = readNumber();
	this.multiplier = readNumber();
	this.numVerAngles = readNumber();
	this.numHorAngles = readNumber();
	this.gonioType = readNumber();
	this.units = readNumber();
	this.width = readNumber();
	this.length = readNumber();
	this.height = readNumber();
	this.ballFactor = readNumber();

	const secondFactor = readNumber();

	// Only legacy formats use the second field as a multiplier.

	this.blpFactor = version < 1995 ? secondFactor : 1;
	this.inputWatts = readNumber();

	this.verAngles = readArray( this.numVerAngles );
	this.horAngles = readArray( this.numHorAngles );

	this.candelaValues = [];
	let maxVal = 0;
	for ( let h = 0; h < this.numHorAngles; h ++ ) {

		const values = readArray( this.numVerAngles );
		for ( let v = 0; v < values.length; v ++ ) {

			values[ v ] *= this.multiplier * this.ballFactor * this.blpFactor;
			maxVal = Math.max( maxVal, values[ v ] );

		}

		this.candelaValues.push( values );

	}

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
