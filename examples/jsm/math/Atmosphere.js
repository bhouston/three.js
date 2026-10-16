/**
 * A spherical, single-scattering atmosphere for the physical sky example.
 * Distances are kilometres, scattering coefficients are per kilometre, incident
 * sunlight is lux and returned sky radiance is cd/m² (linear sRGB).
 * This intentionally omits multiple scattering, clouds and ground interreflection.
 */
const atmosphere = {
	planetRadius: 6360,
	atmosphereRadius: 6460,
	observerAltitude: 0.002,
	rayleigh: [ 0.005802, 0.013558, 0.0331 ],
	mieScattering: 0.003996,
	mieExtinction: 0.004440,
	ozone: [ 0.000650, 0.001881, 0.000085 ],
	aerosol: 1,
	rayleighScaleHeight: 8,
	mieScaleHeight: 1.2,
	mieG: 0.8,
	ozoneCenter: 25,
	ozoneHalfWidth: 15,
	sunIlluminance: 128000,
	sunRadius: 0.2666
};

const radians = Math.PI / 180;
const luminance = rgb => 0.2126 * rgb[ 0 ] + 0.7152 * rgb[ 1 ] + 0.0722 * rgb[ 2 ];

function sunDirection( altitude, azimuth, northRotation = 0 ) {

	const h = altitude * radians, a = ( azimuth + northRotation ) * radians;
	return [ Math.cos( h ) * Math.sin( a ), Math.sin( h ), - Math.cos( h ) * Math.cos( a ) ];

}

function discVisibility( altitude, radius = atmosphere.sunRadius ) {

	const d = Math.max( - 1, Math.min( 1, altitude / radius ) );
	return 1 - ( Math.acos( d ) - d * Math.sqrt( 1 - d * d ) ) / Math.PI;

}

// Positive distance to the outer shell, and a test for planet occlusion.
function rayLength( radius, cosine, params ) {

	const b = radius * cosine;
	const ground = b * b - radius * radius + params.planetRadius ** 2;
	if ( b < 0 && ground > 0 ) return - 1;
	return - b + Math.sqrt( b * b - radius * radius + params.atmosphereRadius ** 2 );

}

function density( height, params ) {

	return [ Math.exp( - height / params.rayleighScaleHeight ), Math.exp( - height / params.mieScaleHeight ), Math.max( 0, 1 - Math.abs( height - params.ozoneCenter ) / params.ozoneHalfWidth ) ];

}

function extinction( opticalDepth, channel, params ) {

	return params.rayleigh[ channel ] * opticalDepth[ 0 ] + params.mieExtinction * params.aerosol * opticalDepth[ 1 ] + params.ozone[ channel ] * opticalDepth[ 2 ];

}

function transmittance( radius, cosine, params = atmosphere, steps = 64 ) {

	const length = rayLength( radius, cosine, params );
	if ( length < 0 ) return [ 0, 0, 0 ];
	const step = length / steps, opticalDepth = [ 0, 0, 0 ];
	for ( let i = 0; i < steps; i ++ ) {

		const distance = ( i + 0.5 ) * step;
		const height = Math.sqrt( radius * radius + distance * distance + 2 * radius * cosine * distance ) - params.planetRadius;
		const rho = density( height, params );
		for ( let c = 0; c < 3; c ++ ) opticalDepth[ c ] += rho[ c ] * step;

	}

	return opticalDepth.map( ( _, c ) => Math.exp( - extinction( opticalDepth, c, params ) ) );

}

function sunLightState( altitude, params = atmosphere ) {

	const transmission = transmittance( params.planetRadius + params.observerAltitude, Math.sin( Math.max( 0, altitude ) * radians ), params );
	const y = luminance( transmission );
	const visibility = discVisibility( altitude, params.sunRadius );
	return { transmission, visibility, illuminance: params.sunIlluminance * y * visibility, color: transmission.map( value => value / Math.max( y, 1e-8 ) ) };

}

/** Integrate the radiative transfer equation along a viewing ray. No sun disc. */
function skyRadiance( direction, sun, params = atmosphere ) {

	const radius = params.planetRadius + params.observerAltitude;
	const length = rayLength( radius, direction[ 1 ], params );
	const result = [ 0.00016, 0.00020, 0.00028 ]; // natural night sky floor
	if ( length < 0 ) return result;
	const cosine = direction.reduce( ( sum, value, i ) => sum + value * sun[ i ], 0 );
	const rayleighPhase = 3 / ( 16 * Math.PI ) * ( 1 + cosine * cosine );
	const g = params.mieG;
	const miePhase = 3 / ( 8 * Math.PI ) * ( 1 - g * g ) * ( 1 + cosine * cosine ) / ( ( 2 + g * g ) * ( 1 + g * g - 2 * g * cosine ) ** 1.5 );
	const step = length / 32, opticalDepth = [ 0, 0, 0 ];
	for ( let i = 0; i < 32; i ++ ) {

		const distance = ( i + 0.5 ) * step;
		const point = [ direction[ 0 ] * distance, radius + direction[ 1 ] * distance, direction[ 2 ] * distance ];
		const pointRadius = Math.hypot( ...point );
		const rho = density( pointRadius - params.planetRadius, params );
		const sunCosine = point.reduce( ( sum, value, c ) => sum + value * sun[ c ], 0 ) / pointRadius;
		const sunlight = transmittance( pointRadius, sunCosine, params );
		for ( let c = 0; c < 3; c ++ ) opticalDepth[ c ] += rho[ c ] * step * 0.5;
		for ( let c = 0; c < 3; c ++ ) {

			const scattering = params.rayleigh[ c ] * rho[ 0 ] * rayleighPhase + params.mieScattering * params.aerosol * rho[ 1 ] * miePhase;
			result[ c ] += params.sunIlluminance * sunlight[ c ] * Math.exp( - extinction( opticalDepth, c, params ) ) * scattering * step;

		}

		for ( let c = 0; c < 3; c ++ ) opticalDepth[ c ] += rho[ c ] * step * 0.5;

	}

	return result;

}

function exposure( horizontalIlluminance, manualEV = null ) {

	const ev = manualEV === null ? Math.max( - 3, Math.min( 17, Math.log2( Math.max( horizontalIlluminance, 1e-8 ) * 100 / 250 ) ) ) : manualEV;
	return { ev, scale: 1 / ( 1.2 * 2 ** ev ) };

}

export { atmosphere, luminance, sunDirection, discVisibility, transmittance, sunLightState, skyRadiance, exposure };
