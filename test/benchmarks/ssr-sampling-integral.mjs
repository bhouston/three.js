/**
 * Numerical audit of the SSR sampling estimator, independent of rendering.
 * Run: node test/benchmarks/ssr-sampling-integral.mjs [sampleCount]
 *
 * Reproduces SpecularHelpers' bounded isotropic GGX sampler and separable Smith
 * weight, SSRNode's mirror-bias warp, and its one deterministic retry. White
 * incident radiance, dielectric F0=0.04. The reference is mirrorBias=0 with no
 * retry. These are reflected energies, NOT quantities expected to equal one:
 * Fresnel, masking and single-scattering loss are part of the evaluated BRDF.
 * The correlated column changes the BRDF to the Smith model used by the
 * material's DFG LUT; it is included to quantify that separate model mismatch.
 * This verifies estimator math, not GPU precision or screen-space rendering.
 */

import process from 'node:process';
import console from 'node:console';

const count = Number( process.argv[ 2 ] ?? 1000000 );
if ( ! Number.isSafeInteger( count ) || count <= 0 ) throw new Error( 'sampleCount must be a positive integer' );

function randomGenerator() {

	let state = 20261024;
	return () => {

		state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0;
		return ( state + 0.5 ) / 4294967296;

	};

}

function integrate( noV, roughness ) {

	const random = randomGenerator();
	const alpha = Math.max( roughness * roughness, 0.001 );
	const alpha2 = alpha * alpha;
	const vx = Math.sqrt( 1 - noV * noV );
	const tV = Math.sqrt( alpha2 * vx * vx + noV * noV );
	const viewX = alpha * vx / tV;
	const viewZ = noV / tV;
	const s2 = ( 1 + vx ) ** 2;
	const k = ( 1 - alpha2 ) * s2 / ( s2 + alpha2 * noV * noV );
	const smithV = 2 * noV / ( noV + tV );
	const sample = ( x, y ) => {

		const phi = 2 * Math.PI * x;
		const z = 1 - y * ( 1 + k * viewZ );
		const sinTheta = Math.sqrt( Math.max( 0, 1 - z * z ) );
		let hx = alpha * ( sinTheta * Math.cos( phi ) + viewX );
		let hy = alpha * sinTheta * Math.sin( phi );
		let hz = Math.max( 0, z + viewZ );
		const length = Math.hypot( hx, hy, hz );
		hx /= length; hy /= length; hz /= length;
		const voH = Math.max( 0, vx * hx + noV * hz );
		const rawNoL = 2 * voH * hz - noV;
		const noL = Math.max( 0, rawNoL );
		const tL = Math.sqrt( alpha2 + ( 1 - alpha2 ) * noL * noL );
		const smithL = 2 * noL / ( noL + tL );
		const fresnel = 0.04 + 0.96 * ( 1 - voH ) ** 5;
		return {
			belowHorizon: rawNoL < 0,
			separable: fresnel * smithL * smithV * ( k * noV + tV ) / Math.max( 2 * noV, 1e-4 ),
			correlated: fresnel * noL * ( k * noV + tV ) / ( noL * tV + noV * tL )
		};

	};
	let baseline = 0, mirrorBias = 0, retry = 0, correlated = 0, rejected = 0;
	for ( let i = 0; i < count; i ++ ) {

		const x = random(), y = random();
		random(); // Xi.z (march jitter) is unused by the BRDF sampler.
		const w = random();
		const original = sample( x, y );
		baseline += original.separable;
		correlated += original.correlated;
		mirrorBias += sample( x, y * ( 1 - 0.5 * Math.sqrt( w ) ) ).separable;
		retry += original.belowHorizon ? sample( ( 8 * x ) % 1, ( 8 * y ) % 1 ).separable : original.separable;
		if ( original.belowHorizon ) rejected ++;

	}
	const round = x => Number( x.toFixed( 6 ) );
	return {
		noV, roughness,
		baseline: round( baseline / count ),
		mirrorBias05ChangePercent: round( 100 * ( mirrorBias / baseline - 1 ) ),
		retryChangePercent: round( 100 * ( retry / baseline - 1 ) ),
		belowHorizonPercent: round( 100 * rejected / count ),
		correlatedEnergy: round( correlated / count ),
		separableVsCorrelatedPercent: round( 100 * ( baseline / correlated - 1 ) )
	};

}

console.log( `SSR estimator audit: ${ count } samples/configuration; LCG seed 20261024; F0=.04; white incident radiance.` );
console.log( 'Changes are relative to mirrorBias=0/no retry; correlatedEnergy evaluates a different Smith geometry model.' );
const results = [];
for ( const noV of [ 1, 0.5, 0.1 ] ) for ( const roughness of [ 0.3, 0.6, 0.8, 1 ] ) results.push( integrate( noV, roughness ) );
console.table( results );
