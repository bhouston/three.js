// Extend convergence.py's uniform-field experiment to all three alternatives.
// No rendering or generation timings: this measures accumulated sampling error.
import fs from 'node:fs';
import { PNG } from 'pngjs';
import { generate as filterRank } from './methods/filter-rank.js';
import { generate as hierarchy } from './methods/hierarchy.js';
import { generate as localEnergy } from './methods/local-energy.js';

const size = 128, depth = 32, plane = size * size;
const frames = [ 1, 2, 4, 8, 16, 32, 128, 256, 512, 1024 ];
const output = process.argv[ 2 ] || 'utils/fastnoise/results/convergence-alternatives.json';
function atlas( name ) {

	const image = PNG.sync.read( fs.readFileSync( new URL( `../../examples/textures/noises/fast/${name}.png`, import.meta.url ) ) );
	if ( image.width !== 256 || image.height !== 4096 ) throw new Error( 'Unexpected FAST atlas dimensions' );
	return Float64Array.from( { length: plane * depth }, ( _, i ) => {

		const z = Math.floor( i / plane ), p = i % plane;
		return ( image.data[ ( ( z * size + Math.floor( p / size ) ) * image.width + p % size ) * 4 ] + 0.5 ) / 256;

	} );

}

const quantize = v => Float64Array.from( v, x => ( Math.floor( Math.min( 0.999999, Math.max( 0, x ) ) * 256 ) + 0.5 ) / 256 );
const spatial = atlas( 'gauss1-ema01-spatial' ), temporal = atlas( 'gauss1-ema01-spatio-temporal' );
const fetch = ( volume, n, p ) => {

	const cycle = Math.floor( n / depth ), y = Math.floor( p / size ), x = p % size;
	return volume[ n % depth * plane + ( y + 71 * cycle ) % size * size + ( x + 37 * cycle ) % size ];

};

let state = 1;
const white = () => {

	state = ( state + 0x6D2B79F5 ) >>> 0;
	let v = state;
	v = Math.imul( v ^ v >>> 15, v | 1 ); v ^= v + Math.imul( v ^ v >>> 7, v | 61 );
	return ( ( v ^ v >>> 14 ) >>> 0 ) / 4294967296;

};

const alternatives = [[ 'Filter / Rank', filterRank ], [ 'Hierarchy', hierarchy ], [ 'Cached Threshold Energy', localEnergy ]];
const series = [
	[ 'White', white ],
	[ 'FAST Spatial', ( n, p ) => fetch( spatial, n, p ) ],
	[ 'FAST Spatio-Temporal', ( n, p ) => fetch( temporal, n, p ) ],
	[ 'Golden Ratio', ( n, p ) => ( spatial[ p ] + ( n * 2654435769 % 4294967296 ) / 4294967296 ) % 1 ],
];
for ( const [ name, generate ] of alternatives ) {

	const volume = quantize( generate( { seed: 1 } ) );
	series.push( [ name, ( n, p ) => fetch( volume, n, p ) ] );

}

const kernel = Array.from( { length: 7 }, ( _, i ) => Math.exp( - 0.5 * ( i - 3 ) ** 2 ) );
const sum = kernel.reduce( ( a, b ) => a + b );
const weights = kernel.map( v => v / sum );
function blur( src ) {

	for ( const axis of [ 0, 1 ] ) {

		const dst = new Float64Array( plane );
		for ( let y = 0; y < size; y ++ ) for ( let x = 0; x < size; x ++ ) for ( let k = - 3; k <= 3; k ++ ) dst[ y * size + x ] += weights[ k + 3 ] * src[ ( y + ( axis ? k : 0 ) + size ) % size * size + ( x + ( axis ? 0 : k ) + size ) % size ];
		src = dst;

	}

	return src;

}

const std = values => {

	const mean = values.reduce( ( a, b ) => a + b ) / values.length;
	return Math.sqrt( values.reduce( ( a, b ) => a + ( b - mean ) ** 2, 0 ) / values.length );

};

const result = {
	frames,
	metadata: {
		metric: 'Standard deviation of running average; blur is wrapped sigma=1 Gaussian, radius 3',
		size, depth, seed: 1,
		baseline: 'Original shipped FAST atlases, red channel, first 128 columns; not regenerated FAST reference',
		animation: '32-frame volumes shifted by (71,37) pixels per cycle, matching convergence.py',
		goldenRatio: 'First FAST Spatial slice plus uint32 golden-ratio Weyl offset, matching convergence.py',
		white: 'Independent uniform samples, seeded Mulberry32 (differs from convergence.py NumPy PCG64 realization)',
		quantization: 'FAST bytes and generated alternatives use 8-bit midpoint values; White and Golden Ratio remain continuous, matching original experiment',
		caution: 'Uniform-field error does not establish thresholded stochastic-rendering quality; see ALTERNATIVES.md',
	},
	raw: {}, blur: {},
};
for ( const [ name, sample ] of series ) {

	const acc = new Float64Array( plane ); result.raw[ name ] = []; result.blur[ name ] = [];
	for ( let n = 1; n <= 1024; n ++ ) {

		for ( let p = 0; p < plane; p ++ ) acc[ p ] += sample( n - 1, p );
		if ( frames.includes( n ) ) {

			const avg = acc.map( v => v / n );
			result.raw[ name ].push( std( avg ) ); result.blur[ name ].push( std( blur( avg ) ) );

		}

	}

	console.log( name, result.blur[ name ] );

}

fs.writeFileSync( output, JSON.stringify( result, null, 2 ) + '\n' );
