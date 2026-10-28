// Numerical summary; not a rendered-scene benchmark.
import fs from 'node:fs';
import { gzipSync } from 'node:zlib';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer';
import { wgslSamplers } from './procedural.js';

const root = new URL( './', import.meta.url );
const report = JSON.parse( fs.readFileSync( new URL( 'results.json', root ) ) );
const gpu = JSON.parse( fs.readFileSync( new URL( 'gpu-results.json', root ) ) );
const ids = [ 'white', 'animated-ign', 'r2-weyl', 'hashed-digital', 'hierarchical-digital', 'houston-stratified-noise', 'phase-16-p8', 'phase-32-p0', 'phase-32-p2', 'phase-32-p8', 'phase-64-p8', 'phase-128-p8', 'FAST-scalar' ];
const thresholds = [ .1, .25, .5, .75, .9, .07, .37, .83, .94 ].map( p => 'threshold-' + p );
const smooth = [ 'squared', 'fourth', 'cosine', 'sine7' ];
const rms = values => Math.sqrt( values.reduce( ( sum, v ) => sum + v * v, 0 ) / values.length );
const aggregate = ( id, functions, mode, n, image = false ) => rms( [ 1, 2 ].flatMap( seed => functions.map( fn => report.methods[ `${id}-seed${seed}` ][ image ? 'image_metrics' : 'metrics' ][ `${fn}/${mode}/${n}` ] ) ) );
const columns = [
	[ 'Identity 32', id => aggregate( id, [ 'identity' ], 'blur', 32 ) ],
	[ 'Threshold 1', id => aggregate( id, thresholds, 'blur', 1 ) ],
	[ 'Threshold 32', id => aggregate( id, thresholds, 'blur', 32 ) ],
	[ 'Threshold 128', id => aggregate( id, thresholds, 'blur', 128 ) ],
	[ 'Threshold EMA', id => aggregate( id, thresholds, 'blur', 'ema' ) ],
	[ 'Smooth 32', id => aggregate( id, smooth, 'blur', 32 ) ],
	[ 'Ramp/sphere 32', id => aggregate( id, [ 'ramp', 'sphere' ], 'blur', 32, true ) ]
];
const summary = {};
let markdown = '# Zero-download runtime comparison\n\nAll entries are RMSE ratios to the shipped FAST scalar B reference; lower is better. Aggregates retain bias and pool both seeds. Thresholds: nine values, including .07/.37/.83/.94; smooth functions: u²/u⁴/cos(2πu)/sin(14πu). Ramp/sphere are spatially varying analytic threshold images. All-phase windows, actual continuous frame indices, wrapped sigma1 spatial filtering. See [RUNTIME.md](RUNTIME.md) for limitations.\n\n| Method | ' + columns.map( c => c[ 0 ] ).join( ' | ' ) + ' | Worst / 160 |\n| --- | ' + columns.map( () => '---:' ).join( ' | ' ) + ' | ---: |\n';
const W = 1420, H = 150 + ids.length * 34;
let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" font-family="Segoe UI,Arial"><rect width="100%" height="100%" fill="#fafafa"/><text x="20" y="28" font-size="20">Zero-download scalar runtime sampling: error relative to FAST</text><text x="20" y="51" font-size="12">Lower is better. Green &lt;1; red &gt;1. Aggregated diagnostics can hide regressions; last column keeps the worst of 160 tests, both seeds.</text>`;
const color = value => {

	const t = Math.min( 1, Math.abs( Math.log2( value ) ) / 2 ), target = value < 1 ? [ 50, 155, 113 ] : [ 207, 90, 76 ];
	return `rgb(${target.map( v => Math.round( 250 + t * ( v - 250 ) ) ).join( ',' )})`;

};

for ( const [ i, [ label ]] of columns.entries() ) svg += `<text x="${290 + i * 132}" y="85" font-size="12">${label}</text>`;
svg += '<text x="1220" y="85" font-size="12">Worst / 160</text>';
for ( const [ rowIndex, id ] of ids.entries() ) {

	const values = columns.map( ( [ , fn ] ) => fn( id ) / fn( 'FAST-scalar' ) );
	const ratios = [ 1, 2 ].flatMap( seed => [ 'metrics', 'image_metrics' ].flatMap( group => Object.entries( report.methods[ `${id}-seed${seed}` ][ group ] ).map( ( [ key, value ] ) => value / report.methods[ `FAST-scalar-seed${seed}` ][ group ][ key ] ) ) );
	const worst = Math.max( ...ratios );
	summary[ id ] = { ratios: Object.fromEntries( columns.map( ( [ label ], i ) => [ label, values[ i ] ] ) ), worstRatio: worst,
		threshold32RMSE: aggregate( id, thresholds, 'blur', 32 ), thresholdEMARMSE: aggregate( id, thresholds, 'blur', 'ema' ),
		gpuSamplesPerSecond: gpu.results.find( x => x.id === id )?.samplesPerSecond ?? null,
		standaloneWGSLGzipBytes: wgslSamplers[ id ] ? gzipSync( wgslSamplers[ id ] ).length : null };
	markdown += `| ${id} | ${values.map( x => x.toFixed( 3 ) ).join( ' | ' )} | ${worst.toFixed( 3 )} |\n`;
	const y = 100 + rowIndex * 34;
	svg += `<text x="20" y="${y + 21}" font-size="13">${id}</text>`;
	for ( const [ i, value ] of [ ...values, worst ].entries() ) svg += `<rect x="${282 + i * 132}" y="${y}" width="125" height="29" fill="${color( value )}"/><text x="${344 + i * 132}" y="${y + 20}" text-anchor="middle" font-size="13">${value.toFixed( 3 )}×</text>`;

}

svg += `<text x="20" y="${H - 12}" font-size="12">Finite scalar development tests; no universal/scene/motion/vector claim. Persistent tile memory and construction cost are separate.</text></svg>`;
fs.writeFileSync( new URL( 'summary.json', root ), JSON.stringify( summary, null, 2 ) + '\n' );
fs.writeFileSync( new URL( 'SUMMARY.md', root ), markdown );
fs.writeFileSync( new URL( 'summary.svg', root ), svg );
const browser = await puppeteer.launch( { executablePath: process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, args: [ '--disable-software-rasterizer' ] } );
try {

	const page = await browser.newPage(); await page.setViewport( { width: W, height: H, deviceScaleFactor: 2 } );
	await page.setContent( `<body style="margin:0">${svg}</body>` ); await page.screenshot( { path: fileURLToPath( new URL( 'summary.png', root ) ) } );

} finally {

	await browser.close();

}
