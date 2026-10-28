// Numerical error-ratio heatmap; SVG/PNG rasterization, not a 3D render.
import fs from 'node:fs';
import path from 'node:path';
import puppeteer from 'puppeteer';

const [ scoreFile = 'utils/fastnoise/results/frontier.json', generationFile = 'utils/fastnoise/results/frontier-generation.json', output = 'utils/fastnoise/results/frontier-summary', selection = 'all' ] = process.argv.slice( 2 );
const report = JSON.parse( fs.readFileSync( scoreFile ) ), generation = JSON.parse( fs.readFileSync( generationFile ) );
if ( ! [ 'all', 'cross' ].includes( selection ) ) throw new Error( 'Selection must be all or cross.' );
const crossMethods = [ 'pr116-annealed', 'pr116-greedy', 'pr116-autocorrelation', 'light-seam-03', 'continuation-time-control', 'pareto-polish' ];
const selectedNames = crossMethods.flatMap( name => [ `${name}-seed1`, `${name}-seed2` ] );
if ( selection === 'cross' ) {

	for ( const name of selectedNames ) if ( ! report.comparisons[ name ] ) throw new Error( `Missing frozen-suite comparison: ${name}` );

}

const rows = ( selection === 'cross' ? selectedNames.map( name => [ name, report.comparisons[ name ] ] ) : Object.entries( report.comparisons ).sort( ( [ a ], [ b ] ) => a.localeCompare( b ) ) );
const metrics = Object.keys( rows[ 0 ][ 1 ].ratios );
if ( metrics.length !== 74 ) throw new Error( 'Expected the unchanged 74-metric frontier suite.' );
for ( const [ name, row ] of rows ) {

	if ( Object.keys( row.ratios ).length !== 74 || metrics.some( metric => ! Number.isFinite( row.ratios[ metric ] ) || row.ratios[ metric ] < 0 ) ) throw new Error( `Invalid metric set: ${name}` );

}

const label = name => name.startsWith( 'pr116-' ) ? `#116 ${name.slice( 6 )}` : `#117 ${name}`;
const groups = [[ 'Original 10', 10 ], [ 'Identity', 4 ], [ 'Squared', 4 ], [ 'Fourth', 4 ], [ 'Cosine', 4 ], [ '5 threshold integrands', 20 ], [ '6 held-out thresholds', 24 ], [ 'Widths', 4 ]];
const L = 310, T = 90, cell = 13, rowHeight = 29, W = 1480, H = T + rowHeight * rows.length + 115;
const escape = text => String( text ).replaceAll( '&', '&amp;' ).replaceAll( '<', '&lt;' ).replaceAll( '>', '&gt;' );
const color = ratio => {

	const value = Math.max( - 1, Math.min( 1, Math.log2( ratio ) ) ), a = Math.abs( value );
	const target = value < 0 ? [ 28, 142, 97 ] : [ 202, 67, 61 ];
	return `rgb(${target.map( v => Math.round( 250 + a * ( v - 250 ) ) ).join( ',' )})`;

};

let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" font-family="Segoe UI,Arial,sans-serif"><rect width="100%" height="100%" fill="#fafafa"/><text x="20" y="29" font-size="19" font-weight="600">PR #116 and #117 research vs unchanged FAST separate: all 74 metrics</text><text x="20" y="51" font-size="12">Green: lower error. Red: higher error. White: near parity. Compare each candidate with its own seed reference.</text>`;
let column = 0;
for ( const [ label, count ] of groups ) {

	svg += `<text x="${L + ( column + count / 2 ) * cell}" y="76" text-anchor="middle" font-size="11">${label}</text>`; column += count;

}

svg += `<text x="${L + metrics.length * cell + 16}" y="76" font-size="11">Strict wins / worst ratio</text>`;
let markdown = `# PR #116 and #117 frozen-suite research results\n\n![All 74 error ratios](${path.basename( output )}.png)\n\nEach candidate is compared with the unchanged FAST separate volume for the same seed.\nCounts are deterministic point comparisons, not confidence statements. Tiny improvements\ncan be visually white; exact ratios are retained in frontier.json.\n\n| Candidate | Better / 74 | Worse | Worst ratio | Total construction seconds |\n| --- | ---: | ---: | ---: | ---: |\n`;
for ( const [ index, [ name, row ]] of rows.entries() ) {

	const y = T + index * rowHeight;
	svg += `<text x="20" y="${y + 16}" font-size="12">${escape( label( name ) )}</text>`;
	metrics.forEach( ( metric, i ) => {

		const ratio = row.ratios[ metric ];
		svg += `<rect x="${L + i * cell}" y="${y}" width="12" height="22" fill="${color( ratio )}"><title>${escape( name )}: ${escape( metric )} = ${ratio}</title></rect>`;

	} );
	svg += `<text x="${L + metrics.length * cell + 16}" y="${y + 16}" font-size="12">${row.better}/74 · ${row.worst_ratio.toFixed( 5 )}×</text>`;
	const run = generation.runs.find( r => r.name === name );
	markdown += `| ${label( name )} | ${row.better} | ${row.worse} | ${row.worst_ratio.toFixed( 6 )} | ${run?.totalSecondsPerVolume?.toFixed( 3 ) ?? '—'} |\n`;

}

const bottom = T + rowHeight * rows.length + 15;
for ( const [ i, ratio ] of [ 0.5, 1, 2 ].entries() ) svg += `<rect x="${20 + i * 140}" y="${bottom}" width="22" height="14" fill="${color( ratio )}" stroke="#bbb"/><text x="${49 + i * 140}" y="${bottom + 12}" font-size="12">${ratio}× error</text>`;
svg += `<text x="20" y="${bottom + 40}" font-size="12">A refinement includes its input FAST construction cost. Extra-budget controls are shown; this is not a same-cost comparison.</text><text x="20" y="${bottom + 60}" font-size="12">Spatial/temporal feature guards do not guarantee all individual metrics. No rendered-scene or universal state-of-the-art claim.</text></svg>`;
if ( selection === 'cross' ) {

	markdown += '\n## Direct cross-PR comparisons\n\nEach row pools the same 74 metrics for seeds 1 and 2 (148 point comparisons). Ratios divide #116 error by #117 error for the same metric and seed; below 1 favors #116. These are correlated measurements on two seeds, not independent trials or evidence of universal superiority. The worst ratio exposes regressions hidden by an aggregate. Construction costs above are unequal.\n\n| #116 candidate | #117 comparison | #116 lower / 148 | Equal | #116 higher | Geometric mean ratio | Worst ratio |\n| --- | --- | ---: | ---: | ---: | ---: | ---: |\n';
	for ( const candidate of crossMethods.slice( 0, 3 ) ) {

		for ( const reference of crossMethods.slice( 3 ) ) {

			const ratios = [ 1, 2 ].flatMap( seed => metrics.map( metric => report.comparisons[ `${candidate}-seed${seed}` ].ratios[ metric ] / report.comparisons[ `${reference}-seed${seed}` ].ratios[ metric ] ) );
			const better = ratios.filter( ratio => ratio < 1 ).length, worse = ratios.filter( ratio => ratio > 1 ).length;
			const geometricMean = Math.exp( ratios.reduce( ( sum, ratio ) => sum + Math.log( ratio ), 0 ) / ratios.length );
			markdown += `| ${candidate.slice( 6 )} | ${reference} | ${better} | ${ratios.length - better - worse} | ${worse} | ${geometricMean.toFixed( 6 )} | ${Math.max( ...ratios ).toFixed( 6 )} |\n`;

		}

	}

	markdown += '\n### Direct comparison with the light-seam-03 candidate, by seed\n\nStrict counts use exact floating-point comparisons with no tolerance; equal means an exactly equal measured error.\n\n| #116 candidate | Seed | Lower / 74 | Equal | Higher | Worst #116 / light-seam-03 ratio |\n| --- | ---: | ---: | ---: | ---: | ---: |\n';
	for ( const candidate of crossMethods.slice( 0, 3 ) ) {

		for ( const seed of [ 1, 2 ] ) {

			const ratios = metrics.map( metric => report.comparisons[ `${candidate}-seed${seed}` ].ratios[ metric ] / report.comparisons[ `light-seam-03-seed${seed}` ].ratios[ metric ] );
			const better = ratios.filter( ratio => ratio < 1 ).length, worse = ratios.filter( ratio => ratio > 1 ).length;
			markdown += `| ${candidate.slice( 6 )} | ${seed} | ${better} | ${ratios.length - better - worse} | ${worse} | ${Math.max( ...ratios ).toFixed( 6 )} |\n`;

		}

	}

}

markdown += `\nSee [FRONTIER.md](../FRONTIER.md) for protocol, controls, limitations and interpretation.\n[Exact scores and ratios](frontier.json) · [construction times and volume hashes](frontier-generation.json) · [SVG](${path.basename( output )}.svg).\n`;
fs.writeFileSync( output + '.svg', svg ); fs.writeFileSync( output + '.md', markdown );
const browser = await puppeteer.launch( { headless: true, executablePath: process.env.CHROME_PATH || ( process.platform === 'win32' ? 'C:/Program Files/Google/Chrome/Application/chrome.exe' : puppeteer.executablePath() ), args: [ '--disable-software-rasterizer' ] } );
try {

	const page = await browser.newPage();
	await page.setViewport( { width: W, height: H, deviceScaleFactor: 2 } );
	await page.setContent( `<body style="margin:0">${svg}</body>` );
	await page.screenshot( { path: output + '.png' } );

} finally {

	await browser.close();

}

