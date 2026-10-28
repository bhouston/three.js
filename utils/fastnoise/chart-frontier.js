// Numerical error-ratio heatmap; SVG/PNG rasterization, not a 3D render.
import fs from 'node:fs';
import puppeteer from 'puppeteer';

const [ scoreFile = 'utils/fastnoise/results/frontier.json', generationFile = 'utils/fastnoise/results/frontier-generation.json', output = 'utils/fastnoise/results/frontier-summary' ] = process.argv.slice( 2 );
const report = JSON.parse( fs.readFileSync( scoreFile ) ), generation = JSON.parse( fs.readFileSync( generationFile ) );
const rows = Object.entries( report.comparisons ).sort( ( [ a ], [ b ] ) => a.localeCompare( b ) );
const metrics = Object.keys( rows[ 0 ][ 1 ].ratios );
const groups = [[ 'Original 10', 10 ], [ 'Identity', 4 ], [ 'Squared', 4 ], [ 'Fourth', 4 ], [ 'Cosine', 4 ], [ '5 threshold integrands', 20 ], [ '6 held-out thresholds', 24 ], [ 'Widths', 4 ]];
const L = 310, T = 90, cell = 13, rowHeight = 29, W = 1480, H = T + rowHeight * rows.length + 115;
const escape = text => String( text ).replaceAll( '&', '&amp;' ).replaceAll( '<', '&lt;' ).replaceAll( '>', '&gt;' );
const color = ratio => {

	const value = Math.max( - 1, Math.min( 1, Math.log2( ratio ) ) ), a = Math.abs( value );
	const target = value < 0 ? [ 28, 142, 97 ] : [ 202, 67, 61 ];
	return `rgb(${target.map( v => Math.round( 250 + a * ( v - 250 ) ) ).join( ',' )})`;

};

let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" font-family="Segoe UI,Arial,sans-serif"><rect width="100%" height="100%" fill="#fafafa"/><text x="20" y="29" font-size="19" font-weight="600">Research candidates vs unchanged FAST separate: all 74 metrics</text><text x="20" y="51" font-size="12">Green: lower error. Red: higher error. White: near parity. Compare each candidate with its own seed reference.</text>`;
let column = 0;
for ( const [ label, count ] of groups ) {

	svg += `<text x="${L + ( column + count / 2 ) * cell}" y="76" text-anchor="middle" font-size="11">${label}</text>`; column += count;

}

svg += `<text x="${L + metrics.length * cell + 16}" y="76" font-size="11">Strict wins / worst ratio</text>`;
let markdown = '# Frozen-suite research results\n\n![All 74 error ratios](frontier-summary.png)\n\nEach candidate is compared with the unchanged FAST separate volume for the same seed.\nCounts are deterministic point comparisons, not confidence statements. Tiny improvements\ncan be visually white; exact ratios are retained in frontier.json.\n\n| Candidate | Better / 74 | Worse | Worst ratio | Total construction seconds |\n| --- | ---: | ---: | ---: | ---: |\n';
for ( const [ index, [ name, row ]] of rows.entries() ) {

	const y = T + index * rowHeight;
	svg += `<text x="20" y="${y + 16}" font-size="12">${escape( name )}</text>`;
	metrics.forEach( ( metric, i ) => {

		const ratio = row.ratios[ metric ];
		svg += `<rect x="${L + i * cell}" y="${y}" width="12" height="22" fill="${color( ratio )}"><title>${escape( name )}: ${escape( metric )} = ${ratio}</title></rect>`;

	} );
	svg += `<text x="${L + metrics.length * cell + 16}" y="${y + 16}" font-size="12">${row.better}/74 · ${row.worst_ratio.toFixed( 5 )}×</text>`;
	const run = generation.runs.find( r => r.name === name );
	markdown += `| ${name} | ${row.better} | ${row.worse} | ${row.worst_ratio.toFixed( 6 )} | ${run?.totalSecondsPerVolume.toFixed( 3 ) ?? '—'} |\n`;

}

const bottom = T + rowHeight * rows.length + 15;
for ( const [ i, ratio ] of [ 0.5, 1, 2 ].entries() ) svg += `<rect x="${20 + i * 140}" y="${bottom}" width="22" height="14" fill="${color( ratio )}" stroke="#bbb"/><text x="${49 + i * 140}" y="${bottom + 12}" font-size="12">${ratio}× error</text>`;
svg += `<text x="20" y="${bottom + 40}" font-size="12">A refinement includes its input FAST construction cost. Extra-budget controls are shown; this is not a same-cost comparison.</text><text x="20" y="${bottom + 60}" font-size="12">Spatial/temporal feature guards do not guarantee all individual metrics. No rendered-scene or universal state-of-the-art claim.</text></svg>`;
markdown += '\nSee [FRONTIER.md](../FRONTIER.md) for protocol, controls, limitations and interpretation.\n[Exact scores and ratios](frontier.json) · [construction times and volume hashes](frontier-generation.json) · [SVG](frontier-summary.svg).\n';
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
