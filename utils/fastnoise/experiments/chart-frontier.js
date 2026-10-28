// Numerical SVG/PNG plot; no WebGL/WebGPU workload or rendering benchmark.
import fs from 'node:fs';
import puppeteer from 'puppeteer';
const [ input, output ] = process.argv.slice( 2 );
const data = JSON.parse( fs.readFileSync( input, 'utf8' ) ).full;
const series = [[ 'identity1-1000-s1', 'Identity-weighted, seed 1', '#dd7633' ], [ 'latin-anneal001-3000-s1', 'Annealed stratified, seed 1', '#16856a' ], [ 'latin-anneal001-3000-s2', 'Annealed stratified, seed 2', '#7561b1' ]];
const W = 1120, H = 850, L = 80, R = 40, width = W - L - R;
let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" font-family="Segoe UI,Arial,sans-serif"><rect width="100%" height="100%" fill="#fcfcfb"/><text x="${L}" y="30" font-size="22" font-weight="600">New generators: broad improvements, remaining regressions</text><text x="${L}" y="54" font-size="13" fill="#555">Error divided by same-seed FAST separate, 10,000 iterations. Below 1 is better; above 1 is worse.</text>`;
for ( const [ i, [ , label, color ]] of series.entries() ) svg += `<line x1="${L + i * 330}" x2="${L + i * 330 + 24}" y1="82" y2="82" stroke="${color}" stroke-width="3"/><text x="${L + i * 330 + 32}" y="86" font-size="13">${label}</text>`;
const keys = [ 'tmpLow', 'spLow', 'bl1', 'bl4', 'pp32', 'bl32', 'pp256', 'bl256', 'ema', 'ema+bl' ];
for ( const panel of [ 0, 1 ] ) {

	const top = panel ? 480 : 150, height = 235;
	const x = u => L + u * width;
	const y = v => top + height * ( 1 - v / 1.3 );
	svg += `<text x="${L}" y="${top - 25}" font-size="17" font-weight="600">${panel ? '308 integration-error diagnostics, sorted from best to worst ratio' : 'All ten original raw-noise metrics improve'}</text>`;
	for ( const v of [ 0, .25, .5, .75, 1, 1.25 ] ) svg += `<line x1="${L}" x2="${W - R}" y1="${y( v )}" y2="${y( v )}" stroke="${v === 1 ? '#333' : '#e1e1dc'}" ${v === 1 ? 'stroke-dasharray="6 4"' : ''}/><text x="${L - 12}" y="${y( v ) + 4}" font-size="12" text-anchor="end" fill="#555">${v.toFixed( 2 )}</text>`;
	for ( const [ name, , color ] of series ) {

		const ratios = data[ name ].ratios;
		const values = panel ? Object.entries( ratios ).filter( ( [ key ] ) => ! key.startsWith( 'legacy/' ) ).map( ( [ , v ] ) => v ).sort( ( a, b ) => a - b ) : keys.map( k => ratios[ 'legacy/' + k ] );
		if ( values.some( v => ! Number.isFinite( v ) || v < 0 || v > 1.3 ) ) throw new Error( 'Update plot range for these data.' );
		svg += `<polyline points="${values.map( ( v, i ) => `${x( i / ( values.length - 1 ) )},${y( v )}` ).join( ' ' )}" fill="none" stroke="${color}" stroke-width="2.5"/>`;
		if ( ! panel ) for ( const [ i, v ] of values.entries() ) svg += `<circle cx="${x( i / 9 )}" cy="${y( v )}" r="3.5" fill="${color}"/>`;

	}

	if ( panel ) {

		for ( const percent of [ 0, 25, 50, 75, 100 ] ) svg += `<text x="${x( percent / 100 )}" y="${top + height + 22}" font-size="12" text-anchor="middle" fill="#555">${percent}%</text>`;

	} else {

		for ( const [ i, key ] of keys.entries() ) svg += `<text x="${x( i / 9 )}" y="${top + height + 22}" font-size="12" text-anchor="middle" fill="#555">${key}</text>`;

	}

}

const a = data[ series[ 1 ][ 0 ] ], b = data[ series[ 2 ][ 0 ] ];
const counts = a.wins === b.wins && a.losses === b.losses && a.ties === b.ties ? `win ${a.wins}, lose ${a.losses}, tie ${a.ties} of 318 on each seed` : `seed 1: ${a.wins}/${a.losses}/${a.ties}; seed 2: ${b.wins}/${b.losses}/${b.ties} (win/lose/tie)`;
svg += `<text x="${L}" y="785" font-size="13" fill="#555">No all-metrics winner: annealed candidates ${counts}.</text><text x="${L}" y="808" font-size="12" fill="#555">Finite correlated development tests; not statistical significance, unseen rendering evidence, or general state-of-the-art proof.</text></svg>`;
fs.writeFileSync( output.replace( /\.png$/, '.svg' ), svg );
const browser = await puppeteer.launch( { executablePath: process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, args: [ '--disable-software-rasterizer' ] } );
try {

	const page = await browser.newPage();
	await page.setViewport( { width: W, height: H, deviceScaleFactor: 2 } );
	await page.setContent( `<body style="margin:0">${svg}</body>` );
	await page.screenshot( { path: output } );

} finally {

	await browser.close();

}
