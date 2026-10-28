// Numerical SVG/PNG plot; no WebGL/WebGPU workload or rendering benchmark.
import fs from 'node:fs';
import puppeteer from 'puppeteer';
const [ input, output ] = process.argv.slice( 2 );
const data = JSON.parse( fs.readFileSync( input, 'utf8' ) ).full;
const series = [
	[ 'identity1-1000-s1', 'Identity-weighted, seed 1', '#dd7633' ],
	[ 'latin-anneal001-3000-s1', 'Annealed stratified, seed 1', '#16856a' ],
	[ 'latin-anneal001-3000-s2', 'Annealed stratified, seed 2', '#7561b1' ],
	[ 'pr117-light-seam-03-s1', '#117 Light playback, seed 1', '#1672bc' ],
	[ 'pr117-seam-polish-s1', '#117 Playback 25%, seed 1', '#c13c6e' ],
	[ 'pr117-pareto-polish-s1', '#117 Pareto polish, seed 1', '#9c8b12' ],
	[ 'pr117-continuation-deep-s1', '#117 Extra FAST: same iterations, seed 1', '#555555', true ],
	[ 'pr117-continuation-time-control-s1', '#117 Extra FAST: similar time, seed 1', '#a2a2a2', true ]
];
const W = 1120, H = 1350, L = 80, R = 40, width = W - L - R;
let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" font-family="Segoe UI,Arial,sans-serif"><rect width="100%" height="100%" fill="#fcfcfb"/><text x="${L}" y="30" font-size="22" font-weight="600">Cross-PR noise comparison: improvements and remaining regressions</text><text x="${L}" y="54" font-size="13" fill="#555">Error divided by same-seed FAST separate, 10,000 iterations. Below 1 is better; above 1 is worse.</text>`;
for ( const [ i, [ , label, color, control ]] of series.entries() ) {

	const left = L + ( i % 2 ) * 500, top = 82 + Math.floor( i / 2 ) * 24;
	svg += `<line x1="${left}" x2="${left + 24}" y1="${top}" y2="${top}" stroke="${color}" stroke-width="3" ${control ? 'stroke-dasharray="5 3"' : ''}/><text x="${left + 32}" y="${top + 4}" font-size="13">${label}</text>`;

}

const keys = [ 'tmpLow', 'spLow', 'bl1', 'bl4', 'pp32', 'bl32', 'pp256', 'bl256', 'ema', 'ema+bl' ];
for ( const panel of [ 0, 1 ] ) {

	const top = panel ? 585 : 230, height = 235;
	const x = u => L + u * width;
	const y = v => top + height * ( 1 - v / 1.3 );
	svg += `<text x="${L}" y="${top - 25}" font-size="17" font-weight="600">${panel ? '308 integration-error diagnostics, sorted from best to worst ratio' : 'Ten original raw-noise metrics (same order for every curve)'}</text>`;
	for ( const v of [ 0, .25, .5, .75, 1, 1.25 ] ) svg += `<line x1="${L}" x2="${W - R}" y1="${y( v )}" y2="${y( v )}" stroke="${v === 1 ? '#333' : '#e1e1dc'}" ${v === 1 ? 'stroke-dasharray="6 4"' : ''}/><text x="${L - 12}" y="${y( v ) + 4}" font-size="12" text-anchor="end" fill="#555">${v.toFixed( 2 )}</text>`;
	for ( const [ name, , color, control ] of series ) {

		const ratios = data[ name ].ratios;
		const values = panel ? Object.entries( ratios ).filter( ( [ key ] ) => ! key.startsWith( 'legacy/' ) ).map( ( [ , v ] ) => v ).sort( ( a, b ) => a - b ) : keys.map( k => ratios[ 'legacy/' + k ] );
		if ( values.some( v => ! Number.isFinite( v ) || v < 0 || v > 1.3 ) ) throw new Error( 'Update plot range for these data.' );
		svg += `<polyline points="${values.map( ( v, i ) => `${x( i / ( values.length - 1 ) )},${y( v )}` ).join( ' ' )}" fill="none" stroke="${color}" stroke-width="2.5" ${control ? 'stroke-dasharray="5 3"' : ''}/>`;
		if ( ! panel ) for ( const [ i, v ] of values.entries() ) svg += `<circle cx="${x( i / 9 )}" cy="${y( v )}" r="3.5" fill="${color}"/>`;

	}

	if ( panel ) {

		for ( const percent of [ 0, 25, 50, 75, 100 ] ) svg += `<text x="${x( percent / 100 )}" y="${top + height + 22}" font-size="12" text-anchor="middle" fill="#555">${percent}%</text>`;

	} else {

		for ( const [ i, key ] of keys.entries() ) svg += `<text x="${x( i / 9 )}" y="${top + height + 22}" font-size="12" text-anchor="middle" fill="#555">${key}</text>`;

	}

}

// Narrow scale exposes the small cross-PR effects hidden near 1 in the full-range panel.
{

	const top = 920, height = 250, x = u => L + u * width;
	const y = v => top + height * ( 1 - ( v - .94 ) / .07 );
	svg += `<text x="${L}" y="${top - 25}" font-size="17" font-weight="600">Zoom: #117 integration-error ratios, including extra-FAST controls</text>`;
	for ( const v of [ .94, .96, .98, 1, 1.01 ] ) svg += `<line x1="${L}" x2="${W - R}" y1="${y( v )}" y2="${y( v )}" stroke="${v === 1 ? '#333' : '#e1e1dc'}" ${v === 1 ? 'stroke-dasharray="6 4"' : ''}/><text x="${L - 12}" y="${y( v ) + 4}" font-size="12" text-anchor="end" fill="#555">${v.toFixed( 2 )}</text>`;
	for ( const [ name, , color, control ] of series.filter( ( [ n ] ) => n.startsWith( 'pr117-' ) ) ) {

		const values = Object.entries( data[ name ].ratios ).filter( ( [ k ] ) => ! k.startsWith( 'legacy/' ) ).map( ( [ , v ] ) => v ).sort( ( a, b ) => a - b );
		if ( values.some( v => v < .94 || v > 1.01 ) ) throw new Error( 'Update zoom range.' );
		svg += `<polyline points="${values.map( ( v, i ) => x( i / ( values.length - 1 ) ) + ',' + y( v ) ).join( ' ' )}" fill="none" stroke="${color}" stroke-width="2.5" ${control ? 'stroke-dasharray="5 3"' : ''}/>`;

	}

	for ( const percent of [ 0, 25, 50, 75, 100 ] ) svg += `<text x="${x( percent / 100 )}" y="${top + height + 22}" font-size="12" text-anchor="middle" fill="#555">${percent}%</text>`;

}

const counts = name => {

	const row = data[ name ];
	if ( row.metrics !== 318 ) throw new Error( 'Unexpected diagnostic count.' );
	return row.wins + '/' + row.losses + '/' + row.ties;

};

svg += `<text x="${L}" y="1250" font-size="13" fill="#555">318-diagnostic wins/losses/ties: light playback ${counts( 'pr117-light-seam-03-s1' )}; playback 25% ${counts( 'pr117-seam-polish-s1' )}; annealed ${counts( 'latin-anneal001-3000-s1' )}.</text><text x="${L}" y="1274" font-size="12" fill="#555">Sorted curves show each method's distribution, not matching diagnostics at matching ranks. No all-metrics winner.</text><text x="${L}" y="1298" font-size="12" fill="#555">#117 inputs: published seed-1 PNGs. Finite development tests; no unseen rendering or general state-of-the-art proof.</text></svg>`;

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
