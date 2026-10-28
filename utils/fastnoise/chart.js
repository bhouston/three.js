import fs from 'fs';
import puppeteer from 'puppeteer';
const [ dataPath, outPath ] = process.argv.slice( 2 );
const raw = JSON.parse( fs.readFileSync( dataPath ) );
const data = { frames: raw.frames, ...raw.blur };
const colors = { 'White': '#2a78d6', 'FAST Spatial': '#eb6834', 'FAST Spatio-Temporal': '#16845b', 'Golden Ratio': '#b78000', 'Filter / Rank': '#9c45c5', 'Hierarchy': '#df4191', 'Cached Threshold Energy': '#303d4c' };
const series = Object.keys( raw.blur ).map( name => [ name, colors[ name ] || '#666666' ] );
const W = 1120, H = 660, L = 85, R = 260, T = 90, B = 86;
const x = n => L + Math.log2( n ) / 10 * ( W - L - R );
const ymin = 10 ** Math.floor( Math.log10( Math.min( ...Object.values( raw.blur ).flat() ) ) ), ymax = 0.1;
const y = v => T + ( Math.log10( ymax ) - Math.log10( v ) ) / ( Math.log10( ymax ) - Math.log10( ymin ) ) * ( H - T - B );
let s = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" font-family="Inter, Segoe UI, Helvetica, Arial, sans-serif"><rect width="100%" height="100%" fill="#fcfcfb"/>`;
s += `<text x="${L}" y="28" font-size="19" font-weight="600" fill="#1a1a19">Error after temporal and spatial blur (lower is better)</text>`;
s += `<text x="${L}" y="49" font-size="12" fill="#5f5e58">Std. dev. of a running average, then a σ = 1 Gaussian · 128×128 · logarithmic axes</text>`;
s += `<text x="${L}" y="69" font-size="12" fill="#5f5e58">${raw.metadata ? 'Uniform noise field · seed 1 alternatives · original shipped FAST textures' : 'Uniform noise field'}</text>`;
for ( let exponent = Math.log10( ymin ); exponent <= - 1; exponent ++ ) for ( const multiplier of [ 1, 2, 5 ] ) {

	const v = multiplier * 10 ** exponent;
	if ( v > ymax ) continue;
	s += `<line x1="${L}" x2="${W - R}" y1="${y( v )}" y2="${y( v )}" stroke="#e6e5e0"/><text x="${L - 8}" y="${y( v ) + 4}" font-size="11" fill="#5f5e58" text-anchor="end">${Number( v.toPrecision( 1 ) )}</text>`;

}

for ( const n of data.frames ) s += `<text x="${x( n )}" y="${H - B + 20}" font-size="11" fill="#5f5e58" text-anchor="middle">${n}</text>`;
s += `<text x="${( L + W - R ) / 2}" y="${H - 42}" font-size="12" fill="#5f5e58" text-anchor="middle">frames accumulated</text>`;
s += `<text x="${L}" y="${H - 15}" font-size="12" fill="#5f5e58">Caution: raw-value convergence does not predict thresholded rendering quality; see ALTERNATIVES.md.</text>`;
const ends = [];
for ( const [ name, color ] of series ) {

	const pts = data.frames.map( ( n, i ) => [ x( n ), y( data[ name ][ i ] ) ] );
	s += `<polyline points="${pts.map( p => p.join( ',' ) ).join( ' ' )}" fill="none" stroke="${color}" stroke-width="2"/>`;
	for ( const [ px, py ] of pts ) s += `<circle cx="${px}" cy="${py}" r="4" fill="${color}" stroke="#fcfcfb" stroke-width="2"/>`;
	ends.push( { name, color, y: pts.at( - 1 )[ 1 ], v: data[ name ].at( - 1 ) } );

}

// Fixed legend rows avoid collisions between seven tightly grouped endpoints.
for ( const [ i, e ] of ends.entries() ) {

	const ly = T + 18 + i * 58;
	s += `<line x1="${W - R + 20}" x2="${W - R + 42}" y1="${ly - 4}" y2="${ly - 4}" stroke="${e.color}" stroke-width="3"/><text x="${W - R + 52}" y="${ly}" font-size="13" font-weight="600" fill="#1a1a19">${e.name}</text><text x="${W - R + 52}" y="${ly + 18}" font-size="12" fill="#5f5e58">${e.v.toPrecision( 3 )} @ 1024 frames</text>`;

}

s += '</svg>';
fs.writeFileSync( outPath.replace( /\.png$/, '.svg' ), s );
const browser = await puppeteer.launch( { headless: true, executablePath: process.env.CHROME_PATH || ( process.platform === 'win32' ? 'C:/Program Files/Google/Chrome/Application/chrome.exe' : puppeteer.executablePath() ), args: [ '--disable-software-rasterizer' ] } );
const page = await browser.newPage();
await page.setViewport( { width: W, height: H, deviceScaleFactor: 2 } );
await page.setContent( `<body style="margin:0">${s}</body>` );
await page.screenshot( { path: outPath } );
await browser.close();
