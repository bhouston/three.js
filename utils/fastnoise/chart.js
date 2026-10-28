import fs from 'fs';
import puppeteer from 'puppeteer';
const [ dataPath, outPath ] = process.argv.slice( 2 );
const raw = JSON.parse( fs.readFileSync( dataPath ) );
const data = { frames: raw.frames, ...raw.blur };
const series = [[ 'White', '#2a78d6' ], [ 'FAST Spatial', '#eb6834' ], [ 'FAST Spatio-Temporal', '#1baf7a' ], [ 'Golden Ratio', '#b98700' ], [ 'Hierarchy', '#9b4bc2' ], [ 'Lattice', '#db4384' ], [ 'Relaxation', '#32636d' ]].filter( ( [ name ] ) => data[ name ] );
const W = 1120, H = 640, L = 84, R = 290, T = 70, B = 90;
const x = n => L + Math.log2( n ) / 10 * ( W - L - R );
const ymin = 1e-5, ymax = 0.1;
const y = v => T + ( Math.log10( ymax ) - Math.log10( v ) ) / ( Math.log10( ymax ) - Math.log10( ymin ) ) * ( H - T - B );
let s = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" font-family="Inter, Segoe UI, Helvetica, Arial, sans-serif"><rect width="100%" height="100%" fill="#fcfcfb"/>`;
s += `<text x="${L}" y="24" font-size="16" font-weight="600" fill="#1a1a19">Error after temporal and spatial blur (lower is better)</text>`;
s += `<text x="${L}" y="42" font-size="12" fill="#5f5e58">Std. dev. of a running average of a uniform noise field, then a σ = 1 gaussian, 128×128, log scale</text>`;
for ( const v of [ 0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1 ] ) s += `<line x1="${L}" x2="${W - R}" y1="${y( v )}" y2="${y( v )}" stroke="#e6e5e0"/><text x="${L - 8}" y="${y( v ) + 4}" font-size="11" fill="#5f5e58" text-anchor="end">${v}</text>`;
for ( const n of data.frames ) s += `<text x="${x( n )}" y="${H - B + 20}" font-size="11" fill="#5f5e58" text-anchor="middle">${n}</text>`;
s += `<text x="${( L + W - R ) / 2}" y="${H - B + 44}" font-size="12" fill="#5f5e58" text-anchor="middle">frames accumulated</text>`;
s += `<text x="${L}" y="${H - 18}" font-size="12" fill="#5f5e58">Scalar-value metric from the original chart; lower error here does not guarantee lower threshold-sampling error.</text>`;
const ends = [];
for ( const [ name, color ] of series ) {

	if ( data[ name ].length !== data.frames.length || data[ name ].some( v => ! Number.isFinite( v ) || v < ymin || v > ymax ) ) throw new Error( `Invalid or out-of-range series: ${name}` );
	const pts = data.frames.map( ( n, i ) => [ x( n ), y( data[ name ][ i ] ) ] );
	s += `<polyline points="${pts.map( p => p.join( ',' ) ).join( ' ' )}" fill="none" stroke="${color}" stroke-width="2"/>`;
	for ( const [ px, py ] of pts ) s += `<circle cx="${px}" cy="${py}" r="4" fill="${color}" stroke="#fcfcfb" stroke-width="2"/>`;
	ends.push( { name, color, y: pts.at( - 1 )[ 1 ], v: data[ name ].at( - 1 ) } );

}

// A fixed legend keeps seven labels readable even when endpoint errors coincide.
ends.sort( ( a, b ) => a.y - b.y );
for ( const [ i, e ] of ends.entries() ) e.ly = T + 15 + i * 56;
for ( const e of ends ) s += `<rect x="${W - R + 24}" y="${e.ly - 9}" width="10" height="10" rx="2" fill="${e.color}"/><text x="${W - R + 44}" y="${e.ly}" font-size="14" font-weight="600" fill="#1a1a19">${e.name}</text><text x="${W - R + 44}" y="${e.ly + 19}" font-size="12" fill="#5f5e58">${e.v.toPrecision( 4 )} @ 1024</text>`;
s += '</svg>';
fs.writeFileSync( outPath.replace( /\.png$/, '.svg' ), s );
const browser = await puppeteer.launch( { headless: true, ...( process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {} ), args: [ '--disable-software-rasterizer' ] } );
try {

	const page = await browser.newPage();
	await page.setViewport( { width: W, height: H, deviceScaleFactor: 2 } );
	await page.setContent( `<body style="margin:0">${s}</body>` );
	await page.screenshot( { path: outPath } );

} finally {

	await browser.close();

}
