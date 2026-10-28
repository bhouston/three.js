import fs from 'fs';
import puppeteer from 'puppeteer';
const [ dataPath, outPath ] = process.argv.slice( 2 );
const raw = JSON.parse( fs.readFileSync( dataPath ) );
const data = { frames: raw.frames, ...raw.blur };
const series = [[ 'White', '#2a78d6' ], [ 'FAST Spatial', '#eb6834' ], [ 'FAST Spatio-Temporal', '#1baf7a' ], [ 'Golden Ratio', '#eda100' ]];
const W = 820, H = 460, L = 70, R = 190, T = 56, B = 56;
const x = n => L + Math.log2( n ) / 10 * ( W - L - R );
const ymin = 1e-4, ymax = 0.1;
const y = v => T + ( Math.log10( ymax ) - Math.log10( v ) ) / ( Math.log10( ymax ) - Math.log10( ymin ) ) * ( H - T - B );
let s = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" font-family="Inter, Segoe UI, Helvetica, Arial, sans-serif"><rect width="100%" height="100%" fill="#fcfcfb"/>`;
s += `<text x="${L}" y="24" font-size="16" font-weight="600" fill="#1a1a19">Error after temporal and spatial blur (lower is better)</text>`;
s += `<text x="${L}" y="42" font-size="12" fill="#5f5e58">Std. dev. of a running average of a uniform noise field, then a σ = 1 gaussian, 128×128, log scale</text>`;
for ( const v of [ 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1 ] ) s += `<line x1="${L}" x2="${W - R}" y1="${y( v )}" y2="${y( v )}" stroke="#e6e5e0"/><text x="${L - 8}" y="${y( v ) + 4}" font-size="11" fill="#5f5e58" text-anchor="end">${v}</text>`;
for ( const n of data.frames ) s += `<text x="${x( n )}" y="${H - B + 20}" font-size="11" fill="#5f5e58" text-anchor="middle">${n}</text>`;
s += `<text x="${( L + W - R ) / 2}" y="${H - 14}" font-size="12" fill="#5f5e58" text-anchor="middle">frames accumulated</text>`;
const ends = [];
for ( const [ name, color ] of series ) {

	const pts = data.frames.map( ( n, i ) => [ x( n ), y( data[ name ][ i ] ) ] );
	s += `<polyline points="${pts.map( p => p.join( ',' ) ).join( ' ' )}" fill="none" stroke="${color}" stroke-width="2"/>`;
	for ( const [ px, py ] of pts ) s += `<circle cx="${px}" cy="${py}" r="4" fill="${color}" stroke="#fcfcfb" stroke-width="2"/>`;
	ends.push( { name, color, y: pts.at( - 1 )[ 1 ], v: data[ name ].at( - 1 ) } );

}

// direct labels at the right edge, nudged apart so close endpoints don't collide
ends.sort( ( a, b ) => a.y - b.y );
for ( let i = 1; i < ends.length; i ++ ) ends[ i ].ly = Math.max( ends[ i ].y, ( ends[ i - 1 ].ly ?? ends[ i - 1 ].y ) + 30 );
ends[ 0 ].ly = ends[ 0 ].y;
for ( const e of ends ) s += `<rect x="${W - R + 12}" y="${e.ly - 9}" width="10" height="10" rx="2" fill="${e.color}"/><text x="${W - R + 28}" y="${e.ly}" font-size="12" font-weight="600" fill="#1a1a19">${e.name}</text><text x="${W - R + 28}" y="${e.ly + 14}" font-size="11" fill="#5f5e58">${e.v.toFixed( 5 )} @ 1024</text>`;
s += '</svg>';
fs.writeFileSync( outPath.replace( /\.png$/, '.svg' ), s );
const browser = await puppeteer.launch( { headless: 'new' } );
const page = await browser.newPage();
await page.setViewport( { width: W, height: H, deviceScaleFactor: 2 } );
await page.setContent( `<body style="margin:0">${s}</body>` );
await page.screenshot( { path: outPath } );
await browser.close();
