import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { createHash } from 'node:crypto';
import { gzipSync } from 'node:zlib';

// Optional dependency root allows running from a worktree without another npm ci.
const directory = path.dirname( fileURLToPath( import.meta.url ) );
const root = path.resolve( directory, '../..' );
const require = createRequire( path.join( process.argv[ 2 ] || root, 'package.json' ) );
const { minify } = require( 'terser' );
const { version } = require( 'terser/package.json' );
const sourcePath = 'examples/jsm/tsl/utils/ProceduralNoise.js';
const source = fs.readFileSync( path.join( root, sourcePath ), 'utf8' ).replace( /\r\n/g, '\n' );
const hash = source.slice( source.indexOf( 'const hashUint =' ), source.indexOf( '\nconst pixelHash =' ) ).trimEnd();
const pixelHash = source.slice( source.indexOf( 'const pixelHash =' ), source.indexOf( '\n/**' ) ).trimEnd();
const hierarchyStart = source.indexOf( '/**' );
const ignStart = source.indexOf( '/**', hierarchyStart + 3 );
if ( hierarchyStart < 0 || ignStart < 0 || ! hash.startsWith( 'const hashUint =' ) ) throw new Error( 'Source layout changed; review standalone extraction.' );

const modules = [

	[ 'Animated IGN', 'Fn, float, frameId, screenCoordinate, uint', [ hash, source.slice( ignStart ).trimEnd() ]],
	[ '3LHW', 'Fn, frameId, screenCoordinate, uint', [ hash, pixelHash, source.slice( hierarchyStart, ignStart ).trimEnd() ]]

];
const options = { module: true, compress: true, mangle: true, format: { comments: false } };
const measurements = [];
for ( const [ name, imports, sections ] of modules ) {

	const input = `import { ${imports} } from 'three/tsl';\n\n${sections.join( '\n\n' )}\n`;
	const { code } = await minify( input, options );
	measurements.push( {
		name,
		rawBytes: Buffer.byteLength( input ),
		minifiedBytes: Buffer.byteLength( code ),
		gzipMinifiedBytes: gzipSync( code, { level: 9 } ).length,
		inputSha256: createHash( 'sha256' ).update( input ).digest( 'hex' )
	} );

}

const data = {

	source: sourcePath,
	scope: 'Standalone TSL JavaScript module per method, including JSDoc, required three/tsl named imports, shared hash helper, and all setup/runtime code. Shared helper is counted independently in each method. LF line endings, UTF-8 bytes. Excludes Three.js engine, generated GPU shader, example UI and benchmark harness. Neither method requires texture precomputation or startup data generation.',
	minifier: { name: 'terser', version, options },
	gzip: { level: 9, node: process.version, zlib: process.versions.zlib },
	measurements

};
fs.writeFileSync( path.join( directory, 'code-size.json' ), JSON.stringify( data, null, 2 ) + '\n' );

const width = 1100, height = 550;
const colors = [ '#3676ce', '#eb7934', '#16845b' ];
const fields = [[ 'Raw source', 'rawBytes' ], [ 'Minified', 'minifiedBytes' ], [ 'Minified + gzip', 'gzipMinifiedBytes' ]];
const max = Math.ceil( Math.max( ...measurements.map( row => row.rawBytes ) ) / 500 ) * 500;
const label = ( x, y, value, size = 16 ) => `<text x="${x}" y="${y}" font-size="${size}">${value}</text>`;
let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" font-family="Segoe UI,Arial,sans-serif" fill="#252525"><rect width="100%" height="100%" fill="#fcfcfb"/>`;
svg += label( 45, 40, 'Procedural noise implementation size', 26 );
svg += label( 45, 70, 'Standalone TSL JavaScript · required helpers included · Three.js engine excluded', 16 );
for ( let i = 0; i < fields.length; i ++ ) {

	svg += `<rect x="${45 + i * 260}" y="95" width="18" height="18" fill="${colors[ i ]}"/>` + label( 73 + i * 260, 110, fields[ i ][ 0 ] );

}

for ( let tick = 0; tick <= max; tick += 500 ) {

	const x = 220 + tick / max * 760;
	svg += `<line x1="${x}" x2="${x}" y1="135" y2="440" stroke="#ddd"/>` + label( x - 12, 465, tick, 13 );

}

for ( const [ index, row ] of measurements.entries() ) {

	const y = 155 + index * 155;
	svg += label( 45, y + 55, row.name, 21 );
	for ( const [ j, [ , field ]] of fields.entries() ) {

		const barWidth = row[ field ] / max * 760;
		svg += `<rect x="220" y="${y + j * 37}" width="${barWidth}" height="27" fill="${colors[ j ]}"/>` + label( 230 + barWidth, y + j * 37 + 20, row[ field ].toLocaleString( 'en-US' ), 15 );

	}

}

svg += label( 530, 493, 'Bytes', 15 );
svg += label( 45, 530, `Raw includes comments and formatting; Terser ${version}; gzip level 9. No precomputed data required.`, 14 ) + '</svg>';
fs.writeFileSync( path.join( directory, 'code-size.svg' ), svg + '\n' );

const puppeteer = require( 'puppeteer' );
const browser = await puppeteer.launch( { headless: true, executablePath: process.env.CHROME_PATH || ( process.platform === 'win32' ? 'C:/Program Files/Google/Chrome/Application/chrome.exe' : puppeteer.executablePath() ), args: [ '--disable-software-rasterizer' ] } );
try {

	const page = await browser.newPage();
	await page.setViewport( { width, height, deviceScaleFactor: 2 } );
	await page.setContent( `<body style="margin:0">${svg}</body>` );
	await page.screenshot( { path: path.join( directory, 'code-size.png' ) } );

} finally {

	await browser.close();

}

console.log( JSON.stringify( data, null, 2 ) );
