import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer';

const directory = path.dirname( fileURLToPath( import.meta.url ) );
const data = JSON.parse( fs.readFileSync( path.join( directory, 'results.json' ) ) );
const names = Object.keys( data.blur );
const colors = [ '#3676ce', '#eb7934', '#16845b', '#b78000', '#a344b4', '#d54178' ];
const W = 1240, H = 760;
function text( x, y, value, size = 14, extra = '' ) {

	return `<text x="${x}" y="${y}" font-size="${size}" ${extra}>${value}</text>`;

}

function background( title, subtitle ) {

	return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" font-family="Segoe UI,Arial,sans-serif" fill="#252525"><rect width="100%" height="100%" fill="#fcfcfb"/>${text( 70, 36, title, 23 )}${text( 70, 62, subtitle, 14 )}`;

}

const browser = await puppeteer.launch( { headless: true, executablePath: process.env.CHROME_PATH || ( process.platform === 'win32' ? 'C:/Program Files/Google/Chrome/Application/chrome.exe' : puppeteer.executablePath() ), args: [ '--disable-software-rasterizer' ] } );
try {

	const page = await browser.newPage();
	await page.setViewport( { width: W, height: H, deviceScaleFactor: 2 } );
	async function save( name, svg ) {

		fs.writeFileSync( path.join( directory, `${name}.svg` ), svg );
		await page.setContent( `<body style="margin:0">${svg}</body>` );
		await page.screenshot( { path: path.join( directory, `${name}.png` ) } );

	}

	for ( const field of [ 'blur', 'raw' ] ) {

		let svg = background( field === 'blur' ? 'Convergence after temporal and spatial filtering' : 'Convergence before spatial filtering', 'Uniform field · 128×128 · arithmetic running average · logarithmic axes · lower is better' );
		const L = 85, T = 110, right = 900, bottom = 650;
		const min = 10 ** Math.floor( Math.log10( Math.min( ...Object.values( data[ field ] ).flat() ) ) );
		const max = field === 'blur' ? .1 : 1;
		const x = n => L + Math.log2( n ) / 10 * ( right - L );
		const y = v => T + Math.log10( max / v ) / Math.log10( max / min ) * ( bottom - T );
		for ( let e = Math.log10( min ); e <= Math.log10( max ); e ++ ) {

			const value = 10 ** e;
			svg += `<line x1="${L}" x2="${right}" y1="${y( value )}" y2="${y( value )}" stroke="#ddd"/>${text( L - 12, y( value ) + 4, value, 12, 'text-anchor="end"' )}`;

		}

		for ( const n of data.frames ) svg += text( x( n ), bottom + 24, n, 12, 'text-anchor="middle"' );
		for ( const [ index, name ] of names.entries() ) {

			const color = colors[ index ];
			const points = data.frames.map( ( n, i ) => `${x( n )},${y( data[ field ][ name ][ i ] )}` ).join( ' ' );
			svg += `<polyline points="${points}" fill="none" stroke="${color}" stroke-width="2.5"/>`;
			for ( const [ i, n ] of data.frames.entries() ) svg += `<circle cx="${x( n )}" cy="${y( data[ field ][ name ][ i ] )}" r="3" fill="${color}"/>`;
			const label = name.replace( 'FAST Spatial (Product)', 'FAST Product' ).replace( 'FAST Spatio-Temporal (Separate)', 'FAST Separate' );
			svg += `<rect x="930" y="${T + index * 75 - 12}" width="20" height="4" fill="${color}"/>${text( 960, T + index * 75, label, 16 )}${text( 960, T + index * 75 + 23, `${data[ field ][ name ].at( - 1 ).toPrecision( 3 )} at 1024 frames`, 13 )}`;

		}

		svg += text( 430, 700, 'Frames accumulated', 14 ) + text( 70, 735, 'Raw-value convergence is not a rendering-quality guarantee. See threshold RMSE and spectrum checks in TABLES.md.', 13 ) + '</svg>';
		await save( `convergence-${field}`, svg );

	}

	let svg = background( 'Texture download and intrinsic setup cost', 'Exact repository PNG bytes; excludes HTTP overhead. Shader compilation and network latency are not zero-cost.' );
	const costs = [[ 'White', 0 ], [ 'FAST Product', data.metadata.textures[ 0 ].bytes ], [ 'FAST Separate', data.metadata.textures[ 1 ].bytes ], [ 'Golden Ratio', data.metadata.textures[ 0 ].bytes ], [ '3LHW', 0 ], [ 'Animated IGN', 0 ]];
	for ( const [ i, [ name, bytes ]] of costs.entries() ) {

		const y = 130 + i * 70;
		svg += text( 70, y + 21, name, 17 ) + `<rect x="250" y="${y}" width="${bytes / 3100000 * 650}" height="30" fill="${colors[ i ]}"/>` + text( 920, y + 21, `${bytes.toLocaleString( 'en-US' )} bytes`, 16 );

	}

	svg += text( 70, 600, 'White / 3LHW / Animated IGN: 0 texture bytes, no mask construction or texture upload.', 16 );
	svg += text( 70, 632, 'FAST / Golden Ratio: offline texture optimization; runtime fetch, PNG decode and GPU upload.', 16 );
	svg += text( 70, 664, 'All methods incur normal shader compilation. Procedural arithmetic executes per sample.', 16 );
	svg += text( 70, 707, 'Measured local first-ready timings are reported separately; they depend on cache, storage, browser and GPU.', 14 ) + '</svg>';
	await save( 'texture-cost', svg );

	const startupFile = path.join( directory, 'startup.json' );
	if ( fs.existsSync( startupFile ) ) {

		const measured = JSON.parse( fs.readFileSync( startupFile, 'utf8' ).replace( /^\uFEFF/, '' ) );
		svg = background( 'Observed local selection-to-ready delay', 'One hardware WebGPU run; includes download/decode/upload and shader compilation. Not an Internet or cold-start ranking.' );
		const maximum = Math.max( ...measured.startup.map( row => row.selectionToGPUCompletedMs ) );
		for ( const [ i, row ] of measured.startup.entries() ) {

			const y = 125 + i * 70;
			svg += text( 70, y + 20, row.label, 17 ) + `<rect x="250" y="${y}" width="${row.selectionToGPUCompletedMs / maximum * 650}" height="30" fill="${colors[ i ]}"/>`;
			svg += text( 920, y + 21, `${row.selectionToGPUCompletedMs.toFixed( 1 )} ms`, 16 );

		}

		svg += text( 70, 600, 'Intrinsic runtime mask generation: 0 ms for White, 3LHW and Animated IGN; no mask or texture is constructed.', 15 );
		svg += text( 70, 632, 'FAST uses preoptimized assets; its readiness also requires an asynchronous image load.', 15 );
		svg += text( 70, 664, 'Default 3LHW was already compiled; Golden Ratio reused Product. The timing samples have different cache states.', 14 );
		svg += text( 70, 707, `Physical ${measured.gpu.vendor} ${measured.gpu.architecture}; common initial page load ${measured.initialPageStartupMs.toFixed( 0 )} ms.`, 14 ) + '</svg>';
		await save( 'startup-cost', svg );

	}

} finally {

	await browser.close();

}
