// Sequential completed-work generation experiments. Software adapters forbidden.
// Usage: node research-run.js PROFILES_JSON OUTPUT_DIRECTORY
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import os from 'node:os';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import puppeteer from 'puppeteer';

const [ profileFile, outputDirectory ] = process.argv.slice( 2 );
if ( ! profileFile || ! outputDirectory ) throw new Error( 'Expected profile JSON and output directory' );
const profiles = JSON.parse( fs.readFileSync( profileFile ) );
const out = path.resolve( outputDirectory ), root = path.dirname( fileURLToPath( import.meta.url ) );
fs.mkdirSync( out, { recursive: true } );
const manifestFile = path.join( out, 'generation.json' );
const manifest = fs.existsSync( manifestFile ) ? JSON.parse( fs.readFileSync( manifestFile ) ) : { environment: { node: process.version, cpu: os.cpus()[ 0 ].model }, runs: [] };
function totalCost( row, seen = new Set() ) {

	if ( seen.has( row.name ) ) throw new Error( 'Generation dependency cycle' );
	seen.add( row.name );
	if ( ! row.profile.initial ) return row.secondsPerVolume;
	const name = row.profile.initial.replace( '{seed}', row.seed );
	const input = manifest.runs.find( v => v.name === name );
	if ( ! input ) throw new Error( 'Missing generation dependency ' + name );
	return row.secondsPerVolume + totalCost( input, seen );

}

for ( const row of manifest.runs ) row.totalSecondsPerVolume = totalCost( row );
const save = () => fs.writeFileSync( manifestFile, JSON.stringify( manifest, null, 2 ) + '\n' );
const record = ( profile, seed, bytes, seconds, count, extra = {} ) => {

	const name = `${profile.name}-seed${seed}`;
	fs.writeFileSync( path.join( out, name + '.f32' ), bytes );
	const row = { name, profile, seed, seconds, count, secondsPerVolume: seconds / count, sha256: createHash( 'sha256' ).update( bytes ).digest( 'hex' ), ...extra };
	if ( profile.initial ) {

		row.inputName = profile.initial.replace( '{seed}', seed );
		const input = manifest.runs.find( v => v.name === row.inputName );
		if ( ! input ) throw new Error( 'Missing input generation cost for ' + row.inputName );
		row.inputSecondsPerVolume = input.totalSecondsPerVolume ?? input.secondsPerVolume;
		row.totalSecondsPerVolume = row.secondsPerVolume + row.inputSecondsPerVolume;

	} else row.totalSecondsPerVolume = row.secondsPerVolume;
	manifest.runs = manifest.runs.filter( v => v.name !== name ); manifest.runs.push( row ); save();
	console.log( name, row.secondsPerVolume, row.sha256 );

};

const server = http.createServer( ( req, res ) => {

	const pathname = new URL( req.url, 'http://localhost' ).pathname;
	if ( pathname === '/probe' ) {

		res.setHeader( 'Content-Type', 'text/html' ); res.end( '<!doctype html><title>GPU probe</title>' ); return;

	}

	let file;
	if ( [ '/generator.html', '/research-generator.html' ].includes( pathname ) ) {

		file = path.join( root, pathname.slice( 1 ) ); res.setHeader( 'Content-Type', 'text/html' );

	} else if ( /^\/data\/[a-zA-Z0-9_-]+\.f32$/.test( pathname ) ) file = path.join( out, path.basename( pathname ) );
	if ( ! file || ! fs.existsSync( file ) ) {

		res.writeHead( 404 ).end(); return;

	}

	res.end( fs.readFileSync( file ) );

} );
await new Promise( resolve => server.listen( 8734, '127.0.0.1', resolve ) );
let browser;
try {

	if ( profiles.some( p => p.kind === 'gpu' ) ) {

		browser = await puppeteer.launch( { headless: true, executablePath: process.env.CHROME_PATH || ( process.platform === 'win32' ? 'C:/Program Files/Google/Chrome/Application/chrome.exe' : puppeteer.executablePath() ), args: [ '--enable-unsafe-webgpu', '--disable-software-rasterizer', '--disable-gpu-vsync', '--disable-frame-rate-limit' ], protocolTimeout: 3600000 } );
		manifest.environment.chrome = await browser.version();
		const page = await browser.newPage();
		await page.goto( 'http://127.0.0.1:8734/probe' );
		manifest.environment.gpu = await page.evaluate( async () => {

			const adapter = await navigator.gpu.requestAdapter();
			if ( ! adapter ) throw new Error( 'Hardware GPU verification blocked: no adapter' );
			const { vendor, architecture, description, isFallbackAdapter } = adapter.info;
			const info = { vendor, architecture, description, isFallbackAdapter };
			if ( isFallbackAdapter || /swiftshader|software|llvmpipe|lavapipe/i.test( JSON.stringify( info ) ) || ! /nvidia|amd|intel|apple|qualcomm/i.test( vendor ) ) throw new Error( 'Unverified physical adapter' );
			return info;

		} );
		console.log( manifest.environment ); save();
		async function runGPU( file, parameters ) {

			let failure;
			const handler = error => {

				failure = error.message;

			};

			page.on( 'pageerror', handler );
			await page.goto( `http://127.0.0.1:8734/${file}?${new URLSearchParams( parameters )}` );
			try {

				const deadline = Date.now() + 1800000;
				while ( true ) {

					const state = await page.evaluate( () => ( { done: !! window.result, failure: window.failure, log: document.querySelector( '#log' )?.textContent } ) );
					if ( failure || state.failure || /wgsl error|device lost/i.test( state.log ) ) throw new Error( failure || state.failure || state.log );
					if ( state.done ) return await page.evaluate( () => ( { result: window.result, log: document.querySelector( '#log' ).textContent } ) );
					if ( Date.now() > deadline ) throw new Error( 'GPU generation timeout' );
					await new Promise( resolve => setTimeout( resolve, 500 ) );

				}

			} finally {

				page.off( 'pageerror', handler );

			}

		}

		for ( const profile of profiles.filter( p => p.kind === 'gpu' ) ) {

			for ( const seed of profile.seeds || [ 1, 2 ] ) {

				if ( process.argv.includes( '--resume' ) && manifest.runs.some( v => v.name === `${profile.name}-seed${seed}` ) ) continue;

				const parameters = { ...profile.parameters, seed };
				if ( profile.initial ) parameters.initial = '/data/' + profile.initial.replace( '{seed}', seed ) + '.f32';
				await runGPU( profile.file, { ...parameters, steps: 100 } );
				await page.goto( 'about:blank' );
				const start = performance.now();
				const value = await runGPU( profile.file, parameters );
				const seconds = ( performance.now() - start ) / 1000;
				record( profile, seed, Buffer.from( value.result, 'base64' ), seconds, 1, { log: value.log, timing: 'navigation through GPU-completed readback and base64 retrieval; warmup excluded' } );
				await page.goto( 'about:blank' );

			}

		}

		await browser.close(); browser = null;

	}

	for ( const profile of profiles.filter( p => p.kind === 'cpu' ) ) {

		const { generate } = await import( new URL( profile.module, import.meta.url ) );
		for ( const seed of profile.seeds || [ 1, 2 ] ) {

			if ( process.argv.includes( '--resume' ) && manifest.runs.some( v => v.name === `${profile.name}-seed${seed}` ) ) continue;

			const options = { ...profile.options, seed };
			if ( profile.initial ) {

				const bytes = fs.readFileSync( path.join( out, profile.initial.replace( '{seed}', seed ) + '.f32' ) );
				options.initialVolume = new Float32Array( bytes.buffer.slice( bytes.byteOffset, bytes.byteOffset + bytes.byteLength ) );

			}

			generate( options );
			let count = 0, volume;
			const start = performance.now();
			do {

				volume = generate( options ); count ++;

			} while ( count < 2 || performance.now() - start < 5000 );

			const seconds = ( performance.now() - start ) / 1000;
			record( profile, seed, Buffer.from( volume.buffer, volume.byteOffset, volume.byteLength ), seconds, count, { timing: 'complete synchronous generations including initialization, excluding output file write; warmup excluded' } );

		}

	}

} finally {

	if ( browser ) await browser.close();
	await new Promise( resolve => server.close( resolve ) );

}
