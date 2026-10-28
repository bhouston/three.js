// node utils/fastnoise/example-textures.js RAW_DIRECTORY [--generate-gpu]
// All assets are scalar, 128x4096, 8-bit grayscale PNGs; runtime does no solving.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { PNG } from 'pngjs';

const rawDirectory = path.resolve( process.argv[ 2 ] || 'example-noise-raw' );
const root = fileURLToPath( new URL( '../../', import.meta.url ) );
const output = path.join( root, 'examples/textures/noises/experimental' );
const pin = '2a065526980261bcf5b129f07ddb12ff7eda1f91';
const profiles = [
	{ name: 'gpu-stratified-annealed', gpu: { steps: 3000, stratified: true, swapFloor: 2, identityWeight: .25, mixedWeight: .2, annealTemperature: .001, annealSteps: 1500 } },
	{ name: 'pr117-light-seam-03', pr117: 'light-seam-03' }
];
fs.mkdirSync( rawDirectory, { recursive: true } );
fs.mkdirSync( output, { recursive: true } );
if ( process.argv.includes( '--generate-gpu' ) ) {

	const plan = path.join( rawDirectory, 'example-gpu-plan.json' );
	fs.writeFileSync( plan, JSON.stringify( profiles.filter( p => p.gpu ).map( p => ( { name: p.name, seed: 1, ...p.gpu } ) ), null, 2 ) );
	const run = spawnSync( process.execPath, [ fileURLToPath( new URL( './experiments/run-gpu.js', import.meta.url ) ), rawDirectory, plan ], { stdio: 'inherit' } );
	if ( run.status !== 0 ) throw new Error( 'Hardware GPU generation failed.' );

}

const sha = bytes => createHash( 'sha256' ).update( bytes ).digest( 'hex' );
const manifest = { layout: '128x4096 scalar R, z/y/x; floor(value*256); decode as (byte+.5)/256; seed 1', profiles: [] };
for ( const profile of profiles ) {

	let volume;
	if ( profile.gpu ) {

		const bytes = fs.readFileSync( path.join( rawDirectory, profile.name + '.f32' ) );
		volume = new Float32Array( bytes.buffer.slice( bytes.byteOffset, bytes.byteOffset + bytes.byteLength ) );

	} else {

		// Import the pinned published PNG, whose 8-bit values are the audit input.
		const file = 'examples/textures/noises/experimental/light-seam-03-seed1.png';
		const windows = process.platform === 'win32';
		const run = spawnSync( windows ? 'cmd.exe' : 'git-dedup', windows ? [ '/d', '/s', '/c', `git-dedup.cmd show ${pin}:${file}` ] : [ 'show', `${pin}:${file}` ], { cwd: root, maxBuffer: 8 * 1024 * 1024 } );
		if ( run.status !== 0 ) throw new Error( 'Fetch the pinned PR117 commit with git-dedup first: ' + run.stderr );
		if ( sha( run.stdout ) !== 'cb36850d1e5e5e1a250c489784086d9a173574118b331b18b7f4a7db24c63b01' ) throw new Error( 'Pinned PNG hash mismatch.' );
		const image = PNG.sync.read( run.stdout );
		if ( image.width !== 128 || image.height !== 4096 ) throw new Error( 'Invalid PR117 dimensions.' );
		volume = Float32Array.from( { length: 128 * 128 * 32 }, ( _, i ) => ( image.data[ i * 4 ] + .5 ) / 256 );
		fs.writeFileSync( path.join( rawDirectory, profile.name + '.f32' ), Buffer.from( volume.buffer ) );

	}

	if ( volume.length !== 128 * 128 * 32 || ! volume.every( v => Number.isFinite( v ) && v >= 0 && v < 1 ) ) throw new Error( 'Invalid volume: ' + profile.name );
	const image = new PNG( { width: 128, height: 4096 } );
	const quantized = new Uint8Array( volume.length );
	for ( let i = 0; i < volume.length; i ++ ) {

		const value = Math.floor( volume[ i ] * 256 );
		quantized[ i ] = value;
		image.data[ i * 4 ] = image.data[ i * 4 + 1 ] = image.data[ i * 4 + 2 ] = value;
		image.data[ i * 4 + 3 ] = 255;

	}

	const bytes = PNG.sync.write( image, { colorType: 0, bitDepth: 8, deflateLevel: 9 } );
	fs.writeFileSync( path.join( output, profile.name + '.png' ), bytes );
	manifest.profiles.push( { ...profile, seed: 1, pr117Commit: profile.pr117 ? pin : undefined, floatSHA256: sha( Buffer.from( volume.buffer ) ), quantizedSHA256: sha( quantized ), pngSHA256: sha( bytes ), bytes: bytes.length } );
	console.log( profile.name, bytes.length );

}

fs.writeFileSync( path.join( output, 'manifest.json' ), JSON.stringify( manifest, null, 2 ) + '\n' );
