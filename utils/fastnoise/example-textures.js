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
const pin = '502a026ac59540f305d6f05fa2e8a53c1301815a';
const profiles = [
	{ name: 'hierarchy', module: './alternatives/hierarchy.js' },
	{ name: 'lattice', module: './alternatives/lattice.js' },
	{ name: 'relaxation', module: './alternatives/relaxation.js' },
	{ name: 'threshold-unconstrained', module: './experiments/threshold-refine.js', options: { passes: 16, preserveColumns: false } },
	{ name: 'threshold-columns', module: './experiments/threshold-refine.js', options: { passes: 8, scalarWeight: 1, thresholdWeight: 4 } },
	{ name: 'stratified', module: './experiments/stratified.js', options: { passes: 128 } },
	{ name: 'stratified-helical', module: './experiments/stratified.js', options: { cycleShift: true } },
	{ name: 'multifeature', module: './experiments/multifeature.js' },
	{ name: 'gpu-identity', gpu: { steps: 1000, identityWeight: 1 } },
	{ name: 'gpu-hybrid', gpu: { steps: 1000, identityWeight: .25, mixedWeight: .2 } },
	{ name: 'gpu-spectral', gpu: { steps: 1000, identityWeight: .25, frequencyWeight: .05 } },
	{ name: 'gpu-mixed', gpu: { steps: 1000, mixedWeight: .2 } },
	{ name: 'gpu-sparse', gpu: { steps: 3000, swapFloor: 2 } },
	{ name: 'gpu-stratified', gpu: { steps: 3000, stratified: true, swapFloor: 2 } },
	{ name: 'gpu-stratified-hybrid', gpu: { steps: 3000, stratified: true, swapFloor: 2, identityWeight: .25, mixedWeight: .2 } },
	{ name: 'gpu-stratified-psd', gpu: { steps: 3000, stratified: true, swapFloor: 2, identityWeight: .25, mixedWeight: .02, mixedKernel: 'binomial' } },
	{ name: 'gpu-stratified-annealed', gpu: { steps: 3000, stratified: true, swapFloor: 2, identityWeight: .25, mixedWeight: .2, annealTemperature: .001, annealSteps: 1500 } },
	... [ 'filter-rank', 'hierarchy', 'local-energy' ].map( method => ( { name: 'pr117-' + method, pr117: method } ) )
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

		let generator;
		if ( profile.pr117 ) {

			const file = `utils/fastnoise/methods/${profile.pr117}.js`;
			const args = process.platform === 'win32' ? [ '/d', '/s', '/c', `git-dedup.cmd show ${pin}:${file}` ] : [ 'show', `${pin}:${file}` ];
			const run = spawnSync( process.platform === 'win32' ? 'cmd.exe' : 'git-dedup', args, { cwd: root, encoding: 'utf8' } );
			if ( run.status !== 0 ) throw new Error( `Fetch the pinned PR117 commit with git-dedup before baking: ${run.stderr}` );
			generator = await import( 'data:text/javascript;base64,' + Buffer.from( run.stdout ).toString( 'base64' ) );

		} else {

			generator = await import( profile.module );

		}

		volume = generator.generate( { seed: 1, ...profile.options } );
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
