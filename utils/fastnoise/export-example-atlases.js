// Pack previously generated seed-1 scalar volumes; never run an optimizer here.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { PNG } from 'pngjs';

const definitions = [

	[ 'light-seam-03', 'Light seam 3% (FAST polish)', 'PR117 restarted FAST refinement with a 3% seam objective; selected for its low regression count.' ],
	[ 'pr116-annealed', 'PR116 annealed stratification', 'PR116 annealed temporal-stratum-preserving construction; selected for its broad improvement and faster generation.' ]

];

const [ rawDirectory, destination = fileURLToPath( new URL( '../../examples/textures/noises/experimental/', import.meta.url ) ) ] = process.argv.slice( 2 );
if ( ! rawDirectory || process.argv.length > 4 ) throw new Error( 'Usage: node utils/fastnoise/export-example-atlases.js RAW_DIR [DEST_DIR]' );
const size = 128, depth = 32, count = size * size * depth;
const sha256 = bytes => createHash( 'sha256' ).update( bytes ).digest( 'hex' );
const entries = [];

// Validate every input before writing any output. Float32 files use little endian.
const volumes = definitions.map( ( [ id, label, description ] ) => {

	const source = `${id}-seed1.f32`;
	const raw = fs.readFileSync( path.join( rawDirectory, source ) );
	assert.equal( raw.length, count * 4, `${source}: expected ${count} Float32 samples` );
	const pixels = Buffer.alloc( count );
	for ( let i = 0; i < count; i ++ ) {

		const value = raw.readFloatLE( i * 4 );
		assert.ok( Number.isFinite( value ) && value >= 0 && value < 1, `${source}: invalid sample at ${i}: ${value}` );
		pixels[ i ] = Math.floor( Math.min( 1, Math.max( 0, value ) ) * 256 );

	}

	return { id, label, description, source, raw, pixels };

} );

fs.mkdirSync( destination, { recursive: true } );
for ( const { id, label, description, source, raw, pixels } of volumes ) {

	const png = PNG.sync.write( { width: size, height: size * depth, data: pixels }, { inputColorType: 0, colorType: 0, inputHasAlpha: false, bitDepth: 8 } );
	const decoded = PNG.sync.read( png );
	assert.equal( decoded.width, size );
	assert.equal( decoded.height, size * depth );
	for ( let i = 0; i < count; i ++ ) {

		for ( let channel = 0; channel < 3; channel ++ ) assert.equal( decoded.data[ i * 4 + channel ], pixels[ i ], `${id}: PNG round-trip mismatch` );
		assert.equal( decoded.data[ i * 4 + 3 ], 255 );

	}

	const file = `${id}-seed1.png`;
	fs.writeFileSync( path.join( destination, file ), png );
	entries.push( {

		id, label, description, file, seed: 1,
		width: size, height: size * depth, size, depth,
		layout: 'z-major vertical slices; atlas(x, z * size + y)',
		channels: 'scalar grayscale; no vector channels',
		quantization: 'floor(value * 256); runtime midpoint (byte + 0.5) / 256',
		volumeSHA256: sha256( raw ), pngSHA256: sha256( png ),
		source: { file: source, format: 'little-endian Float32, z/y/x order', generation: 'Offline seed-1 research volume; see utils/fastnoise/EXAMPLE.md and FRONTIER.md' }

	} );

}

fs.writeFileSync( path.join( destination, 'manifest.json' ), JSON.stringify( { version: 1, entries }, null, 2 ) + '\n' );
console.log( `Exported and verified ${entries.length} scalar PNG atlases in ${destination}` );
