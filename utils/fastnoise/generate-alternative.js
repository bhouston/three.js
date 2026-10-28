// node utils/fastnoise/generate-alternative.js hierarchy output.f32 [seed] [atlas.png]
import fs from 'node:fs';
import { performance } from 'node:perf_hooks';
import { PNG } from 'pngjs';

const [ method, output, seedText = '1', atlas ] = process.argv.slice( 2 );
if ( ! [ 'hierarchy', 'lattice', 'relaxation' ].includes( method ) || ! output || ! Number.isInteger( Number( seedText ) ) ) {

	throw new Error( 'Usage: node generate-alternative.js hierarchy|lattice|relaxation output.f32 [integer-seed] [atlas.png]' );

}

const { generate } = await import( `./alternatives/${method}.js` );
const start = performance.now();
const volume = generate( { seed: Number( seedText ) } );
console.log( `${method}: ${( performance.now() - start ).toFixed( 3 )} ms (CPU generation only)` );
fs.writeFileSync( output, Buffer.from( volume.buffer, volume.byteOffset, volume.byteLength ) );

if ( atlas ) {

	// Scalar comparison atlas: repeat the scalar into xyz; w uses FAST's spatial
	// decorrelation offset. This does not construct optimized vector2 samples.
	const image = new PNG( { width: 256, height: 4096 } );
	for ( let z = 0; z < 32; z ++ ) for ( let y = 0; y < 128; y ++ ) for ( let x = 0; x < 256; x ++ ) {

		const right = x >= 128;
		const sx = right ? ( x - 128 + 128 - 49 ) % 128 : x;
		const sy = right ? ( y + 128 - 83 ) % 128 : y;
		const value = Math.floor( volume[ z * 16384 + sy * 128 + sx ] * 256 );
		const index = ( ( z * 128 + y ) * 256 + x ) * 4;
		image.data[ index ] = image.data[ index + 1 ] = image.data[ index + 2 ] = value;
		image.data[ index + 3 ] = 255;

	}

	fs.writeFileSync( atlas, PNG.sync.write( image ) );

}
