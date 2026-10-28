// Sequential quality-screen runner. CPU wall times here are exploratory only.
import fs from 'node:fs';
import path from 'node:path';
import { PNG } from 'pngjs';
const [ output, planFile ] = process.argv.slice( 2 );
const plan = JSON.parse( fs.readFileSync( planFile, 'utf8' ) );
fs.mkdirSync( output, { recursive: true } );
const report = [];
for ( const item of plan ) {

	if ( ! /^[a-zA-Z0-9_-]+$/.test( item.name ) || ! [ 'threshold-refine', 'stratified', 'multifeature' ].includes( item.method ) ) throw new Error( 'Unknown experiment.' );
	const { generate } = await import( `./${item.method}.js` );
	const options = { ...item.options };
	if ( item.initial ) {

		if ( item.initial === 'FAST-shipped' && ! fs.existsSync( path.join( output, 'FAST-shipped.f32' ) ) ) {

			const image = PNG.sync.read( fs.readFileSync( new URL( '../../../examples/textures/noises/fast/gauss1-ema01-spatio-temporal.png', import.meta.url ) ) );
			const volume = new Float32Array( 128 * 128 * 32 );
			for ( let z = 0; z < 32; z ++ ) for ( let y = 0; y < 128; y ++ ) for ( let x = 0; x < 128; x ++ ) volume[ ( z * 128 + y ) * 128 + x ] = ( image.data[ ( ( z * 128 + y ) * 256 + x ) * 4 + 2 ] + .5 ) / 256;
			fs.writeFileSync( path.join( output, 'FAST-shipped.f32' ), Buffer.from( volume.buffer ) );

		}

		const bytes = fs.readFileSync( path.join( output, item.initial + '.f32' ) );
		options.initial = new Float32Array( bytes.buffer.slice( bytes.byteOffset, bytes.byteOffset + bytes.byteLength ) );

	}

	const start = performance.now();
	const values = generate( options );
	const milliseconds = performance.now() - start;
	if ( ! values.every( x => Number.isFinite( x ) && x >= 0 && x < 1 ) ) throw new Error( 'Invalid output.' );
	fs.writeFileSync( path.join( output, item.name + '.f32' ), Buffer.from( values.buffer ) );
	report.push( { ...item, milliseconds } );
	fs.writeFileSync( path.join( output, 'cpu-screen-' + path.basename( planFile ) ), JSON.stringify( report, null, 2 ) );
	console.log( item.name, milliseconds );

}
