// Sequential quality-screen runner. CPU wall times here are exploratory only.
import fs from 'node:fs';
import path from 'node:path';
const [ output, planFile ] = process.argv.slice( 2 );
const plan = JSON.parse( fs.readFileSync( planFile, 'utf8' ) );
fs.mkdirSync( output, { recursive: true } );
const report = [];
for ( const item of plan ) {

	if ( ! /^[a-zA-Z0-9_-]+$/.test( item.name ) || ! [ 'threshold-refine', 'stratified', 'multifeature' ].includes( item.method ) ) throw new Error( 'Unknown experiment.' );
	const { generate } = await import( `./${item.method}.js` );
	const options = { ...item.options };
	if ( item.initial ) {

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
