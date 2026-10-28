// Full-stream validation, independently of the numerical scorer.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';

const [ modelDirectory, gpuDirectory ] = process.argv.slice( 2 );
if ( ! modelDirectory || ! gpuDirectory ) throw new Error( 'Usage: verify-streams.js CPU_DATA_DIR GPU_DATA_DIR' );
const manifest = JSON.parse( fs.readFileSync( path.join( modelDirectory, 'streams.json' ) ) );
const checks = [];
for ( const file of fs.readdirSync( gpuDirectory ).filter( file => file.endsWith( '.f32' ) ) ) {

	const bytes = fs.readFileSync( path.join( gpuDirectory, file ) );
	assert.equal( bytes.length, 159 * 128 * 128 * 4 );
	const hash = createHash( 'sha256' ).update( bytes ).digest( 'hex' );
	assert.equal( hash, manifest.methods.find( row => row.file === file )?.sha256, file );
	checks.push( { file, sha256: hash } );

}

assert.equal( checks.length, 18 );
console.log( JSON.stringify( { verifiedSamples: 18 * 159 * 128 * 128, exactCPUAndGPU: true, checks }, null, 2 ) );
