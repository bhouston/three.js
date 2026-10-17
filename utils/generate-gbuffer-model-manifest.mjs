import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const repositoryRoot = resolve( dirname( fileURLToPath( import.meta.url ) ), '..' );
const manifestPath = 'examples/models/gltf/gbuffer-model-manifest.json';

// Use the index, rather than a directory walk: ignored and untracked downloads
// must not become choices in a commit-hosted example. NUL separators preserve
// spaces, Unicode, and other legal filename characters without Git quoting.
export function getTrackedModelPaths( root = repositoryRoot ) {

	const options = { cwd: root, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024, windowsHide: true };
	const tracked = process.platform === 'win32'
		? execFileSync( process.env.ComSpec || 'cmd.exe', [ '/d', '/s', '/c', 'git-dedup.cmd ls-files --cached --full-name -z' ], options )
		: execFileSync( 'git-dedup', [ 'ls-files', '--cached', '--full-name', '-z' ], options );

	// Default string sorting is locale independent. Do not lowercase the paths:
	// servers may be case sensitive even when this generator runs on Windows.
	return [ ...new Set( tracked.split( '\0' ).filter( path => /\.(gltf|glb)$/i.test( path ) ) ) ].sort();

}

export function generateModelManifest( { root = repositoryRoot, check = false } = {} ) {

	const paths = getTrackedModelPaths( root );
	const output = resolve( root, manifestPath );
	const content = JSON.stringify( paths, null, '\t' ) + '\n';

	if ( check ) {

		let existing;

		try {

			// A Windows checkout may expand LF according to core.autocrlf.
			existing = readFileSync( output, 'utf8' ).replace( /\r\n/g, '\n' );

		} catch ( error ) {

			if ( error.code !== 'ENOENT' ) throw error;

		}

		if ( existing !== content ) {

			throw new Error( `${ manifestPath } is missing or stale. Run node utils/generate-gbuffer-model-manifest.mjs.` );

		}

	} else {

		mkdirSync( dirname( output ), { recursive: true } );
		writeFileSync( output, content );

	}

	return paths;

}

if ( process.argv[ 1 ] && resolve( process.argv[ 1 ] ) === fileURLToPath( import.meta.url ) ) {

	try {

		const args = process.argv.slice( 2 );
		if ( args.some( argument => argument !== '--check' ) ) throw new Error( 'Usage: node utils/generate-gbuffer-model-manifest.mjs [--check]' );
		const check = args.includes( '--check' );
		const paths = generateModelManifest( { check } );
		process.stdout.write( `${ check ? 'Verified' : 'Generated' } ${ paths.length } tracked glTF/GLB models in ${ manifestPath }\n` );

	} catch ( error ) {

		process.stderr.write( `${ error.message }\n` );
		process.exitCode = 1;

	}

}
