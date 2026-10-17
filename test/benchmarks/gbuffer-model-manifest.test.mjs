import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import process from 'node:process';
import test from 'node:test';
import { generateModelManifest, getTrackedModelPaths } from '../../utils/generate-gbuffer-model-manifest.mjs';

function runGitDedup( command, cwd ) {

	const options = { cwd, encoding: 'utf8', windowsHide: true };
	return process.platform === 'win32'
		? execFileSync( process.env.ComSpec || 'cmd.exe', [ '/d', '/s', '/c', `git-dedup.cmd ${ command }` ], options )
		: execFileSync( 'git-dedup', command.split( ' ' ), options );

}

test( 'checked-in manifest covers the current complete tracked model tree', () => {

	const paths = generateModelManifest( { check: true } );
	assert.ok( paths.length > 0 );
	assert.equal( new Set( paths ).size, paths.length );

	for ( const path of paths ) {

		assert.ok( /\.(gltf|glb)$/i.test( path ), path );
		assert.ok( ! path.includes( '\\' ), path );
		assert.ok( ! path.startsWith( '/' ) && ! path.startsWith( '../' ), path );

	}

} );

test( 'discovers nested mixed-case assets outside the usual model directory and rejects stale coverage', t => {

	const root = mkdtempSync( join( tmpdir(), 'three-gbuffer-manifest-' ) );
	t.after( () => rmSync( root, { recursive: true, force: true } ) );
	runGitDedup( 'init --quiet', root );

	const paths = [
		'docs/fixtures/Outside.GLB',
		'examples/models/gltf/Nested variant/Scene.gLtF',
		'examples/models/gltf/scene.glb'
	];

	for ( const path of [ ...paths, 'examples/models/gltf/not-a-model.glb.png' ] ) {

		mkdirSync( dirname( join( root, path ) ), { recursive: true } );
		writeFileSync( join( root, path ), '{}' );

	}

	runGitDedup( 'add --all', root );
	writeFileSync( join( root, 'untracked.glb' ), '{}' );
	assert.deepEqual( getTrackedModelPaths( root ), paths );
	assert.throws( () => generateModelManifest( { root, check: true } ), /missing or stale/ );
	assert.deepEqual( generateModelManifest( { root } ), paths );
	const output = join( root, 'examples/models/gltf/gbuffer-model-manifest.json' );
	const first = readFileSync( output, 'utf8' );
	generateModelManifest( { root } );
	assert.equal( readFileSync( output, 'utf8' ), first, 'regeneration must be byte-identical' );
	generateModelManifest( { root, check: true } );
	writeFileSync( output, first.replace( /\n/g, '\r\n' ) );
	generateModelManifest( { root, check: true } );

	writeFileSync( join( root, 'new-model.Gltf' ), '{}' );
	runGitDedup( 'add --all', root );
	assert.throws( () => generateModelManifest( { root, check: true } ), /missing or stale/ );
	generateModelManifest( { root } );
	generateModelManifest( { root, check: true } );

} );
