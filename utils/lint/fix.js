// ESLint repeats --fix up to 10 passes per run, oxlint does a single pass.
// Rerun oxlint until no more fixes apply so lint-fix matches ESLint's output.

import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const oxlint = fileURLToPath( new URL( '../../node_modules/oxlint/bin/oxlint', import.meta.url ) );
const args = [ oxlint, '--fix', '--format', 'unix', ...process.argv.slice( 2 ) ];

let result, last;

for ( let i = 0; i < 10; i ++ ) {

	result = spawnSync( process.execPath, args, { encoding: 'utf8' } );

	const report = result.stdout.split( '\n' ).sort().join( '\n' );
	if ( report === last ) break;
	last = report;

}

process.stdout.write( result.stdout );
process.stderr.write( result.stderr );
process.exitCode = result.status;
