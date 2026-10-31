// Preserve the example-name CLI while delegating lifecycle, retries and sharding to Playwright.
import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';

const require = createRequire( import.meta.url );
const examples = [];
const args = [];
const env = { ...process.env };

const cli = process.argv.slice( 2 );

for ( let i = 0; i < cli.length; i ++ ) {

	const arg = cli[ i ];

	if ( arg === '--make' ) env.THREE_SCREENSHOTS = '1';
	else if ( arg === '--webgpu' ) env.THREE_WEBGPU = '1';
	else if ( arg.startsWith( '-' ) ) {

		args.push( arg );
		if ( [ '--workers', '-j', '--shard', '--retries', '--timeout', '--grep', '-g', '--grep-invert', '--reporter', '--project', '--repeat-each' ].includes( arg ) ) {

			if ( cli[ i + 1 ] === undefined ) throw new Error( `Missing value for ${ arg }` );
			args.push( cli[ ++ i ] );

		}

	} else examples.push( arg.replace( /\.html$/, '' ) );

}

env.THREE_EXAMPLES = JSON.stringify( examples );

const child = spawn( process.execPath, [ require.resolve( '@playwright/test/cli' ), 'test', '--config=test/e2e/playwright.config.js', ...args ], { stdio: 'inherit', env } );
child.on( 'error', error => {

	console.error( error );
	process.exitCode = 1;

} );
child.on( 'exit', code => {

	process.exitCode = code ?? 1;

} );
