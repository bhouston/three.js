// Preserve the example-name CLI while delegating lifecycle, retries and sharding to Playwright.
import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';

const require = createRequire( import.meta.url );
const examples = [];
const args = [];
const env = { ...process.env };

for ( const arg of process.argv.slice( 2 ) ) {

	if ( arg === '--make' ) env.THREE_SCREENSHOTS = '1';
	else if ( arg === '--webgpu' ) env.THREE_WEBGPU = '1';
	else if ( arg.startsWith( '-' ) ) args.push( arg );
	else examples.push( arg.replace( /\.html$/, '' ) );

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
