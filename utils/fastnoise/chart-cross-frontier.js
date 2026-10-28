// Focused cross-PR view of the unchanged frontier suite; no rescoring or new baseline.
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const [ scores = 'utils/fastnoise/results/frontier.json', generation = 'utils/fastnoise/results/frontier-generation.json', output = 'utils/fastnoise/results/cross-frontier-summary' ] = process.argv.slice( 2 );
const result = spawnSync( process.execPath, [ fileURLToPath( new URL( './chart-frontier.js', import.meta.url ) ), scores, generation, output, 'cross' ], { stdio: 'inherit' } );
if ( result.error ) throw result.error;
process.exitCode = result.status ?? 1;
