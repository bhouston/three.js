/** Independent checks against pinned PR116 published numerical outputs.
 * Usage: node utils/fastnoise/check-cross-frontier.js [frontier.json]
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const published = JSON.parse( readFileSync( new URL( './results/pr116-frontier-published.json', import.meta.url ), 'utf8' ) );
const report = JSON.parse( readFileSync( process.argv[ 2 ] || new URL( './results/frontier.json', import.meta.url ), 'utf8' ) );
assert.equal( report.protocol, 'frontier-v1' );
const names = {
	'annealed': 'latin-anneal001-3000',
	'greedy': 'latin-hybrid-3000',
	'autocorrelation': 'latin-psd-3000'
};
const checked = [];
for ( const [ method, upstream ] of Object.entries( names ) ) {

	for ( const seed of [ 1, 2 ] ) {

		const name = `pr116-${method}-seed${seed}`;
		const row = report.methods[ name ];
		assert.ok( row, `Missing ${name}` );
		assert.match( row.sha256, /^[a-f0-9]{64}$/ );
		const values = Object.values( row.metrics );
		assert.equal( values.length, 74, `${name}: frozen metric count` );
		assert.ok( values.every( value => Number.isFinite( value ) && value >= 0 ), `${name}: finite nonnegative metrics` );
		const baseline = report.methods[ `fast-seed${seed}` ];
		assert.ok( baseline, `Missing baseline seed ${seed}` );
		assert.deepEqual( Object.keys( row.metrics ).sort(), Object.keys( baseline.metrics ).sort() );
		const ratios = Object.fromEntries( Object.entries( row.metrics ).map( ( [ key, value ] ) => [ key, value / baseline.metrics[ key ] ] ) );
		const comparison = report.comparisons[ name ];
		assert.ok( comparison, `Missing comparison ${name}` );
		for ( const [ key, value ] of Object.entries( ratios ) ) assert.equal( comparison.ratios[ key ], value );
		assert.equal( comparison.better, Object.values( ratios ).filter( value => value < 1 ).length );
		assert.equal( comparison.equal, Object.values( ratios ).filter( value => value === 1 ).length );
		assert.equal( comparison.worse, Object.values( ratios ).filter( value => value > 1 ).length );
		assert.equal( comparison.all_strictly_better, Object.values( ratios ).every( value => value < 1 ) );
		assert.equal( comparison.worst_ratio, Math.max( ...Object.values( ratios ) ) );
		assert.equal( comparison.ratios[ comparison.worst_metric ], comparison.worst_ratio );
		const original = published.methods[ `${upstream}-s${seed}` ];
		if ( ! original ) {

			assert.equal( method, 'autocorrelation' );
			assert.equal( seed, 2 );
			checked.push( { name, published: false, note: 'New confirmation: PSD seed 2 was not published in pinned PR116.' } );
			continue;

		}

		assert.equal( Object.keys( original.metrics ).length, 58 );
		const identical = row.sha256 === original.sha256;
		if ( identical ) {

			for ( const [ key, value ] of Object.entries( original.metrics ) ) {

				assert.ok( Math.abs( row.metrics[ key ] - value ) <= 1e-12, `${name}/${key}: disagrees with independent published scorer` );

			}

		}

		checked.push( { name, published: true, byteIdentical: identical, independentlyCheckedMetrics: identical ? 58 : 0,
			note: identical ? 'Published scalar output and compatible metrics agree.' : 'Different output bytes: published numerical equality is not asserted.' } );

	}

}

console.log( JSON.stringify( { sourceCommit: published.source.commit, absoluteTolerance: 1e-12, checked }, null, 2 ) );
