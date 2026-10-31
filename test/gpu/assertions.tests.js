import { describe, expect, test } from 'vitest';
import { Kind, diffComponents } from './assertions.js';

describe( 'GPU readback comparisons', () => {

	for ( const kind of [ Kind.CLOSE_ABS, Kind.CLOSE_REL ] ) {

		test( `${ kind } rejects NaN in either GPU buffer`, () => {

			for ( const [ actual, expected ] of [[ NaN, 1 ], [ 1, NaN ], [ NaN, NaN ]] ) {

				expect( diffComponents( [ actual ], [ expected ], 0.01, kind )[ 0 ].bad ).toBe( true );

			}

		} );

		test( `${ kind } preserves finite tolerance comparisons`, () => {

			expect( diffComponents( [ 1 ], [ 1.001 ], 0.01, kind )[ 0 ].bad ).toBe( false );
			expect( diffComponents( [ 1 ], [ 2 ], 0.01, kind )[ 0 ].bad ).toBe( true );

		} );

	}

} );
