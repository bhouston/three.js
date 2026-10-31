export const Kind = {
	EQ: 'eq',
	CLOSE_ABS: 'closeAbs',
	CLOSE_REL: 'closeRel',
	GT: 'greaterThan',
	GE: 'greaterThanOrEqual',
	LT: 'lessThan',
	LE: 'lessThanOrEqual'
};

export function diffComponents( actual, expected, tolerance, kind ) {

	const diffs = [];

	for ( let i = 0; i < expected.length; i ++ ) {

		const a = actual[ i ];
		const e = expected[ i ];
		let delta, bad;

		if ( kind === Kind.CLOSE_REL ) {

			delta = Math.abs( a - e ) / Math.max( Math.abs( a ), Math.abs( e ), 1e-12 );
			bad = ! ( delta <= tolerance );

		} else if ( kind === Kind.CLOSE_ABS ) {

			delta = Math.abs( a - e );
			bad = ! ( delta <= tolerance );

		} else if ( kind === Kind.GT ) {

			bad = ! ( a > e );
			delta = e - a;

		} else if ( kind === Kind.GE ) {

			bad = ! ( a >= e );
			delta = e - a;

		} else if ( kind === Kind.LT ) {

			bad = ! ( a < e );
			delta = a - e;

		} else if ( kind === Kind.LE ) {

			bad = ! ( a <= e );
			delta = a - e;

		} else { // EQ

			delta = Math.abs( a - e );
			bad = a !== e;

		}

		diffs.push( { index: i, actual: a, expected: e, delta, bad } );

	}

	return diffs;

}
