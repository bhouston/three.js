import { describe, expect, it } from 'vitest';
import {
	triangleWave,
	positionalEncodingSize,
	computePositionalEncoding
} from '../../../../examples/jsm/ntc/NTCPositionalEncoding.js';

describe( 'Addons > NTC > NTCPositionalEncoding', () => {

	describe( 'triangleWave', () => {

		it( 'has period 1', () => {

			for ( const x of [ 0, 0.1, 0.37, 0.5, 0.9 ] ) {

				expect( triangleWave( x ) ).toBeCloseTo( triangleWave( x + 1 ), 10 );
				expect( triangleWave( x ) ).toBeCloseTo( triangleWave( x - 1 ), 10 );
				expect( triangleWave( x ) ).toBeCloseTo( triangleWave( x + 5 ), 10 );

			}

		} );

		it( 'stays within [-1, 1]', () => {

			for ( let i = 0; i <= 100; i ++ ) {

				const value = triangleWave( i / 33.333 );
				expect( value ).toBeGreaterThanOrEqual( - 1 - 1e-9 );
				expect( value ).toBeLessThanOrEqual( 1 + 1e-9 );

			}

		} );

		it( 'reaches its minimum (-1) at x = 0 and its maximum (1) at x = 0.5', () => {

			expect( triangleWave( 0 ) ).toBeCloseTo( - 1, 10 );
			expect( triangleWave( 0.5 ) ).toBeCloseTo( 1, 10 );

		} );

		it( 'is continuous (no jump) across a period boundary', () => {

			const epsilon = 1e-6;
			expect( triangleWave( 1 - epsilon ) ).toBeCloseTo( triangleWave( 0 ), 4 );
			expect( triangleWave( epsilon ) ).toBeCloseTo( triangleWave( 0 ), 4 );

		} );

		it( 'is symmetric about its peak (0.5) within one period', () => {

			expect( triangleWave( 0.5 - 0.1 ) ).toBeCloseTo( triangleWave( 0.5 + 0.1 ), 10 );

		} );

	} );

	describe( 'positionalEncodingSize', () => {

		it( 'is 4 values per octave (2 axes x 2 phases)', () => {

			expect( positionalEncodingSize( 0 ) ).toBe( 0 );
			expect( positionalEncodingSize( 1 ) ).toBe( 4 );
			expect( positionalEncodingSize( 2 ) ).toBe( 8 );
			expect( positionalEncodingSize( 3 ) ).toBe( 12 );

		} );

	} );

	describe( 'computePositionalEncoding', () => {

		it( 'returns exactly positionalEncodingSize(octaves) values', () => {

			for ( const octaves of [ 1, 2, 3 ] ) {

				const values = computePositionalEncoding( 0.3, 0.7, octaves );
				expect( values.length ).toBe( positionalEncodingSize( octaves ) );

			}

		} );

		it( 'every value stays within [-1, 1]', () => {

			for ( const [ tx, ty ] of [ [ 0, 0 ], [ 0.25, 0.75 ], [ 0.99, 0.01 ], [ 0.5, 0.5 ] ] ) {

				for ( const value of computePositionalEncoding( tx, ty, 3 ) ) {

					expect( value ).toBeGreaterThanOrEqual( - 1 - 1e-9 );
					expect( value ).toBeLessThanOrEqual( 1 + 1e-9 );

				}

			}

		} );

		it( 'matches hand-computed triangleWave calls at each octave/phase/axis, in the documented order', () => {

			const tx = 0.2, ty = 0.6, octaves = 2;
			const values = computePositionalEncoding( tx, ty, octaves );

			// octave 0 (frequency 1)
			expect( values[ 0 ] ).toBeCloseTo( triangleWave( tx ), 10 );
			expect( values[ 1 ] ).toBeCloseTo( triangleWave( tx + 0.25 ), 10 );
			expect( values[ 2 ] ).toBeCloseTo( triangleWave( ty ), 10 );
			expect( values[ 3 ] ).toBeCloseTo( triangleWave( ty + 0.25 ), 10 );

			// octave 1 (frequency 2)
			expect( values[ 4 ] ).toBeCloseTo( triangleWave( tx * 2 ), 10 );
			expect( values[ 5 ] ).toBeCloseTo( triangleWave( tx * 2 + 0.25 ), 10 );
			expect( values[ 6 ] ).toBeCloseTo( triangleWave( ty * 2 ), 10 );
			expect( values[ 7 ] ).toBeCloseTo( triangleWave( ty * 2 + 0.25 ), 10 );

		} );

		it( 'the two phases at the same octave/axis are not identical (they actually carry different information)', () => {

			const values = computePositionalEncoding( 0.2, 0.6, 1 );
			expect( values[ 0 ] ).not.toBeCloseTo( values[ 1 ], 5 );
			expect( values[ 2 ] ).not.toBeCloseTo( values[ 3 ], 5 );

		} );

		it( 'higher octaves vary faster across tx/ty than lower ones', () => {

			// Sweep tx across a small range - the octave-1 (frequency 2) term
			// should change roughly 2x as fast as the octave-0 (frequency 1)
			// term for the same phase.
			const deltaAt = ( octaves, index, txA, txB ) => {

				const a = computePositionalEncoding( txA, 0, octaves )[ index ];
				const b = computePositionalEncoding( txB, 0, octaves )[ index ];
				return Math.abs( b - a );

			};

			const octave0Delta = deltaAt( 2, 0, 0.1, 0.15 );
			const octave1Delta = deltaAt( 2, 4, 0.1, 0.15 );

			expect( octave1Delta ).toBeGreaterThan( octave0Delta );

		} );

		it( 'is deterministic', () => {

			expect( computePositionalEncoding( 0.33, 0.77, 3 ) ).toEqual( computePositionalEncoding( 0.33, 0.77, 3 ) );

		} );

	} );

} );
