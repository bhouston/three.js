import {
	float, vec2, vec3,
	parabola, gain, pcurve, sinc,
	oscSine, oscSquare, oscTriangle, oscSawtooth,
	rotate,
	floatBitsToInt, floatBitsToUint, intBitsToFloat, uintBitsToFloat,
	int, uint
} from 'three/tsl';
import { gpuTest } from './gpu-test-utils.js';

// Coverage for standalone TSL "curve"/remap-shape helpers (src/nodes/math/MathUtils.js),
// timer-driven oscillators (src/nodes/utils/Oscillators.js), the rotate()
// utility (src/nodes/utils/RotateNode.js), and bit-reinterpretation casts
// (src/nodes/math/BitcastNode.js). Every expected value below is derived
// independently (hand-computed from each function's own documented formula,
// or from plain JS Math), never by re-running the same TSL expression under
// test -- see TSLMath.tests.js's file header for why that matters
// (https://ben3d.ca/blog/the-rise-of-test-theater).
export default QUnit.module( 'TSL', () => {

	QUnit.module( 'curve/remap-shape helpers', () => {

		gpuTest( 'parabola() maps [0,1] corners to 0 and the center to 1', ( { assert } ) => {

			// parabola(x, k) == (4*x*(1-x))^k -- from the function's own doc
			// comment/formula, computed independently in JS.
			assert.closeAbs( parabola( float( 0 ), float( 1 ) ), float( 0 ), 1e-5, 'parabola(0, 1) == 0 -- left corner' );
			assert.closeAbs( parabola( float( 1 ), float( 1 ) ), float( 0 ), 1e-5, 'parabola(1, 1) == 0 -- right corner' );
			assert.closeAbs( parabola( float( 0.5 ), float( 1 ) ), float( 1 ), 1e-5, 'parabola(0.5, 1) == 1 -- center' );

			// k=2 squares the parabola's shape: parabola(0.5, k) is always 1
			// (1^k == 1), but a non-center point should differ from the k=1 case.
			const p25k1 = Math.pow( 4 * 0.25 * ( 1 - 0.25 ), 1 );
			const p25k2 = Math.pow( 4 * 0.25 * ( 1 - 0.25 ), 2 );
			assert.closeAbs( parabola( float( 0.25 ), float( 1 ) ), float( p25k1 ), 1e-5, 'parabola(0.25, 1) matches the hand-computed formula' );
			assert.closeAbs( parabola( float( 0.25 ), float( 2 ) ), float( p25k2 ), 1e-5, 'parabola(0.25, 2) matches the hand-computed formula (different from k=1)' );

		} );

		gpuTest( 'gain() keeps the endpoints and midpoint fixed, k=1 is the identity', ( { assert } ) => {

			// gain()'s doc comment: "k=1 is the identity curve" -- checked at
			// several points, not just the fixed ones.
			assert.closeAbs( gain( float( 0.2 ), float( 1 ) ), float( 0.2 ), 1e-4, 'gain(0.2, k=1) == 0.2 (identity)' );
			assert.closeAbs( gain( float( 0.7 ), float( 1 ) ), float( 0.7 ), 1e-4, 'gain(0.7, k=1) == 0.7 (identity)' );

			// Regardless of k, gain() fixes 0, 0.5 and 1 (a documented property
			// of this remap shape).
			assert.closeAbs( gain( float( 0 ), float( 3 ) ), float( 0 ), 1e-5, 'gain(0, k) == 0 for any k' );
			assert.closeAbs( gain( float( 1 ), float( 3 ) ), float( 1 ), 1e-5, 'gain(1, k) == 1 for any k' );
			assert.closeAbs( gain( float( 0.5 ), float( 3 ) ), float( 0.5 ), 1e-4, 'gain(0.5, k) == 0.5 for any k' );

		} );

		gpuTest( 'pcurve() matches its own documented formula and vanishes at the domain edges', ( { assert } ) => {

			// pcurve(x, a, b) == (x^a / (x^a + (1-x)^b))^(1/a) -- from the
			// function's own doc comment, computed independently in JS.
			const x = 0.3, a = 2, b = 3;
			const expected = Math.pow( Math.pow( x, a ) / ( Math.pow( x, a ) + Math.pow( 1 - x, b ) ), 1 / a );
			assert.closeAbs( pcurve( float( x ), float( a ), float( b ) ), float( expected ), 1e-4, 'pcurve(0.3, 2, 3) matches the hand-computed formula' );

			assert.closeAbs( pcurve( float( 0 ), float( 2 ), float( 3 ) ), float( 0 ), 1e-4, 'pcurve(0, a, b) == 0' );
			assert.closeAbs( pcurve( float( 1 ), float( 2 ), float( 3 ) ), float( 1 ), 1e-4, 'pcurve(1, a, b) == 1' );

		} );

		gpuTest( 'sinc() starts and ends at zero over one full bounce period', ( { assert } ) => {

			// sinc(x, k) == sin(PI*(k*x-1)) / (PI*(k*x-1)) -- from the function's
			// own doc comment. At x=0 the argument is -PI, giving sin(-PI)/(-PI) == 0.
			assert.closeAbs( sinc( float( 0 ), float( 1 ) ), float( 0 ), 1e-4, 'sinc(0, k=1) == 0' );

			// At x == 1/k the argument is exactly 0 -- naively that's sin(0)/0,
			// but sinc()'s removable-singularity guard (see MathUtils.js) returns
			// the analytic limit of 1 instead of evaluating the 0/0 division, so
			// this is exercised at full precision, not just approximately.
			assert.closeAbs( sinc( float( 1 ), float( 1 ) ), float( 1 ), 1e-4, 'sinc(1/k, k) == 1 -- the sinc() peak' );

			// At x == 2/k the argument is +PI, giving sin(PI)/PI == 0 again.
			assert.closeAbs( sinc( float( 2 ), float( 1 ) ), float( 0 ), 1e-3, 'sinc(2/k, k) == 0 again' );

		} );

	} );

	QUnit.module( 'timer-driven oscillators', () => {

		gpuTest( 'oscSine/oscSquare/oscTriangle/oscSawtooth at known phases', ( { assert } ) => {

			// Every oscillator here is fed an explicit `t`, never the default
			// (real-time) timer -- that keeps every expected value a pure,
			// independently hand-computed function of t.

			// oscSine(t) == 0.5*sin(2*PI*(t+0.75)) + 0.5 -- the +0.75 phase
			// offset puts the trough at t=0 and the peak at t=0.5 (not the
			// t=0.25/t=0.75 a naive sin() phase might suggest).
			assert.closeAbs( oscSine( float( 0 ) ), float( 0.5 * Math.sin( 2 * Math.PI * 0.75 ) + 0.5 ), 1e-4, 'oscSine(0)' );
			assert.closeAbs( oscSine( float( 0.5 ) ), float( 1 ), 1e-4, 'oscSine(0.5) reaches its peak of 1' );
			assert.closeAbs( oscSine( float( 0 ) ), float( 0 ), 1e-4, 'oscSine(0) reaches its trough of 0' );

			// oscSquare(t) == round(fract(t)) -- 0 for the first half of each
			// unit period, 1 for the second half.
			assert.eq( oscSquare( float( 0.2 ) ), float( 0 ), 'oscSquare(0.2) is in the low half' );
			assert.eq( oscSquare( float( 0.8 ) ), float( 1 ), 'oscSquare(0.8) is in the high half' );
			assert.eq( oscSquare( float( 1.2 ) ), float( 0 ), 'oscSquare(1.2) repeats the low half of the next period' );

			// oscTriangle(t) == abs(2*fract(t+0.5)-1) -- 0 at integer t, 1 at
			// the half-integer point, ramping linearly between.
			assert.closeAbs( oscTriangle( float( 0 ) ), float( 0 ), 1e-5, 'oscTriangle(0) is at its trough' );
			assert.closeAbs( oscTriangle( float( 0.5 ) ), float( 1 ), 1e-5, 'oscTriangle(0.5) is at its peak' );
			assert.closeAbs( oscTriangle( float( 0.25 ) ), float( 0.5 ), 1e-4, 'oscTriangle(0.25) is exactly midway up the ramp' );

			// oscSawtooth(t) == fract(t) -- a plain linear ramp per period.
			assert.closeAbs( oscSawtooth( float( 0.3 ) ), float( 0.3 ), 1e-5, 'oscSawtooth(0.3) == 0.3' );
			assert.closeAbs( oscSawtooth( float( 1.3 ) ), float( 0.3 ), 1e-5, 'oscSawtooth(1.3) wraps back to 0.3' );

		} );

	} );

	QUnit.module( 'rotate()', () => {

		gpuTest( 'rotate() on a 2D position by known angles', ( { assert } ) => {

			// A 90-degree (PI/2) counter-clockwise rotation sends +X to +Y.
			const rotated90 = rotate( vec2( 1, 0 ), float( Math.PI / 2 ) );
			assert.closeAbs( rotated90, vec2( 0, 1 ), 1e-4, 'rotate((1,0), PI/2) == (0,1)' );

			// A full 2*PI rotation is the identity (up to floating-point drift).
			const rotatedFull = rotate( vec2( 3, -2 ), float( Math.PI * 2 ) );
			assert.closeAbs( rotatedFull, vec2( 3, -2 ), 1e-3, 'rotate(v, 2*PI) returns to the original vector' );

			// Zero rotation is exactly the identity.
			assert.closeAbs( rotate( vec2( 5, 7 ), float( 0 ) ), vec2( 5, 7 ), 1e-5, 'rotate(v, 0) is the identity' );

		} );

		gpuTest( 'rotate() on a 3D position about a single axis matches the 2D case', ( { assert } ) => {

			// Rotating only about Z (x/y Euler angles held at 0) must behave
			// exactly like the 2D rotation above, with Z passed through
			// unchanged -- an independent cross-check between the 2D and 3D
			// code paths inside RotateNode.setup().
			const rotated = rotate( vec3( 1, 0, 5 ), vec3( 0, 0, Math.PI / 2 ) );
			assert.closeAbs( rotated, vec3( 0, 1, 5 ), 1e-4, 'rotating (1,0,5) by PI/2 about Z gives (0,1,5)' );

		} );

		// The X and Y single-axis paths weren't independently exercised when
		// RotateNode's 3D branch had its column/row transpose bug (see
		// tsl-unit-test-findings.md) -- only Z was empirically checked, even
		// though the fix touched all three per-axis matrices identically.
		// These two tests close that gap by applying the same 2D-vs-3D
		// cross-check to X and Y, with the "other two" coordinates held fixed
		// as an independent pass-through check (mirroring the Z test's own Z
		// pass-through of `5`).
		gpuTest( 'rotate() on a 3D position about the X axis only matches the 2D case', ( { assert } ) => {

			// Rotating about X only: (y,z) behaves exactly like the 2D case's
			// (x,y), with X passed through unchanged.
			const rotated = rotate( vec3( 5, 1, 0 ), vec3( Math.PI / 2, 0, 0 ) );
			assert.closeAbs( rotated, vec3( 5, 0, 1 ), 1e-4, 'rotating (5,1,0) by PI/2 about X gives (5,0,1)' );

		} );

		gpuTest( 'rotate() on a 3D position about the Y axis only matches the 2D case', ( { assert } ) => {

			// Rotating about Y only, with Y passed through unchanged. Unlike
			// the X and Z axes, RotateNode's Y-axis matrix is built with its
			// sin/-sin terms swapped relative to the X/Z pattern (see
			// RotateNode.js's rotationYMatrix), so this rotates (x,z) the
			// opposite way round from what the X/Z pattern alone would
			// suggest: x' = x*cos + z*sin, z' = -x*sin + z*cos.
			const rotated = rotate( vec3( 1, 5, 0 ), vec3( 0, Math.PI / 2, 0 ) );
			assert.closeAbs( rotated, vec3( 0, 5, -1 ), 1e-4, 'rotating (1,5,0) by PI/2 about Y gives (0,5,-1)' );

		} );

	} );

	QUnit.module( 'bit-reinterpretation casts', () => {

		gpuTest( 'floatBitsToInt/floatBitsToUint/intBitsToFloat/uintBitsToFloat round trip and match IEEE-754 bit patterns', ( { assert } ) => {

			// 1.0f's IEEE-754 bit pattern is the well-known constant
			// 0x3F800000 == 1065353216 -- an independently known fact about
			// float encoding, not something derived from the node under test.
			assert.eq( floatBitsToInt( float( 1.0 ) ), int( 1065353216 ), 'floatBitsToInt(1.0) == 0x3F800000' );
			assert.eq( floatBitsToUint( float( 1.0 ) ), uint( 1065353216 ), 'floatBitsToUint(1.0) == 0x3F800000' );

			// -2.0f's bit pattern is 0xC0000000, which as a signed 32-bit int
			// is -1073741824 (sign bit set) -- again, an independently known
			// IEEE-754 fact.
			assert.eq( floatBitsToInt( float( -2.0 ) ), int( -1073741824 ), 'floatBitsToInt(-2.0) == 0xC0000000 (signed)' );

			// Round trips: reinterpreting bits out and back must recover the
			// exact original value (bit-reinterpretation is lossless, unlike a
			// numeric cast).
			assert.eq( intBitsToFloat( floatBitsToInt( float( 3.140625 ) ) ), float( 3.140625 ), 'int bit round trip recovers the exact float' );
			assert.eq( uintBitsToFloat( floatBitsToUint( float( -7.5 ) ) ), float( -7.5 ), 'uint bit round trip recovers the exact float' );

		} );

	} );

} );
