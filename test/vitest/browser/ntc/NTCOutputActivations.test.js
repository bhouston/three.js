import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { float } from 'three/tsl';
import {
	applyChannelActivation,
	channelActivationDerivativeFromOutput
} from '../../../../examples/jsm/ntc/NTCOutputActivations.js';
import { withTestRenderer, evalScalar } from '../helpers/webgpuEval.js';

// NeuralOutputActivations builds TSL node graphs - `applyChannelActivation`
// takes a node and returns a node, it never runs any JS math itself. The
// only way to know it's numerically correct (as opposed to just "builds a
// graph without throwing") is to actually run it on the GPU, which is what
// this file does, backed by plain-JS reference implementations of the same
// formulas.

function referenceForward( z, activation ) {

	if ( activation === 'sigmoid' ) return 1 / ( 1 + Math.exp( - z ) );
	if ( activation === 'tanh' ) return Math.tanh( z );
	if ( activation === 'softplus' ) return Math.log( 1 + Math.exp( z ) );

	return z; // linear / unrecognized falls through unchanged

}

function referenceDerivativeFromOutput( a, activation ) {

	if ( activation === 'sigmoid' ) return a * ( 1 - a );
	if ( activation === 'tanh' ) return 1 - a * a;
	if ( activation === 'softplus' ) return 1 - Math.exp( - a );

	return 1;

}

describe( 'Addons > Neural > NeuralOutputActivations (real WebGPU)', () => {

	const getRenderer = withTestRenderer( { beforeAll, afterAll } );

	const activations = [ 'sigmoid', 'tanh', 'softplus', 'linear' ];
	// Includes values comfortably inside softplus's overflow-prone region
	// (exp(z) would overflow a naive log(1+e^z) around z >~ 88) - the whole
	// point of the numerically-stable formulation documented in the source.
	const zValues = [ - 50, - 5, - 1, 0, 1, 5, 50, 200 ];

	describe( 'applyChannelActivation (forward, z -> a)', () => {

		for ( const activation of activations ) {

			for ( const z of zValues ) {

				it( `${activation}(${z}) matches the reference formula`, async () => {

					const gpuResult = await evalScalar( getRenderer(), () => applyChannelActivation( float( z ), activation ) );
					const expected = referenceForward( z, activation );

					expect( gpuResult ).toBeCloseTo( expected, 3 );

				} );

			}

		}

		it( 'stays finite for softplus at large positive z (no exp() overflow)', async () => {

			const gpuResult = await evalScalar( getRenderer(), () => applyChannelActivation( float( 200 ), 'softplus' ) );

			expect( Number.isFinite( gpuResult ) ).toBe( true );
			expect( gpuResult ).toBeCloseTo( 200, 1 ); // softplus(z) ~= z for large z

		} );

		it( 'passes an unrecognized activation name through unchanged (linear fallback)', async () => {

			const gpuResult = await evalScalar( getRenderer(), () => applyChannelActivation( float( 3.25 ), 'not-a-real-activation' ) );

			expect( gpuResult ).toBeCloseTo( 3.25, 5 );

		} );

	} );

	describe( 'channelActivationDerivativeFromOutput (backward, da/dz from a)', () => {

		for ( const activation of activations ) {

			for ( const z of zValues ) {

				it( `d/dz ${activation} at a = ${activation}(${z}) matches the reference formula`, async () => {

					const a = referenceForward( z, activation );
					const gpuResult = await evalScalar( getRenderer(), () => channelActivationDerivativeFromOutput( float( a ), activation ) );
					const expected = referenceDerivativeFromOutput( a, activation );

					expect( gpuResult ).toBeCloseTo( expected, 3 );

				} );

			}

		}

	} );

	it( 'the forward/backward pair is numerically consistent under finite differencing', async () => {

		// Cross-check applyChannelActivation and channelActivationDerivativeFromOutput
		// against each other (not just against the reference formulas above):
		// da/dz estimated by finite-differencing the forward activation should
		// match the analytic derivative computed from its own output.
		const renderer = getRenderer();
		const activation = 'sigmoid';
		const z = 0.7;
		const epsilon = 1e-3;

		const aPlus = await evalScalar( renderer, () => applyChannelActivation( float( z + epsilon ), activation ) );
		const aMinus = await evalScalar( renderer, () => applyChannelActivation( float( z - epsilon ), activation ) );
		const finiteDifference = ( aPlus - aMinus ) / ( 2 * epsilon );

		const a = await evalScalar( renderer, () => applyChannelActivation( float( z ), activation ) );
		const analyticDerivative = await evalScalar( renderer, () => channelActivationDerivativeFromOutput( float( a ), activation ) );

		expect( analyticDerivative ).toBeCloseTo( finiteDifference, 2 );

	} );

} );
