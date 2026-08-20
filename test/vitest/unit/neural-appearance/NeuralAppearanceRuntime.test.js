import { describe, expect, it } from 'vitest';
import { CHANNELS_PER_LEVEL } from '../../../../examples/jsm/neural-appearance/NeuralAppearanceFormat.js';
import {
	evaluateNeuralAppearanceJson,
	evaluateNeuralAppearanceOutputs,
	evaluateIBLHead,
	evaluateNeuralPrefilteredIBL,
	evaluateNeuralIBLWhiteFurnace,
	integrateNeuralBRDFWhiteFurnace,
	sampleRuntimeLatents,
	evaluateDecoderLayers,
	mixArray
} from '../../../../examples/jsm/neural-appearance/NeuralAppearanceRuntime.js';

// Minimal hand-built manifest helpers. Weights are deliberately zeroed
// wherever the test doesn't care about the linear-algebra itself, so every
// MLP head collapses to "output = bias" - a plain, independently-verifiable
// identity/pass-through case (no dependency on reading the source's own
// matmul formula back at itself). Where the matmul itself is under test
// (evaluateDecoderLayers' 2-layer case below), the expected numbers are
// worked out by hand from the standard dense-layer definition
// (out = W.x + b), not by re-running the function under test.

function makeLayer( inputSize, outputSize, weights, biases, activation ) {

	return { inputSize, outputSize, weights, biases, activation };

}

function zeros( n ) {

	return new Array( n ).fill( 0 );

}

// One 2x2 level, CHANNELS_PER_LEVEL(=4) channels/texel, all latent values 0
// (so decoder input content from the grid is uniformly zero and only the
// bias of each zero-weight layer determines its output).
function zeroLevel() {

	return { width: 2, height: 2, wrap: 'repeat', data: zeros( 2 * 2 * CHANNELS_PER_LEVEL ) };

}

// A single-level (LATENT_CHANNELS = CHANNELS_PER_LEVEL = 4) manifest with a
// bias-only brdf decoder and bias-only IBL head - the smallest shape
// evaluateNeuralAppearanceOutputs will accept without throwing.
function makeBaseManifest( { brdfBias = [ 0.2, 0.3, 0.4 ], iblBias = [ 1, 2, 2, 0.75 ] } = {} ) {

	const latentChannels = CHANNELS_PER_LEVEL; // 1 level
	const decoderInputSize = latentChannels + 12;
	const iblInputSize = latentChannels + 6;

	return {
		latents: { levels: [ zeroLevel() ] },
		outputs: {
			brdf: {
				rotation: { weights: zeros( latentChannels * 12 ) },
				layers: [ makeLayer( decoderInputSize, 3, zeros( 3 * decoderInputSize ), brdfBias.slice(), 'linear' ) ],
				outputActivation: { type: 'raw' }
			},
			ibl: {
				layers: [ makeLayer( iblInputSize, 4, zeros( 4 * iblInputSize ), iblBias.slice(), 'linear' ) ],
				outputActivation: { type: 'raw' }
			}
		}
	};

}

describe( 'Addons > Neural > NeuralAppearance > NeuralAppearanceRuntime', () => {

	describe( 'evaluateDecoderLayers', () => {

		it( 'a zero-weight layer passes the bias through unchanged under a "raw" output activation', () => {

			const layers = [ makeLayer( 2, 3, zeros( 6 ), [ 0.2, - 0.3, 0.4 ], 'linear' ) ];

			const result = evaluateDecoderLayers( layers, [ 5, - 7 ], { type: 'raw' } );

			expect( result ).toEqual( [ 0.2, - 0.3, 0.4 ] );

		} );

		it( 'computes a hand-worked two-layer forward pass (matmul + bias, relu hidden layer, linear output)', () => {

			// Layer 1: identity weights, biases [-0.5, 0.5], relu.
			//   pre-activation = W.x + b = [1*1 + 0*(-1) - 0.5, 0*1 + 1*(-1) + 0.5] = [0.5, -0.5]
			//   relu -> [0.5, 0]
			// Layer 2: weights [2, 3], bias [0], linear (no relu clamp inside the loop).
			//   value = 2*0.5 + 3*0 + 0 = 1
			const layer1 = makeLayer( 2, 2, [ 1, 0, 0, 1 ], [ - 0.5, 0.5 ], 'relu' );
			const layer2 = makeLayer( 2, 1, [ 2, 3 ], [ 0 ], 'linear' );

			const result = evaluateDecoderLayers( [ layer1, layer2 ], [ 1, - 1 ], { type: 'raw' } );

			expect( result[ 0 ] ).toBeCloseTo( 1, 10 );

		} );

		it( 'the default output activation ("linear") is actually a zero-floor, not a true identity - negative values are clamped to 0', () => {

			const layers = [ makeLayer( 1, 1, [ 0 ], [ - 2 ], 'linear' ) ];

			// No outputActivation argument -> defaults to { type: 'linear' }, which
			// falls through every named branch (scaledSigmoid/exp/sigmoid/raw) to
			// the final `values.map(v => Math.max(v, 0))`.
			const result = evaluateDecoderLayers( layers, [ 0 ] );

			expect( result ).toEqual( [ 0 ] );

		} );

		it( 'the default output activation passes a non-negative value through unchanged', () => {

			const layers = [ makeLayer( 1, 1, [ 0 ], [ 3.5 ], 'linear' ) ];

			expect( evaluateDecoderLayers( layers, [ 0 ] ) ).toEqual( [ 3.5 ] );

		} );

		it( 'scaledSigmoid applies 1/(1+exp(-x)) scaled by the given factor (default scale 1)', () => {

			const layers = [ makeLayer( 1, 1, [ 0 ], [ 0.75 ], 'linear' ) ];

			const resultDefault = evaluateDecoderLayers( layers, [ 0 ], { type: 'scaledSigmoid' } );
			const resultScaled = evaluateDecoderLayers( layers, [ 0 ], { type: 'scaledSigmoid', scale: 4 } );

			const expectedSigmoid = 1 / ( 1 + Math.exp( - 0.75 ) ); // hand-computed logistic function
			expect( resultDefault[ 0 ] ).toBeCloseTo( expectedSigmoid, 10 );
			expect( resultScaled[ 0 ] ).toBeCloseTo( 4 * expectedSigmoid, 10 );

		} );

		it( 'exp applies exp(x + offset), offset defaulting to 0', () => {

			const layers = [ makeLayer( 1, 1, [ 0 ], [ 1 ], 'linear' ) ];

			const resultDefault = evaluateDecoderLayers( layers, [ 0 ], { type: 'exp' } );
			const resultOffset = evaluateDecoderLayers( layers, [ 0 ], { type: 'exp', offset: 0.5 } );

			expect( resultDefault[ 0 ] ).toBeCloseTo( Math.E, 10 ); // exp(1)
			expect( resultOffset[ 0 ] ).toBeCloseTo( Math.exp( 1.5 ), 10 );

		} );

		it( 'sigmoid applies the plain logistic function', () => {

			const layers = [ makeLayer( 1, 1, [ 0 ], [ - 1.2 ], 'linear' ) ];

			const result = evaluateDecoderLayers( layers, [ 0 ], { type: 'sigmoid' } );

			expect( result[ 0 ] ).toBeCloseTo( 1 / ( 1 + Math.exp( 1.2 ) ), 10 );

		} );

		it( 'raw passes negative values through with no clamping at all', () => {

			const layers = [ makeLayer( 1, 1, [ 0 ], [ - 6.25 ], 'linear' ) ];

			expect( evaluateDecoderLayers( layers, [ 0 ], { type: 'raw' } ) ).toEqual( [ - 6.25 ] );

		} );

	} );

	describe( 'sampleRuntimeLatents', () => {

		it( 'reproduces a texel\'s exact stored value when sampled at its texel center (1 level)', () => {

			// 2x2 grid, CHANNELS_PER_LEVEL(=4) channels/texel, row-major layout.
			const data = [];
			for ( let i = 0; i < 2 * 2 * CHANNELS_PER_LEVEL; i ++ ) data.push( i + 1 );
			const level = { width: 2, height: 2, wrap: 'repeat', data };

			// texel (1, 0) center.
			const uv = [ 1.5 / 2, 0.5 / 2 ];
			const latents = sampleRuntimeLatents( { latents: { levels: [ level ] } }, uv );

			const offset = ( 0 * 2 + 1 ) * CHANNELS_PER_LEVEL;
			for ( let c = 0; c < CHANNELS_PER_LEVEL; c ++ ) {

				expect( latents[ c ] ).toBeCloseTo( data[ offset + c ], 10 );

			}

		} );

		it( 'averages two horizontally-adjacent texels at the exact midpoint UV', () => {

			// 2x2, channel 0 values: (0,0)=10 (1,0)=20 (0,1)=30 (1,1)=40, other channels 0.
			const data = zeros( 2 * 2 * CHANNELS_PER_LEVEL );
			data[ ( 0 * 2 + 0 ) * CHANNELS_PER_LEVEL ] = 10;
			data[ ( 0 * 2 + 1 ) * CHANNELS_PER_LEVEL ] = 20;
			data[ ( 1 * 2 + 0 ) * CHANNELS_PER_LEVEL ] = 30;
			data[ ( 1 * 2 + 1 ) * CHANNELS_PER_LEVEL ] = 40;
			const level = { width: 2, height: 2, wrap: 'repeat', data };

			// Midpoint between texel (0,0) and (1,0) centers, at row 0's center.
			const uv = [ 1 / 2, 0.5 / 2 ];
			const latents = sampleRuntimeLatents( { latents: { levels: [ level ] } }, uv );

			expect( latents[ 0 ] ).toBeCloseTo( ( 10 + 20 ) / 2, 10 );

		} );

		it( 'concatenates each level\'s CHANNELS_PER_LEVEL channels in level order', () => {

			const levelA = { width: 2, height: 2, wrap: 'repeat', data: new Array( 2 * 2 * CHANNELS_PER_LEVEL ).fill( 1 ) };
			const levelB = { width: 2, height: 2, wrap: 'repeat', data: new Array( 2 * 2 * CHANNELS_PER_LEVEL ).fill( 9 ) };

			const latents = sampleRuntimeLatents( { latents: { levels: [ levelA, levelB ] } }, [ 0.5, 0.5 ] );

			expect( latents.length ).toBe( 2 * CHANNELS_PER_LEVEL );
			for ( let c = 0; c < CHANNELS_PER_LEVEL; c ++ ) expect( latents[ c ] ).toBeCloseTo( 1, 10 );
			for ( let c = 0; c < CHANNELS_PER_LEVEL; c ++ ) expect( latents[ CHANNELS_PER_LEVEL + c ] ).toBeCloseTo( 9, 10 );

		} );

		it( '"repeat" wrap blends the last and first texel across the UV=0 seam; "clamp" does not', () => {

			// 2x2, channel 0: (0,0)=10 (1,0)=20 (0,1)=30 (1,1)=40.
			const data = zeros( 2 * 2 * CHANNELS_PER_LEVEL );
			data[ ( 0 * 2 + 0 ) * CHANNELS_PER_LEVEL ] = 10;
			data[ ( 0 * 2 + 1 ) * CHANNELS_PER_LEVEL ] = 20;
			data[ ( 1 * 2 + 0 ) * CHANNELS_PER_LEVEL ] = 30;
			data[ ( 1 * 2 + 1 ) * CHANNELS_PER_LEVEL ] = 40;

			// uv[0] = 0 => x = -0.5 => x0 = -1, tx = 0.5; row 0's center for uv[1].
			const uv = [ 0, 0.5 / 2 ];

			const repeatLatents = sampleRuntimeLatents( { latents: { levels: [ { width: 2, height: 2, wrap: 'repeat', data } ] } }, uv );
			const clampLatents = sampleRuntimeLatents( { latents: { levels: [ { width: 2, height: 2, wrap: 'clamp', data } ] } }, uv );

			// repeat: blend of wrapped texel (1,0)=20 and texel (0,0)=10 -> average.
			expect( repeatLatents[ 0 ] ).toBeCloseTo( ( 20 + 10 ) / 2, 10 );

			// clamp: both taps (x0=-1 and x0+1=0) clamp to the same texel x=0 (=10),
			// so the blend degenerates to exactly that texel's value regardless of tx.
			expect( clampLatents[ 0 ] ).toBeCloseTo( 10, 10 );

			// The two wrap modes must disagree here - this is the whole point of
			// the test (independent, physically-meaningful edge case: sampling at
			// the wrap seam of an asymmetric texel pattern).
			expect( repeatLatents[ 0 ] ).not.toBeCloseTo( clampLatents[ 0 ], 5 );

		} );

	} );

	describe( 'mixArray', () => {

		it( 'linearly interpolates each component independently', () => {

			expect( mixArray( [ 0, 10, - 4 ], [ 10, 0, 4 ], 0.25 ) ).toEqual( [ 2.5, 7.5, - 2 ] );

		} );

		it( 'amount = 0 returns a\'s values, amount = 1 returns b\'s values', () => {

			const a = [ 1, 2, 3 ];
			const b = [ 4, 5, 6 ];

			expect( mixArray( a, b, 0 ) ).toEqual( a );
			expect( mixArray( a, b, 1 ) ).toEqual( b );

		} );

	} );

	describe( 'evaluateNeuralAppearanceOutputs / evaluateNeuralAppearanceJson (full forward pass)', () => {

		it( 'a bias-only decoder + IBL head produces the biases directly, independent of wi/wo/uv', () => {

			const json = makeBaseManifest( { brdfBias: [ 0.2, 0.3, 0.4 ], iblBias: [ 1, 2, 2, 0.75 ] } );
			const reference = { uv: [ 0.5, 0.5 ], wi: [ 0, 0, 1 ], wo: [ 0.3, 0.4, 0.866 ] };

			const outputs = evaluateNeuralAppearanceOutputs( json, reference );

			expect( outputs.brdf ).toEqual( [ 0.2, 0.3, 0.4 ] );

			// unpackIBLOutput: direction = normalize([1, 2, 2]) (a 1:2:2 vector has
			// length 3, this is a hand-checkable Pythagorean triple: 1+4+4=9=3^2),
			// roughness = logistic(0.75), computed here from first principles, not
			// by importing the source's own sigmoid/normalize helpers.
			const len = Math.sqrt( 1 * 1 + 2 * 2 + 2 * 2 ); // = 3
			expect( outputs.ibl.direction[ 0 ] ).toBeCloseTo( 1 / len, 10 );
			expect( outputs.ibl.direction[ 1 ] ).toBeCloseTo( 2 / len, 10 );
			expect( outputs.ibl.direction[ 2 ] ).toBeCloseTo( 2 / len, 10 );
			expect( outputs.ibl.roughness ).toBeCloseTo( 1 / ( 1 + Math.exp( - 0.75 ) ), 10 );

			expect( evaluateNeuralAppearanceJson( json, reference ) ).toEqual( outputs.brdf );

		} );

		it( 'omits indirectRadiance/indirectIrradiance/indirect/emission/opacity when the manifest has no such heads', () => {

			const json = makeBaseManifest();
			const outputs = evaluateNeuralAppearanceOutputs( json, { wi: [ 0, 0, 1 ], wo: [ 0, 0, 1 ] } );

			expect( outputs.indirectRadiance ).toBeUndefined();
			expect( outputs.indirectIrradiance ).toBeUndefined();
			expect( outputs.indirect ).toBeUndefined();
			expect( outputs.emission ).toBeUndefined();
			expect( outputs.opacity ).toBeUndefined();

		} );

		it( 'bias-only emission/opacity heads read from the raw latent vector and produce the biases', () => {

			const json = makeBaseManifest();
			const latentChannels = CHANNELS_PER_LEVEL;

			json.outputs.emission = {
				layers: [ makeLayer( latentChannels, 3, zeros( 3 * latentChannels ), [ 0.1, 0.2, 0.3 ], 'linear' ) ],
				outputActivation: { type: 'raw' }
			};
			json.outputs.opacity = {
				layers: [ makeLayer( latentChannels, 1, zeros( latentChannels ), [ - 4 ], 'linear' ) ],
				outputActivation: { type: 'sigmoid' }
			};

			const outputs = evaluateNeuralAppearanceOutputs( json, { wi: [ 0, 0, 1 ], wo: [ 0, 0, 1 ] } );

			expect( outputs.emission ).toEqual( [ 0.1, 0.2, 0.3 ] );
			expect( outputs.opacity ).toBeCloseTo( 1 / ( 1 + Math.exp( 4 ) ), 10 ); // logistic(-4)

		} );

		it( 'sums indirectRadiance and indirectIrradiance element-wise into "indirect"', () => {

			const json = makeBaseManifest();
			const latentChannels = CHANNELS_PER_LEVEL;
			const probeInputSize = latentChannels + 6;

			json.outputs.indirectRadiance = {
				layers: [ makeLayer( probeInputSize, 3, zeros( 3 * probeInputSize ), [ 0.1, 0.2, 0.3 ], 'linear' ) ],
				outputActivation: { type: 'raw' }
			};
			json.outputs.indirectIrradiance = {
				layers: [ makeLayer( probeInputSize, 3, zeros( 3 * probeInputSize ), [ 0.05, 0.05, 0.05 ], 'linear' ) ],
				outputActivation: { type: 'raw' }
			};

			const outputs = evaluateNeuralAppearanceOutputs( json, { wi: [ 0, 0, 1 ], wo: [ 0, 0, 1 ] } );

			expect( outputs.indirectRadiance ).toEqual( [ 0.1, 0.2, 0.3 ] );
			expect( outputs.indirectIrradiance ).toEqual( [ 0.05, 0.05, 0.05 ] );
			expect( outputs.indirect[ 0 ] ).toBeCloseTo( 0.15, 10 );
			expect( outputs.indirect[ 1 ] ).toBeCloseTo( 0.25, 10 );
			expect( outputs.indirect[ 2 ] ).toBeCloseTo( 0.35, 10 );

		} );

		it( 'indirectRadiance reads reference.iblIncoming in preference to reference.prefilteredSpecular, falling back to [1,1,1] when neither is given', () => {

			const json = makeBaseManifest();
			const latentChannels = CHANNELS_PER_LEVEL; // 4
			const probeInputSize = latentChannels + 6; // 10: latents(4) + wo(3) + probe(3)
			const probeRedIndex = latentChannels + 3; // first probe component (R)

			// Isolate the probe's R channel: weight 1 only on that one input index,
			// output channel 0, everything else 0 - so output[0] == probe[0]
			// exactly, letting each fallback level be checked independently of the
			// (irrelevant here) latent/wo content.
			const weights = zeros( 3 * probeInputSize );
			weights[ 0 * probeInputSize + probeRedIndex ] = 1;

			json.outputs.indirectRadiance = {
				layers: [ makeLayer( probeInputSize, 3, weights, [ 0, 0, 0 ], 'linear' ) ],
				outputActivation: { type: 'raw' }
			};

			const wiwo = { wi: [ 0, 0, 1 ], wo: [ 0, 0, 1 ] };

			const withBoth = evaluateNeuralAppearanceOutputs( json, { ...wiwo, iblIncoming: [ 0.7, 0, 0 ], prefilteredSpecular: [ 0.3, 0, 0 ] } );
			const withSpecularOnly = evaluateNeuralAppearanceOutputs( json, { ...wiwo, prefilteredSpecular: [ 0.9, 0, 0 ] } );
			const withNeither = evaluateNeuralAppearanceOutputs( json, wiwo );

			expect( withBoth.indirectRadiance[ 0 ] ).toBeCloseTo( 0.7, 10 );
			expect( withSpecularOnly.indirectRadiance[ 0 ] ).toBeCloseTo( 0.9, 10 );
			expect( withNeither.indirectRadiance[ 0 ] ).toBeCloseTo( 1, 10 );

		} );

	} );

	describe( 'evaluateIBLHead (standalone)', () => {

		it( 'unpacks a bias-only IBL head\'s output into a normalized direction and a logistic-mapped roughness', () => {

			const json = makeBaseManifest( { iblBias: [ 0, 0, 5, - 2 ] } );
			const latents = zeros( CHANNELS_PER_LEVEL );

			const result = evaluateIBLHead( json, latents, [ 0, 0, 1 ] );

			expect( result.direction ).toEqual( [ 0, 0, 1 ] ); // (0,0,5) normalized
			expect( result.roughness ).toBeCloseTo( 1 / ( 1 + Math.exp( 2 ) ), 10 ); // logistic(-2)

		} );

	} );

	describe( 'evaluateNeuralPrefilteredIBL', () => {

		it( 'returns [0,0,0] when the manifest has no indirect heads at all', () => {

			const json = makeBaseManifest();

			expect( evaluateNeuralPrefilteredIBL( json, { wi: [ 0, 0, 1 ], wo: [ 0, 0, 1 ] } ) ).toEqual( [ 0, 0, 0 ] );

		} );

		it( 'returns the same "indirect" sum evaluateNeuralAppearanceOutputs computes when indirect heads are present', () => {

			const json = makeBaseManifest();
			const latentChannels = CHANNELS_PER_LEVEL;
			const probeInputSize = latentChannels + 6;

			json.outputs.indirectRadiance = {
				layers: [ makeLayer( probeInputSize, 3, zeros( 3 * probeInputSize ), [ 0.2, 0.2, 0.2 ], 'linear' ) ],
				outputActivation: { type: 'raw' }
			};

			const reference = { wi: [ 0, 0, 1 ], wo: [ 0, 0, 1 ] };
			const result = evaluateNeuralPrefilteredIBL( json, reference );

			expect( result[ 0 ] ).toBeCloseTo( 0.2, 10 );
			expect( result[ 1 ] ).toBeCloseTo( 0.2, 10 );
			expect( result[ 2 ] ).toBeCloseTo( 0.2, 10 );

		} );

	} );

	describe( 'evaluateNeuralIBLWhiteFurnace', () => {

		it( 'forces iblIncoming/iblIrradiance/prefilteredSpecular to [1,1,1], overriding whatever the reference sample already carried', () => {

			const json = makeBaseManifest();
			const latentChannels = CHANNELS_PER_LEVEL;
			const probeInputSize = latentChannels + 6;
			const probeRedIndex = latentChannels + 3;

			const weights = zeros( 3 * probeInputSize );
			weights[ 0 * probeInputSize + probeRedIndex ] = 1; // output[0] == probe's R component

			json.outputs.indirectRadiance = {
				layers: [ makeLayer( probeInputSize, 3, weights, [ 0, 0, 0 ], 'linear' ) ],
				outputActivation: { type: 'raw' }
			};

			// The reference explicitly asks for a non-white probe (0.2); the white
			// furnace evaluation must ignore that and use 1 instead.
			const reference = { wi: [ 0, 0, 1 ], wo: [ 0, 0, 1 ], iblIncoming: [ 0.2, 0, 0 ] };

			const result = evaluateNeuralIBLWhiteFurnace( json, reference );

			expect( result[ 0 ] ).toBeCloseTo( 1, 10 );

		} );

	} );

	describe( 'integrateNeuralBRDFWhiteFurnace', () => {

		it( 'for a wi/wo-independent (bias-only) BRDF, the hemisphere-cosine integral collapses exactly to bias * PI, for any sample count', () => {

			// total[c] accumulates brdf[c]/sampleCount * PI over `sampleCount`
			// samples. If brdf[c] is the same constant `bias` on every sample
			// (true here since the decoder's weights are all zero, so its output
			// cannot depend on the per-sample wi the integrator generates), then
			// total[c] = sampleCount * (bias / sampleCount * PI) = bias * PI
			// exactly, regardless of how many samples were taken. This is a
			// closed-form result derived independently of the source, not a
			// re-statement of its loop.
			const bias = 0.2;
			const json = makeBaseManifest( { brdfBias: [ bias, bias, bias ] } );
			const reference = { uv: [ 0.5, 0.5 ], wo: [ 0, 0, 1 ] };

			for ( const sampleCount of [ 1, 8, 64 ] ) {

				const total = integrateNeuralBRDFWhiteFurnace( json, reference, sampleCount );

				for ( let c = 0; c < 3; c ++ ) {

					expect( total[ c ] ).toBeCloseTo( bias * Math.PI, 8 );

				}

			}

		} );

		it( 'defaults sampleCount to 64 when omitted (matches the explicit-64 result)', () => {

			const json = makeBaseManifest( { brdfBias: [ 0.1, 0.1, 0.1 ] } );
			const reference = { uv: [ 0.5, 0.5 ], wo: [ 0, 0, 1 ] };

			const withDefault = integrateNeuralBRDFWhiteFurnace( json, reference );
			const withExplicit64 = integrateNeuralBRDFWhiteFurnace( json, reference, 64 );

			expect( withDefault ).toEqual( withExplicit64 );

		} );

	} );

} );
