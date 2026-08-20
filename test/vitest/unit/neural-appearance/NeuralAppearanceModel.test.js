import { describe, expect, it } from 'vitest';
import {
	buildDecoderInput,
	buildIBLInput,
	buildIndirectProbeInput,
	createModel,
	forwardDecoderInput,
	forwardIBLInput,
	getSampleWeightSum,
	sampleLatents,
	unpackIBLOutput,
	wrapIndex
} from '../../../../examples/jsm/neural-appearance/NeuralAppearanceModel.js';
import { cross, dot, normalize } from '../../../../examples/jsm/neural/NeuralVectorMath.js';
import { computeGridLevels } from '../../../../examples/jsm/neural/NeuralGridModel.js';
import {
	IBL_INPUT_SIZE,
	IBL_OUTPUT_SIZE,
	INDIRECT_INPUT_SIZE,
	LATENT_CHANNELS
} from '../../../../examples/jsm/neural-appearance/NeuralAppearanceFormat.js';

// `NeuralAppearanceModel.js` re-exports `dot`/`cross`/`normalize` unchanged
// from `NeuralVectorMath.js` (already covered by NeuralVectorMath.test.js) -
// not re-tested here.

function makeGrid( width, height, channels, data ) {

	return { width, height, channels, data };

}

describe( 'Addons > Neural > NeuralAppearance > NeuralAppearanceModel', () => {

	describe( 'wrapIndex', () => {

		it( 'passes through in-range values unchanged', () => {

			expect( wrapIndex( 0, 4 ) ).toBe( 0 );
			expect( wrapIndex( 3, 4 ) ).toBe( 3 );

		} );

		it( 'wraps negative values to the top of the range', () => {

			expect( wrapIndex( - 1, 4 ) ).toBe( 3 );
			expect( wrapIndex( - 5, 4 ) ).toBe( 3 );

		} );

		it( 'wraps values >= size back to zero', () => {

			expect( wrapIndex( 4, 4 ) ).toBe( 0 );
			expect( wrapIndex( 5, 4 ) ).toBe( 1 );

		} );

	} );

	describe( 'sampleLatents', () => {

		it( 'concatenates every grid\'s channel count into the flat output length', () => {

			const grids = [
				makeGrid( 4, 4, 1, new Array( 16 ).fill( 0 ) ),
				makeGrid( 4, 4, 2, new Array( 32 ).fill( 0 ) ),
				makeGrid( 4, 4, 3, new Array( 48 ).fill( 0 ) )
			];

			const result = sampleLatents( grids, [ 0.5, 0.5 ] );

			expect( result.output.length ).toBe( 1 + 2 + 3 );

		} );

		it( 'reproduces a texel\'s exact value when sampled at its center', () => {

			// 4x4, 2 channels, row-major (y * width + x) * channels
			const data = [];
			for ( let i = 0; i < 4 * 4 * 2; i ++ ) data.push( i + 1 );
			const grid = makeGrid( 4, 4, 2, data );

			const x = 2;
			const y = 1;
			const uv = [ ( x + 0.5 ) / grid.width, ( y + 0.5 ) / grid.height ];

			const result = sampleLatents( [ grid ], uv );
			const offset = ( y * grid.width + x ) * grid.channels;

			expect( result.output[ 0 ] ).toBeCloseTo( data[ offset ], 10 );
			expect( result.output[ 1 ] ).toBeCloseTo( data[ offset + 1 ], 10 );

		} );

		it( 'averages two horizontally-adjacent texels at the midpoint UV', () => {

			// 2x2, 1 channel: (0,0)=10 (1,0)=20 (0,1)=30 (1,1)=40
			const grid = makeGrid( 2, 2, 1, [ 10, 20, 30, 40 ] );

			// Midpoint between texel (0,0) and (1,0) centers, at row y=0's center.
			const uv = [ 1 / grid.width, 0.5 / grid.height ];

			const result = sampleLatents( [ grid ], uv );

			expect( result.output[ 0 ] ).toBeCloseTo( ( 10 + 20 ) / 2, 10 );

		} );

		it( 'wraps addressing at the UV=0 edge to blend the last and first texel', () => {

			// 2x2, 1 channel: (0,0)=10 (1,0)=20 (0,1)=30 (1,1)=40
			const grid = makeGrid( 2, 2, 1, [ 10, 20, 30, 40 ] );

			// uv[0] = 0 => x = -0.5 => x0 = -1 (wraps to width-1 = 1), tx = 0.5
			// uv[1] = 0.5/height (texel row 0 center) => ty = 0
			const uv = [ 0, 0.5 / grid.height ];

			const result = sampleLatents( [ grid ], uv );

			// Blend of wrapped texel (1,0)=20 and texel (0,0)=10, i.e. average.
			expect( result.output[ 0 ] ).toBeCloseTo( ( 20 + 10 ) / 2, 10 );

		} );

	} );

	describe( 'forwardDecoderInput / buildDecoderInput', () => {

		it( 'reduces to the identity basis when rotationWeights are all zero', () => {

			const latents = new Array( LATENT_CHANNELS ).fill( 0 ).map( ( _, i ) => i * 0.1 );
			const rotationWeights = new Array( LATENT_CHANNELS * 12 ).fill( 0 );
			const wi = [ 0.3, 0.4, 0.5 ];
			const wo = [ 0.1, 0.2, 0.9 ];

			const result = forwardDecoderInput( latents, rotationWeights, wi, wo );

			expect( result.output.length ).toBe( latents.length + 12 );

			// rawN = (0,0,1), rawT = (1,0,0) => n=(0,0,1), t=(1,0,0), b=(0,1,0):
			// identical for both frames since rotationWeights is all zero.
			const expectedFrame = [ wi[ 0 ], wi[ 1 ], wi[ 2 ], wo[ 0 ], wo[ 1 ], wo[ 2 ] ];

			expect( result.output.slice( latents.length, latents.length + 6 ) ).toEqual(
				expectedFrame.map( ( v ) => expect.closeTo( v, 10 ) )
			);
			expect( result.output.slice( latents.length + 6, latents.length + 12 ) ).toEqual(
				expectedFrame.map( ( v ) => expect.closeTo( v, 10 ) )
			);

			for ( const frame of result.frames ) {

				expect( frame.n ).toEqual( [ 0, 0, 1 ] );
				expect( frame.t ).toEqual( [ 1, 0, 0 ] );
				expect( frame.b ).toEqual( [ 0, 1, 0 ] );

			}

		} );

		it( 'matches buildDecoderInput\'s output array exactly', () => {

			const latents = new Array( LATENT_CHANNELS ).fill( 0.2 );
			const rotationWeights = new Array( LATENT_CHANNELS * 12 ).fill( 0 );
			const wi = [ 0, 1, 0 ];
			const wo = [ 1, 0, 0 ];

			expect( buildDecoderInput( latents, rotationWeights, wi, wo ) ).toEqual(
				forwardDecoderInput( latents, rotationWeights, wi, wo ).output
			);

		} );

		it( 'produces unit-length, mutually orthogonal (n, t, b) frames for nonzero rotationWeights', () => {

			const latents = new Array( LATENT_CHANNELS ).fill( 0 ).map( ( _, i ) => Math.sin( i + 1 ) * 0.5 );
			let seed = 0;
			const rotationWeights = new Array( LATENT_CHANNELS * 12 ).fill( 0 ).map( () => {

				seed += 1;
				return Math.sin( seed * 1.37 ) * 0.7;

			} );
			const wi = [ 0.2, - 0.4, 0.7 ];
			const wo = [ - 0.6, 0.1, 0.3 ];

			const result = forwardDecoderInput( latents, rotationWeights, wi, wo );

			expect( result.frames.length ).toBe( 2 );

			for ( const frame of result.frames ) {

				expect( Math.hypot( ...frame.n ) ).toBeCloseTo( 1, 8 );
				expect( Math.hypot( ...frame.t ) ).toBeCloseTo( 1, 8 );
				expect( Math.hypot( ...frame.b ) ).toBeCloseTo( 1, 8 );

				// b = normalize(cross(n, t)) is perpendicular to both n and t by
				// construction; n and t themselves are each independently
				// normalized (not Gram-Schmidt orthogonalized), so n.t is not
				// guaranteed to be zero.
				expect( dot( frame.n, frame.b ) ).toBeCloseTo( 0, 6 );
				expect( dot( frame.t, frame.b ) ).toBeCloseTo( 0, 6 );

				const expectedB = normalize( cross( frame.n, frame.t ) );
				expect( frame.b[ 0 ] ).toBeCloseTo( expectedB[ 0 ], 8 );
				expect( frame.b[ 1 ] ).toBeCloseTo( expectedB[ 1 ], 8 );
				expect( frame.b[ 2 ] ).toBeCloseTo( expectedB[ 2 ], 8 );

			}

		} );

	} );

	describe( 'forwardIBLInput / buildIBLInput', () => {

		it( 'has length IBL_INPUT_SIZE and content latents + per-frame dot(wo, t/b/n) with wi fixed at (0,0,1)', () => {

			const latents = new Array( LATENT_CHANNELS ).fill( 0 ).map( ( _, i ) => i * 0.05 );
			const rotationWeights = new Array( LATENT_CHANNELS * 12 ).fill( 0 );
			const wo = [ 0.4, - 0.3, 0.8 ];

			const forward = forwardIBLInput( latents, rotationWeights, wo );
			const built = buildIBLInput( latents, rotationWeights, wo );

			expect( built ).toEqual( forward.output );
			expect( built.length ).toBe( IBL_INPUT_SIZE );

			// With rotationWeights all zero, every frame reduces to the identity
			// basis (t=(1,0,0), b=(0,1,0), n=(0,0,1)) regardless of `wi`, so each
			// frame contributes exactly wo's own components.
			const expected = latents.concat( [ wo[ 0 ], wo[ 1 ], wo[ 2 ], wo[ 0 ], wo[ 1 ], wo[ 2 ] ] );

			for ( let i = 0; i < expected.length; i ++ ) {

				expect( built[ i ] ).toBeCloseTo( expected[ i ], 10 );

			}

		} );

	} );

	describe( 'buildIndirectProbeInput', () => {

		it( 'concatenates latents, wo, and probe into a single flat array', () => {

			const latents = new Array( LATENT_CHANNELS ).fill( 0.1 );
			const wo = [ 0.2, 0.3, 0.4 ];
			const probe = [ 0.5, 0.6, 0.7 ];

			const input = buildIndirectProbeInput( latents, wo, probe );

			expect( input ).toEqual( [ ...latents, ...wo, ...probe ] );
			expect( input.length ).toBe( INDIRECT_INPUT_SIZE );

		} );

		it( 'defaults probe to [1, 1, 1] when omitted', () => {

			const latents = new Array( LATENT_CHANNELS ).fill( 0 );
			const wo = [ 0, 0, 1 ];

			const input = buildIndirectProbeInput( latents, wo );

			expect( input.slice( - 3 ) ).toEqual( [ 1, 1, 1 ] );

		} );

		it( 'defaults probe to [1, 1, 1] when explicitly undefined', () => {

			const latents = new Array( LATENT_CHANNELS ).fill( 0 );
			const wo = [ 0, 0, 1 ];

			const input = buildIndirectProbeInput( latents, wo, undefined );

			expect( input.slice( - 3 ) ).toEqual( [ 1, 1, 1 ] );

		} );

		it( 'the length check can no longer actually fire - it validates against a size *derived from* latents.length, not a fixed constant', () => {

			// buildIndirectProbeInput's expected size is `latents.length + 6`
			// (see NeuralAppearanceFormat.js's computeIndirectInputSize) - not
			// the fixed INDIRECT_INPUT_SIZE constant, which is only correct
			// for the default levels (see NeuralAppearanceModel.levels-bug.
			// test.js). That makes the check a tautology relative to
			// `latents`' own length (any length is self-consistent by
			// construction), and since `wo[0..2]`/`probe[0..2]` are always
			// individually pushed regardless of the input arrays' actual
			// length (a too-short `wo` just contributes `undefined` entries,
			// not fewer array slots), this throw path is effectively
			// unreachable for any input shape - which is fine: the real
			// invariant it used to catch (a caller passing `latents` sized
			// for the wrong model) is a `levels`/model-consistency concern
			// this function has no way to check on its own once the constant
			// it used to compare against became one this function itself
			// computed the model with.
			const latents = new Array( LATENT_CHANNELS + 1 ).fill( 0 );

			expect( () => buildIndirectProbeInput( latents, [ 0, 0, 1 ], [ 1, 1, 1 ] ) ).not.toThrow();

		} );

	} );

	describe( 'unpackIBLOutput', () => {

		it( 'throws when values.length !== IBL_OUTPUT_SIZE', () => {

			expect( () => unpackIBLOutput( [ 1, 2, 3 ] ) ).toThrow();
			expect( () => unpackIBLOutput( [ 1, 2, 3, 4, 5 ] ) ).toThrow();

		} );

		it( 'returns the normalized direction and sigmoid(roughness logit) for a valid input', () => {

			const values = [ 1, 2, 2, 0.75 ];

			const result = unpackIBLOutput( values );

			expect( result.direction ).toEqual( normalize( values.slice( 0, 3 ) ) );
			expect( Math.hypot( ...result.direction ) ).toBeCloseTo( 1, 10 );

			const expectedRoughness = 1 / ( 1 + Math.exp( - values[ 3 ] ) );
			expect( result.roughness ).toBeCloseTo( expectedRoughness, 10 );

			expect( values.length ).toBe( IBL_OUTPUT_SIZE );

		} );

	} );

	describe( 'getSampleWeightSum', () => {

		it( 'sums explicit weights and counts weightless samples as 1', () => {

			const samples = [
				{ weight: 2 },
				{},
				{ weight: 0.5 },
				{ weight: undefined }
			];

			expect( getSampleWeightSum( samples ) ).toBeCloseTo( 2 + 1 + 0.5 + 1, 10 );

		} );

		it( 'returns 0 for an empty sample list', () => {

			expect( getSampleWeightSum( [] ) ).toBe( 0 );

		} );

	} );

	describe( 'createModel', () => {

		const baseOptions = { hiddenSize: 8, baseResolution: 2, growthFactor: 2 };

		it( 'builds every head and a zero-initialized rotationWeights array', () => {

			const model = createModel( baseOptions, () => 0.5 );

			expect( model.decoder ).toBeDefined();
			expect( model.iblHead ).toBeDefined();
			expect( model.indirectRadianceHead ).toBeDefined();
			expect( model.indirectIrradianceHead ).toBeDefined();

			expect( model.rotationWeights.length ).toBe( LATENT_CHANNELS * 12 );
			expect( model.rotationWeights.every( ( v ) => v === 0 ) ).toBe( true );

		} );

		it( 'sizes resolutions/latentGrids from computeGridLevels', () => {

			const model = createModel( baseOptions, () => 0.5 );
			const expectedResolutions = computeGridLevels( baseOptions.baseResolution, baseOptions.growthFactor, model.levels );

			expect( model.resolutions ).toEqual( expectedResolutions );
			expect( model.latentGrids.length ).toBe( expectedResolutions.length );

		} );

		it( 'leaves emissionHead/opacityHead null when outputFeatures is unset', () => {

			const model = createModel( baseOptions, () => 0.5 );

			expect( model.emissionHead ).toBeNull();
			expect( model.opacityHead ).toBeNull();

		} );

		it( 'builds small no-hidden-layer emission/opacity heads when requested', () => {

			const options = { ...baseOptions, outputFeatures: { emission: true, opacity: true } };
			const model = createModel( options, () => 0.5 );

			expect( model.emissionHead ).not.toBeNull();
			expect( model.emissionHead.layers.length ).toBe( 1 );
			expect( model.emissionHead.layers[ 0 ].inputSize ).toBe( LATENT_CHANNELS );
			expect( model.emissionHead.layers[ 0 ].outputSize ).toBe( 3 );

			expect( model.opacityHead ).not.toBeNull();
			expect( model.opacityHead.layers.length ).toBe( 1 );
			expect( model.opacityHead.layers[ 0 ].inputSize ).toBe( LATENT_CHANNELS );
			expect( model.opacityHead.layers[ 0 ].outputSize ).toBe( 1 );

		} );

		it( 'does not throw during decorrelateLinearRgbHead and leaves the decoder\'s RGB output shape intact', () => {

			const options = { ...baseOptions, outputFeatures: { emission: true, opacity: true } };

			expect( () => createModel( options, () => 0.5 ) ).not.toThrow();

			const model = createModel( options, () => 0.5 );
			const lastLayer = model.decoder.layers[ model.decoder.layers.length - 1 ];

			expect( lastLayer.outputSize ).toBe( 3 );

		} );

	} );

} );
