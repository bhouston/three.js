import {
	createModel,
	createLatentGrid,
	sampleLatents,
	forwardDecoderInput,
	buildIBLInput
} from '../../../../examples/jsm/neural/NeuralAppearanceModel.js';
import { forwardMLP } from '../../../../examples/jsm/neural/NeuralAppearanceMLP.js';

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'Neural', () => {

		QUnit.module( 'NeuralAppearanceModel', () => {

			QUnit.test( 'creates multi-level latent mip grids', ( assert ) => {

				const random = () => 0.5;
				const model = createModel( {
					resolution: 4,
					hiddenSize: 8,
					outputFeatures: { emission: true, opacity: true }
				}, random );

				assert.strictEqual( model.latentGrids.length, 3, 'creates 4x4, 2x2, 1x1 grids' );
				assert.strictEqual( model.latentGrids[ 0 ].width, 4, 'base grid width is 4' );
				assert.strictEqual( model.latentGrids[ 2 ].width, 1, 'finest grid width is 1' );
				assert.ok( model.emissionHead !== null, 'creates emission head' );
				assert.ok( model.opacityHead !== null, 'creates opacity head' );
				assert.ok( model.iblHead !== null, 'creates required IBL head' );
				assert.ok( model.indirectHead !== null, 'creates required indirect head' );
				assert.strictEqual( model.iblHead.layers[ 1 ].outputSize, 4, 'IBL query head predicts direction and roughness' );
				assert.strictEqual( model.indirectHead.layers[ 0 ].inputSize, 17, 'indirect head consumes latents, view, radiance, and irradiance' );
				assert.strictEqual( model.rotationWeights.length, 96, 'allocates 8 * 12 rotation weights' );

			} );

			QUnit.test( 'initializes decoder RGB near gray with all channels active', ( assert ) => {

				function createRandom( seed ) {

					let state = seed >>> 0;

					return function random() {

						state = ( state + 0x6D2B79F5 ) | 0;
						let value = Math.imul( state ^ state >>> 15, 1 | state );
						value ^= value + Math.imul( value ^ value >>> 7, 61 | value );

						return ( ( value ^ value >>> 14 ) >>> 0 ) / 4294967296;

					};

				}

				function sampleHemisphere( random ) {

					const z = random();
					const phi = 2 * Math.PI * random();
					const r = Math.sqrt( Math.max( 0, 1 - z * z ) );

					return [ r * Math.cos( phi ), r * Math.sin( phi ), z ];

				}

				const model = createModel( { resolution: 8, hiddenSize: 32 }, createRandom( 1 ) );
				const evalRandom = createRandom( 1000 );
				const channelSums = [ 0, 0, 0 ];
				let clampedCount = 0;
				let chromaSum = 0;
				const winningChannels = new Set();

				for ( let i = 0; i < 64; i ++ ) {

					const latents = sampleLatents( model.latentGrid, [ evalRandom(), evalRandom() ] ).output;
					const input = forwardDecoderInput(
						latents,
						model.rotationWeights,
						sampleHemisphere( evalRandom ),
						sampleHemisphere( evalRandom )
					).output;
					const rgb = forwardMLP( model.decoder, input ).output;
					let brightest = 0;

					for ( let c = 0; c < 3; c ++ ) {

						if ( rgb[ c ] <= 0 ) clampedCount ++;
						channelSums[ c ] += rgb[ c ];
						if ( rgb[ c ] > rgb[ brightest ] ) brightest = c;

					}

					chromaSum += Math.max( rgb[ 0 ], rgb[ 1 ], rgb[ 2 ] ) - Math.min( rgb[ 0 ], rgb[ 1 ], rgb[ 2 ] );
					winningChannels.add( brightest );

				}

				const means = channelSums.map( ( sum ) => sum / 64 );
				const meanRgb = ( means[ 0 ] + means[ 1 ] + means[ 2 ] ) / 3;
				const maxMeanDiff = Math.max( ...means.map( ( mean ) => Math.abs( mean - meanRgb ) ) );

				assert.strictEqual( clampedCount, 0, 'keeps all RGB channels above the non-negative clamp' );
				assert.ok( meanRgb > 0.15 && meanRgb < 0.5, 'starts near a mid-gray BRDF' );
				assert.ok( maxMeanDiff < 0.12, 'keeps RGB channel means balanced instead of a single-channel cast' );
				assert.ok( chromaSum / 64 > 0.03, 'varies hue instead of only lighter and darker gray' );
				assert.ok( winningChannels.size >= 2, 'lets different samples peak in different RGB channels' );

			} );

			QUnit.test( 'samples latents bilinearly', ( assert ) => {

				const grid = createLatentGrid( 2, 2, () => 0.5 );
				const uv = [ 0.25, 0.25 ];
				const run = sampleLatents( grid, uv );

				assert.strictEqual( run.output.length, 8, 'samples 8 latent channels' );
				assert.strictEqual( run.taps.length, 4, 'samples 4 taps' );

				const tapWeightSum = run.taps.reduce( ( sum, tap ) => sum + tap.weight, 0 );
				assert.ok( Math.abs( tapWeightSum - 1.0 ) < 1e-12, 'bilinear weights sum to 1' );
				assert.ok( run.output.every( ( value ) => value === 0 ), 'constant random seed initializes zero-valued latents' );

			} );

			QUnit.test( 'evaluates learned-frame decoder input', ( assert ) => {

				const latents = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 ];
				const rotationWeights = new Array( 96 ).fill( 0 );
				const wi = [ 0, 0, 1 ];
				const wo = [ 0, 0, 1 ];

				const inputRun = forwardDecoderInput( latents, rotationWeights, wi, wo );
				assert.strictEqual( inputRun.output.length, 20, 'produces 20-channel decoder input (8 latents + 2 * 6 frame projections)' );
				assert.deepEqual( inputRun.output.slice( 8, 14 ), [ 0, 0, 1, 0, 0, 1 ], 'projects directions into the first default frame' );
				assert.strictEqual( buildIBLInput( latents, rotationWeights, wo ).length, 14, 'produces 14-channel IBL input (8 latents + 2 * 3 view projections)' );

			} );

		} );

	} );

} );
