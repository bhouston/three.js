import {
	createModel,
	sampleLatents,
	forwardDecoderInput,
	buildIBLInput
} from '../../../../examples/jsm/neural-appearance/NeuralAppearanceModel.js';
import { createLatentGrid } from '../../../../examples/jsm/neural/NeuralGridModel.js';
import { forwardMLP } from '../../../../examples/jsm/neural/NeuralMLP.js';

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'Neural', () => {

		QUnit.module( 'NeuralAppearanceModel', () => {

			QUnit.test( 'creates a multiresolution latent grid pyramid', ( assert ) => {

				const random = () => 0.5;
				const model = createModel( {
					levels: 3,
					baseResolution: 2,
					targetResolution: 8,
					hiddenSize: 8,
					outputFeatures: { emission: true, opacity: true }
				}, random );

				assert.strictEqual( model.latentGrids.length, 3, 'creates one grid per level' );
				assert.strictEqual( model.latentGrids[ 0 ].width, 2, 'coarsest grid uses baseResolution' );
				assert.strictEqual( model.latentGrids[ 2 ].width, 8, 'finest grid uses targetResolution' );
				assert.strictEqual( model.latentGrids[ 0 ].channels, 4, 'each level contributes 4 channels' );
				assert.ok( model.emissionHead !== null, 'creates emission head' );
				assert.ok( model.opacityHead !== null, 'creates opacity head' );
				assert.ok( model.iblHead !== null, 'creates required IBL head' );
				assert.ok( model.indirectRadianceHead !== null, 'creates required radiance IBL head' );
				assert.ok( model.indirectIrradianceHead !== null, 'creates required irradiance IBL head' );
				assert.strictEqual( model.iblHead.layers[ 1 ].outputSize, 4, 'IBL query head predicts direction and roughness' );
				assert.strictEqual( model.indirectRadianceHead.layers[ 0 ].inputSize, 22, 'radiance IBL head consumes latents, view, and radiance' );
				assert.strictEqual( model.indirectIrradianceHead.layers[ 0 ].inputSize, 22, 'irradiance IBL head consumes latents, view, and irradiance' );
				assert.strictEqual( model.rotationWeights.length, 192, 'allocates 16 * 12 rotation weights' );

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

				const model = createModel( { levels: 4, baseResolution: 8, targetResolution: 8, hiddenSize: 32 }, createRandom( 1 ) );
				const evalRandom = createRandom( 1000 );
				const channelSums = [ 0, 0, 0 ];
				let clampedCount = 0;
				let chromaSum = 0;
				const winningChannels = new Set();

				for ( let i = 0; i < 64; i ++ ) {

					const latents = sampleLatents( model.latentGrids, [ evalRandom(), evalRandom() ] ).output;
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

			QUnit.test( 'samples latents bilinearly across every grid level', ( assert ) => {

				const grids = [
					createLatentGrid( 2, 2, 4, () => 0.5 ),
					createLatentGrid( 2, 2, 4, () => 0.5 )
				];
				const uv = [ 0.25, 0.25 ];
				const run = sampleLatents( grids, uv );

				assert.strictEqual( run.output.length, 8, 'samples 4 latent channels per level' );
				assert.strictEqual( run.levelTaps.length, 2, 'samples every grid level' );
				assert.strictEqual( run.levelTaps[ 0 ].taps.length, 4, 'samples 4 taps per level' );

				const tapWeightSum = run.levelTaps[ 0 ].taps.reduce( ( sum, tap ) => sum + tap.weight, 0 );
				assert.ok( Math.abs( tapWeightSum - 1.0 ) < 1e-12, 'bilinear weights sum to 1' );
				assert.ok( run.output.every( ( value ) => value === 0 ), 'constant random seed initializes zero-valued latents' );

			} );

			QUnit.test( 'evaluates learned-frame decoder input', ( assert ) => {

				const latents = new Array( 16 ).fill( 0 ).map( ( _, i ) => ( i + 1 ) / 20 );
				const rotationWeights = new Array( 192 ).fill( 0 );
				const wi = [ 0, 0, 1 ];
				const wo = [ 0, 0, 1 ];

				const inputRun = forwardDecoderInput( latents, rotationWeights, wi, wo );
				assert.strictEqual( inputRun.output.length, 28, 'produces 28-channel decoder input (16 latents + 2 * 6 frame projections)' );
				assert.deepEqual( inputRun.output.slice( 16, 22 ), [ 0, 0, 1, 0, 0, 1 ], 'projects directions into the first default frame' );
				assert.strictEqual( buildIBLInput( latents, rotationWeights, wo ).length, 22, 'produces 22-channel IBL input (16 latents + 2 * 3 view projections)' );

			} );

		} );

	} );

} );
