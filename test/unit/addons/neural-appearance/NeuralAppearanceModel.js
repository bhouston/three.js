import { describe, test, expect } from 'vitest';
import {
	createModel,
	sampleLatents,
	forwardDecoderInput,
	buildIBLInput
} from '../../../../examples/jsm/neural-appearance/NeuralAppearanceModel.js';
import { createLatentGrid } from '../../../../examples/jsm/neural/NeuralGridModel.js';
import { forwardMLP } from '../../../../examples/jsm/neural/NeuralMLP.js';

describe( 'Addons', () => {

	describe( 'Neural', () => {

		describe( 'NeuralAppearanceModel', () => {

			test( 'creates a multiresolution latent grid pyramid', () => {

				const random = () => 0.5;
				const model = createModel( {
					levels: 3,
					baseResolution: 2,
					targetResolution: 8,
					hiddenSize: 8,
					outputFeatures: { emission: true, opacity: true }
				}, random );

				expect( model.latentGrids.length ).toBe( 3 );
				expect( model.latentGrids[ 0 ].width ).toBe( 2 );
				expect( model.latentGrids[ 2 ].width ).toBe( 8 );
				expect( model.latentGrids[ 0 ].channels ).toBe( 4 );
				expect( model.emissionHead !== null ).toBeTruthy();
				expect( model.opacityHead !== null ).toBeTruthy();
				expect( model.iblHead !== null ).toBeTruthy();
				expect( model.indirectRadianceHead !== null ).toBeTruthy();
				expect( model.indirectIrradianceHead !== null ).toBeTruthy();
				expect( model.iblHead.layers[ 1 ].outputSize ).toBe( 4 );
				expect( model.indirectRadianceHead.layers[ 0 ].inputSize ).toBe( 18 );
				expect( model.indirectIrradianceHead.layers[ 0 ].inputSize ).toBe( 18 );
				expect( model.rotationWeights.length ).toBe( 144 );

			} );

			test( 'initializes decoder RGB near gray with all channels active', () => {

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

				expect( clampedCount ).toBe( 0 );
				expect( meanRgb > 0.15 && meanRgb < 0.5 ).toBeTruthy();
				expect( maxMeanDiff < 0.12 ).toBeTruthy();
				expect( chromaSum / 64 > 0.03 ).toBeTruthy();
				expect( winningChannels.size >= 2 ).toBeTruthy();

			} );

			test( 'samples latents bilinearly across every grid level', () => {

				const grids = [
					createLatentGrid( 2, 2, 4, () => 0.5 ),
					createLatentGrid( 2, 2, 4, () => 0.5 )
				];
				const uv = [ 0.25, 0.25 ];
				const run = sampleLatents( grids, uv );

				expect( run.output.length ).toBe( 8 );
				expect( run.levelTaps.length ).toBe( 2 );
				expect( run.levelTaps[ 0 ].taps.length ).toBe( 4 );

				const tapWeightSum = run.levelTaps[ 0 ].taps.reduce( ( sum, tap ) => sum + tap.weight, 0 );
				expect( Math.abs( tapWeightSum - 1.0 ) < 1e-12 ).toBeTruthy();
				expect( run.output.every( ( value ) => value === 0 ) ).toBeTruthy();

			} );

			test( 'evaluates learned-frame decoder input', () => {

				const latents = new Array( 16 ).fill( 0 ).map( ( _, i ) => ( i + 1 ) / 20 );
				const rotationWeights = new Array( 192 ).fill( 0 );
				const wi = [ 0, 0, 1 ];
				const wo = [ 0, 0, 1 ];

				const inputRun = forwardDecoderInput( latents, rotationWeights, wi, wo );
				expect( inputRun.output.length ).toBe( 28 );
				expect( inputRun.output.slice( 16, 22 ) ).toEqual( [ 0, 0, 1, 0, 0, 1 ] );
				expect( buildIBLInput( latents, rotationWeights, wo ).length ).toBe( 22 );

			} );

		} );

	} );

} );
