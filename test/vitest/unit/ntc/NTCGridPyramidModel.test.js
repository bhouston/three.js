import { describe, expect, it } from 'vitest';
import { computeGridLevels, createLatentGrid, createNTCGridPyramidModel } from '../../../../examples/jsm/ntc/training/NTCGridPyramidModel.js';

describe( 'Addons > NTC > NTCGridPyramidModel', () => {

	describe( 'createNTCGridPyramidModel', () => {

		it( 'derives resolutions from computeGridLevels and produces one grid per resolution', () => {

			const options = { channels: 2, levels: 3, baseResolution: 8, mipsPerLevel: 1 };
			const model = createNTCGridPyramidModel( options, () => 0.5 );

			const expectedResolutions = computeGridLevels( options.baseResolution, options.levels, options.mipsPerLevel );

			expect( model.resolutions ).toEqual( expectedResolutions );
			expect( model.grids.length ).toBe( expectedResolutions.length );

			for ( let i = 0; i < model.grids.length; i ++ ) {

				const grid = model.grids[ i ];
				const resolution = expectedResolutions[ i ];

				expect( grid.width ).toBe( resolution );
				expect( grid.height ).toBe( resolution );
				expect( grid.channels ).toBe( options.channels );
				expect( grid.data.length ).toBe( resolution * resolution * options.channels );

			}

		} );

		it( 'matches createLatentGrid content exactly for a given seeded random sequence', () => {

			// A stateful "random" makes the point that model construction visits the
			// grids (and only the grids) in resolution order, consuming exactly as
			// many values as createLatentGrid would on its own.
			let calls = 0;
			const random = () => {

				calls += 1;
				return ( Math.sin( calls ) + 1 ) / 2;

			};

			const options = { channels: 2, levels: 2, baseResolution: 4, mipsPerLevel: 1, hiddenSizes: [ 3 ], outputChannels: 2 };
			const model = createNTCGridPyramidModel( options, random );

			let independentCalls = 0;
			const independentRandom = () => {

				independentCalls += 1;
				return ( Math.sin( independentCalls ) + 1 ) / 2;

			};

			const expectedGrids = model.resolutions.map( ( resolution ) => createLatentGrid( resolution, resolution, options.channels, independentRandom ) );

			for ( let i = 0; i < model.grids.length; i ++ ) {

				expect( Array.from( model.grids[ i ].data ) ).toEqual( Array.from( expectedGrids[ i ].data ) );

			}

		} );

		it( 'sizes the decoder input as channels + 1 (one selected level tap + LOD), regardless of level count', () => {

			for ( const levels of [ 1, 2, 3, 5 ] ) {

				const options = { channels: 4, levels, baseResolution: 8, hiddenSizes: [ 5 ], outputChannels: 3 };
				const model = createNTCGridPyramidModel( options, () => 0.5 );

				expect( model.decoder.layers[ 0 ].inputSize ).toBe( options.channels + 1 );

			}

		} );

		it( 'sizes the decoder output layer to match options.outputChannels', () => {

			const options = { channels: 2, levels: 2, baseResolution: 4, hiddenSizes: [ 6, 6 ], outputChannels: 7 };
			const model = createNTCGridPyramidModel( options, () => 0.5 );

			const outputLayer = model.decoder.layers[ model.decoder.layers.length - 1 ];

			expect( outputLayer.outputSize ).toBe( options.outputChannels );
			expect( model.decoder.layers.length ).toBe( options.hiddenSizes.length + 1 );

		} );

		it( 'applies documented defaults when options are omitted', () => {

			const model = createNTCGridPyramidModel( {}, () => 0.5 );

			expect( model.channels ).toBe( 4 );
			expect( model.levels ).toBe( 4 );
			expect( model.mipsPerLevel ).toBe( 2 );
			expect( model.hiddenSizes ).toEqual( [ 32, 32 ] );
			expect( model.outputChannels ).toBe( 3 );

			// baseResolution = 128 (finest, no textureResolution given), mipsPerLevel = 2, levels = 4
			// -> 128, 32, 8, 2 (each level a quarter the resolution of the last).
			expect( model.resolutions ).toEqual( [ 128, 32, 8, 2 ] );

			// decoder: input = channels + 1 = 5, 2 hidden layers + 1 output layer
			expect( model.decoder.layers.length ).toBe( 3 );
			expect( model.decoder.layers[ 0 ].inputSize ).toBe( 5 );
			expect( model.decoder.layers[ model.decoder.layers.length - 1 ].outputSize ).toBe( 3 );

		} );

		it( 'produces deterministic grid and decoder contents for random() = 0.5 (midpoint => all zero weights)', () => {

			const options = { channels: 2, levels: 2, baseResolution: 4, mipsPerLevel: 1, hiddenSizes: [ 4 ], outputChannels: 3 };
			const model = createNTCGridPyramidModel( options, () => 0.5 );

			for ( const grid of model.grids ) {

				for ( const value of grid.data ) {

					expect( value ).toBe( 0 );

				}

			}

			for ( const layer of model.decoder.layers ) {

				for ( const weight of layer.weights ) {

					expect( weight ).toBe( 0 );

				}

			}

			// Linear 3-wide output layer gets the fixed RGB gray bias regardless of
			// random(), since createMLP forces it rather than deriving it from scale.
			const outputLayer = model.decoder.layers[ model.decoder.layers.length - 1 ];
			expect( outputLayer.biases ).toEqual( [ 0.3, 0.3, 0.3 ] );

		} );

		it( 'is deterministic given the same seeded random function across two independent builds', () => {

			const makeRandom = () => {

				let calls = 0;
				return () => {

					calls += 1;
					return ( Math.sin( calls * 1.7 ) + 1 ) / 2;

				};

			};

			const options = { channels: 3, levels: 3, baseResolution: 4, mipsPerLevel: 1, hiddenSizes: [ 6 ], outputChannels: 2 };

			const modelA = createNTCGridPyramidModel( options, makeRandom() );
			const modelB = createNTCGridPyramidModel( options, makeRandom() );

			for ( let i = 0; i < modelA.grids.length; i ++ ) {

				expect( Array.from( modelA.grids[ i ].data ) ).toEqual( Array.from( modelB.grids[ i ].data ) );

			}

			for ( let i = 0; i < modelA.decoder.layers.length; i ++ ) {

				expect( modelA.decoder.layers[ i ].weights ).toEqual( modelB.decoder.layers[ i ].weights );
				expect( modelA.decoder.layers[ i ].biases ).toEqual( modelB.decoder.layers[ i ].biases );

			}

		} );

	} );

	describe( 'maxLod / textureResolution', () => {

		it( 'derives maxLod from an explicit textureResolution as ceil(log2(textureResolution))', () => {

			const options = { channels: 2, levels: 2, baseResolution: 8, textureResolution: 4096 };
			const model = createNTCGridPyramidModel( options, () => 0.5 );

			expect( model.textureResolution ).toBe( 4096 );
			expect( model.maxLod ).toBe( Math.ceil( Math.log2( 4096 ) ) );

		} );

		it( 'falls back to the finest stored grid resolution when textureResolution is omitted', () => {

			const options = { channels: 2, levels: 3, baseResolution: 64, mipsPerLevel: 2 };
			const model = createNTCGridPyramidModel( options, () => 0.5 );

			expect( model.textureResolution ).toBe( 64 );
			expect( model.maxLod ).toBe( Math.ceil( Math.log2( 64 ) ) );

		} );

		it( 'can be exactly 0 for a 1x1 texture (no artificial floor at 1)', () => {

			const model = createNTCGridPyramidModel( { textureResolution: 1, levels: 1 }, () => 0.5 );

			expect( model.maxLod ).toBe( 0 );

		} );

		it( 'defaults baseResolution (the finest grid level) to the resolved textureResolution when not given explicitly', () => {

			const model = createNTCGridPyramidModel( { textureResolution: 512, levels: 1 }, () => 0.5 );

			expect( model.resolutions[ 0 ] ).toBe( 512 );

		} );

		it( 'an explicit baseResolution still overrides the textureResolution-derived default', () => {

			const model = createNTCGridPyramidModel( { textureResolution: 512, baseResolution: 64, levels: 1 }, () => 0.5 );

			expect( model.resolutions[ 0 ] ).toBe( 64 );

		} );

	} );

} );
