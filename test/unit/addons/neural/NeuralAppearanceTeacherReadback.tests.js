import {
	computeFootprintArea,
	createGaussianSampleKernel,
	getGaussianSampleGridSize,
	prefilterLeanNormalRoughness
} from '../../../../examples/jsm/neural/NeuralAppearanceFilterUtils.js';
import {
	readSamplePixel,
	readFilteredSample,
	renderAndReadTeacher
} from '../../../../examples/jsm/neural/NeuralAppearanceTeacherReadback.js';

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'Neural', () => {

		QUnit.module( 'NeuralAppearanceTeacherReadback', () => {

			QUnit.test( 'builds Gaussian footprint filters and LEAN normal moments', ( assert ) => {

				const fineArea = computeFootprintArea( [ 1 / 64, 0 ], [ 0, 1 / 64 ], 64, 64 );
				const coarseArea = computeFootprintArea( [ 4 / 64, 0 ], [ 0, 4 / 64 ], 64, 64 );
				const fineGrid = getGaussianSampleGridSize( fineArea, 1, 64 );
				const coarseGrid = getGaussianSampleGridSize( coarseArea, 1, 64 );
				const kernel = createGaussianSampleKernel( coarseGrid, 8 );
				const filtered = prefilterLeanNormalRoughness( [
					{ normal: [ - 0.5, 0, 0.8660254 ], roughness: 0.2 },
					{ normal: [ 0.5, 0, 0.8660254 ], roughness: 0.2 }
				] );

				assert.strictEqual( fineGrid, 1, 'uses one target sample for a one-texel footprint' );
				assert.strictEqual( coarseGrid, 4, 'increases the sample grid with filter area' );
				assert.strictEqual( kernel.length, 16, 'creates the requested stratified Gaussian taps' );
				assert.ok( Math.abs( kernel.reduce( ( sum, sample ) => sum + sample.weight, 0 ) - 1 ) < 1e-12, 'normalizes Gaussian weights' );
				assert.ok( Math.abs( filtered.normal[ 0 ] ) < 1e-12, 'preserves the mean normal direction' );
				assert.ok( filtered.roughness > 0.2, 'adds unresolved normal variance to roughness' );
				assert.ok( filtered.secondMoment[ 0 ] > 0, 'retains the LEAN slope second moment' );

			} );

			QUnit.test( 'averages Gaussian teacher pixels across coarse footprints', ( assert ) => {

				const tileSize = 8;
				const atlasColumns = 1;
				const atlasWidth = 8;
				const pixels = new Float32Array( 8 * 8 * 4 );

				for ( let y = 0; y < 8; y ++ ) {

					for ( let x = 0; x < 8; x ++ ) {

						const offset = ( y * 8 + x ) * 4;
						pixels[ offset ] = Math.pow( x / 7, 2 );
						pixels[ offset + 1 ] = Math.pow( y / 7, 2 );

					}

				}

				const point = readSamplePixel( pixels, 0, atlasColumns, atlasWidth, tileSize );
				const filtered = readFilteredSample( pixels, 0, {
					duvDx: [ 2 / 8, 0 ],
					duvDy: [ 0, 2 / 8 ]
				}, {
					atlasColumns,
					atlasWidth,
					tileSize,
					sourceResolution: 8
				} );

				assert.ok( filtered[ 0 ] > point[ 0 ], 'integrates nonlinear spatial variation instead of reading the center pixel' );
				assert.strictEqual( filtered[ 0 ], filtered[ 1 ], 'uses an isotropic Gaussian for an isotropic footprint' );
				assert.strictEqual( filtered[ 2 ], 0, 'preserves a constant channel' );

			} );

			QUnit.test( 'rejects non-half-float teacher readback', async ( assert ) => {

				const renderer = {
					toneMapping: 0,
					getRenderTarget: () => null,
					getClearAlpha: () => 1,
					getClearColor() {},
					setClearColor() {},
					setRenderTarget() {},
					render() {},
					readRenderTargetPixelsAsync: async () => new Uint8Array( 4 )
				};

				await assert.rejects(
					renderAndReadTeacher( renderer, {}, {}, {}, 1, 1 ),
					/Half-float teacher readback is required/,
					'does not silently train from LDR pixels'
				);

			} );

		} );

	} );

} );
