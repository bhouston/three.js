import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import {
	computeModelLayout,
	NeuralAppearanceGPUModel,
	DIRECT_SAMPLE_SIZE
} from '../../../../examples/jsm/neural-appearance/NeuralAppearanceGPUModel.js';
import {
	computeLatentChannels,
	computeDecoderInputSize,
	computeIblInputSize,
	computeIndirectInputSize,
	CHANNELS_PER_LEVEL,
	IBL_OUTPUT_SIZE,
	INDIRECT_OUTPUT_SIZE,
	IBL_TARGET_SIZE
} from '../../../../examples/jsm/neural-appearance/NeuralAppearanceFormat.js';
import { computeGridLevels } from '../../../../examples/jsm/neural/NeuralGridModel.js';
import { withTestRenderer, evalFloats } from '../helpers/webgpuEval.js';

// NeuralAppearanceGPUModel.js allocates the storage buffers that
// NeuralAppearanceGPUComputeTSL.js's training kernels index into with hand-
// written byte-offset arithmetic (row-major "offset + j*stride + i"
// addressing for weight matrices, and literal integer offsets for the
// per-sample fields). Nothing in the language enforces that the two files
// agree - a layout change on one side that isn't mirrored on the other
// silently reads/writes the wrong slot.
//
// Per the vitest/README.md test-theater guidance, none of the checks below
// re-derive their "expected" values from computeModelLayout()'s own offset
// arithmetic (which would just test the source against itself). Instead:
//   - block *widths* are independently re-derived here from the network
//     architecture (decoderInputSize -> hiddenSize -> hiddenSize -> 3, etc.)
//     and from NeuralAppearanceFormat.js / NeuralGridModel.js, which are
//     separate modules from the one under test;
//   - the buffer *offsets* returned by computeModelLayout() are then only
//     ever compared against that independently-accumulated running total,
//     never against each other's own field names;
//   - the cross-file sample/weight agreement checks write through
//     NeuralAppearanceGPUModel's public API (uploadSamples/initFromCPUModel)
//     using sentinel values chosen in this file, then read the real GPU
//     buffer back using the literal offset expressions copied verbatim from
//     NeuralAppearanceGPUComputeTSL.js - so a real mismatch between the two
//     files (wrong literal, transposed weight indexing, wrong field order)
//     shows up as a real numeric mismatch on real hardware, not a tautology.

function buildExpectedWeightRegions( { levels, hiddenSize, iblHiddenSize, supportsEmission, supportsOpacity } ) {

	const latentChannels = computeLatentChannels( levels );
	const decoderInputSize = computeDecoderInputSize( levels );
	const iblInputSize = computeIblInputSize( levels );
	const indirectInputSize = computeIndirectInputSize( levels );

	const regions = [
		{ field: 'rotationOffset', width: latentChannels * 12 },
		{ field: 'layer0WeightsOffset', width: decoderInputSize * hiddenSize },
		{ field: 'layer0BiasesOffset', width: hiddenSize },
		{ field: 'layer1WeightsOffset', width: hiddenSize * hiddenSize },
		{ field: 'layer1BiasesOffset', width: hiddenSize },
		{ field: 'layer2WeightsOffset', width: hiddenSize * 3 },
		{ field: 'layer2BiasesOffset', width: 3 }
	];

	if ( supportsEmission ) {

		regions.push( { field: 'emissionWeightsOffset', width: latentChannels * 3 } );
		regions.push( { field: 'emissionBiasesOffset', width: 3 } );

	}

	if ( supportsOpacity ) {

		regions.push( { field: 'opacityWeightsOffset', width: latentChannels * 1 } );
		regions.push( { field: 'opacityBiasesOffset', width: 1 } );

	}

	regions.push( { field: 'iblLayer0WeightsOffset', width: iblInputSize * iblHiddenSize, directBoundary: true } );
	regions.push( { field: 'iblLayer0BiasesOffset', width: iblHiddenSize } );
	regions.push( { field: 'iblLayer1WeightsOffset', width: iblHiddenSize * IBL_OUTPUT_SIZE } );
	regions.push( { field: 'iblLayer1BiasesOffset', width: IBL_OUTPUT_SIZE } );

	for ( const head of [ 'indirectRadiance', 'indirectIrradiance' ] ) {

		regions.push( { field: `${ head }Layer0WeightsOffset`, width: indirectInputSize * iblHiddenSize } );
		regions.push( { field: `${ head }Layer0BiasesOffset`, width: iblHiddenSize } );
		regions.push( { field: `${ head }Layer1WeightsOffset`, width: iblHiddenSize * INDIRECT_OUTPUT_SIZE } );
		regions.push( { field: `${ head }Layer1BiasesOffset`, width: INDIRECT_OUTPUT_SIZE } );

	}

	return regions;

}

function buildExpectedActivationRegions( { levels, hiddenSize, iblHiddenSize, supportsEmission, supportsOpacity } ) {

	const latentChannels = computeLatentChannels( levels );
	const decoderInputSize = computeDecoderInputSize( levels );
	const iblInputSize = computeIblInputSize( levels );
	const indirectInputSize = computeIndirectInputSize( levels );

	const regions = [
		{ field: 'actA0Offset', width: decoderInputSize },
		{ field: 'actZ1Offset', width: hiddenSize },
		{ field: 'actA1Offset', width: hiddenSize },
		{ field: 'actZ2Offset', width: hiddenSize },
		{ field: 'actA2Offset', width: hiddenSize },
		{ field: 'actZ3Offset', width: 3 },
		{ field: 'actDelta3Offset', width: 3 },
		{ field: 'actDelta2Offset', width: hiddenSize },
		{ field: 'actDelta1Offset', width: hiddenSize },
		{ field: 'actGradA0Offset', width: decoderInputSize },
		{ field: 'actGradLatentsOffset', width: latentChannels }
	];

	if ( supportsEmission ) {

		regions.push( { field: 'actEmissionOffset', width: 6 } );

	}

	if ( supportsOpacity ) {

		regions.push( { field: 'actOpacityOffset', width: 2 } );

	}

	regions.push( { field: 'actIblA0Offset', width: iblInputSize } );
	regions.push( { field: 'actIblZ1Offset', width: iblHiddenSize } );
	regions.push( { field: 'actIblA1Offset', width: iblHiddenSize } );
	regions.push( { field: 'actIblZ2Offset', width: IBL_OUTPUT_SIZE } );
	regions.push( { field: 'actIblDelta2Offset', width: IBL_OUTPUT_SIZE } );
	regions.push( { field: 'actIblDelta1Offset', width: iblHiddenSize } );
	regions.push( { field: 'actIndirectA0Offset', width: indirectInputSize } );
	regions.push( { field: 'actIndirectZ1Offset', width: iblHiddenSize } );
	regions.push( { field: 'actIndirectA1Offset', width: iblHiddenSize } );
	regions.push( { field: 'actIndirectZ2Offset', width: INDIRECT_OUTPUT_SIZE } );
	regions.push( { field: 'actIndirectDelta2Offset', width: INDIRECT_OUTPUT_SIZE } );
	regions.push( { field: 'actIndirectDelta1Offset', width: iblHiddenSize } );

	return regions;

}

// Walks `regions` in the exact order NeuralAppearanceGPUModel.js declares
// them, asserting each region's actual offset (read from the real
// `layout`) equals a running total built purely from the independently
// derived `width`s above - i.e. that the regions exactly tile the buffer
// with no gap and no overlap, in this order. Returns the running total
// (== the buffer's real total size) and, if any region was flagged
// `directBoundary`, the offset immediately before it.
function assertRegionsContiguous( layout, regions ) {

	let offset = 0;
	let directBoundary;

	for ( const region of regions ) {

		if ( region.directBoundary ) directBoundary = offset;

		expect( layout[ region.field ], region.field ).toBe( offset );
		offset += region.width;

	}

	return { total: offset, directBoundary };

}

const LEVEL_CONFIGS = [ 2, 3, 4, 5, 6, 8 ];

describe( 'Addons > Neural > NeuralAppearance > NeuralAppearanceGPUModel storage buffer layout', () => {

	describe( 'computeModelLayout() weight-buffer region tiling (independent widths, no GPU needed)', () => {

		it.each( LEVEL_CONFIGS )( 'weight blocks tile totalWeights with no gap/overlap, levels=%i, no aux heads', ( levels ) => {

			const options = { levels, hiddenSize: 24, iblHiddenSize: 20 };
			const layout = computeModelLayout( options );
			const regions = buildExpectedWeightRegions( { levels, hiddenSize: 24, iblHiddenSize: 20, supportsEmission: false, supportsOpacity: false } );

			const { total, directBoundary } = assertRegionsContiguous( layout, regions );

			expect( layout.totalWeights ).toBe( total );
			expect( layout.directWeightCount ).toBe( directBoundary );
			expect( layout.iblWeightCount ).toBe( total - directBoundary );

		} );

		it( 'weight blocks tile totalWeights with no gap/overlap when emission+opacity heads are both enabled', () => {

			const options = { levels: 4, hiddenSize: 32, iblHiddenSize: 32, outputFeatures: { emission: true, opacity: true } };
			const layout = computeModelLayout( options );
			const regions = buildExpectedWeightRegions( { levels: 4, hiddenSize: 32, iblHiddenSize: 32, supportsEmission: true, supportsOpacity: true } );

			const { total, directBoundary } = assertRegionsContiguous( layout, regions );

			expect( layout.totalWeights ).toBe( total );
			expect( layout.directWeightCount ).toBe( directBoundary );

		} );

		it( 'enabling only the emission head (not opacity) inserts exactly latentChannels*3+3 floats between layer2 and the IBL head', () => {

			const base = computeModelLayout( { levels: 4, hiddenSize: 16, iblHiddenSize: 16 } );
			const withEmission = computeModelLayout( { levels: 4, hiddenSize: 16, iblHiddenSize: 16, outputFeatures: { emission: true } } );

			const latentChannels = computeLatentChannels( 4 );
			expect( withEmission.directWeightCount - base.directWeightCount ).toBe( latentChannels * 3 + 3 );
			// The IBL head onward must be pushed by exactly that same amount.
			expect( withEmission.iblLayer0WeightsOffset - base.iblLayer0WeightsOffset ).toBe( latentChannels * 3 + 3 );

		} );

	} );

	describe( 'computeModelLayout() activation-buffer region tiling (independent widths, no GPU needed)', () => {

		it.each( LEVEL_CONFIGS )( 'per-sample activation slots tile activationStride with no gap/overlap, levels=%i', ( levels ) => {

			const options = { levels, hiddenSize: 24, iblHiddenSize: 20, outputFeatures: { emission: true, opacity: true } };
			const layout = computeModelLayout( options );
			const regions = buildExpectedActivationRegions( { levels, hiddenSize: 24, iblHiddenSize: 20, supportsEmission: true, supportsOpacity: true } );

			const { total } = assertRegionsContiguous( layout, regions );

			expect( layout.activationStride ).toBe( total );

		} );

	} );

	describe( 'latent grid geometry (independent oracle: NeuralGridModel.computeGridLevels + CHANNELS_PER_LEVEL)', () => {

		it.each( LEVEL_CONFIGS )( 'gridLevels count matches configured `levels` and offsets tile totalLatents, levels=%i', ( levels ) => {

			const baseResolution = 8;
			const targetResolution = 128;
			const layout = computeModelLayout( { levels, baseResolution, targetResolution } );

			const resolutions = computeGridLevels( baseResolution, targetResolution, levels );
			expect( layout.gridLevels.length ).toBe( levels );
			expect( layout.gridLevels.length ).toBe( resolutions.length );

			let offset = 0;
			for ( let i = 0; i < resolutions.length; i ++ ) {

				const resolution = resolutions[ i ];
				const expectedFloatCount = resolution * resolution * CHANNELS_PER_LEVEL;

				expect( layout.gridLevels[ i ].width ).toBe( resolution );
				expect( layout.gridLevels[ i ].height ).toBe( resolution );
				expect( layout.gridLevels[ i ].offset ).toBe( offset );
				expect( layout.gridLevels[ i ].floatCount ).toBe( expectedFloatCount );

				offset += expectedFloatCount;

			}

			expect( layout.totalLatents ).toBe( offset );

		} );

	} );

	describe( 'sample buffer field-count constants (hand-counted against the field list, independent of any source arithmetic)', () => {

		it( 'DIRECT_SAMPLE_SIZE (17) matches the hand-counted direct field list: uv(2)+wi(3)+wo(3)+weight(1)+target(3)+emissionFlag(1)+emissionTarget(3)+opacityTarget(1)', () => {

			expect( DIRECT_SAMPLE_SIZE ).toBe( 2 + 3 + 3 + 1 + 3 + 1 + 3 + 1 );

		} );

		it( 'IBL_TARGET_SIZE (17) matches the hand-counted IBL field list: iblWeight(1)+direction(3)+roughness(1)+incoming(3)+irradiance(3)+indirectRadiance(3)+indirectIrradiance(3)', () => {

			expect( IBL_TARGET_SIZE ).toBe( 1 + 3 + 1 + 3 + 3 + 3 + 3 );

		} );

		it( 'sampleStride equals DIRECT_SAMPLE_SIZE + IBL_TARGET_SIZE for every levels config (sampleStride has no levels dependency)', () => {

			for ( const levels of LEVEL_CONFIGS ) {

				const layout = computeModelLayout( { levels } );
				expect( layout.sampleStride ).toBe( DIRECT_SAMPLE_SIZE + IBL_TARGET_SIZE );

			}

		} );

	} );

	describe( 'iblHiddenSize default clamp (documented behavior: clamp(hiddenSize, 16, 32) when not explicitly set)', () => {

		it.each( [
			[ 4, 16 ],
			[ 8, 16 ],
			[ 16, 16 ],
			[ 24, 24 ],
			[ 32, 32 ],
			[ 64, 32 ]
		] )( 'hiddenSize=%i -> iblHiddenSize=%i', ( hiddenSize, expected ) => {

			const layout = computeModelLayout( { hiddenSize } );
			expect( layout.iblHiddenSize ).toBe( expected );

		} );

	} );

	describe( 'StorageBufferAttribute allocation sizes match the computed layout', () => {

		it.each( [
			{ levels: 4, hiddenSize: 32, iblHiddenSize: 32, batchSize: 8, outputFeatures: {} },
			{ levels: 6, hiddenSize: 16, iblHiddenSize: 16, batchSize: 4, outputFeatures: { emission: true, opacity: true } },
			{ levels: 2, hiddenSize: 8, iblHiddenSize: 16, batchSize: 2, outputFeatures: { emission: true } }
		] )( 'allocates every buffer to exactly its layout field ($levels levels, batchSize=$batchSize)', ( options ) => {

			const model = new NeuralAppearanceGPUModel( options );
			const { layout } = model;

			expect( model.weightsBuffers.attribute.array.length ).toBe( layout.totalWeights );
			expect( model.weightsBuffers.gradAttribute.array.length ).toBe( layout.totalWeights );
			expect( model.weightsBuffers.mAttribute.array.length ).toBe( layout.totalWeights );
			expect( model.weightsBuffers.vAttribute.array.length ).toBe( layout.totalWeights );

			expect( model.latentsBuffers.attribute.array.length ).toBe( layout.totalLatents );
			expect( model.latentsBuffers.gradAttribute.array.length ).toBe( layout.totalLatents );
			expect( model.latentsBuffers.mAttribute.array.length ).toBe( layout.totalLatents );
			expect( model.latentsBuffers.vAttribute.array.length ).toBe( layout.totalLatents );

			expect( model.samplesAttribute.array.length ).toBe( options.batchSize * layout.sampleStride );
			expect( model.activationsAttribute.array.length ).toBe( options.batchSize * layout.activationStride );

			expect( model.lossAttribute.array.length ).toBe( 3 );
			expect( model.gradNormAttribute.array.length ).toBe( 1 );

		} );

	} );

	describe( 'sample buffer: real GPU agreement between uploadSamples() writes and the literal offsets NeuralAppearanceGPUComputeTSL.js reads', () => {

		const getRenderer = withTestRenderer( { beforeAll, afterAll } );

		// These literal element offsets are copied verbatim from
		// createTrainBatchComputeNode()/trainIndirectProbeHead() in
		// NeuralAppearanceGPUComputeTSL.js (sampleOffset.add(N)) - the point
		// of this test is to prove uploadSamples() actually puts each named
		// field at the exact slot the kernel reads it from, using sentinel
		// values chosen independently of any file's internals.
		it( 'direct fields (uv/wi/wo/weight/target/emissionFlag/emissionTarget/opacityTarget) land at offsets 0-16', async () => {

			const renderer = getRenderer();
			const model = new NeuralAppearanceGPUModel( { levels: 4, hiddenSize: 8, iblHiddenSize: 16, batchSize: 1, outputFeatures: { emission: true, opacity: true } } );

			model.uploadSamples( [ {
				uv: [ 0.11, 0.22 ],
				wi: [ 0.31, 0.32, 0.33 ],
				wo: [ 0.41, 0.42, 0.43 ],
				weight: 0.55,
				target: [ 0.61, 0.62, 0.63 ],
				emissionTarget: [ 0.71, 0.72, 0.73 ],
				opacityTarget: 0.81
			} ] );

			const values = await evalFloats( renderer, 17, ( out ) => {

				for ( let i = 0; i < 17; i ++ ) out.element( i ).assign( model.samplesStorage.element( i ) );

			} );

			const expected = [
				0.11, 0.22, // uv (offsets 0,1)
				0.31, 0.32, 0.33, // wi (2,3,4)
				0.41, 0.42, 0.43, // wo (5,6,7)
				0.55, // weight (8)
				0.61, 0.62, 0.63, // target (9,10,11)
				1.0, // emission flag (12)
				0.71, 0.72, 0.73, // emissionTarget (13,14,15)
				0.81 // opacityTarget (16)
			];

			for ( let i = 0; i < 17; i ++ ) {

				expect( values[ i ], `offset ${ i }` ).toBeCloseTo( expected[ i ], 5 );

			}

		} );

		it( 'IBL fields (weight/direction/roughness/incoming/irradiance/indirect targets) land at the exact offsets trainIndirectProbeHead() reads (17, 18-20, 21, 22-24, 25-27, 28-30, 31-33)', async () => {

			const renderer = getRenderer();
			const model = new NeuralAppearanceGPUModel( { levels: 4, hiddenSize: 8, iblHiddenSize: 16, batchSize: 1 } );

			model.uploadSamples( [ {
				iblWeight: 0.91,
				iblDirection: [ 0.01, 0.02, 0.03 ],
				iblRoughness: 0.44,
				iblIncoming: [ 1.1, 1.2, 1.3 ],
				iblIrradiance: [ 2.1, 2.2, 2.3 ],
				iblIndirectRadiance: [ 3.1, 3.2, 3.3 ],
				iblIndirectIrradiance: [ 4.1, 4.2, 4.3 ]
			} ] );

			// Mirrors GPUComputeTSL's literal reads exactly:
			//   iblWeight               -> sampleOffset.add(17)
			//   targetDirection         -> sampleOffset.add(18,19,20)
			//   targetRough             -> sampleOffset.add(21)
			//   radiance probe input    -> sampleOffset.add(22,23,24)   (probeSampleOffset:22)
			//   radiance probe target   -> sampleOffset.add(28,29,30)   (targetSampleOffset:28)
			//   irradiance probe input  -> sampleOffset.add(25,26,27)   (probeSampleOffset:25)
			//   irradiance probe target -> sampleOffset.add(31,32,33)   (targetSampleOffset:31)
			const offsets = [ 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33 ];

			const values = await evalFloats( renderer, offsets.length, ( out ) => {

				offsets.forEach( ( offset, i ) => out.element( i ).assign( model.samplesStorage.element( offset ) ) );

			} );

			const expected = [
				0.91, // 17
				0.01, 0.02, 0.03, // 18-20
				0.44, // 21
				1.1, 1.2, 1.3, // 22-24 (radiance probe input)
				2.1, 2.2, 2.3, // 25-27 (irradiance probe input)
				3.1, 3.2, 3.3, // 28-30 (radiance probe target)
				4.1, 4.2, 4.3 // 31-33 (irradiance probe target)
			];

			for ( let i = 0; i < offsets.length; i ++ ) {

				expect( values[ i ], `offset ${ offsets[ i ] }` ).toBeCloseTo( expected[ i ], 5 );

			}

		} );

	} );

	describe( 'weight buffer: real GPU agreement between initFromCPUModel() writes and the row-major addressing NeuralAppearanceGPUComputeTSL.js uses', () => {

		const getRenderer = withTestRenderer( { beforeAll, afterAll } );

		// Builds a synthetic CPU model whose flattened layer.weights[k]
		// equals `tag*1e6 + k` (biases: `tag*1e6 + 5e5 + j`), for whatever
		// (outSize, inSize) shape NeuralAppearanceGPUModel.initFromCPUModel
		// expects for the given layout - independent of computeModelLayout's
		// own offset arithmetic (only the shapes come from the layout; the
		// content is a deterministic marker this test invents).
		function makeLayer( outSize, inSize, tag ) {

			const weights = new Float32Array( outSize * inSize );
			for ( let k = 0; k < weights.length; k ++ ) weights[ k ] = tag * 1e6 + k;

			const biases = new Float32Array( outSize );
			for ( let j = 0; j < outSize; j ++ ) biases[ j ] = tag * 1e6 + 5e5 + j;

			return { weights, biases };

		}

		function makeSyntheticCpuModel( layout ) {

			const { latentChannels, decoderInputSize, hiddenSize, iblInputSize, iblHiddenSize, indirectInputSize, gridLevels } = layout;

			return {
				rotationWeights: new Float32Array( latentChannels * 12 ).map( ( _, k ) => 9e6 + k ),
				decoder: { layers: [
					makeLayer( hiddenSize, decoderInputSize, 1 ),
					makeLayer( hiddenSize, hiddenSize, 2 ),
					makeLayer( 3, hiddenSize, 3 )
				] },
				iblHead: { layers: [
					makeLayer( iblHiddenSize, iblInputSize, 4 ),
					makeLayer( IBL_OUTPUT_SIZE, iblHiddenSize, 5 )
				] },
				indirectRadianceHead: { layers: [
					makeLayer( iblHiddenSize, indirectInputSize, 6 ),
					makeLayer( INDIRECT_OUTPUT_SIZE, iblHiddenSize, 7 )
				] },
				indirectIrradianceHead: { layers: [
					makeLayer( iblHiddenSize, indirectInputSize, 8 ),
					makeLayer( INDIRECT_OUTPUT_SIZE, iblHiddenSize, 9 )
				] },
				latentGrids: gridLevels.map( ( level, g ) => ( {
					data: new Float32Array( level.floatCount ).map( ( _, k ) => 100 + g * 1000 + k )
				} ) )
			};

		}

		it.each( [ 4, 2, 6 ] )( 'decoder layer0 weight[j][i] lands at layer0WeightsOffset + j*decoderInputSize + i (levels=%i)', async ( levels ) => {

			const renderer = getRenderer();
			const model = new NeuralAppearanceGPUModel( { levels, hiddenSize: 6, iblHiddenSize: 16, baseResolution: 4, targetResolution: 8, batchSize: 1 } );
			const cpuModel = makeSyntheticCpuModel( model.layout );

			model.initFromCPUModel( cpuModel );

			const { layer0WeightsOffset, decoderInputSize, hiddenSize } = model.layout;
			// Probe first row, last row, and a middle (j,i) pair - mirrors
			// GPUComputeTSL's `rowOffset = layer0WeightsOffset + j*decoderInputSize; ... .add(i)`.
			const probes = [ [ 0, 0 ], [ hiddenSize - 1, decoderInputSize - 1 ], [ 1, 2 ] ];

			const values = await evalFloats( renderer, probes.length, ( out ) => {

				probes.forEach( ( [ j, i ], idx ) => {

					out.element( idx ).assign( model.weightsBuffers.valuesStorage.element( layer0WeightsOffset + j * decoderInputSize + i ) );

				} );

			} );

			probes.forEach( ( [ j, i ], idx ) => {

				const expected = 1e6 + ( j * decoderInputSize + i );
				expect( values[ idx ], `j=${ j } i=${ i }` ).toBeCloseTo( expected, 1 );

			} );

		} );

		it.each( [ 4, 2, 6 ] )( 'both indirect probe heads\' layer1 weight[j][i] and bias[j] land at their own layout offsets, not aliasing each other (levels=%i)', async ( levels ) => {

			const renderer = getRenderer();
			const model = new NeuralAppearanceGPUModel( { levels, hiddenSize: 6, iblHiddenSize: 12, baseResolution: 4, targetResolution: 8, batchSize: 1 } );
			const cpuModel = makeSyntheticCpuModel( model.layout );

			model.initFromCPUModel( cpuModel );

			const {
				indirectRadianceLayer1WeightsOffset, indirectRadianceLayer1BiasesOffset,
				indirectIrradianceLayer1WeightsOffset, indirectIrradianceLayer1BiasesOffset,
				iblHiddenSize
			} = model.layout;

			const j = INDIRECT_OUTPUT_SIZE - 1;
			const i = iblHiddenSize - 1;

			const values = await evalFloats( renderer, 4, ( out ) => {

				out.element( 0 ).assign( model.weightsBuffers.valuesStorage.element( indirectRadianceLayer1WeightsOffset + j * iblHiddenSize + i ) );
				out.element( 1 ).assign( model.weightsBuffers.valuesStorage.element( indirectRadianceLayer1BiasesOffset + j ) );
				out.element( 2 ).assign( model.weightsBuffers.valuesStorage.element( indirectIrradianceLayer1WeightsOffset + j * iblHiddenSize + i ) );
				out.element( 3 ).assign( model.weightsBuffers.valuesStorage.element( indirectIrradianceLayer1BiasesOffset + j ) );

			} );

			// tag 7 = indirectRadiance layer1, tag 9 = indirectIrradiance layer1 (see makeSyntheticCpuModel).
			expect( values[ 0 ] ).toBeCloseTo( 7e6 + ( j * iblHiddenSize + i ), 1 );
			expect( values[ 1 ] ).toBeCloseTo( 7e6 + 5e5 + j, 1 );
			expect( values[ 2 ] ).toBeCloseTo( 9e6 + ( j * iblHiddenSize + i ), 1 );
			expect( values[ 3 ] ).toBeCloseTo( 9e6 + 5e5 + j, 1 );

		} );

		it( 'syncToCPU() round-trips weights and latent grids written by initFromCPUModel() back out unchanged', async () => {

			const renderer = getRenderer();
			const model = new NeuralAppearanceGPUModel( { levels: 3, hiddenSize: 6, iblHiddenSize: 12, baseResolution: 4, targetResolution: 8, batchSize: 1 } );
			const cpuModel = makeSyntheticCpuModel( model.layout );

			model.initFromCPUModel( cpuModel );

			// getArrayBufferAsync() (used by syncToCPU below) requires the backend
			// to already have created a GPU-resident buffer for these attributes -
			// which normally happens as a side effect of a training compute pass.
			// Force that here with a trivial no-op dispatch that touches both.
			await evalFloats( renderer, 1, ( out ) => {

				out.element( 0 ).assign( model.weightsBuffers.valuesStorage.element( 0 ).add( model.latentsBuffers.valuesStorage.element( 0 ) ) );

			} );

			const roundTripped = makeSyntheticCpuModel( model.layout ); // fresh zeroed-shape target
			for ( const layer of roundTripped.decoder.layers ) { layer.weights.fill( 0 ); layer.biases.fill( 0 ); }
			roundTripped.rotationWeights.fill( 0 );
			for ( const grid of roundTripped.latentGrids ) grid.data.fill( 0 );

			await model.syncToCPU( roundTripped, renderer );

			expect( Array.from( roundTripped.decoder.layers[ 0 ].weights ) ).toEqual( Array.from( cpuModel.decoder.layers[ 0 ].weights ) );
			expect( Array.from( roundTripped.decoder.layers[ 0 ].biases ) ).toEqual( Array.from( cpuModel.decoder.layers[ 0 ].biases ) );
			expect( Array.from( roundTripped.rotationWeights ) ).toEqual( Array.from( cpuModel.rotationWeights ) );
			expect( roundTripped.latentGrids.length ).toBe( cpuModel.latentGrids.length );
			for ( let g = 0; g < cpuModel.latentGrids.length; g ++ ) {

				expect( Array.from( roundTripped.latentGrids[ g ].data ) ).toEqual( Array.from( cpuModel.latentGrids[ g ].data ) );

			}

		} );

	} );

} );
