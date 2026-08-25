import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { Fn, instanceIndex } from 'three/tsl';
import {
	computeTextureModelLayout,
	NTCGPUModel
} from '../../../../examples/jsm/ntc/training/NTCGPUModel.js';
import { withTestRenderer } from '../helpers/webgpuEval.js';

// NTCGPUModel.js only *computes* a storage-buffer layout (offsets,
// counts) and allocates buffers sized from it - the actual GPU kernels that
// read/write those buffers live in NeuralTextureGPUComputeTSL.js, and they
// don't reference the layout via any shared constant: every offset used
// there is read straight off `gpuModel.layout` at kernel-build time (see e.g.
// `layout.a0Offset`, `layout.mlpLayers[l].weightsOffset`, `layout.gradA0Offset`
// throughout createTextureTrainBatchComputeNode). So the two files can only
// disagree if the *shape* the model believes it allocated doesn't actually
// match the index arithmetic the compute kernel independently performs
// against it.
//
// Two independent kinds of check are used here, deliberately avoiding
// "test theater" (asserting a value against a re-derivation of the same
// formula the source already used, which passes by construction):
//
//  1. Hand-computed literal expectations for two small configurations,
//     worked out on paper (see the comments beside each number below) and
//     checked against computeTextureModelLayout's actual output - this can
//     only pass if the source's arithmetic matches the documented layout
//     contract, not just itself.
//  2. Index-formula replays transcribed from NeuralTextureGPUComputeTSL.js's
//     own kernel code (a different, independently-written file) - these
//     stand in for "what the training kernel actually reads/writes" and are
//     checked for in-bounds addressing and exact, non-overlapping buffer
//     coverage against the layout the model computed. A layout field that
//     drifted out of sync with how the kernel indexes it (an off-by-one, a
//     swapped offset, a buffer sized from the wrong field) would show up
//     here as an out-of-bounds index, a gap, or an overlap.

/**
 * Replays the grid-tap addressing formula from
 * createTextureTrainBatchComputeNode's forward pass (step 1):
 *   off = level.offset + (tapY * level.width + tapX) * channels
 * for every texel in every grid level, and returns the full set of float
 * indices the kernel can address into the latents buffer.
 */
function simulateLatentIndices( layout ) {

	const indices = new Set();

	for ( const level of layout.gridLevels ) {

		for ( let y = 0; y < level.height; y ++ ) {

			for ( let x = 0; x < level.width; x ++ ) {

				const base = level.offset + ( y * level.width + x ) * layout.channels;

				for ( let c = 0; c < layout.channels; c ++ ) indices.add( base + c );

			}

		}

	}

	return indices;

}

/**
 * Replays the a0 addressing from the same forward pass: the *selected*
 * grid level's `channels`-wide tap always lands at the same shared
 * `a0Offset + c` slot (not a per-level slot - every level's contribution is
 * summed into it, gated by an exact 0/1 selector, see
 * createTextureTrainBatchComputeNode step 1), followed by the LOD value at
 * `a0Offset + channels`.
 */
function simulateA0Indices( layout ) {

	const indices = new Set();

	for ( let c = 0; c < layout.channels; c ++ ) indices.add( layout.a0Offset + c );
	indices.add( layout.a0Offset + layout.channels );

	return indices;

}

/**
 * Replays the per-sample activation-buffer addressing used across the
 * forward pass (step 2), loss/delta (step 3), and backward pass (step 4):
 * a0, then per-layer z (all layers) and a (non-output layers only), then
 * per-layer deltas (all layers), then gradA0.
 */
function simulateActivationRegions( layout ) {

	const regions = {};
	regions.inputA0 = rangeSet( layout.a0Offset, layout.inputSize );

	layout.mlpLayers.forEach( ( layer, l ) => {

		const acts = layout.layerActs[ l ];
		regions[ `z${l}` ] = rangeSet( acts.zOffset, layer.outputSize );

		if ( layer.isOutput === false ) regions[ `act${l}` ] = rangeSet( acts.aOffset, layer.outputSize );

	} );

	layout.mlpLayers.forEach( ( layer, l ) => {

		regions[ `delta${l}` ] = rangeSet( layout.deltaOffsets[ l ], layer.outputSize );

	} );

	regions.gradA0 = rangeSet( layout.gradA0Offset, layout.inputSize );

	return regions;

}

/**
 * Replays the per-layer weight/bias addressing from the forward pass (step 2)
 * and backward pass (step 4): weight row `weightsOffset + j * inputSize + i`
 * for output j / input i, and bias `biasesOffset + j`.
 */
function simulateWeightRegions( layout ) {

	const regions = {};

	layout.mlpLayers.forEach( ( layer, l ) => {

		const weightIndices = new Set();

		for ( let j = 0; j < layer.outputSize; j ++ ) {

			for ( let i = 0; i < layer.inputSize; i ++ ) weightIndices.add( layer.weightsOffset + j * layer.inputSize + i );

		}

		regions[ `weights${l}` ] = weightIndices;
		regions[ `biases${l}` ] = rangeSet( layer.biasesOffset, layer.biasesCount );

	} );

	return regions;

}

function rangeSet( offset, count ) {

	const set = new Set();
	for ( let i = 0; i < count; i ++ ) set.add( offset + i );

	return set;

}

/** Asserts that a collection of index Sets exactly tiles [0, total) - no gaps, no overlaps. */
function expectExactPartition( regions, total ) {

	const seen = new Set();

	for ( const [ name, indices ] of Object.entries( regions ) ) {

		for ( const index of indices ) {

			expect( index, `${name} index ${index} out of [0, ${total})` ).toBeGreaterThanOrEqual( 0 );
			expect( index, `${name} index ${index} out of [0, ${total})` ).toBeLessThan( total );
			expect( seen.has( index ), `${name} index ${index} overlaps another region` ).toBe( false );
			seen.add( index );

		}

	}

	expect( seen.size, `regions leave gaps in [0, ${total})` ).toBe( total );

}

const configs = {
	'default options': {},
	'single hidden layer': { hiddenSizes: [ 16 ] },
	'deep MLP, 3 hidden layers': { hiddenSizes: [ 24, 24, 24 ] },
	'material-style 9-channel output': { outputChannels: 9, hiddenSizes: [ 48, 48 ] },
	'single grid level': { levels: 1, baseResolution: 8 },
	'channels = 2': { channels: 2 },
	'no hidden layers (direct input -> output)': { hiddenSizes: [], levels: 1, baseResolution: 2, channels: 1, outputChannels: 1 }
};

describe( 'Addons > NeuralTexture > NTCGPUModel (storage buffer layout)', () => {

	describe( 'computeTextureModelLayout - hand-computed layouts (worked out on paper, not re-derived from the source)', () => {

		it( 'two-level grid, one hidden layer, 2-channel output', () => {

			// channels=2, levels=2, baseResolution=2, mipsPerLevel=2 (default) ->
			// resolutions [2, 1] (2 halvings per level: 2 -> 1 -> floor(0.5)=>1).
			// hiddenSizes=[3], outputChannels=2. Hand-traced:
			//   level0: 2x2 texels * 2 channels = 8 floats @ offset 0
			//   level1: 1x1 texels * 2 channels = 2 floats @ offset 8
			//   totalLatents = 10
			//   inputSize = channels + 1 = 3 (one selected level's tap + LOD -
			//   fixed regardless of level count, see NTCGridPyramidModel.js)
			//   layer sizes [3, 3, 2]:
			//     layer0: 3*3=9 weights @0, 3 biases @9 -> next offset 12
			//     layer1 (output): 3*2=6 weights @12, 2 biases @18 -> totalWeights 20
			//   activation buffer: a0 [0,3) -> z0 [3,6) -> a0-layer [6,9) (hidden,
			//   not output) -> z1(output) [9,11) -> delta0 [11,14) -> delta1 [14,16)
			//   -> gradA0 [16,19) -> activationStride 19
			const layout = computeTextureModelLayout( {
				channels: 2, levels: 2, baseResolution: 2,
				hiddenSizes: [ 3 ], outputChannels: 2
			} );

			expect( layout.resolutions ).toEqual( [ 2, 1 ] );
			expect( layout.gridLevels[ 0 ] ).toMatchObject( { width: 2, height: 2, offset: 0, texelCount: 4, floatCount: 8 } );
			expect( layout.gridLevels[ 1 ] ).toMatchObject( { width: 1, height: 1, offset: 8, texelCount: 1, floatCount: 2 } );
			expect( layout.totalLatents ).toBe( 10 );

			expect( layout.inputSize ).toBe( 3 );

			expect( layout.mlpLayers[ 0 ] ).toMatchObject( {
				inputSize: 3, outputSize: 3, weightsOffset: 0, weightsCount: 9, biasesOffset: 9, biasesCount: 3, isOutput: false
			} );
			expect( layout.mlpLayers[ 1 ] ).toMatchObject( {
				inputSize: 3, outputSize: 2, weightsOffset: 12, weightsCount: 6, biasesOffset: 18, biasesCount: 2, isOutput: true
			} );
			expect( layout.totalWeights ).toBe( 20 );

			expect( layout.a0Offset ).toBe( 0 );
			expect( layout.layerActs[ 0 ] ).toEqual( { zOffset: 3, aOffset: 6 } );
			expect( layout.layerActs[ 1 ] ).toEqual( { zOffset: 9, aOffset: - 1 } );
			expect( layout.deltaOffsets ).toEqual( [ 11, 14 ] );
			expect( layout.gradA0Offset ).toBe( 16 );
			expect( layout.activationStride ).toBe( 19 );

		} );

		it( 'minimal degenerate config: single 1x1 texel level, no hidden layers, 1-channel everything', () => {

			// channels=1, levels=1, baseResolution=1, hiddenSizes=[],
			// outputChannels=1. Hand-traced:
			//   single 1x1 grid level -> 1 float @ offset 0 -> totalLatents 1
			//   inputSize = channels + 1 = 2
			//   layer sizes [2, 1]: one layer, immediately the output:
			//     2*1=2 weights @0, 1 bias @2 -> totalWeights 3
			//   activation buffer: a0 [0,2) -> z0(output, no a-slot) [2,3) ->
			//   delta0 [3,4) -> gradA0 [4,6) -> activationStride 6
			const layout = computeTextureModelLayout( {
				channels: 1, levels: 1, baseResolution: 1,
				hiddenSizes: [], outputChannels: 1
			} );

			expect( layout.resolutions ).toEqual( [ 1 ] );
			expect( layout.gridLevels[ 0 ] ).toMatchObject( { width: 1, height: 1, offset: 0, texelCount: 1, floatCount: 1 } );
			expect( layout.totalLatents ).toBe( 1 );

			expect( layout.inputSize ).toBe( 2 );

			expect( layout.mlpLayers ).toHaveLength( 1 );
			expect( layout.mlpLayers[ 0 ] ).toMatchObject( {
				inputSize: 2, outputSize: 1, weightsOffset: 0, weightsCount: 2, biasesOffset: 2, biasesCount: 1, isOutput: true
			} );
			expect( layout.totalWeights ).toBe( 3 );

			expect( layout.layerActs[ 0 ] ).toEqual( { zOffset: 2, aOffset: - 1 } );
			expect( layout.deltaOffsets ).toEqual( [ 3 ] );
			expect( layout.gradA0Offset ).toBe( 4 );
			expect( layout.activationStride ).toBe( 6 );

		} );

	} );

	describe( 'cross-check against NeuralTextureGPUComputeTSL.js kernel index formulas', () => {

		for ( const [ label, options ] of Object.entries( configs ) ) {

			describe( label, () => {

				const layout = computeTextureModelLayout( options );

				it( 'the grid-tap addressing formula (createTextureTrainBatchComputeNode step 1) stays in bounds and covers every latent exactly once', () => {

					const latentIndices = simulateLatentIndices( layout );

					for ( const index of latentIndices ) {

						expect( index ).toBeGreaterThanOrEqual( 0 );
						expect( index ).toBeLessThan( layout.totalLatents );

					}

					expect( latentIndices.size ).toBe( layout.totalLatents );

				} );

				it( 'the a0 addressing formula (grid taps) exactly tiles [0, inputSize)', () => {

					const a0Indices = simulateA0Indices( layout );
					expect( a0Indices.size ).toBe( layout.inputSize );

					for ( const index of a0Indices ) {

						expect( index ).toBeGreaterThanOrEqual( layout.a0Offset );
						expect( index ).toBeLessThan( layout.a0Offset + layout.inputSize );

					}

				} );

				it( 'the weight-row/bias addressing formulas (forward + backward kernels) exactly tile [0, totalWeights) with no gaps/overlaps', () => {

					const regions = simulateWeightRegions( layout );
					expectExactPartition( regions, layout.totalWeights );

				} );

				it( 'the activation-buffer addressing (a0/z/a/delta/gradA0, forward + loss + backward kernels) exactly tiles [0, activationStride) with no gaps/overlaps', () => {

					const regions = simulateActivationRegions( layout );
					expectExactPartition( regions, layout.activationStride );

				} );

			} );

		}

	} );

	describe( 'NTCGPUModel buffer allocation', () => {

		const getRenderer = withTestRenderer( { beforeAll, afterAll } );

		it( 'allocates every storage buffer at exactly the size the training/Adam/gradient-norm kernels dispatch against', () => {

			// Touch the renderer getter so the shared WebGPU device from
			// withTestRenderer is actually initialized for this describe block,
			// matching the "real WebGPU" browser-project convention even though
			// buffer allocation itself is plain JS.
			expect( getRenderer() ).toBeTruthy();

			const model = new NTCGPUModel( { batchSize: 64, outputChannels: 5 } );
			const { totalWeights, totalLatents, activationStride } = model.layout;

			// These four pairs are independent JS expressions (`new
			// Float32Array/Int32Array(totalWeights)` etc.), not a shared formula -
			// a copy-paste slip sizing e.g. gradWeightsAttribute from totalLatents
			// instead of totalWeights would be caught here.
			expect( model.weightsBuffers.attribute.array.length ).toBe( totalWeights );
			expect( model.weightsBuffers.gradAttribute.array.length ).toBe( totalWeights );
			expect( model.weightsBuffers.mAttribute.array.length ).toBe( totalWeights );
			expect( model.weightsBuffers.vAttribute.array.length ).toBe( totalWeights );

			expect( model.latentsBuffers.attribute.array.length ).toBe( totalLatents );
			expect( model.latentsBuffers.gradAttribute.array.length ).toBe( totalLatents );
			expect( model.latentsBuffers.mAttribute.array.length ).toBe( totalLatents );
			expect( model.latentsBuffers.vAttribute.array.length ).toBe( totalLatents );

			// createTextureTrainBatchComputeNode dispatches `batchSize`
			// invocations, each indexing its private activation-buffer slice at
			// `instanceIndex * activationStride` (see actBase in
			// NeuralTextureGPUComputeTSL.js) - so the buffer must hold exactly
			// batchSize * activationStride floats.
			expect( model.activationsAttribute.array.length ).toBe( model.batchSize * activationStride );

			expect( model.lossAttribute.array.length ).toBe( 1 );
			expect( model.gradNormAttribute.array.length ).toBe( 1 );

			// createAccumulateGradientNormComputeNode dispatches
			// totalWeights + totalLatents invocations, routing [0, totalWeights)
			// into gradWeightsAtomic and [totalWeights, totalWeights+totalLatents)
			// (offset back down to [0, totalLatents)) into gradLatentsAtomic - so
			// both buffers together must exactly cover that dispatch range.
			expect( model.weightsBuffers.gradAttribute.array.length + model.latentsBuffers.gradAttribute.array.length )
				.toBe( totalWeights + totalLatents );

		} );

		it( 'initFromCPUModel/syncToCPU round-trip through the real WebGPU device, placing each MLP layer and grid level at the layout offsets the kernels read/write', async () => {

			const model = new NTCGPUModel( { batchSize: 16, hiddenSizes: [ 8 ], levels: 2, baseResolution: 4 } );

			// A tiny stand-in CPU model with recognizable, distinct weight/bias/
			// grid values per layer/level so a mislaid offset would show up as a
			// value landing in the wrong slot rather than silently matching.
			const cpuModel = {
				decoder: {
					layers: model.layout.mlpLayers.map( ( layer, l ) => ( {
						weights: new Float32Array( layer.weightsCount ).fill( l + 1 ),
						biases: new Float32Array( layer.biasesCount ).fill( - ( l + 1 ) )
					} ) )
				},
				grids: model.layout.gridLevels.map( ( level, g ) => ( {
					data: new Float32Array( level.floatCount ).fill( 100 + g )
				} ) )
			};

			model.initFromCPUModel( cpuModel );

			model.layout.mlpLayers.forEach( ( layer, l ) => {

				for ( let i = 0; i < layer.weightsCount; i ++ ) {

					expect( model.weightsBuffers.attribute.array[ layer.weightsOffset + i ] ).toBe( l + 1 );

				}

				for ( let i = 0; i < layer.biasesCount; i ++ ) {

					expect( model.weightsBuffers.attribute.array[ layer.biasesOffset + i ] ).toBe( - ( l + 1 ) );

				}

			} );

			model.layout.gridLevels.forEach( ( level, g ) => {

				for ( let i = 0; i < level.floatCount; i ++ ) {

					expect( model.latentsBuffers.attribute.array[ level.offset + i ] ).toBe( 100 + g );

				}

			} );

			// Round-trip through syncToCPU's readback path (getArrayBufferAsync)
			// using the real WebGPU device, reading the same offsets back out.
			// getArrayBufferAsync requires the backend to have already materialized
			// a GPU buffer for the attribute, which normally happens the first time
			// a kernel reads/writes it - so run a trivial identity compute pass over
			// both storages first (mirroring what every real training kernel does
			// implicitly before any readback in the trainer).
			const renderer = getRenderer();
			model.weightsBuffers.attribute.needsUpdate = true;
			model.latentsBuffers.attribute.needsUpdate = true;

			const identityWeights = Fn( () => {

				model.weightsBuffers.valuesStorage.element( instanceIndex ).assign( model.weightsBuffers.valuesStorage.element( instanceIndex ) );

			} )().compute( model.layout.totalWeights );

			const identityLatents = Fn( () => {

				model.latentsBuffers.valuesStorage.element( instanceIndex ).assign( model.latentsBuffers.valuesStorage.element( instanceIndex ) );

			} )().compute( model.layout.totalLatents );

			await renderer.computeAsync( identityWeights );
			await renderer.computeAsync( identityLatents );

			const readbackCpuModel = {
				decoder: {
					layers: model.layout.mlpLayers.map( layer => ( {
						weights: new Float32Array( layer.weightsCount ),
						biases: new Float32Array( layer.biasesCount )
					} ) )
				},
				grids: model.layout.gridLevels.map( level => ( { data: new Float32Array( level.floatCount ) } ) )
			};

			await model.syncToCPU( readbackCpuModel, renderer );

			model.layout.mlpLayers.forEach( ( layer, l ) => {

				expect( Array.from( readbackCpuModel.decoder.layers[ l ].weights ) ).toEqual( Array.from( cpuModel.decoder.layers[ l ].weights ) );
				expect( Array.from( readbackCpuModel.decoder.layers[ l ].biases ) ).toEqual( Array.from( cpuModel.decoder.layers[ l ].biases ) );

			} );

			model.layout.gridLevels.forEach( ( level, g ) => {

				expect( Array.from( readbackCpuModel.grids[ g ].data ) ).toEqual( Array.from( cpuModel.grids[ g ].data ) );

			} );

		} );

	} );

} );
