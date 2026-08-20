import { StorageBufferAttribute } from 'three/webgpu';
import { storage, uniform } from 'three/tsl';
import {
	IBL_OUTPUT_SIZE,
	INDIRECT_OUTPUT_SIZE,
	IBL_TARGET_SIZE,
	CHANNELS_PER_LEVEL,
	computeLatentChannels,
	computeDecoderInputSize,
	computeIblInputSize,
	computeIndirectInputSize,
	resolveNeuralAppearanceModelOptions
} from './NeuralAppearanceFormat.js';
import { computeGridLevels } from '../neural/NeuralGridModel.js';
import { FIXED_POINT_SCALE } from '../neural/NeuralGPUTrainingConstants.js';
import { createAdamParameterBuffers, disposeAdamParameterBuffers } from '../neural/NeuralGPUComputeTSL.js';

// Fields making up the "direct" part of each uploaded training sample
// (everything before the IBL query/probe block) - see uploadSamples() below.
const DIRECT_SAMPLE_SIZE = 17;

function allocateIndirectProbeHead( currentOffset, iblHiddenSize, indirectInputSize ) {

	const layer0WeightsOffset = currentOffset;
	const layer0WeightsCount = indirectInputSize * iblHiddenSize;
	const layer0BiasesOffset = layer0WeightsOffset + layer0WeightsCount;
	const layer0BiasesCount = iblHiddenSize;
	const layer1WeightsOffset = layer0BiasesOffset + layer0BiasesCount;
	const layer1WeightsCount = iblHiddenSize * INDIRECT_OUTPUT_SIZE;
	const layer1BiasesOffset = layer1WeightsOffset + layer1WeightsCount;
	const layer1BiasesCount = INDIRECT_OUTPUT_SIZE;

	return {
		layer0WeightsOffset,
		layer0WeightsCount,
		layer0BiasesOffset,
		layer0BiasesCount,
		layer1WeightsOffset,
		layer1WeightsCount,
		layer1BiasesOffset,
		layer1BiasesCount,
		nextOffset: layer1BiasesOffset + layer1BiasesCount
	};

}

/**
 * Computes buffer layouts and offsets for GPU-based neural appearance training.
 */
function computeModelLayout( options = {} ) {

	// See NeuralAppearanceFormat.resolveNeuralAppearanceModelOptions's doc
	// comment: this is the single source of truth for these defaults,
	// shared with createModel (CPU authoring) so the two can't silently
	// disagree.
	const { levels: requestedLevels, hiddenSize, iblHiddenSize, baseResolution, growthFactor, peOctaves, supportsEmission, supportsOpacity } = resolveNeuralAppearanceModelOptions( options );

	// `computeGridLevels` may return fewer levels than requested when a
	// level's resolution would exceed `MAX_GRID_RESOLUTION` (see
	// NeuralGridModel.js) - computed up front here (rather than down where
	// the grid layout itself is built, below) so every input-size/channel-
	// count derivation uses the actual level count, matching `createModel`'s
	// CPU-side layout exactly.
	const resolutions = computeGridLevels( baseResolution, growthFactor, requestedLevels );
	const levels = resolutions.length;
	const latentChannels = computeLatentChannels( levels );
	const decoderInputSize = computeDecoderInputSize( levels, peOctaves );
	const iblInputSize = computeIblInputSize( levels, peOctaves );
	const indirectInputSize = computeIndirectInputSize( levels, peOctaves );
	// Offset of the peOctaves*2 tiled-PE block within each of the 3 input
	// vectors above - always immediately after that vector's fixed portion
	// (12 frame dots for the decoder, 6 direction dots for IBL/indirect - see
	// NeuralAppearanceGPUComputeTSL.js's forward pass for where these get
	// written/copied).
	const decoderPeOffset = decoderInputSize - peOctaves * 2;
	const iblPeOffset = iblInputSize - peOctaves * 2;
	const indirectPeOffset = indirectInputSize - peOctaves * 2;

	// Weights Layout:
	// 0..(latentChannels*12-1): rotation weights (latentChannels channels * 12 outputs)
	const rotationOffset = 0;
	const rotationCount = latentChannels * 12;

	// Layer 0: decoderInputSize -> hiddenSize
	const layer0WeightsOffset = rotationOffset + rotationCount;
	const layer0WeightsCount = decoderInputSize * hiddenSize;
	const layer0BiasesOffset = layer0WeightsOffset + layer0WeightsCount;
	const layer0BiasesCount = hiddenSize;

	// Layer 1: hiddenSize -> hiddenSize
	const layer1WeightsOffset = layer0BiasesOffset + layer0BiasesCount;
	const layer1WeightsCount = hiddenSize * hiddenSize;
	const layer1BiasesOffset = layer1WeightsOffset + layer1WeightsCount;
	const layer1BiasesCount = hiddenSize;

	// Layer 2: hiddenSize -> 3
	const layer2WeightsOffset = layer1BiasesOffset + layer1BiasesCount;
	const layer2WeightsCount = hiddenSize * 3;
	const layer2BiasesOffset = layer2WeightsOffset + layer2WeightsCount;
	const layer2BiasesCount = 3;

	let currentOffset = layer2BiasesOffset + layer2BiasesCount;

	// Auxiliary: Emission Head (latentChannels -> 3)
	let emissionWeightsOffset = - 1;
	let emissionBiasesOffset = - 1;
	let emissionWeightsCount = 0;
	let emissionBiasesCount = 0;
	if ( supportsEmission ) {

		emissionWeightsOffset = currentOffset;
		emissionWeightsCount = latentChannels * 3;
		emissionBiasesOffset = emissionWeightsOffset + emissionWeightsCount;
		emissionBiasesCount = 3;
		currentOffset = emissionBiasesOffset + emissionBiasesCount;

	}

	// Auxiliary: Opacity Head (latentChannels -> 1)
	let opacityWeightsOffset = - 1;
	let opacityBiasesOffset = - 1;
	let opacityWeightsCount = 0;
	let opacityBiasesCount = 0;
	if ( supportsOpacity ) {

		opacityWeightsOffset = currentOffset;
		opacityWeightsCount = latentChannels * 1;
		opacityBiasesOffset = opacityWeightsOffset + opacityWeightsCount;
		opacityBiasesCount = 1;
		currentOffset = opacityBiasesOffset + opacityBiasesCount;

	}

	const directWeightCount = currentOffset;

	// IBL Head last so BRDF+aux Adam can update a contiguous prefix.
	const iblLayer0WeightsOffset = currentOffset;
	const iblLayer0WeightsCount = iblInputSize * iblHiddenSize;
	const iblLayer0BiasesOffset = iblLayer0WeightsOffset + iblLayer0WeightsCount;
	const iblLayer0BiasesCount = iblHiddenSize;
	const iblLayer1WeightsOffset = iblLayer0BiasesOffset + iblLayer0BiasesCount;
	const iblLayer1WeightsCount = iblHiddenSize * IBL_OUTPUT_SIZE;
	const iblLayer1BiasesOffset = iblLayer1WeightsOffset + iblLayer1WeightsCount;
	const iblLayer1BiasesCount = IBL_OUTPUT_SIZE;
	currentOffset = iblLayer1BiasesOffset + iblLayer1BiasesCount;

	const indirectRadiance = allocateIndirectProbeHead( currentOffset, iblHiddenSize, indirectInputSize );
	currentOffset = indirectRadiance.nextOffset;
	const indirectIrradiance = allocateIndirectProbeHead( currentOffset, iblHiddenSize, indirectInputSize );
	currentOffset = indirectIrradiance.nextOffset;
	const iblWeightCount = currentOffset - directWeightCount;

	const totalWeights = currentOffset;

	// Multiresolution Latent Grid Layout (instant-ngp / NVIDIA NTC style - same
	// geometry as neural-texture / neural-material, see NeuralGridModel.js):
	// `levels` grids starting at `baseResolution` and multiplying by
	// `growthFactor` for each additional level, each contributing
	// `CHANNELS_PER_LEVEL` features that get concatenated (not summed) into
	// the decoder input. (`resolutions`/`levels` were already resolved above,
	// alongside latentChannels/decoderInputSize/iblInputSize/
	// indirectInputSize, so they can't disagree.)
	const gridLevels = [];
	let latentOffset = 0;

	for ( const resolution of resolutions ) {

		const texelCount = resolution * resolution;
		const floatCount = texelCount * CHANNELS_PER_LEVEL;
		gridLevels.push( { width: resolution, height: resolution, offset: latentOffset, texelCount, floatCount } );
		latentOffset += floatCount;

	}

	const totalLatents = latentOffset;

	// Per-Sample Activations Layout:
	// a0: decoderInputSize
	// z1: H
	// a1: H
	// z2: H
	// a2: H
	// z3: 3
	// delta3: 3
	// delta2: H
	// delta1: H
	// gradA0: decoderInputSize
	// gradLatents: latentChannels
	const actA0Offset = 0;
	const actZ1Offset = actA0Offset + decoderInputSize;
	const actA1Offset = actZ1Offset + hiddenSize;
	const actZ2Offset = actA1Offset + hiddenSize;
	const actA2Offset = actZ2Offset + hiddenSize;
	const actZ3Offset = actA2Offset + hiddenSize;
	const actDelta3Offset = actZ3Offset + 3;
	const actDelta2Offset = actDelta3Offset + 3;
	const actDelta1Offset = actDelta2Offset + hiddenSize;
	const actGradA0Offset = actDelta1Offset + hiddenSize;
	const actGradLatentsOffset = actGradA0Offset + decoderInputSize;
	let actCurrent = actGradLatentsOffset + latentChannels;

	let actEmissionOffset = - 1;
	if ( supportsEmission ) {

		actEmissionOffset = actCurrent;
		actCurrent += 6; // z_em(3) + delta_em(3)

	}

	let actOpacityOffset = - 1;
	if ( supportsOpacity ) {

		actOpacityOffset = actCurrent;
		actCurrent += 2; // z_op(1) + delta_op(1)

	}

	const actIblA0Offset = actCurrent;
	const actIblZ1Offset = actIblA0Offset + iblInputSize;
	const actIblA1Offset = actIblZ1Offset + iblHiddenSize;
	const actIblZ2Offset = actIblA1Offset + iblHiddenSize;
	const actIblDelta2Offset = actIblZ2Offset + IBL_OUTPUT_SIZE;
	const actIblDelta1Offset = actIblDelta2Offset + IBL_OUTPUT_SIZE;
	actCurrent = actIblDelta1Offset + iblHiddenSize;

	const actIndirectA0Offset = actCurrent;
	const actIndirectZ1Offset = actIndirectA0Offset + indirectInputSize;
	const actIndirectA1Offset = actIndirectZ1Offset + iblHiddenSize;
	const actIndirectZ2Offset = actIndirectA1Offset + iblHiddenSize;
	const actIndirectDelta2Offset = actIndirectZ2Offset + INDIRECT_OUTPUT_SIZE;
	const actIndirectDelta1Offset = actIndirectDelta2Offset + INDIRECT_OUTPUT_SIZE;
	actCurrent = actIndirectDelta1Offset + iblHiddenSize;

	const activationStride = actCurrent;

	const sampleStride = DIRECT_SAMPLE_SIZE + IBL_TARGET_SIZE;

	return {
		hiddenSize,
		iblHiddenSize,
		supportsEmission,
		supportsOpacity,
		totalWeights,
		directWeightCount,
		iblWeightCount,
		rotationOffset,
		rotationCount,
		layer0WeightsOffset,
		layer0WeightsCount,
		layer0BiasesOffset,
		layer0BiasesCount,
		layer1WeightsOffset,
		layer1WeightsCount,
		layer1BiasesOffset,
		layer1BiasesCount,
		layer2WeightsOffset,
		layer2WeightsCount,
		layer2BiasesOffset,
		layer2BiasesCount,
		iblLayer0WeightsOffset,
		iblLayer0WeightsCount,
		iblLayer0BiasesOffset,
		iblLayer0BiasesCount,
		iblLayer1WeightsOffset,
		iblLayer1WeightsCount,
		iblLayer1BiasesOffset,
		iblLayer1BiasesCount,
		indirectRadianceLayer0WeightsOffset: indirectRadiance.layer0WeightsOffset,
		indirectRadianceLayer0WeightsCount: indirectRadiance.layer0WeightsCount,
		indirectRadianceLayer0BiasesOffset: indirectRadiance.layer0BiasesOffset,
		indirectRadianceLayer0BiasesCount: indirectRadiance.layer0BiasesCount,
		indirectRadianceLayer1WeightsOffset: indirectRadiance.layer1WeightsOffset,
		indirectRadianceLayer1WeightsCount: indirectRadiance.layer1WeightsCount,
		indirectRadianceLayer1BiasesOffset: indirectRadiance.layer1BiasesOffset,
		indirectRadianceLayer1BiasesCount: indirectRadiance.layer1BiasesCount,
		indirectIrradianceLayer0WeightsOffset: indirectIrradiance.layer0WeightsOffset,
		indirectIrradianceLayer0WeightsCount: indirectIrradiance.layer0WeightsCount,
		indirectIrradianceLayer0BiasesOffset: indirectIrradiance.layer0BiasesOffset,
		indirectIrradianceLayer0BiasesCount: indirectIrradiance.layer0BiasesCount,
		indirectIrradianceLayer1WeightsOffset: indirectIrradiance.layer1WeightsOffset,
		indirectIrradianceLayer1WeightsCount: indirectIrradiance.layer1WeightsCount,
		indirectIrradianceLayer1BiasesOffset: indirectIrradiance.layer1BiasesOffset,
		indirectIrradianceLayer1BiasesCount: indirectIrradiance.layer1BiasesCount,
		emissionWeightsOffset,
		emissionWeightsCount,
		emissionBiasesOffset,
		emissionBiasesCount,
		opacityWeightsOffset,
		opacityWeightsCount,
		opacityBiasesOffset,
		opacityBiasesCount,
		levels,
		latentChannels,
		peOctaves,
		decoderInputSize,
		iblInputSize,
		indirectInputSize,
		decoderPeOffset,
		iblPeOffset,
		indirectPeOffset,
		gridLevels,
		totalLatents,
		activationStride,
		actA0Offset,
		actZ1Offset,
		actA1Offset,
		actZ2Offset,
		actA2Offset,
		actZ3Offset,
		actDelta3Offset,
		actDelta2Offset,
		actDelta1Offset,
		actGradA0Offset,
		actGradLatentsOffset,
		actEmissionOffset,
		actOpacityOffset,
		actIblA0Offset,
		actIblZ1Offset,
		actIblA1Offset,
		actIblZ2Offset,
		actIblDelta2Offset,
		actIblDelta1Offset,
		actIndirectA0Offset,
		actIndirectZ1Offset,
		actIndirectA1Offset,
		actIndirectZ2Offset,
		actIndirectDelta2Offset,
		actIndirectDelta1Offset,
		sampleStride
	};

}

/**
 * Encapsulates the GPU StorageBuffers, uniforms, and CPU synchronizers for neural training.
 */
class NeuralAppearanceGPUModel {

	constructor( options = {} ) {

		this.options = options;
		this.batchSize = options.batchSize || 1024;
		this.layout = computeModelLayout( options );

		// 1. Weight buffers
		const { totalWeights, totalLatents, sampleStride, activationStride } = this.layout;
		const batchSize = this.batchSize;

		this.weightsBuffers = createAdamParameterBuffers( totalWeights );

		// 2. Latent buffers
		this.latentsBuffers = createAdamParameterBuffers( totalLatents );

		// 3. Sample buffer
		this.samplesAttribute = new StorageBufferAttribute( new Float32Array( batchSize * sampleStride ), 1, Float32Array );
		this.samplesStorage = storage( this.samplesAttribute, 'float', batchSize * sampleStride );

		// 4. Activation buffer
		this.activationsAttribute = new StorageBufferAttribute( new Float32Array( batchSize * activationStride ), 1, Float32Array );
		this.activationsStorage = storage( this.activationsAttribute, 'float', batchSize * activationStride );

		// 5. Batch loss atomic buffer
		this.lossAttribute = new StorageBufferAttribute( new Int32Array( 3 ), 1, Int32Array );
		this.lossAtomic = storage( this.lossAttribute, 'int', 3 ).toAtomic();

		// 6. Gradient clipping accumulator
		this.gradNormAttribute = new StorageBufferAttribute( new Int32Array( 1 ), 1, Int32Array );
		this.gradNormAtomic = storage( this.gradNormAttribute, 'int', 1 ).toAtomic();

		// 7. Uniforms
		this.invBatchUniform = uniform( 1.0 / batchSize );
		this.learningRateUniform = uniform( options.learningRate || 0.001 );
		this.stepUniform = uniform( 1 );
		this.maxGradientNormUniform = uniform( options.maxGradientNorm || 1 );

	}

	initFromCPUModel( cpuModel ) {

		const weights = this.weightsBuffers.attribute.array;
		const latents = this.latentsBuffers.attribute.array;

		weights.fill( 0 );
		latents.fill( 0 );

		// Rotation weights - this.layout.latentChannels (not the fixed
		// LATENT_CHANNELS constant) is this model's actual latentChannels*12
		// rotation-weight block size, matching cpuModel.rotationWeights'
		// actual length (see NeuralAppearanceModel.js's createModel). Using
		// the fixed constant here would read/write the wrong number of
		// entries whenever this model's `levels` isn't the default, silently
		// aliasing into the decoder's weight block right after it.
		for ( let i = 0; i < this.layout.latentChannels * 12; i ++ ) {

			weights[ this.layout.rotationOffset + i ] = cpuModel.rotationWeights[ i ] || 0;

		}

		// Decoder Layer 0 (decoderInputSize -> H)
		copyLayerWeightsToGPU( cpuModel.decoder.layers[ 0 ], weights, this.layout.layer0WeightsOffset, this.layout.layer0BiasesOffset );

		// Decoder Layer 1 (H -> H)
		copyLayerWeightsToGPU( cpuModel.decoder.layers[ 1 ], weights, this.layout.layer1WeightsOffset, this.layout.layer1BiasesOffset );

		// Decoder Layer 2 (H -> 3)
		copyLayerWeightsToGPU( cpuModel.decoder.layers[ 2 ], weights, this.layout.layer2WeightsOffset, this.layout.layer2BiasesOffset );

		// IBL Head
		copyLayerWeightsToGPU( cpuModel.iblHead.layers[ 0 ], weights, this.layout.iblLayer0WeightsOffset, this.layout.iblLayer0BiasesOffset );
		copyLayerWeightsToGPU( cpuModel.iblHead.layers[ 1 ], weights, this.layout.iblLayer1WeightsOffset, this.layout.iblLayer1BiasesOffset );
		copyLayerWeightsToGPU( cpuModel.indirectRadianceHead.layers[ 0 ], weights, this.layout.indirectRadianceLayer0WeightsOffset, this.layout.indirectRadianceLayer0BiasesOffset );
		copyLayerWeightsToGPU( cpuModel.indirectRadianceHead.layers[ 1 ], weights, this.layout.indirectRadianceLayer1WeightsOffset, this.layout.indirectRadianceLayer1BiasesOffset );
		copyLayerWeightsToGPU( cpuModel.indirectIrradianceHead.layers[ 0 ], weights, this.layout.indirectIrradianceLayer0WeightsOffset, this.layout.indirectIrradianceLayer0BiasesOffset );
		copyLayerWeightsToGPU( cpuModel.indirectIrradianceHead.layers[ 1 ], weights, this.layout.indirectIrradianceLayer1WeightsOffset, this.layout.indirectIrradianceLayer1BiasesOffset );

		// Emission Head (latentChannels -> 3)
		if ( cpuModel.emissionHead ) {

			copyLayerWeightsToGPU( cpuModel.emissionHead.layers[ 0 ], weights, this.layout.emissionWeightsOffset, this.layout.emissionBiasesOffset );

		}

		// Opacity Head (latentChannels -> 1)
		if ( cpuModel.opacityHead ) {

			copyLayerWeightsToGPU( cpuModel.opacityHead.layers[ 0 ], weights, this.layout.opacityWeightsOffset, this.layout.opacityBiasesOffset );

		}

		// Latent Grid Levels
		for ( let g = 0; g < cpuModel.latentGrids.length; g ++ ) {

			const level = this.layout.gridLevels[ g ];
			const grid = cpuModel.latentGrids[ g ];
			for ( let i = 0; i < level.floatCount; i ++ ) {

				latents[ level.offset + i ] = grid.data[ i ];

			}

		}

		this.weightsBuffers.attribute.needsUpdate = true;
		this.latentsBuffers.attribute.needsUpdate = true;

	}

	uploadSamples( samples, learningRate = 0.001, step = 1, maxGradientNorm = this.options.maxGradientNorm || 1 ) {

		const data = this.samplesAttribute.array;
		data.fill( 0 );

		if ( samples.length > this.batchSize ) {

			throw new Error( `THREE.NeuralAppearanceGPUModel: Batch contains ${ samples.length } samples, but GPU buffers were allocated for ${ this.batchSize }.` );

		}

		let weightSum = 0;
		for ( const sample of samples ) {

			weightSum += sample.weight !== undefined ? sample.weight : 1.0;

		}

		const invBatch = weightSum > 0 ? 1 / weightSum : 1 / Math.max( 1, samples.length );
		this.invBatchUniform.value = invBatch;
		this.learningRateUniform.value = learningRate;
		this.stepUniform.value = step;
		this.maxGradientNormUniform.value = maxGradientNorm;

		const count = Math.min( samples.length, this.batchSize );
		const stride = this.layout.sampleStride;

		for ( let i = 0; i < count; i ++ ) {

			const sample = samples[ i ];
			const base = i * stride;

			const uv = sample.uv || [ 0.5, 0.5 ];
			const wi = sample.wi || [ 0, 0, 1 ];
			const wo = sample.wo || [ 0, 0, 1 ];
			const target = sample.target || [ 0, 0, 0 ];
			const weight = sample.weight !== undefined ? sample.weight : 1.0;

			data[ base + 0 ] = uv[ 0 ];
			data[ base + 1 ] = uv[ 1 ];
			data[ base + 2 ] = wi[ 0 ];
			data[ base + 3 ] = wi[ 1 ];
			data[ base + 4 ] = wi[ 2 ];
			data[ base + 5 ] = wo[ 0 ];
			data[ base + 6 ] = wo[ 1 ];
			data[ base + 7 ] = wo[ 2 ];
			data[ base + 8 ] = weight;
			data[ base + 9 ] = target[ 0 ];
			data[ base + 10 ] = target[ 1 ];
			data[ base + 11 ] = target[ 2 ];

			if ( sample.emissionTarget ) {

				data[ base + 12 ] = 1.0;
				data[ base + 13 ] = sample.emissionTarget[ 0 ];
				data[ base + 14 ] = sample.emissionTarget[ 1 ];
				data[ base + 15 ] = sample.emissionTarget[ 2 ];

			} else {

				data[ base + 12 ] = 0.0;
				data[ base + 13 ] = 0;
				data[ base + 14 ] = 0;
				data[ base + 15 ] = 0;

			}

			if ( Number.isFinite( sample.opacityTarget ) ) {

				data[ base + 16 ] = sample.opacityTarget;

			} else {

				data[ base + 16 ] = - 1.0;

			}

			data[ base + 17 ] = sample.iblWeight !== undefined ? sample.iblWeight : 0;
			data[ base + 18 ] = ( sample.iblDirection || [ 0, 0, 1 ] )[ 0 ];
			data[ base + 19 ] = ( sample.iblDirection || [ 0, 0, 1 ] )[ 1 ];
			data[ base + 20 ] = ( sample.iblDirection || [ 0, 0, 1 ] )[ 2 ];
			data[ base + 21 ] = Number.isFinite( sample.iblRoughness ) ? sample.iblRoughness : 1;
			data[ base + 22 ] = ( sample.iblIncoming || [ 0, 0, 0 ] )[ 0 ];
			data[ base + 23 ] = ( sample.iblIncoming || [ 0, 0, 0 ] )[ 1 ];
			data[ base + 24 ] = ( sample.iblIncoming || [ 0, 0, 0 ] )[ 2 ];
			data[ base + 25 ] = ( sample.iblIrradiance || [ 0, 0, 0 ] )[ 0 ];
			data[ base + 26 ] = ( sample.iblIrradiance || [ 0, 0, 0 ] )[ 1 ];
			data[ base + 27 ] = ( sample.iblIrradiance || [ 0, 0, 0 ] )[ 2 ];
			data[ base + 28 ] = ( sample.iblIndirectRadiance || [ 0, 0, 0 ] )[ 0 ];
			data[ base + 29 ] = ( sample.iblIndirectRadiance || [ 0, 0, 0 ] )[ 1 ];
			data[ base + 30 ] = ( sample.iblIndirectRadiance || [ 0, 0, 0 ] )[ 2 ];
			data[ base + 31 ] = ( sample.iblIndirectIrradiance || [ 0, 0, 0 ] )[ 0 ];
			data[ base + 32 ] = ( sample.iblIndirectIrradiance || [ 0, 0, 0 ] )[ 1 ];
			data[ base + 33 ] = ( sample.iblIndirectIrradiance || [ 0, 0, 0 ] )[ 2 ];

		}

		this.samplesAttribute.needsUpdate = true;

	}

	resetLoss() {

		this.lossAttribute.array[ 0 ] = 0;
		this.lossAttribute.array[ 1 ] = 0;
		this.lossAttribute.array[ 2 ] = 0;
		this.lossAttribute.needsUpdate = true;

	}

	async syncToCPU( cpuModel, renderer ) {

		const [ weightsBuffer, latentsBuffer ] = await Promise.all( [
			renderer.getArrayBufferAsync( this.weightsBuffers.attribute ),
			renderer.getArrayBufferAsync( this.latentsBuffers.attribute )
		] );

		const weights = new Float32Array( weightsBuffer );
		const latents = new Float32Array( latentsBuffer );

		// Copy rotation weights - see initFromCPUModel's matching comment on
		// why this.layout.latentChannels, not the fixed LATENT_CHANNELS.
		for ( let i = 0; i < this.layout.latentChannels * 12; i ++ ) {

			cpuModel.rotationWeights[ i ] = weights[ this.layout.rotationOffset + i ];

		}

		// Copy decoder layers
		copyLayerWeightsFromGPU( cpuModel.decoder.layers[ 0 ], weights, this.layout.layer0WeightsOffset, this.layout.layer0BiasesOffset );
		copyLayerWeightsFromGPU( cpuModel.decoder.layers[ 1 ], weights, this.layout.layer1WeightsOffset, this.layout.layer1BiasesOffset );
		copyLayerWeightsFromGPU( cpuModel.decoder.layers[ 2 ], weights, this.layout.layer2WeightsOffset, this.layout.layer2BiasesOffset );
		copyLayerWeightsFromGPU( cpuModel.iblHead.layers[ 0 ], weights, this.layout.iblLayer0WeightsOffset, this.layout.iblLayer0BiasesOffset );
		copyLayerWeightsFromGPU( cpuModel.iblHead.layers[ 1 ], weights, this.layout.iblLayer1WeightsOffset, this.layout.iblLayer1BiasesOffset );
		copyLayerWeightsFromGPU( cpuModel.indirectRadianceHead.layers[ 0 ], weights, this.layout.indirectRadianceLayer0WeightsOffset, this.layout.indirectRadianceLayer0BiasesOffset );
		copyLayerWeightsFromGPU( cpuModel.indirectRadianceHead.layers[ 1 ], weights, this.layout.indirectRadianceLayer1WeightsOffset, this.layout.indirectRadianceLayer1BiasesOffset );
		copyLayerWeightsFromGPU( cpuModel.indirectIrradianceHead.layers[ 0 ], weights, this.layout.indirectIrradianceLayer0WeightsOffset, this.layout.indirectIrradianceLayer0BiasesOffset );
		copyLayerWeightsFromGPU( cpuModel.indirectIrradianceHead.layers[ 1 ], weights, this.layout.indirectIrradianceLayer1WeightsOffset, this.layout.indirectIrradianceLayer1BiasesOffset );

		if ( cpuModel.emissionHead ) {

			copyLayerWeightsFromGPU( cpuModel.emissionHead.layers[ 0 ], weights, this.layout.emissionWeightsOffset, this.layout.emissionBiasesOffset );

		}

		if ( cpuModel.opacityHead ) {

			copyLayerWeightsFromGPU( cpuModel.opacityHead.layers[ 0 ], weights, this.layout.opacityWeightsOffset, this.layout.opacityBiasesOffset );

		}

		// Copy latent grid levels
		for ( let g = 0; g < cpuModel.latentGrids.length; g ++ ) {

			const level = this.layout.gridLevels[ g ];
			const grid = cpuModel.latentGrids[ g ];
			for ( let i = 0; i < level.floatCount; i ++ ) {

				grid.data[ i ] = latents[ level.offset + i ];

			}

		}

	}

	async readLoss( renderer ) {

		const losses = await this.readLosses( renderer );
		return losses.loss;

	}

	async readLosses( renderer ) {

		const buffer = await renderer.getArrayBufferAsync( this.lossAttribute );
		const array = new Int32Array( buffer );
		const losses = {
			loss: array[ 0 ] / FIXED_POINT_SCALE,
			directLoss: array[ 1 ] / FIXED_POINT_SCALE,
			iblLoss: array[ 2 ] / FIXED_POINT_SCALE
		};

		this.lossAttribute.array[ 0 ] = 0;
		this.lossAttribute.array[ 1 ] = 0;
		this.lossAttribute.array[ 2 ] = 0;
		this.lossAttribute.needsUpdate = true;

		return losses;

	}

	// Frees the GPU-resident storage buffers (weights, weight/latent
	// gradients, weight/latent Adam moments, samples, activations, loss,
	// gradient-norm accumulator) backing this model. Without this, every
	// training run leaks its full set of compute buffers for the lifetime of
	// the WebGPU device -- StorageBufferAttribute.dispose() alone is not
	// enough here (attributes.js only wires a GPU-side cleanup listener for
	// attributes attached to a BufferGeometry via a render object; these are
	// standalone compute-only buffers, never geometry-attached, so nothing
	// currently listens for their 'dispose' event). `renderer.backend` is
	// reached into directly to actually release the GPU buffer -- there is
	// currently no higher-level public renderer API for deallocating a
	// standalone storage buffer.
	dispose( renderer ) {

		disposeAdamParameterBuffers( this.weightsBuffers, renderer );
		disposeAdamParameterBuffers( this.latentsBuffers, renderer );

		for ( const attribute of [ this.samplesAttribute, this.activationsAttribute, this.lossAttribute, this.gradNormAttribute ] ) {

			if ( renderer && renderer.backend && renderer.backend.has( attribute ) ) {

				renderer.backend.destroyAttribute( attribute );

			}

			attribute.dispose();

		}

	}

}

function copyLayerWeightsToGPU( layer, target, weightsOffset, biasesOffset ) {

	for ( let i = 0; i < layer.weights.length; i ++ ) {

		target[ weightsOffset + i ] = layer.weights[ i ];

	}

	for ( let i = 0; i < layer.biases.length; i ++ ) {

		target[ biasesOffset + i ] = layer.biases[ i ];

	}

}

function copyLayerWeightsFromGPU( layer, source, weightsOffset, biasesOffset ) {

	for ( let i = 0; i < layer.weights.length; i ++ ) {

		layer.weights[ i ] = source[ weightsOffset + i ];

	}

	for ( let i = 0; i < layer.biases.length; i ++ ) {

		layer.biases[ i ] = source[ biasesOffset + i ];

	}

}

export {
	computeModelLayout,
	NeuralAppearanceGPUModel,
	DIRECT_SAMPLE_SIZE
};
