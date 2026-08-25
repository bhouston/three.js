import { StorageBufferAttribute } from 'three/webgpu';
import { storage, uniform } from 'three/tsl';
import { FIXED_POINT_SCALE } from './NTCGPUTrainingConstants.js';
import { createAdamParameterBuffers, disposeAdamParameterBuffers } from './NTCGPUKernelsTSL.js';
import { computeGridLevels } from './NTCGridModel.js';
import { resolveNTCGridPyramidOptions } from './NTCGridPyramidModel.js';
import { resolveQuantizationConfig } from './NTCQuantization.js';
import { computeAllNaturalLods } from '../NTCMipPyramid.js';

/**
 * Computes buffer layouts and offsets for GPU-based neural texture training:
 * a multiresolution feature grid pyramid plus a small MLP decoder, sized
 * generically from an arbitrary hidden-layer configuration.
 *
 * Mirrors `createNTCGridPyramidModel` (NTCGridPyramidModel.js) exactly for
 * `mipPyramid`/`inputSize` - both derive the CPU model and this GPU layout
 * from the same `resolveNTCGridPyramidOptions`, so a trainer's CPU model and
 * its GPU buffer layout can never disagree about whether an LOD input slot
 * exists or how many grid levels feed it.
 */
function computeTextureModelLayout( options = {} ) {

	const { channels, levels: requestedLevels, baseResolution, growthFactor, hiddenSizes, outputChannels, enableMipPyramid, textureResolution } = resolveNTCGridPyramidOptions( options );
	// One entry per output channel naming its output nonlinearity (see
	// ./NTCOutputActivations.js); undefined/omitted entries (the
	// default, `options.channelActivations` unset) mean plain linear, i.e.
	// today's behavior for every neural-texture caller that doesn't pass
	// this - only neural-material does (see NeuralMaterialFormat.js).
	const channelActivations = options.channelActivations || null;

	// `computeGridLevels` may return fewer levels than requested when a
	// level's resolution would exceed `MAX_GRID_RESOLUTION` (see
	// NeuralGridModel.js) - `levels` below is reassigned to the actual grid
	// count so `inputSize` matches `NeuralTextureModel`'s CPU decoder exactly.
	const resolutions = computeGridLevels( baseResolution, growthFactor, requestedLevels );
	const levels = resolutions.length;

	const gridLevels = [];
	let latentOffset = 0;

	for ( const resolution of resolutions ) {

		const texelCount = resolution * resolution;
		const floatCount = texelCount * channels;
		gridLevels.push( { width: resolution, height: resolution, offset: latentOffset, texelCount, floatCount } );
		latentOffset += floatCount;

	}

	const totalLatents = latentOffset;

	// Mip-pyramid metadata (see NTCMipPyramid.js) - `null` when disabled,
	// matching `createNTCGridPyramidModel`'s `mipPyramid` field exactly.
	let mipPyramid = null;

	if ( enableMipPyramid ) {

		const resolvedTextureResolution = textureResolution || resolutions[ resolutions.length - 1 ];
		const naturalLods = computeAllNaturalLods( resolutions, resolvedTextureResolution );
		const maxLod = Math.max( 1, Math.ceil( Math.log2( Math.max( 1, resolvedTextureResolution ) ) ) );
		mipPyramid = { textureResolution: resolvedTextureResolution, naturalLods, maxLod };

	}

	// MLP weight layout: input = concatenated multiresolution grid features,
	// plus one extra slot for the normalized LOD value when mip-pyramid
	// training is enabled (see NTCGPUComputeTSL.js step 1, which appends it
	// after every grid level's taps).
	const inputSize = levels * channels + ( mipPyramid ? 1 : 0 );
	const sizes = [ inputSize, ...hiddenSizes, outputChannels ];
	const mlpLayers = [];
	let weightOffset = 0;

	for ( let i = 0; i < sizes.length - 1; i ++ ) {

		const inSize = sizes[ i ];
		const outSize = sizes[ i + 1 ];
		const weightsOffset = weightOffset;
		const weightsCount = inSize * outSize;
		const biasesOffset = weightsOffset + weightsCount;
		const biasesCount = outSize;
		weightOffset = biasesOffset + biasesCount;

		mlpLayers.push( {
			inputSize: inSize,
			outputSize: outSize,
			weightsOffset,
			weightsCount,
			biasesOffset,
			biasesCount,
			isOutput: i === sizes.length - 2
		} );

	}

	const totalWeights = weightOffset;

	// Per-sample activation buffer layout (forward a0/z/a, backward deltas, gradA0).
	let cursor = 0;
	const a0Offset = cursor; cursor += inputSize;
	const layerActs = [];

	for ( const layer of mlpLayers ) {

		const zOffset = cursor; cursor += layer.outputSize;
		let aOffset = - 1;

		if ( layer.isOutput === false ) {

			aOffset = cursor; cursor += layer.outputSize;

		}

		layerActs.push( { zOffset, aOffset } );

	}

	const deltaOffsets = [];

	for ( const layer of mlpLayers ) {

		deltaOffsets.push( cursor );
		cursor += layer.outputSize;

	}

	const gradA0Offset = cursor; cursor += inputSize;
	const activationStride = cursor;

	return {
		channels,
		levels,
		resolutions,
		hiddenSizes,
		outputChannels,
		channelActivations,
		mipPyramid,
		inputSize,
		gridLevels,
		totalLatents,
		mlpLayers,
		totalWeights,
		a0Offset,
		layerActs,
		deltaOffsets,
		gradA0Offset,
		activationStride
	};

}

/**
 * Copies weights/biases and latent-grid data between the CPU reference
 * model and a flat GPU-layout-shaped array, in either direction - the two
 * are identical index-for-index copy loops with only the assignment side
 * flipped, so `initFromCPUModel`/`syncToCPU` both delegate here rather than
 * maintaining that loop twice.
 */
function copyModel( cpuModel, layout, weights, latents, direction ) {

	for ( let l = 0; l < cpuModel.decoder.layers.length; l ++ ) {

		const layer = cpuModel.decoder.layers[ l ];
		const layerLayout = layout.mlpLayers[ l ];

		if ( direction === 'toGPU' ) {

			for ( let i = 0; i < layer.weights.length; i ++ ) weights[ layerLayout.weightsOffset + i ] = layer.weights[ i ];
			for ( let i = 0; i < layer.biases.length; i ++ ) weights[ layerLayout.biasesOffset + i ] = layer.biases[ i ];

		} else {

			for ( let i = 0; i < layer.weights.length; i ++ ) layer.weights[ i ] = weights[ layerLayout.weightsOffset + i ];
			for ( let i = 0; i < layer.biases.length; i ++ ) layer.biases[ i ] = weights[ layerLayout.biasesOffset + i ];

		}

	}

	for ( let g = 0; g < cpuModel.grids.length; g ++ ) {

		const grid = cpuModel.grids[ g ];
		const level = layout.gridLevels[ g ];

		if ( direction === 'toGPU' ) {

			for ( let i = 0; i < grid.data.length; i ++ ) latents[ level.offset + i ] = grid.data[ i ];

		} else {

			for ( let i = 0; i < grid.data.length; i ++ ) grid.data[ i ] = latents[ level.offset + i ];

		}

	}

}

/**
 * Encapsulates the GPU StorageBuffers, uniforms, and CPU synchronizers for
 * neural texture training. `weightsBuffers`/`latentsBuffers` shapes
 * intentionally mirror `NeuralAppearanceGPUModel`'s so the Adam / gradient-
 * clip compute kernels can be reused verbatim.
 */
class NTCGPUModel {

	constructor( options = {} ) {

		this.options = options;
		this.batchSize = options.batchSize || 4096;
		this.layout = computeTextureModelLayout( options );

		const { totalWeights, totalLatents, activationStride } = this.layout;
		const batchSize = this.batchSize;

		this.weightsBuffers = createAdamParameterBuffers( totalWeights );
		this.latentsBuffers = createAdamParameterBuffers( totalLatents );

		this.activationsAttribute = new StorageBufferAttribute( new Float32Array( batchSize * activationStride ), 1, Float32Array );
		this.activationsStorage = storage( this.activationsAttribute, 'float', batchSize * activationStride );

		this.lossAttribute = new StorageBufferAttribute( new Int32Array( 1 ), 1, Int32Array );
		this.lossAtomic = storage( this.lossAttribute, 'int', 1 ).toAtomic();

		this.gradNormAttribute = new StorageBufferAttribute( new Int32Array( 1 ), 1, Int32Array );
		this.gradNormAtomic = storage( this.gradNormAttribute, 'int', 1 ).toAtomic();

		this.invBatchUniform = uniform( 1.0 / batchSize );
		this.learningRateUniform = uniform( options.learningRate || 0.01 );
		this.stepUniform = uniform( 1 );
		this.maxGradientNormUniform = uniform( options.maxGradientNorm || 1 );

		// QAT (see NeuralQuantization.js) - `quantization.mode !== 'none'`
		// makes NeuralTextureGPUComputeTSL.js's forward pass quantize every
		// bilinear-sampled latent before it's consumed by the MLP. One
		// [min, max] uniform pair per grid level, initialized to a generous
		// placeholder range (latents are init'd within
		// [-LATENT_INIT_SCALE, LATENT_INIT_SCALE], see NeuralGridModel.js, but
		// grow during training) so nothing is clipped before the first 'auto'
		// range refresh (see NTCTrainer.js) actually measures the
		// real range - a fixed `range` tuple is written once here and never
		// refreshed.
		this.quantization = resolveQuantizationConfig( options );
		this.quantizationRangeUniforms = this.layout.gridLevels.map( () => ( {
			min: uniform( this.quantization.range === 'auto' ? - 1 : this.quantization.range[ 0 ] ),
			max: uniform( this.quantization.range === 'auto' ? 1 : this.quantization.range[ 1 ] )
		} ) );

	}

	/**
	 * Updates every per-level quantization-range uniform from a freshly
	 * computed `[min, max]` tuple array (see
	 * `NeuralQuantization.computeLatentRanges`) - called periodically (and
	 * once more at the very end) by NTCTrainer.js when
	 * `quantization.range === 'auto'`.
	 */
	setQuantizationRange( ranges ) {

		for ( let g = 0; g < this.quantizationRangeUniforms.length; g ++ ) {

			this.quantizationRangeUniforms[ g ].min.value = ranges[ g ][ 0 ];
			this.quantizationRangeUniforms[ g ].max.value = ranges[ g ][ 1 ];

		}

	}

	/**
	 * Reads back the current per-level quantization range as plain
	 * `[min, max]` arrays (not TSL uniform nodes) - used to freeze the final
	 * range at the end of training (see NTCTrainer.js).
	 */
	getQuantizationRange() {

		return this.quantizationRangeUniforms.map( ( { min, max } ) => [ min.value, max.value ] );

	}

	initFromCPUModel( cpuModel ) {

		const weights = this.weightsBuffers.attribute.array;
		const latents = this.latentsBuffers.attribute.array;
		weights.fill( 0 );
		latents.fill( 0 );

		copyModel( cpuModel, this.layout, weights, latents, 'toGPU' );

		this.weightsBuffers.attribute.needsUpdate = true;
		this.latentsBuffers.attribute.needsUpdate = true;

	}

	resetLoss() {

		this.lossAttribute.array[ 0 ] = 0;
		this.lossAttribute.needsUpdate = true;

	}

	async syncToCPU( cpuModel, renderer ) {

		const [ weightsBuffer, latentsBuffer ] = await Promise.all( [
			renderer.getArrayBufferAsync( this.weightsBuffers.attribute ),
			renderer.getArrayBufferAsync( this.latentsBuffers.attribute )
		] );
		const weights = new Float32Array( weightsBuffer );
		const latents = new Float32Array( latentsBuffer );

		copyModel( cpuModel, this.layout, weights, latents, 'toCPU' );

	}

	/**
	 * Releases every GPU storage buffer this instance owns. Training GPU
	 * models are cheap to construct but hold 11 real GPU-side buffers - a
	 * fresh trainer run that doesn't dispose the previous one leaks them.
	 */
	dispose() {

		disposeAdamParameterBuffers( this.weightsBuffers );
		disposeAdamParameterBuffers( this.latentsBuffers );

		this.activationsAttribute.dispose();
		this.lossAttribute.dispose();
		this.gradNormAttribute.dispose();

	}

	async readLoss( renderer ) {

		const buffer = await renderer.getArrayBufferAsync( this.lossAttribute );
		const array = new Int32Array( buffer );
		// The kernel accumulates the raw (un-batch-averaged) per-sample loss
		// sum - see NeuralTextureGPUComputeTSL.js for why - so the mean loss is
		// recovered here by dividing by batchSize as well as FIXED_POINT_SCALE.
		const loss = array[ 0 ] / ( FIXED_POINT_SCALE * this.batchSize );

		this.lossAttribute.array[ 0 ] = 0;
		this.lossAttribute.needsUpdate = true;

		return loss;

	}

}

export { computeTextureModelLayout, NTCGPUModel };
