import { StorageBufferAttribute } from 'three/webgpu';
import { storage, uniform } from 'three/tsl';
import { FIXED_POINT_SCALE } from '../neural/NeuralGPUTrainingConstants.js';
import { computeGridLevels } from '../neural/NeuralGridModel.js';

/**
 * Computes buffer layouts and offsets for GPU-based neural texture training:
 * a multiresolution feature grid pyramid plus a small MLP decoder, sized
 * generically from an arbitrary hidden-layer configuration.
 */
function computeTextureModelLayout( options = {} ) {

	const channels = options.channels || 4;
	const levels = options.levels || 4;
	const baseResolution = options.baseResolution || 16;
	const targetResolution = options.targetResolution || 256;
	const hiddenSizes = options.hiddenSizes || [ 32, 32 ];
	const outputChannels = options.outputChannels || 3;
	const includeUV = options.includeUV || false;

	const resolutions = computeGridLevels( baseResolution, targetResolution, levels );

	const gridLevels = [];
	let latentOffset = 0;

	for ( const resolution of resolutions ) {

		const texelCount = resolution * resolution;
		const floatCount = texelCount * channels;
		gridLevels.push( { width: resolution, height: resolution, offset: latentOffset, texelCount, floatCount } );
		latentOffset += floatCount;

	}

	const totalLatents = latentOffset;

	// MLP weight layout: input = concatenated multiresolution grid features,
	// optionally followed by the raw (u, v) sample coordinate (see
	// NeuralTextureModel.js for why).
	const uvInputOffset = levels * channels;
	const inputSize = uvInputOffset + ( includeUV ? 2 : 0 );
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
		includeUV,
		inputSize,
		uvInputOffset,
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
 * Encapsulates the GPU StorageBuffers, uniforms, and CPU synchronizers for
 * neural texture training. Field names intentionally mirror
 * `NeuralAppearanceGPUModel` so the Adam / gradient-clip compute kernels can
 * be reused verbatim.
 */
class NeuralTextureGPUModel {

	constructor( options = {} ) {

		this.options = options;
		this.batchSize = options.batchSize || 4096;
		this.layout = computeTextureModelLayout( options );

		const { totalWeights, totalLatents, activationStride } = this.layout;
		const batchSize = this.batchSize;

		this.weightsAttribute = new StorageBufferAttribute( new Float32Array( totalWeights ), 1, Float32Array );
		this.gradWeightsAttribute = new StorageBufferAttribute( new Int32Array( totalWeights ), 1, Int32Array );
		this.mWeightsAttribute = new StorageBufferAttribute( new Float32Array( totalWeights ), 1, Float32Array );
		this.vWeightsAttribute = new StorageBufferAttribute( new Float32Array( totalWeights ), 1, Float32Array );

		this.weightsStorage = storage( this.weightsAttribute, 'float', totalWeights );
		this.gradWeightsAtomic = storage( this.gradWeightsAttribute, 'int', totalWeights ).toAtomic();
		this.mWeightsStorage = storage( this.mWeightsAttribute, 'float', totalWeights );
		this.vWeightsStorage = storage( this.vWeightsAttribute, 'float', totalWeights );

		this.latentsAttribute = new StorageBufferAttribute( new Float32Array( totalLatents ), 1, Float32Array );
		this.gradLatentsAttribute = new StorageBufferAttribute( new Int32Array( totalLatents ), 1, Int32Array );
		this.mLatentsAttribute = new StorageBufferAttribute( new Float32Array( totalLatents ), 1, Float32Array );
		this.vLatentsAttribute = new StorageBufferAttribute( new Float32Array( totalLatents ), 1, Float32Array );

		this.latentsStorage = storage( this.latentsAttribute, 'float', totalLatents );
		this.gradLatentsAtomic = storage( this.gradLatentsAttribute, 'int', totalLatents ).toAtomic();
		this.mLatentsStorage = storage( this.mLatentsAttribute, 'float', totalLatents );
		this.vLatentsStorage = storage( this.vLatentsAttribute, 'float', totalLatents );

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

	}

	initFromCPUModel( cpuModel ) {

		const weights = this.weightsAttribute.array;
		const latents = this.latentsAttribute.array;
		weights.fill( 0 );
		latents.fill( 0 );

		for ( let l = 0; l < cpuModel.decoder.layers.length; l ++ ) {

			const layer = cpuModel.decoder.layers[ l ];
			const layout = this.layout.mlpLayers[ l ];

			for ( let i = 0; i < layer.weights.length; i ++ ) weights[ layout.weightsOffset + i ] = layer.weights[ i ];
			for ( let i = 0; i < layer.biases.length; i ++ ) weights[ layout.biasesOffset + i ] = layer.biases[ i ];

		}

		for ( let g = 0; g < cpuModel.grids.length; g ++ ) {

			const grid = cpuModel.grids[ g ];
			const level = this.layout.gridLevels[ g ];

			for ( let i = 0; i < grid.data.length; i ++ ) latents[ level.offset + i ] = grid.data[ i ];

		}

		this.weightsAttribute.needsUpdate = true;
		this.latentsAttribute.needsUpdate = true;

	}

	resetLoss() {

		this.lossAttribute.array[ 0 ] = 0;
		this.lossAttribute.needsUpdate = true;

	}

	async syncToCPU( cpuModel, renderer ) {

		const weightsBuffer = await renderer.getArrayBufferAsync( this.weightsAttribute );
		const latentsBuffer = await renderer.getArrayBufferAsync( this.latentsAttribute );
		const weights = new Float32Array( weightsBuffer );
		const latents = new Float32Array( latentsBuffer );

		for ( let l = 0; l < cpuModel.decoder.layers.length; l ++ ) {

			const layer = cpuModel.decoder.layers[ l ];
			const layout = this.layout.mlpLayers[ l ];

			for ( let i = 0; i < layer.weights.length; i ++ ) layer.weights[ i ] = weights[ layout.weightsOffset + i ];
			for ( let i = 0; i < layer.biases.length; i ++ ) layer.biases[ i ] = weights[ layout.biasesOffset + i ];

		}

		for ( let g = 0; g < cpuModel.grids.length; g ++ ) {

			const grid = cpuModel.grids[ g ];
			const level = this.layout.gridLevels[ g ];

			for ( let i = 0; i < grid.data.length; i ++ ) grid.data[ i ] = latents[ level.offset + i ];

		}

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

export { computeTextureModelLayout, NeuralTextureGPUModel };
