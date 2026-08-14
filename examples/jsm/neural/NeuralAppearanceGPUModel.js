import { StorageBufferAttribute } from 'three/webgpu';
import { Vector4 } from 'three';
import { storage, uniform, uniformArray } from 'three/tsl';
import { IBL_INPUT_SIZE, IBL_OUTPUT_SIZE, LATENT_CHANNELS } from './NeuralAppearanceFormat.js';

const FIXED_POINT_SCALE = 1e5;
const GRADIENT_NORM_SCALE = 1e5;

/**
 * Computes buffer layouts and offsets for GPU-based neural appearance training.
 */
function computeModelLayout( options = {} ) {

	const hiddenSize = options.hiddenSize || 32;
	const iblHiddenSize = options.iblHiddenSize || Math.min( Math.max( hiddenSize, 16 ), 32 );
	const supportsEmission = Boolean( options.outputFeatures && options.outputFeatures.emission );
	const supportsOpacity = Boolean( options.outputFeatures && options.outputFeatures.opacity );

	// Weights Layout:
	// 0..95: rotation weights (8 channels * 12 outputs)
	const rotationOffset = 0;
	const rotationCount = LATENT_CHANNELS * 12;

	// Layer 0: 20 -> hiddenSize
	const layer0WeightsOffset = rotationOffset + rotationCount;
	const layer0WeightsCount = 20 * hiddenSize;
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

	// IBL Head: 14 -> iblHiddenSize -> 13
	const iblLayer0WeightsOffset = currentOffset;
	const iblLayer0WeightsCount = IBL_INPUT_SIZE * iblHiddenSize;
	const iblLayer0BiasesOffset = iblLayer0WeightsOffset + iblLayer0WeightsCount;
	const iblLayer0BiasesCount = iblHiddenSize;
	const iblLayer1WeightsOffset = iblLayer0BiasesOffset + iblLayer0BiasesCount;
	const iblLayer1WeightsCount = iblHiddenSize * IBL_OUTPUT_SIZE;
	const iblLayer1BiasesOffset = iblLayer1WeightsOffset + iblLayer1WeightsCount;
	const iblLayer1BiasesCount = IBL_OUTPUT_SIZE;
	currentOffset = iblLayer1BiasesOffset + iblLayer1BiasesCount;

	// Auxiliary: Emission Head (8 -> 3)
	let emissionWeightsOffset = - 1;
	let emissionBiasesOffset = - 1;
	let emissionWeightsCount = 0;
	let emissionBiasesCount = 0;
	if ( supportsEmission ) {

		emissionWeightsOffset = currentOffset;
		emissionWeightsCount = LATENT_CHANNELS * 3;
		emissionBiasesOffset = emissionWeightsOffset + emissionWeightsCount;
		emissionBiasesCount = 3;
		currentOffset = emissionBiasesOffset + emissionBiasesCount;

	}

	// Auxiliary: Opacity Head (8 -> 1)
	let opacityWeightsOffset = - 1;
	let opacityBiasesOffset = - 1;
	let opacityWeightsCount = 0;
	let opacityBiasesCount = 0;
	if ( supportsOpacity ) {

		opacityWeightsOffset = currentOffset;
		opacityWeightsCount = LATENT_CHANNELS * 1;
		opacityBiasesOffset = opacityWeightsOffset + opacityWeightsCount;
		opacityBiasesCount = 1;
		currentOffset = opacityBiasesOffset + opacityBiasesCount;

	}

	const totalWeights = currentOffset;

	// Latents Mip Pyramid Layout
	const baseResolution = options.resolution || 8;
	const mipLevels = [];
	let currentWidth = baseResolution;
	let currentHeight = baseResolution;
	let latentOffset = 0;

	while ( true ) {

		const texelCount = currentWidth * currentHeight;
		const floatCount = texelCount * LATENT_CHANNELS;
		mipLevels.push( {
			width: currentWidth,
			height: currentHeight,
			offset: latentOffset,
			texelCount,
			floatCount
		} );
		latentOffset += floatCount;

		if ( currentWidth === 1 && currentHeight === 1 ) break;
		currentWidth = Math.max( 1, currentWidth >> 1 );
		currentHeight = Math.max( 1, currentHeight >> 1 );

	}

	const totalLatents = latentOffset;

	// Per-Sample Activations Layout:
	// a0: 20
	// z1: H
	// a1: H
	// z2: H
	// a2: H
	// z3: 3
	// delta3: 3
	// delta2: H
	// delta1: H
	// gradA0: 20
	// gradLatents: 8
	const actA0Offset = 0;
	const actZ1Offset = actA0Offset + 20;
	const actA1Offset = actZ1Offset + hiddenSize;
	const actZ2Offset = actA1Offset + hiddenSize;
	const actA2Offset = actZ2Offset + hiddenSize;
	const actZ3Offset = actA2Offset + hiddenSize;
	const actDelta3Offset = actZ3Offset + 3;
	const actDelta2Offset = actDelta3Offset + 3;
	const actDelta1Offset = actDelta2Offset + hiddenSize;
	const actGradA0Offset = actDelta1Offset + hiddenSize;
	const actGradLatentsOffset = actGradA0Offset + 20;
	let actCurrent = actGradLatentsOffset + LATENT_CHANNELS;

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

	const activationStride = actCurrent;

	// Sample Buffer Stride: 20 direct/aux floats + 13 IBL target floats.
	const sampleStride = 20 + IBL_OUTPUT_SIZE;

	return {
		hiddenSize,
		iblHiddenSize,
		supportsEmission,
		supportsOpacity,
		totalWeights,
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
		emissionWeightsOffset,
		emissionWeightsCount,
		emissionBiasesOffset,
		emissionBiasesCount,
		opacityWeightsOffset,
		opacityWeightsCount,
		opacityBiasesOffset,
		opacityBiasesCount,
		mipLevels,
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
		const { totalWeights, totalLatents, sampleStride, activationStride, mipLevels } = this.layout;
		const batchSize = this.batchSize;

		this.weightsAttribute = new StorageBufferAttribute( new Float32Array( totalWeights ), 1, Float32Array );
		this.gradWeightsAttribute = new StorageBufferAttribute( new Int32Array( totalWeights ), 1, Int32Array );
		this.mWeightsAttribute = new StorageBufferAttribute( new Float32Array( totalWeights ), 1, Float32Array );
		this.vWeightsAttribute = new StorageBufferAttribute( new Float32Array( totalWeights ), 1, Float32Array );

		this.weightsStorage = storage( this.weightsAttribute, 'float', totalWeights );
		this.gradWeightsAtomic = storage( this.gradWeightsAttribute, 'int', totalWeights ).toAtomic();
		this.mWeightsStorage = storage( this.mWeightsAttribute, 'float', totalWeights );
		this.vWeightsStorage = storage( this.vWeightsAttribute, 'float', totalWeights );

		// 2. Latent buffers
		this.latentsAttribute = new StorageBufferAttribute( new Float32Array( totalLatents ), 1, Float32Array );
		this.gradLatentsAttribute = new StorageBufferAttribute( new Int32Array( totalLatents ), 1, Int32Array );
		this.mLatentsAttribute = new StorageBufferAttribute( new Float32Array( totalLatents ), 1, Float32Array );
		this.vLatentsAttribute = new StorageBufferAttribute( new Float32Array( totalLatents ), 1, Float32Array );

		this.latentsStorage = storage( this.latentsAttribute, 'float', totalLatents );
		this.gradLatentsAtomic = storage( this.gradLatentsAttribute, 'int', totalLatents ).toAtomic();
		this.mLatentsStorage = storage( this.mLatentsAttribute, 'float', totalLatents );
		this.vLatentsStorage = storage( this.vLatentsAttribute, 'float', totalLatents );

		// 3. Sample buffer
		this.samplesAttribute = new StorageBufferAttribute( new Float32Array( batchSize * sampleStride ), 1, Float32Array );
		this.samplesStorage = storage( this.samplesAttribute, 'float', batchSize * sampleStride );

		// 4. Activation buffer
		this.activationsAttribute = new StorageBufferAttribute( new Float32Array( batchSize * activationStride ), 1, Float32Array );
		this.activationsStorage = storage( this.activationsAttribute, 'float', batchSize * activationStride );

		// 5. Batch loss atomic buffer
		this.lossAttribute = new StorageBufferAttribute( new Int32Array( 1 ), 1, Int32Array );
		this.lossAtomic = storage( this.lossAttribute, 'int', 1 ).toAtomic();

		// 6. Gradient clipping accumulator
		this.gradNormAttribute = new StorageBufferAttribute( new Int32Array( 1 ), 1, Int32Array );
		this.gradNormAtomic = storage( this.gradNormAttribute, 'int', 1 ).toAtomic();

		// 7. Uniforms
		this.invBatchUniform = uniform( 1.0 / batchSize );
		this.learningRateUniform = uniform( options.learningRate || 0.001 );
		this.stepUniform = uniform( 1 );
		this.maxGradientNormUniform = uniform( options.maxGradientNorm || 1 );

		// Pack mip information: (width, height, baseOffset, 0)
		const mipVectors = mipLevels.map( ( m ) => new Vector4( m.width, m.height, m.offset, 0 ) );
		this.mipInfoArray = uniformArray( mipVectors, 'vec4' );

	}

	initFromCPUModel( cpuModel ) {

		const weights = this.weightsAttribute.array;
		const latents = this.latentsAttribute.array;

		weights.fill( 0 );
		latents.fill( 0 );

		// Rotation weights
		for ( let i = 0; i < 96; i ++ ) {

			weights[ this.layout.rotationOffset + i ] = cpuModel.rotationWeights[ i ] || 0;

		}

		// Decoder Layer 0 (20 -> H)
		copyLayerWeightsToGPU( cpuModel.decoder.layers[ 0 ], weights, this.layout.layer0WeightsOffset, this.layout.layer0BiasesOffset );

		// Decoder Layer 1 (H -> H)
		copyLayerWeightsToGPU( cpuModel.decoder.layers[ 1 ], weights, this.layout.layer1WeightsOffset, this.layout.layer1BiasesOffset );

		// Decoder Layer 2 (H -> 3)
		copyLayerWeightsToGPU( cpuModel.decoder.layers[ 2 ], weights, this.layout.layer2WeightsOffset, this.layout.layer2BiasesOffset );

		// IBL Head (14 -> H_ibl -> 13)
		copyLayerWeightsToGPU( cpuModel.iblHead.layers[ 0 ], weights, this.layout.iblLayer0WeightsOffset, this.layout.iblLayer0BiasesOffset );
		copyLayerWeightsToGPU( cpuModel.iblHead.layers[ 1 ], weights, this.layout.iblLayer1WeightsOffset, this.layout.iblLayer1BiasesOffset );

		// Emission Head (8 -> 3)
		if ( cpuModel.emissionHead ) {

			copyLayerWeightsToGPU( cpuModel.emissionHead.layers[ 0 ], weights, this.layout.emissionWeightsOffset, this.layout.emissionBiasesOffset );

		}

		// Opacity Head (8 -> 1)
		if ( cpuModel.opacityHead ) {

			copyLayerWeightsToGPU( cpuModel.opacityHead.layers[ 0 ], weights, this.layout.opacityWeightsOffset, this.layout.opacityBiasesOffset );

		}

		// Latent Grids
		for ( let m = 0; m < cpuModel.latentGrids.length; m ++ ) {

			const mip = this.layout.mipLevels[ m ];
			const grid = cpuModel.latentGrids[ m ];
			for ( let i = 0; i < mip.floatCount; i ++ ) {

				latents[ mip.offset + i ] = grid.data[ i ];

			}

		}

		this.weightsAttribute.needsUpdate = true;
		this.latentsAttribute.needsUpdate = true;

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
			const duvDx = sample.duvDx || [ 0, 0 ];
			const wi = sample.wi || [ 0, 0, 1 ];
			const wo = sample.wo || [ 0, 0, 1 ];
			const target = sample.target || [ 0, 0, 0 ];
			const weight = sample.weight !== undefined ? sample.weight : 1.0;
			const mip = sample.mip || 0;

			data[ base + 0 ] = uv[ 0 ];
			data[ base + 1 ] = uv[ 1 ];
			data[ base + 2 ] = duvDx[ 0 ];
			data[ base + 3 ] = duvDx[ 1 ];
			data[ base + 4 ] = wi[ 0 ];
			data[ base + 5 ] = wi[ 1 ];
			data[ base + 6 ] = wi[ 2 ];
			data[ base + 7 ] = mip;
			data[ base + 8 ] = wo[ 0 ];
			data[ base + 9 ] = wo[ 1 ];
			data[ base + 10 ] = wo[ 2 ];
			data[ base + 11 ] = weight;
			data[ base + 12 ] = target[ 0 ];
			data[ base + 13 ] = target[ 1 ];
			data[ base + 14 ] = target[ 2 ];

			if ( sample.emissionTarget ) {

				data[ base + 15 ] = 1.0;
				data[ base + 16 ] = sample.emissionTarget[ 0 ];
				data[ base + 17 ] = sample.emissionTarget[ 1 ];
				data[ base + 18 ] = sample.emissionTarget[ 2 ];

			} else {

				data[ base + 15 ] = 0.0;
				data[ base + 16 ] = 0;
				data[ base + 17 ] = 0;
				data[ base + 18 ] = 0;

			}

			if ( Number.isFinite( sample.opacityTarget ) ) {

				data[ base + 19 ] = sample.opacityTarget;

			} else {

				data[ base + 19 ] = - 1.0;

			}

			const iblTarget = sample.iblTarget || [];
			for ( let channel = 0; channel < IBL_OUTPUT_SIZE; channel ++ ) {

				data[ base + 20 + channel ] = Number.isFinite( iblTarget[ channel ] ) ? iblTarget[ channel ] : 0;

			}

		}

		this.samplesAttribute.needsUpdate = true;

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

		// Copy rotation weights
		for ( let i = 0; i < 96; i ++ ) {

			cpuModel.rotationWeights[ i ] = weights[ this.layout.rotationOffset + i ];

		}

		// Copy decoder layers
		copyLayerWeightsFromGPU( cpuModel.decoder.layers[ 0 ], weights, this.layout.layer0WeightsOffset, this.layout.layer0BiasesOffset );
		copyLayerWeightsFromGPU( cpuModel.decoder.layers[ 1 ], weights, this.layout.layer1WeightsOffset, this.layout.layer1BiasesOffset );
		copyLayerWeightsFromGPU( cpuModel.decoder.layers[ 2 ], weights, this.layout.layer2WeightsOffset, this.layout.layer2BiasesOffset );
		copyLayerWeightsFromGPU( cpuModel.iblHead.layers[ 0 ], weights, this.layout.iblLayer0WeightsOffset, this.layout.iblLayer0BiasesOffset );
		copyLayerWeightsFromGPU( cpuModel.iblHead.layers[ 1 ], weights, this.layout.iblLayer1WeightsOffset, this.layout.iblLayer1BiasesOffset );

		if ( cpuModel.emissionHead ) {

			copyLayerWeightsFromGPU( cpuModel.emissionHead.layers[ 0 ], weights, this.layout.emissionWeightsOffset, this.layout.emissionBiasesOffset );

		}

		if ( cpuModel.opacityHead ) {

			copyLayerWeightsFromGPU( cpuModel.opacityHead.layers[ 0 ], weights, this.layout.opacityWeightsOffset, this.layout.opacityBiasesOffset );

		}

		// Copy latent grids
		for ( let m = 0; m < cpuModel.latentGrids.length; m ++ ) {

			const mip = this.layout.mipLevels[ m ];
			const grid = cpuModel.latentGrids[ m ];
			for ( let i = 0; i < mip.floatCount; i ++ ) {

				grid.data[ i ] = latents[ mip.offset + i ];

			}

		}

	}

	async readLoss( renderer ) {

		const buffer = await renderer.getArrayBufferAsync( this.lossAttribute );
		const array = new Int32Array( buffer );
		const loss = array[ 0 ] / FIXED_POINT_SCALE;

		// Reset loss accumulator
		this.lossAttribute.array[ 0 ] = 0;
		this.lossAttribute.needsUpdate = true;

		return loss;

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
	FIXED_POINT_SCALE,
	GRADIENT_NORM_SCALE,
	computeModelLayout,
	NeuralAppearanceGPUModel
};
