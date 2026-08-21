// Exported as .neuralTexture (JSON content, format: 'three-neural-texture') - see FORMAT/VERSION below

import { encodeUint8Base64, encodeMLPLayersBase64 } from '../neural/NeuralBinaryCodec.js';
import { computeLatentRanges } from '../neural/NeuralQuantization.js';

const FORMAT = 'three-neural-texture';
const VERSION = 1;

/**
 * Concatenates every grid level's flat `Float32Array`/`Array` `data` into a
 * single flat `Float32Array`, alongside the `{ offset, floatCount }` layout
 * `NeuralQuantization.computeLatentRanges` expects (the same shape
 * `NeuralTextureGPUModel.computeTextureModelLayout`'s `layout.gridLevels`
 * already uses) - so a plain export-time min/max scan can reuse that shared
 * helper instead of re-deriving its own reduction loop.
 */
function concatenateGridData( grids ) {

	let offset = 0;
	const gridLevels = [];

	for ( const grid of grids ) {

		gridLevels.push( { offset, floatCount: grid.data.length } );
		offset += grid.data.length;

	}

	const flat = new Float32Array( offset );

	for ( let g = 0; g < grids.length; g ++ ) {

		flat.set( grids[ g ].data, gridLevels[ g ].offset );

	}

	return { flat, gridLevels };

}

/**
 * Resolves the per-level `[min, max]` quantization range used to encode the
 * exported latent grid as uint8: an explicit `options.quantizationRange`,
 * else `cpuModel.quantizationRange` (set by `NeuralTextureTrainer.train()`
 * when QAT was actually used - see NeuralQuantization.js), else a plain
 * min/max scan over the model's *final* (post-training) latents via
 * `computeLatentRanges`.
 *
 * Note: scanning a range at export time for a model that was never
 * trained with quantization-aware training (QAT) still produces a
 * numerically valid uint8-quantized manifest, but the network's weights
 * were never exposed to the rounding error that quantization introduces -
 * so the exported model may reconstruct slightly less accurately than one
 * actually trained with `quantization.mode: 'uint8'` (see
 * NeuralTextureTrainer.js's `quantization` option). Prefer training with
 * QAT enabled when export-time compactness matters.
 */
function resolveQuantizationRange( cpuModel, options ) {

	if ( options.quantizationRange ) return options.quantizationRange;
	if ( cpuModel.quantizationRange ) return cpuModel.quantizationRange;

	const { flat, gridLevels } = concatenateGridData( cpuModel.grids );

	return computeLatentRanges( flat, gridLevels, true );

}

/**
 * Builds the compact binary `.neuralTexture` manifest for a trained
 * `NeuralTextureTrainer`/`createNeuralTextureModel` CPU model: each latent
 * grid level is quantized to uint8 against its own `[min, max]` range (see
 * `resolveQuantizationRange`), and the decoder MLP's weights/biases are
 * packed as float16 - both via NeuralBinaryCodec.js, so a
 * `NeuralTextureLoader` decode is an exact (uint8-quantization-lossy /
 * float16-precision-lossy, but otherwise exact) inverse of this.
 */
function createNeuralTextureManifest( cpuModel, options = {} ) {

	const ranges = resolveQuantizationRange( cpuModel, options );

	const levels = cpuModel.grids.map( ( grid, index ) => {

		const [ min, max ] = ranges[ index ];

		return {
			width: grid.width,
			height: grid.height,
			channels: grid.channels,
			wrap: options.wrap || 'repeat',
			dtype: 'uint8',
			min,
			max,
			dataBase64: encodeUint8Base64( grid.data, min, max )
		};

	} );

	return {
		format: FORMAT,
		version: VERSION,
		name: options.name,
		source: options.source || 'THREE.NeuralTextureTrainer',
		latents: {
			channelsPerLevel: cpuModel.channels,
			wrap: options.wrap || 'repeat',
			levels
		},
		outputChannels: cpuModel.outputChannels,
		mlp: encodeMLPLayersBase64( cpuModel.decoder.layers )
	};

}

/**
 * Convenience wrapper matching the `exportNeuralXxx`/`createNeuralXxxManifest`
 * split every neural-* manifest module uses (see
 * NeuralAppearanceManifest.js's `exportNeuralAppearance`/
 * `createNeuralAppearanceManifest`) - for `.neuralTexture` there's no
 * teacher-driven reference-evaluation step to also compute, so this is
 * presently just `createNeuralTextureManifest` under the `exportNeuralXxx`
 * name every caller expects.
 */
function exportNeuralTexture( cpuModel, options = {} ) {

	return createNeuralTextureManifest( cpuModel, options );

}

export { FORMAT, VERSION, createNeuralTextureManifest, exportNeuralTexture };
