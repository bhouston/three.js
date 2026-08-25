import { encodeNTC } from '../ntc/NTCManifest.js';

/**
 * An exporter for `.ntc` (Neural Texture Compression) assets.
 *
 * Encodes an already-trained `NTCNodeMaterial` (see
 * `three/addons/ntc/NTCNodeMaterial.js` / `NTCTrainer.js`) - its shared
 * multiresolution latent grid plus MLP decoder, and the channel layout it
 * was fit against - into the compact JSON manifest `NTCLoader` reads back
 * (see `../ntc/NTCManifest.js` for the format itself).
 *
 * `material` must be an `NTCNodeMaterial` instance (or any object exposing
 * the same `cpuModel`/`activeChannels`/`channels` fields that constructor
 * leaves on `this` - see NTCNodeMaterial.js), not a general `Material`;
 * there is no path from an arbitrary material back to a trained grid+MLP
 * model.
 *
 * ```js
 * const exporter = new NTCExporter();
 * const manifest = exporter.parse( ntcNodeMaterial, { name: 'Gold' } );
 * const blob = new Blob( [ JSON.stringify( manifest ) ], { type: 'application/json' } );
 * ```
 *
 * @three_import import { NTCExporter } from 'three/addons/exporters/NTCExporter.js';
 */
class NTCExporter {

	/**
	 * Parses the given trained NTC material and generates the `.ntc` manifest.
	 *
	 * @param {Object} material - A trained `NTCNodeMaterial` (must carry `cpuModel`, `activeChannels`, `channels`).
	 * @param {NTCExporter~Options} [options] - The export options.
	 * @return {Object} The `.ntc` manifest - JSON-serializable as-is (`JSON.stringify( manifest )`).
	 */
	parse( material, options = {} ) {

		if ( ! material || ! material.cpuModel || ! material.activeChannels ) {

			throw new Error( 'THREE.NTCExporter: material must be a trained NTCNodeMaterial (missing cpuModel/activeChannels).' );

		}

		const channelClassification = {
			activeChannels: material.activeChannels,
			constantValues: material._constantValues || {},
			totalChannels: material.cpuModel.outputChannels,
			packCount: Math.ceil( material.cpuModel.outputChannels / 4 ),
			renderFlags: { side: material.side, transparent: material.transparent }
		};

		return encodeNTC( material.cpuModel, channelClassification, options );

	}

}

/**
 * NTC exporter options.
 *
 * @typedef {Object} NTCExporter~Options
 * @property {string} [name] - A display name embedded in the manifest.
 * @property {string} [source] - A free-form provenance string embedded in the manifest.
 * @property {('repeat'|'clamp')} [wrap='repeat'] - The latent grid's wrap mode.
 * @property {Array<[number,number]>} [quantizationRanges] - Explicit per-level `[min, max]` quantization ranges, overriding both `material.cpuModel.quantizationRange` (QAT) and a plain min/max scan.
 **/

export { NTCExporter };
