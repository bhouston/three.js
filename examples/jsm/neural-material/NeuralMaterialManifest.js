// Exported as .neuralMaterial (JSON content, format: 'three-neural-material') - see FORMAT/VERSION below

import { createNeuralTextureManifest } from '../neural-texture/NeuralTextureManifest.js';

const FORMAT = 'three-neural-material';
const VERSION = 1;

/**
 * Unlike neural-texture (one grid+MLP per texture) or neural-appearance (one
 * grid+MLP per BRDF), a `NeuralMaterialNodeMaterial` is trained as *one*
 * shared `NeuralTextureTrainer` model whose output channels are packed
 * across every spatially-varying PBR channel the source material actually
 * uses (see `NeuralMaterialSource.classifyMaterialChannels` /
 * `NeuralMaterialFormat.layoutChannels` - `channelActivations` is what tells
 * the trainer which output nonlinearity each packed channel gets). So this
 * manifest embeds exactly one `.neuralTexture`-shaped manifest (via
 * `createNeuralTextureManifest`), plus the channel layout/constant-value
 * metadata `classifyMaterialChannels` produced, needed to slice that single
 * decoder's output back into named PBR channels at load time (see
 * `NeuralMaterialNodeMaterial`'s constructor / `sliceChannels`).
 *
 * `channelClassification` is whatever `NeuralMaterialSource.
 * classifyMaterialChannels` returned - `{ activeChannels, constantValues,
 * totalChannels, packCount }` - passed through as-is, matching how
 * `NeuralMaterialNodeMaterial`'s own constructor already consumes it.
 */
function createNeuralMaterialManifest( cpuModel, channelClassification, options = {} ) {

	const texture = createNeuralTextureManifest( cpuModel, { ...options, source: options.source || 'THREE.NeuralMaterialNodeMaterial' } );

	return {
		format: FORMAT,
		version: VERSION,
		name: options.name,
		texture,
		// See NeuralMaterialSource.resolveRenderFlags's doc comment - `side`/
		// `transparent` aren't channels (nothing for the network to fit), but
		// still need to round-trip so a loaded material's transmission pass
		// count (and therefore its attenuation tint strength) matches the
		// source material it was fit against. `undefined` on a
		// classification built without a source material (e.g. hand-assembled
		// constant-only classification) round-trips as a plain `null`.
		renderFlags: channelClassification.renderFlags || null,
		channels: {
			// Only the channel *keys* need to be persisted, in layout order -
			// every other field on an `activeChannels` entry (`size`,
			// `activation`, `nodeKeys`, `clampRange`, `defaultValue`, `offset`)
			// is a fixed property of that key already declared in
			// NeuralMaterialFormat.CHANNELS/`getChannel`, and `offset`/
			// `totalChannels`/`packCount` are re-derived deterministically from
			// the key list alone via `layoutChannels` - see
			// NeuralMaterialLoader.js. Storing only the keys means this manifest
			// can never disagree with NeuralMaterialFormat.js about what a
			// channel's own size/activation/clampRange is.
			activeKeys: channelClassification.activeChannels.map( ( channel ) => channel.key ),
			constantValues: channelClassification.constantValues
		}
	};

}

function exportNeuralMaterial( cpuModel, channelClassification, options = {} ) {

	return createNeuralMaterialManifest( cpuModel, channelClassification, options );

}

export { FORMAT, VERSION, createNeuralMaterialManifest, exportNeuralMaterial };
