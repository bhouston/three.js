import { vec3 } from 'three/tsl';

/**
 * Full vocabulary of standard PBR channels this trainer knows how to fit,
 * jointly, with one shared grid + MLP network (NVIDIA neural texture
 * compression style: one small decoder, many correlated output channels).
 *
 * A given material only *trains* the subset of these channels it actually
 * drives with a node graph (texture/procedural) - see
 * `NeuralMaterialSource.classifyMaterialChannels` - so the network's output
 * width, and the sample cost of every training iteration, scale with how
 * much of the material is actually spatially-varying rather than always
 * paying for all of them. `nodeKeys` is what that classification checks:
 * a channel is "active" (worth training) iff at least one of its node
 * properties is set on the source material; otherwise it's read once as a
 * plain constant and applied directly, bypassing the network entirely.
 *
 * `activation` is the output nonlinearity applied to this channel's slice of
 * the decoder's (always-linear, see NeuralTextureModel.js) raw output - see
 * ../neural/NeuralOutputActivations.js - chosen to match each channel's natural
 * physical range instead of forcing every channel through the same
 * unbounded linear output:
 *  - 'sigmoid': bounded [0,1] scalars/colors (reflectance, coverage,
 *    roughness-like properties).
 *  - 'tanh': bounded [-1,1] signed vectors (tangent-space normal offsets,
 *    anisotropy direction) - trained/consumed directly in that range, no
 *    extra `*0.5+0.5` remap needed.
 *  - 'softplus': unbounded, non-negative HDR values (emission) - smooth and
 *    non-exploding, unlike a raw exponential, for an undertrained decoder's
 *    large pre-activations.
 * Anything without an explicit `activation` falls back to plain linear
 * (matching every other MLP output in this codebase).
 */

const CHANNELS = [
	{ key: 'albedo', size: 3, activation: 'sigmoid', nodeKeys: [ 'colorNode' ] },
	{ key: 'opacity', size: 1, activation: 'sigmoid', nodeKeys: [ 'opacityNode' ] },
	// 2-component tangent-space (dx, dy) offset, not a full xyz vector - z is
	// reconstructed at consumption time as sqrt(1 - dx*dx - dy*dy), see
	// NeuralMaterialNodeMaterial.reconstructFinalNormal.
	{ key: 'normal', size: 2, activation: 'tanh', nodeKeys: [ 'normalNode' ] },
	{ key: 'roughness', size: 1, activation: 'sigmoid', nodeKeys: [ 'roughnessNode' ] },
	{ key: 'metalness', size: 1, activation: 'sigmoid', nodeKeys: [ 'metalnessNode' ] },
	{ key: 'clearcoat', size: 1, activation: 'sigmoid', nodeKeys: [ 'clearcoatNode' ] },
	{ key: 'clearcoatRoughness', size: 1, activation: 'sigmoid', nodeKeys: [ 'clearcoatRoughnessNode' ] },
	{ key: 'clearcoatNormal', size: 2, activation: 'tanh', nodeKeys: [ 'clearcoatNormalNode' ] },
	{ key: 'transmission', size: 1, activation: 'sigmoid', nodeKeys: [ 'transmissionNode' ] },
	{ key: 'emissive', size: 3, activation: 'softplus', nodeKeys: [ 'emissiveNode' ] },
	// Anisotropy strength+rotation trained as a single signed 2D direction
	// vector (ax, ay) = (cos(rotation)*strength, sin(rotation)*strength),
	// exactly what MeshPhysicalMaterial.anisotropyNode itself consumes - see
	// NeuralMaterialSource.resolveAnisotropyNodes. This avoids training a
	// wrapping [0,1] angle (which has a hard 0/1 discontinuity and an
	// atan2(0,0) singularity at zero strength) through a bounded activation.
	{ key: 'anisotropy', size: 2, activation: 'tanh', nodeKeys: [ 'anisotropyNode' ] },
	{ key: 'sheenColor', size: 3, activation: 'sigmoid', nodeKeys: [ 'sheenNode' ] },
	{ key: 'sheenRoughness', size: 1, activation: 'sigmoid', nodeKeys: [ 'sheenRoughnessNode' ] }
];

const MAX_TOTAL_CHANNELS = CHANNELS.reduce( ( sum, c ) => sum + c.size, 0 ); // 24, if every channel is active

function getChannel( key ) {

	const channel = CHANNELS.find( ( c ) => c.key === key );
	if ( channel === undefined ) throw new Error( `THREE.NeuralMaterialFormat: unknown channel "${key}".` );

	return channel;

}

/**
 * Assigns contiguous flat offsets (and a total/pack count) to an arbitrary
 * subset of CHANNELS, in CHANNELS order - used both for the "active,
 * trained" subset (network output layout) and, incidentally, for computing
 * the full 24-channel layout when every channel is active.
 */
function layoutChannels( channelSubset ) {

	let offset = 0;
	const layout = [];

	for ( const channel of channelSubset ) {

		layout.push( { ...channel, offset } );
		offset += channel.size;

	}

	const totalChannels = offset;

	return { channels: layout, totalChannels, packCount: Math.ceil( totalChannels / 4 ) };

}

/**
 * Flattens an (offset-assigned) active-channel layout - see `layoutChannels`
 * - into a per-output-component `activation` array, e.g. for
 * `NeuralTextureTrainer`'s `channelActivations` option (see
 * NeuralTextureGPUComputeTSL.js) and for `NeuralMaterialNodeMaterial`'s
 * `sliceChannels`/`applyChannelActivation` at inference time - both index
 * this the same way `layoutChannels` assigns offsets, so a component's
 * activation always lands at the same flat index its value does.
 */
function buildChannelActivations( activeChannels ) {

	const activations = [];

	for ( const channel of activeChannels ) {

		for ( let i = 0; i < channel.size; i ++ ) activations.push( channel.activation );

	}

	return activations;

}

/**
 * Turns one channel's raw value into an unlit RGB preview color, for a
 * per-channel "debug view" - shared between the teacher (raw physical
 * value, needs remapping into display space to match) and the trained
 * network (whose output already came through this same channel's
 * `activation`, so `alreadyEncoded: true` skips the extra remap). Scalars
 * are broadcast to grayscale; 'tanh' vectors (normals, anisotropy) map to
 * the familiar bluish normal-map-style preview.
 */
function previewColor( valueNode, channel, alreadyEncoded ) {

	let value = valueNode;

	if ( channel.activation === 'tanh' && ! alreadyEncoded ) value = value.mul( 0.5 ).add( 0.5 );

	// A trained (as opposed to measured-constant) value has no guarantee of
	// landing in [0,1] - even a sigmoid/tanh-activated channel only
	// approaches its bound asymptotically, and softplus (emission) is
	// unbounded above by design. Clamp for display so this reads as "not
	// converged yet" (a muted, but plausible, color) rather than
	// hard-saturating to a misleadingly bold primary color.
	value = value.clamp( 0, 1 );

	if ( channel.size === 1 ) return vec3( value, value, value );
	if ( channel.size === 2 ) return vec3( value.x, value.y, 0 );

	return value;

}

export { CHANNELS, MAX_TOTAL_CHANNELS, getChannel, layoutChannels, buildChannelActivations, previewColor };
