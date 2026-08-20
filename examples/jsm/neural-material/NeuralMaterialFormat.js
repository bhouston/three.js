import { transformNormalToView, vec3, vec4 } from 'three/tsl';

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

/**
 * `clampRange` and `defaultValue` (alongside `activation`/`nodeKeys`/`size`
 * above) are what let `NeuralMaterialNodeMaterial`'s constructor and
 * `NeuralMaterialSource`'s node/constant resolvers be driven by a loop
 * instead of one hand-written branch per channel - see `SIMPLE_SCALAR_KEYS`
 * below for which channels are regular enough to do that.
 *
 * `clampRange` is the range this channel's *trained* value is clamped to
 * before use (a trained value has no hard guarantee of landing in its
 * activation's nominal range - see `previewColor`'s doc comment). `null`
 * means "don't clamp" - used for vector-valued channels that get a
 * different, channel-specific treatment instead (normal reconstruction,
 * anisotropy's hypot/atan2, unbounded emission).
 *
 * `defaultValue` is what a *constant* (untrained) channel falls back to when
 * the source material doesn't set even the corresponding plain property -
 * matching three.js's own `MeshPhysicalMaterial` defaults for that property.
 */
const CHANNELS = [
	{ key: 'albedo', size: 3, activation: 'sigmoid', nodeKeys: [ 'colorNode' ], clampRange: null, defaultValue: [ 1, 1, 1 ] },
	{ key: 'opacity', size: 1, activation: 'sigmoid', nodeKeys: [ 'opacityNode' ], clampRange: [ 0, 1 ], defaultValue: 1 },
	// 2-component tangent-space (dx, dy) offset, not a full xyz vector - z is
	// reconstructed at consumption time as sqrt(1 - dx*dx - dy*dy), see
	// NeuralMaterialNodeMaterial.reconstructFinalNormal.
	{ key: 'normal', size: 2, activation: 'tanh', nodeKeys: [ 'normalNode' ], clampRange: null, defaultValue: [ 0, 0, 1 ] },
	{ key: 'roughness', size: 1, activation: 'sigmoid', nodeKeys: [ 'roughnessNode' ], clampRange: [ 0.02, 1 ], defaultValue: 1 },
	{ key: 'metalness', size: 1, activation: 'sigmoid', nodeKeys: [ 'metalnessNode' ], clampRange: [ 0, 1 ], defaultValue: 0 },
	{ key: 'clearcoat', size: 1, activation: 'sigmoid', nodeKeys: [ 'clearcoatNode' ], clampRange: [ 0, 1 ], defaultValue: 0 },
	{ key: 'clearcoatRoughness', size: 1, activation: 'sigmoid', nodeKeys: [ 'clearcoatRoughnessNode' ], clampRange: [ 0.02, 1 ], defaultValue: 0 },
	{ key: 'clearcoatNormal', size: 2, activation: 'tanh', nodeKeys: [ 'clearcoatNormalNode' ], clampRange: null, defaultValue: [ 0, 0, 1 ] },
	{ key: 'transmission', size: 1, activation: 'sigmoid', nodeKeys: [ 'transmissionNode' ], clampRange: [ 0, 1 ], defaultValue: 0 },
	{ key: 'emissive', size: 3, activation: 'softplus', nodeKeys: [ 'emissiveNode' ], clampRange: null, defaultValue: [ 0, 0, 0 ] },
	// Anisotropy strength+rotation trained as a single signed 2D direction
	// vector (ax, ay) = (cos(rotation)*strength, sin(rotation)*strength),
	// exactly what MeshPhysicalMaterial.anisotropyNode itself consumes - see
	// NeuralMaterialSource.resolveAnisotropyNodes. This avoids training a
	// wrapping [0,1] angle (which has a hard 0/1 discontinuity and an
	// atan2(0,0) singularity at zero strength) through a bounded activation.
	{ key: 'anisotropy', size: 2, activation: 'tanh', nodeKeys: [ 'anisotropyNode' ], clampRange: null, defaultValue: [ 0, 0 ] },
	{ key: 'sheenColor', size: 3, activation: 'sigmoid', nodeKeys: [ 'sheenNode' ], clampRange: [ 0, 1 ], defaultValue: [ 0, 0, 0 ] },
	{ key: 'sheenRoughness', size: 1, activation: 'sigmoid', nodeKeys: [ 'sheenRoughnessNode' ], clampRange: [ 0.02, 1 ], defaultValue: 1 }
];

/**
 * Channels regular enough - a single `${key}Node`/`${key}` node/property
 * pair, a scalar clamp range, and a scalar default - to be driven by one
 * loop in both `NeuralMaterialSource` (node/constant resolution) and
 * `NeuralMaterialNodeMaterial`'s constructor, instead of one hand-written
 * branch per channel. Channels left out of this list each need real
 * special-casing: `albedo`/`sheenColor` resolve from a `Color` + separate
 * intensity property, `normal`/`clearcoatNormal` reconstruct a full vector
 * from a 2-component offset, and `anisotropy` decomposes into legacy
 * strength/rotation properties - see `NeuralMaterialSource.
 * resolveMaterialChannelNodes` and `NeuralMaterialNodeMaterial`'s
 * constructor for each.
 */
const SIMPLE_SCALAR_KEYS = [ 'opacity', 'roughness', 'metalness', 'clearcoat', 'clearcoatRoughness', 'transmission', 'sheenRoughness' ];

const MAX_TOTAL_CHANNELS = CHANNELS.reduce( ( sum, c ) => sum + c.size, 0 ); // 24, if every channel is active

/**
 * Debug-only views of the mesh's raw tangent-space frame itself
 * (`tangentWorld`/`bitangentWorld`) - not trained channels, just the
 * geometry-derived basis every 'tanh'-activated normal/clearcoatNormal/
 * anisotropy channel is defined relative to. Both the teacher
 * (buildChannelPreviewMaterials) and the neural material (setDebugView)
 * expose these identically so a "normal" debug-view mismatch between the
 * two can be isolated to either the trained (dx, dy) content or this
 * shared frame - if *this* view already disagrees, the bug is in how the
 * frame itself is built/consumed, not in anything the network learned.
 */
const FRAME_VIEWS = [ 'tangent', 'bitangent' ];

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

/**
 * Builds one channel's unlit debug-view color node - widening `normal`/
 * `clearcoatNormal`'s described size to 3 first, since by the time a value
 * reaches here it's always the fully TBN-reconstructed 3-component vector
 * (whether trained-and-reconstructed or the [0,0,1] constant fallback), not
 * the raw 2-component (dx, dy) the network itself trains against - feeding
 * the real 2-component channel descriptor would hit `previewColor`'s
 * `size === 2` branch, which hardcodes blue to 0 and silently discards z.
 * Shared between `NeuralMaterialSource.buildChannelPreviewMaterials` (the
 * teacher side) and `NeuralMaterialNodeMaterial.setDebugView` (the neural
 * side) so the two can't drift apart on this widening.
 */
function buildDebugViewColorNode( channel, valueNode ) {

	const isNormalChannel = channel.key === 'normal' || channel.key === 'clearcoatNormal';
	const previewChannel = isNormalChannel ? { ...channel, size: 3 } : channel;

	return vec4( previewColor( valueNode, previewChannel, false ), 1 );

}

/**
 * Builds the debug-view color node for one of `FRAME_VIEWS` - the mesh's
 * raw tangent-space frame itself, not a trained channel. Shared for the
 * same reason as `buildDebugViewColorNode` above.
 */
function buildFrameViewColorNode( frameNode ) {

	return vec4( previewColor( transformNormalToView( frameNode ), { activation: 'tanh', size: 3 }, false ), 1 );

}

export {
	CHANNELS,
	SIMPLE_SCALAR_KEYS,
	MAX_TOTAL_CHANNELS,
	FRAME_VIEWS,
	getChannel,
	layoutChannels,
	buildChannelActivations,
	previewColor,
	buildDebugViewColorNode,
	buildFrameViewColorNode
};
