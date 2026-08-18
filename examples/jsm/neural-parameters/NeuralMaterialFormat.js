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
 * Vector channels that can be negative (normals, anisotropy direction) are
 * stored network-side in [0,1] via `n * 0.5 + 0.5` and decoded back with
 * `n * 2 - 1` wherever they're consumed - both baking and runtime
 * reconstruction go through `encode`/`decode` below so the mapping only
 * has to be defined once. This "encode" metadata is unrelated to `offset`,
 * which is only assigned once a channel is known to be active - see
 * `layoutChannels`.
 */

const CHANNELS = [
	{ key: 'albedo', size: 3, encode: 'linear', nodeKeys: [ 'colorNode' ] },
	{ key: 'opacity', size: 1, encode: 'linear', nodeKeys: [ 'opacityNode' ] },
	{ key: 'normal', size: 3, encode: 'signed', nodeKeys: [ 'normalNode' ] },
	{ key: 'roughness', size: 1, encode: 'linear', nodeKeys: [ 'roughnessNode' ] },
	{ key: 'metalness', size: 1, encode: 'linear', nodeKeys: [ 'metalnessNode' ] },
	{ key: 'clearcoat', size: 1, encode: 'linear', nodeKeys: [ 'clearcoatNode' ] },
	{ key: 'clearcoatRoughness', size: 1, encode: 'linear', nodeKeys: [ 'clearcoatRoughnessNode' ] },
	{ key: 'clearcoatNormal', size: 3, encode: 'signed', nodeKeys: [ 'clearcoatNormalNode' ] },
	{ key: 'transmission', size: 1, encode: 'linear', nodeKeys: [ 'transmissionNode' ] },
	{ key: 'emissive', size: 3, encode: 'linear', nodeKeys: [ 'emissiveNode' ] },
	{ key: 'anisotropyStrength', size: 1, encode: 'linear', nodeKeys: [ 'anisotropyNode' ] },
	{ key: 'anisotropyRotation', size: 1, encode: 'angle', nodeKeys: [ 'anisotropyNode' ] },
	{ key: 'sheenColor', size: 3, encode: 'linear', nodeKeys: [ 'sheenNode' ] },
	{ key: 'sheenRoughness', size: 1, encode: 'linear', nodeKeys: [ 'sheenRoughnessNode' ] }
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
 * Turns one channel's raw value into an unlit RGB preview color, for a
 * per-channel "debug view" - shared between the teacher (raw physical
 * value, needs `encode`-ing to match) and the trained network (whose output
 * is already in the same encoded [0,1] space it was trained against, so
 * `alreadyEncoded: true` skips the extra step). Scalars are broadcast to
 * grayscale; 'signed' vectors (normals) map to the familiar bluish
 * normal-map preview.
 */
function previewColor( valueNode, channel, alreadyEncoded ) {

	let value = valueNode;

	if ( channel.encode === 'signed' && ! alreadyEncoded ) value = value.mul( 0.5 ).add( 0.5 );

	if ( channel.size === 1 ) return vec3( value, value, value );

	return value;

}

export { CHANNELS, MAX_TOTAL_CHANNELS, getChannel, layoutChannels, previewColor };
