import { vec3 } from 'three/tsl';

/**
 * Fixed 20-channel schema for jointly fitting a standard PBR material's
 * texture inputs with one shared grid + MLP network (NVIDIA neural texture
 * compression style: one small decoder, many correlated output channels).
 *
 * Channels are grouped into 5 RGBA "packs" so they can be baked (and later
 * sampled as training targets) as 5 textures instead of 20 - this is purely
 * a packing convenience, unrelated to the physical meaning of each channel.
 *
 * Vector channels that can be negative (normals, anisotropy direction) are
 * stored network-side in [0,1] via `n * 0.5 + 0.5` and decoded back with
 * `n * 2 - 1` wherever they're consumed - both baking and runtime
 * reconstruction go through `encode`/`decode` below so the mapping only
 * has to be defined once.
 */

const CHANNELS = [
	{ key: 'albedo', size: 3, encode: 'linear' },
	{ key: 'opacity', size: 1, encode: 'linear' },
	{ key: 'normal', size: 3, encode: 'signed' },
	{ key: 'roughness', size: 1, encode: 'linear' },
	{ key: 'metalness', size: 1, encode: 'linear' },
	{ key: 'clearcoat', size: 1, encode: 'linear' },
	{ key: 'clearcoatRoughness', size: 1, encode: 'linear' },
	{ key: 'transmission', size: 1, encode: 'linear' },
	{ key: 'emissive', size: 3, encode: 'linear' },
	{ key: 'anisotropyStrength', size: 1, encode: 'linear' },
	{ key: 'clearcoatNormal', size: 3, encode: 'signed' },
	{ key: 'anisotropyRotation', size: 1, encode: 'angle' }
];

let offset = 0;

for ( const channel of CHANNELS ) {

	channel.offset = offset;
	offset += channel.size;

}

const TOTAL_CHANNELS = offset; // 20
const PACK_COUNT = Math.ceil( TOTAL_CHANNELS / 4 ); // 5 RGBA textures

function getChannel( key ) {

	const channel = CHANNELS.find( ( c ) => c.key === key );
	if ( channel === undefined ) throw new Error( `THREE.NeuralMaterialFormat: unknown channel "${key}".` );

	return channel;

}

/**
 * Maps a flat channel index (0..19) to which pack texture and which of its
 * 4 components (r/g/b/a) it lives in.
 */
function componentLocation( flatIndex ) {

	return { pack: Math.floor( flatIndex / 4 ), component: flatIndex % 4 };

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

export { CHANNELS, TOTAL_CHANNELS, PACK_COUNT, getChannel, componentLocation, previewColor };
