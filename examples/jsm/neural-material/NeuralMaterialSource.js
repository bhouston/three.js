import * as THREE from 'three';
import { atan, float, length, vec2, vec3, vec4 } from 'three/tsl';
import { bakeColorNodeToTexture } from '../neural-texture/NeuralTextureSource.js';
import { CHANNELS, layoutChannels, previewColor } from './NeuralMaterialFormat.js';

const TWO_PI = Math.PI * 2;

function encodeSigned( vectorNode ) {

	return vectorNode.mul( 0.5 ).add( 0.5 );

}

function encodeChannelNode( valueNode, channel ) {

	return channel.encode === 'signed' ? encodeSigned( valueNode ) : valueNode;

}

function resolveScalarNode( material, nodeKey, propertyKey, fallback ) {

	if ( material[ nodeKey ] ) return float( material[ nodeKey ] );
	if ( material[ propertyKey ] !== undefined ) return float( material[ propertyKey ] );

	return float( fallback );

}

function resolveColorNode( material, nodeKey, propertyKey, fallback ) {

	if ( material[ nodeKey ] ) return vec3( material[ nodeKey ] );
	if ( material[ propertyKey ] !== undefined ) return vec3( material[ propertyKey ] );

	return vec3( fallback );

}

function resolveAnisotropyNodes( material ) {

	let anisoVec;

	if ( material.anisotropyNode ) {

		anisoVec = vec2( material.anisotropyNode );

	} else {

		const strength = material.anisotropy !== undefined ? material.anisotropy : 0;
		const rotation = material.anisotropyRotation !== undefined ? material.anisotropyRotation : 0;
		anisoVec = vec2( Math.cos( rotation ) * strength, Math.sin( rotation ) * strength );

	}

	const strengthNode = length( anisoVec );
	// atan2(0,0) is a degenerate call whose result is implementation-defined
	// (can come back NaN on some GPU backends) - and (0,0) is exactly what
	// every material with no anisotropy at all (the common case) produces
	// here. A single NaN sample poisons the *entire* backward pass, not just
	// this channel, since the output layer's gradient into the shared hidden
	// layer sums over every active channel - so nudge x away from the
	// singularity by an amount far too small to affect any real (non-zero-
	// strength) anisotropy direction. Moot in practice now that a constant
	// (node-less) anisotropy never reaches this path at all - see
	// classifyMaterialChannels - but kept as a defensive guard for the case
	// where anisotropyNode *is* set but legitimately evaluates to zero
	// somewhere on the surface.
	const rotationNode = atan( anisoVec.y, anisoVec.x.add( 1e-6 ) ).div( TWO_PI ).add( 0.5 );

	return { strengthNode, rotationNode };

}

/**
 * Resolves every channel in NeuralMaterialFormat.CHANNELS to its raw
 * (physical, un-encoded) TSL node - falling back to the material's plain
 * scalar/color property when a channel has no node graph of its own. This
 * is the single place that knows how to pull each channel off a
 * MeshPhysicalNodeMaterial; baking, channel-preview materials, and constant-
 * channel detection all build on top of it (or its JS-side counterpart,
 * `resolveConstantValue`, for channels classified as constant).
 */
function resolveMaterialChannelNodes( material ) {

	const emissiveColor = resolveColorNode( material, 'emissiveNode', 'emissive', new THREE.Color( 0, 0, 0 ) );
	const emissiveIntensity = material.emissiveIntensity !== undefined ? material.emissiveIntensity : 1;
	const sheenColor = resolveColorNode( material, 'sheenNode', 'sheenColor', new THREE.Color( 0, 0, 0 ) );
	const sheenStrength = material.sheen !== undefined ? material.sheen : 1;
	const { strengthNode: anisotropyStrength, rotationNode: anisotropyRotation } = resolveAnisotropyNodes( material );

	return {
		albedo: material.colorNode ? vec3( material.colorNode ) : vec3( material.color || new THREE.Color( 1, 1, 1 ) ),
		opacity: resolveScalarNode( material, 'opacityNode', 'opacity', 1 ),
		normal: material.normalNode ? vec3( material.normalNode ) : vec3( 0, 0, 1 ),
		roughness: resolveScalarNode( material, 'roughnessNode', 'roughness', 1 ),
		metalness: resolveScalarNode( material, 'metalnessNode', 'metalness', 0 ),
		clearcoat: resolveScalarNode( material, 'clearcoatNode', 'clearcoat', 0 ),
		clearcoatRoughness: resolveScalarNode( material, 'clearcoatRoughnessNode', 'clearcoatRoughness', 0 ),
		clearcoatNormal: material.clearcoatNormalNode ? vec3( material.clearcoatNormalNode ) : vec3( 0, 0, 1 ),
		transmission: resolveScalarNode( material, 'transmissionNode', 'transmission', 0 ),
		emissive: emissiveColor.mul( emissiveIntensity ),
		anisotropyStrength,
		anisotropyRotation,
		sheenColor: sheenColor.mul( sheenStrength ),
		sheenRoughness: resolveScalarNode( material, 'sheenRoughnessNode', 'sheenRoughness', 1 )
	};

}

/**
 * JS-side (not TSL) counterpart of resolveMaterialChannelNodes, used only
 * for channels classifyMaterialChannels has determined are constant -
 * returns a plain number or [r,g,b]/[x,y,z] array, ready to assign directly
 * to the reconstructed material's ordinary (non-node) property.
 */
function resolveConstantValue( material, key ) {

	switch ( key ) {

		case 'albedo': {

			const c = material.color || new THREE.Color( 1, 1, 1 );
			return [ c.r, c.g, c.b ];

		}

		case 'opacity': return material.opacity !== undefined ? material.opacity : 1;
		case 'normal': return [ 0, 0, 1 ];
		case 'roughness': return material.roughness !== undefined ? material.roughness : 1;
		case 'metalness': return material.metalness !== undefined ? material.metalness : 0;
		case 'clearcoat': return material.clearcoat !== undefined ? material.clearcoat : 0;
		case 'clearcoatRoughness': return material.clearcoatRoughness !== undefined ? material.clearcoatRoughness : 0;
		case 'clearcoatNormal': return [ 0, 0, 1 ];
		case 'transmission': return material.transmission !== undefined ? material.transmission : 0;
		case 'emissive': {

			const c = material.emissive || new THREE.Color( 0, 0, 0 );
			const i = material.emissiveIntensity !== undefined ? material.emissiveIntensity : 1;
			return [ c.r * i, c.g * i, c.b * i ];

		}

		case 'anisotropyStrength': return material.anisotropy !== undefined ? material.anisotropy : 0;
		case 'anisotropyRotation': return material.anisotropyRotation !== undefined ? material.anisotropyRotation : 0;
		case 'sheenColor': {

			const c = material.sheenColor || new THREE.Color( 0, 0, 0 );
			const s = material.sheen !== undefined ? material.sheen : 1;
			return [ c.r * s, c.g * s, c.b * s ];

		}

		case 'sheenRoughness': return material.sheenRoughness !== undefined ? material.sheenRoughness : 1;
		default: throw new Error( `THREE.NeuralMaterialSource: unknown channel "${key}".` );

	}

}

/**
 * Splits a material's channels into the ones worth training (driven by a
 * node graph - texture or procedural, so potentially spatially-varying) and
 * the ones that are just a flat constant (a plain scalar/color property,
 * with no node at all) - so the network's output width, and the cost of
 * every training sample, scale with how much of the material actually
 * varies rather than always paying for the full 24-channel vocabulary.
 * Constant channels are applied directly as ordinary material properties at
 * reconstruction time, bypassing the network entirely (see
 * NeuralMaterialNodeMaterial.js).
 *
 * This is a node-presence heuristic, not a pixel-level analysis: a channel
 * is "active" (trained) iff at least one of its `nodeKeys` is set on the
 * material, regardless of whether that node's output actually varies once
 * evaluated - a texture-driven channel that happens to be uniform everywhere
 * still gets trained (harmlessly - a constant target is trivially fit, see
 * the flat-velvet-color regression test in NeuralTextureTrainer). An earlier
 * version of this function baked and read back each candidate channel to
 * measure its real variance and reclassify accordingly; that added real
 * complexity and failure surface (a GPU readback round trip per material
 * load) without a clear net win, so it's been backed out in favor of this
 * simpler, synchronous check.
 */
function classifyMaterialChannels( material ) {

	const activeList = [];
	const constantValues = {};

	for ( const channel of CHANNELS ) {

		const hasNode = channel.nodeKeys.some( ( key ) => Boolean( material[ key ] ) );

		if ( hasNode ) {

			activeList.push( channel );

		} else {

			constantValues[ channel.key ] = resolveConstantValue( material, channel.key );

		}

	}

	const { channels: activeChannels, totalChannels, packCount } = layoutChannels( activeList );

	return { activeChannels, totalChannels, packCount, constantValues };

}

function flattenChannelComponents( activeChannels, channelNodes ) {

	const components = [];
	const axes = [ 'x', 'y', 'z' ];

	for ( const channel of activeChannels ) {

		const encoded = encodeChannelNode( channelNodes[ channel.key ], channel );

		if ( channel.size === 1 ) {

			components.push( encoded );

		} else {

			for ( let i = 0; i < channel.size; i ++ ) components.push( encoded[ axes[ i ] ] );

		}

	}

	return components;

}

function packComponentsIntoVec4( components ) {

	const packs = [];

	for ( let i = 0; i < components.length; i += 4 ) {

		const group = components.slice( i, i + 4 );
		while ( group.length < 4 ) group.push( float( 0 ) );
		packs.push( vec4( group[ 0 ], group[ 1 ], group[ 2 ], group[ 3 ] ) );

	}

	return packs;

}

/**
 * Builds the packed RGBA colorNodes (see NeuralMaterialFormat.js) used to
 * bake training targets for exactly the given (active/trained) channels, in
 * their layout order - encoding signed channels (normals, anisotropy
 * direction) into [0,1] first.
 */
function buildPackedColorNodes( activeChannels, material ) {

	const channelNodes = resolveMaterialChannelNodes( material );
	const components = flattenChannelComponents( activeChannels, channelNodes );

	return packComponentsIntoVec4( components );

}

/**
 * Bakes every active (trained) PBR channel of a material into
 * `activeChannels.packCount`-worth of RGBA textures via one fullscreen-quad
 * render per pack (see NeuralMaterialFormat.js for how flat channel indices
 * map onto them).
 */
async function bakeMaterialToTextures( renderer, material, resolution, activeChannels ) {

	const colorNodes = buildPackedColorNodes( activeChannels, material );
	const renderTargets = [];

	for ( const colorNode of colorNodes ) {

		renderTargets.push( await bakeColorNodeToTexture( renderer, colorNode, resolution ) );

	}

	return renderTargets;

}

/**
 * Builds one unlit `THREE.NodeMaterial` per channel in NeuralMaterialFormat.
 * CHANNELS (plus the `shaded` key, which is just the original material) so a
 * "debug view" GUI control can swap the teacher mesh's material to inspect
 * any single channel in isolation - active or constant alike - mirroring
 * `NeuralMaterialNodeMaterial.setDebugView` on the neural side, so both are
 * comparable in the same encoded [0,1] space.
 */
function buildChannelPreviewMaterials( material ) {

	const channelNodes = resolveMaterialChannelNodes( material );
	const materials = { shaded: material };

	for ( const channel of CHANNELS ) {

		const previewMaterial = new THREE.NodeMaterial();
		previewMaterial.lights = false;
		previewMaterial.toneMapped = false;
		previewMaterial.colorNode = vec4( previewColor( channelNodes[ channel.key ], channel, false ), 1 );
		materials[ channel.key ] = previewMaterial;

	}

	return materials;

}

export {
	bakeMaterialToTextures,
	buildPackedColorNodes,
	buildChannelPreviewMaterials,
	classifyMaterialChannels,
	resolveMaterialChannelNodes
};
