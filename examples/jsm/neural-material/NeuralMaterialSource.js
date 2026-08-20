import * as THREE from 'three';
import { bitangentWorld, float, tangentWorld, vec2, vec3, vec4 } from 'three/tsl';
import { bakeColorNodeToTexture } from '../neural-texture/NeuralTextureSource.js';
import { CHANNELS, SIMPLE_SCALAR_KEYS, FRAME_VIEWS, getChannel, layoutChannels, buildDebugViewColorNode, buildFrameViewColorNode } from './NeuralMaterialFormat.js';

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

/**
 * Resolves the material's anisotropy direction as a raw signed 2D vector
 * `(strength*cos(rotation), strength*sin(rotation))` - exactly the form
 * `MeshPhysicalMaterial.anisotropyNode` itself expects, and the form the
 * trained network predicts directly via a single 'tanh' `anisotropy`
 * channel (see NeuralMaterialFormat.js) rather than a separate
 * strength/rotation pair - which avoids ever needing an atan2-based angle
 * encoding (with its 0/1 wraparound discontinuity and atan2(0,0) singularity
 * at zero strength) in the first place.
 */
function resolveAnisotropyNodes( material ) {

	if ( material.anisotropyNode ) return vec2( material.anisotropyNode );

	const strength = material.anisotropy !== undefined ? material.anisotropy : 0;
	const rotation = material.anisotropyRotation !== undefined ? material.anisotropyRotation : 0;

	return vec2( Math.cos( rotation ) * strength, Math.sin( rotation ) * strength );

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

	const nodes = {
		albedo: material.colorNode ? vec3( material.colorNode ) : vec3( material.color || new THREE.Color( 1, 1, 1 ) ),
		normal: material.normalNode ? vec3( material.normalNode ) : vec3( 0, 0, 1 ),
		clearcoatNormal: material.clearcoatNormalNode ? vec3( material.clearcoatNormalNode ) : vec3( 0, 0, 1 ),
		emissive: emissiveColor.mul( emissiveIntensity ),
		anisotropy: resolveAnisotropyNodes( material ),
		sheenColor: sheenColor.mul( sheenStrength )
	};

	// The remaining channels are all a single `${key}Node`/`${key}` node/
	// property pair with a scalar default - see SIMPLE_SCALAR_KEYS.
	for ( const key of SIMPLE_SCALAR_KEYS ) {

		const channel = getChannel( key );
		nodes[ key ] = resolveScalarNode( material, key + 'Node', key, channel.defaultValue );

	}

	return nodes;

}

/**
 * JS-side (not TSL) counterpart of resolveMaterialChannelNodes, used only
 * for channels classifyMaterialChannels has determined are constant -
 * returns a plain number or [r,g,b]/[x,y,z] array, ready to assign directly
 * to the reconstructed material's ordinary (non-node) property.
 */
function resolveConstantValue( material, key ) {

	if ( SIMPLE_SCALAR_KEYS.includes( key ) ) {

		const defaultValue = getChannel( key ).defaultValue;
		return material[ key ] !== undefined ? material[ key ] : defaultValue;

	}

	switch ( key ) {

		case 'albedo': {

			const c = material.color || new THREE.Color( 1, 1, 1 );
			return [ c.r, c.g, c.b ];

		}

		case 'normal': return [ 0, 0, 1 ];
		case 'clearcoatNormal': return [ 0, 0, 1 ];
		case 'emissive': {

			const c = material.emissive || new THREE.Color( 0, 0, 0 );
			const i = material.emissiveIntensity !== undefined ? material.emissiveIntensity : 1;
			return [ c.r * i, c.g * i, c.b * i ];

		}

		case 'anisotropy': {

			const strength = material.anisotropy !== undefined ? material.anisotropy : 0;
			const rotation = material.anisotropyRotation !== undefined ? material.anisotropyRotation : 0;
			return [ Math.cos( rotation ) * strength, Math.sin( rotation ) * strength ];

		}

		case 'sheenColor': {

			const c = material.sheenColor || new THREE.Color( 0, 0, 0 );
			const s = material.sheen !== undefined ? material.sheen : 1;
			return [ c.r * s, c.g * s, c.b * s ];

		}

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

		const value = channelNodes[ channel.key ];

		if ( channel.size === 1 ) {

			components.push( value );

		} else {

			// 2-component 'tanh' channels (normal/clearcoatNormal offsets,
			// anisotropy direction) train against only (x, y) - the raw
			// resolved node may still be a full vec3 (e.g. the material's
			// tangent-space normalNode); z is deliberately dropped here and
			// reconstructed at consumption time instead (see
			// NeuralMaterialNodeMaterial.reconstructFinalNormal).
			for ( let i = 0; i < channel.size; i ++ ) {

				let component = value[ axes[ i ] ];

				// The normal/clearcoatNormal offset is baked on a flat quad
				// whose UV.v gets flipped before computeTangents() runs (see
				// bakeColorNodeToTexture in NeuralTextureSource.js - that flip
				// is required for correct scalar-channel UV round-tripping),
				// which flips that quad's bitangent handedness relative to
				// any real mesh's own (un-flipped) tangent basis. Since the
				// quad sits at the identity transform, its tangent/bitangent/
				// normal are otherwise meant to coincide with world X/Y/Z, so
				// the baked y here is the *negated* tangent-space offset the
				// network is meant to learn. Negate it back so the trained
				// (dx, dy) matches the true tangent-space convention that
				// NeuralMaterialNodeMaterial.reconstructFinalNormal() blends
				// through a real mesh's un-flipped tangentWorld/bitangentWorld
				// at inference time - without this, the neural material's
				// reconstructed normal disagrees in sign with the teacher's
				// live-evaluated one (visible as a mismatched "normal" debug
				// view between the two).
				if ( i === 1 && ( channel.key === 'normal' || channel.key === 'clearcoatNormal' ) ) component = component.negate();

				components.push( component );

			}

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
 * their layout order and raw physical units - each channel's `activation`
 * (see NeuralMaterialFormat.js / ../neural/NeuralOutputActivations.js) is what maps
 * the network's own output into this same raw range, so no target-side
 * encoding happens here.
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

	// Every preview below is built as a *clone* of the real (MeshPhysical-
	// NodeMaterial-derived) `material`, not a bare `THREE.NodeMaterial()`.
	// That matters specifically for `bitangentWorld`: it's never a stored
	// vertex attribute (only `tangent.xyz` + a handedness sign are - see
	// TangentUtils/Bitangent.js), so it's *always* re-derived per pixel as
	// normalWorld.cross(tangentWorld) inside a `.once(['NORMAL'])`-cached
	// helper that only gets wired up to a real per-frame varying when the
	// material actually runs a normal-mapping build pass. A bare
	// `NodeMaterial` with only `colorNode` set never triggers that pass, so
	// its `bitangentWorld` was observed to freeze at its first-build value
	// instead of tracking the mesh's rotation - `tangentWorld` doesn't have
	// this problem (it comes straight off the stored vertex attribute), and
	// `NeuralMaterialNodeMaterial` doesn't either, since it's a full
	// MeshPhysicalNodeMaterial subclass that always runs that setup
	// regardless of `lights`. Cloning `material` here (which already has
	// `normalNode`/`clearcoatNormalNode` set) gives every preview the same
	// class and the same setup path, so `bitangentWorld` behaves exactly as
	// it does in the real 'shaded' render and in the neural material.
	function clonePreviewMaterial() {

		const previewMaterial = material.clone();
		previewMaterial.lights = false;
		previewMaterial.toneMapped = false;
		return previewMaterial;

	}

	for ( const channel of CHANNELS ) {

		const previewMaterial = clonePreviewMaterial();
		previewMaterial.colorNode = buildDebugViewColorNode( channel, channelNodes[ channel.key ] );
		materials[ channel.key ] = previewMaterial;

	}

	// Debug-only views of the raw tangentWorld/bitangentWorld frame itself
	// (see FRAME_VIEWS's doc comment) - view-space, same tanh-vector color
	// convention as the 'normal'/'clearcoatNormal' channels above, so this
	// is directly comparable against NeuralMaterialNodeMaterial's own
	// 'tangent'/'bitangent' debug views.
	const frameNodes = { tangent: tangentWorld, bitangent: bitangentWorld };

	for ( const key of FRAME_VIEWS ) {

		const previewMaterial = clonePreviewMaterial();
		previewMaterial.colorNode = buildFrameViewColorNode( frameNodes[ key ] );
		materials[ key ] = previewMaterial;

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
