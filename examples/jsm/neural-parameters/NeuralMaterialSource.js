import * as THREE from 'three';
import { atan, float, length, vec2, vec3, vec4 } from 'three/tsl';
import { bakeColorNodeToTexture } from './NeuralTextureSource.js';
import { CHANNELS, PACK_COUNT, previewColor } from './NeuralMaterialFormat.js';

const TWO_PI = Math.PI * 2;

function encodeSigned( vectorNode ) {

	return vectorNode.mul( 0.5 ).add( 0.5 );

}

/**
 * Resolves a material channel to a TSL node, whether it's driven by a node
 * graph (MaterialX texture / procedural expression) or a plain scalar/color
 * property - so every channel bakes to a texture uniformly, even ones the
 * source material only defines as a constant.
 */
function resolveScalar( material, nodeKey, propertyKey, fallback ) {

	if ( material[ nodeKey ] ) return float( material[ nodeKey ] );
	if ( material[ propertyKey ] !== undefined ) return float( material[ propertyKey ] );

	return float( fallback );

}

function resolveColor( material, nodeKey, propertyKey, fallback ) {

	if ( material[ nodeKey ] ) return vec3( material[ nodeKey ] );
	if ( material[ propertyKey ] !== undefined ) return vec3( material[ propertyKey ] );

	return vec3( fallback );

}

function resolveAnisotropy( material ) {

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
	// layer sums over all 20 channels - so nudge x away from the singularity
	// by an amount far too small to affect any real (non-zero-strength)
	// anisotropy direction.
	const rotationNode = atan( anisoVec.y, anisoVec.x.add( 1e-6 ) ).div( TWO_PI ).add( 0.5 );

	return { strengthNode, rotationNode };

}

/**
 * Resolves every supported PBR channel of a material to its raw (physical,
 * un-encoded) TSL node - falling back to the material's plain scalar/color
 * property when a given channel has no node graph of its own (e.g. a
 * constant roughness rather than a roughness map). This is the single place
 * that knows how to pull each channel off a MeshPhysicalNodeMaterial; both
 * baking (which additionally *encodes* signed channels, see
 * NeuralMaterialFormat.js) and channel-preview materials build on top of it.
 */
function resolveMaterialChannels( material ) {

	const emissiveColor = resolveColor( material, 'emissiveNode', 'emissive', new THREE.Color( 0, 0, 0 ) );
	const emissiveIntensity = material.emissiveIntensity !== undefined ? material.emissiveIntensity : 1;
	const { strengthNode: anisotropyStrength, rotationNode: anisotropyRotation } = resolveAnisotropy( material );

	return {
		albedo: material.colorNode ? vec3( material.colorNode ) : vec3( material.color || new THREE.Color( 1, 1, 1 ) ),
		opacity: resolveScalar( material, 'opacityNode', 'opacity', 1 ),
		normal: material.normalNode ? vec3( material.normalNode ) : vec3( 0, 0, 1 ),
		roughness: resolveScalar( material, 'roughnessNode', 'roughness', 1 ),
		metalness: resolveScalar( material, 'metalnessNode', 'metalness', 0 ),
		clearcoat: resolveScalar( material, 'clearcoatNode', 'clearcoat', 0 ),
		clearcoatRoughness: resolveScalar( material, 'clearcoatRoughnessNode', 'clearcoatRoughness', 0 ),
		transmission: resolveScalar( material, 'transmissionNode', 'transmission', 0 ),
		emissive: emissiveColor.mul( emissiveIntensity ),
		anisotropyStrength,
		clearcoatNormal: material.clearcoatNormalNode ? vec3( material.clearcoatNormalNode ) : vec3( 0, 0, 1 ),
		anisotropyRotation
	};

}

/**
 * Builds the 5 packed RGBA colorNodes (see NeuralMaterialFormat.js) used to
 * bake training targets, encoding signed channels (normals, anisotropy
 * direction) into [0,1] first.
 */
function buildPackedColorNodes( material ) {

	const ch = resolveMaterialChannels( material );

	return [
		vec4( ch.albedo, ch.opacity ),
		vec4( encodeSigned( ch.normal ), ch.roughness ),
		vec4( ch.metalness, ch.clearcoat, ch.clearcoatRoughness, ch.transmission ),
		vec4( ch.emissive, ch.anisotropyStrength ),
		vec4( encodeSigned( ch.clearcoatNormal ), ch.anisotropyRotation )
	];

}

/**
 * Builds one unlit `THREE.NodeMaterial` per supported channel (plus the
 * `shaded` key, which is just the original material) so a "debug view" GUI
 * control can swap the teacher mesh's material to inspect any single channel
 * in isolation - mirroring `NeuralMaterialNodeMaterial.setDebugView` on the
 * neural side, so both are comparable in the same encoded [0,1] space.
 */
function buildChannelPreviewMaterials( material ) {

	const ch = resolveMaterialChannels( material );
	const materials = { shaded: material };

	for ( const channel of CHANNELS ) {

		const previewMaterial = new THREE.NodeMaterial();
		previewMaterial.lights = false;
		previewMaterial.toneMapped = false;
		previewMaterial.colorNode = vec4( previewColor( ch[ channel.key ], channel, false ), 1 );
		materials[ channel.key ] = previewMaterial;

	}

	return materials;

}

/**
 * Bakes every supported PBR channel of a MeshPhysicalNodeMaterial into
 * `PACK_COUNT` RGBA textures via one fullscreen-quad render per pack (see
 * NeuralTextureFormat.js for how flat channel indices map onto them).
 */
async function bakeMaterialToTextures( renderer, material, resolution = 512 ) {

	const colorNodes = buildPackedColorNodes( material );
	if ( colorNodes.length !== PACK_COUNT ) throw new Error( 'THREE.NeuralMaterialSource: pack count mismatch.' );

	const renderTargets = [];

	for ( const colorNode of colorNodes ) {

		renderTargets.push( await bakeColorNodeToTexture( renderer, colorNode, resolution ) );

	}

	return renderTargets;

}

export { bakeMaterialToTextures, buildPackedColorNodes, buildChannelPreviewMaterials, resolveMaterialChannels };
