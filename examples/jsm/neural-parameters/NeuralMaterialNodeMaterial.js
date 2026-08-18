import * as THREE from 'three';
import { cos, float, fract, sin, uv, vec2, vec3, vec4 } from 'three/tsl';
import { buildLevelTextures, evaluateNeuralTextureRaw } from './NeuralTextureNodeMaterial.js';
import { getChannel, previewColor } from './NeuralMaterialFormat.js';

const TWO_PI = Math.PI * 2;

function decodeSigned( vectorNode ) {

	// A plain .normalize() is a 0/0 NaN trap if the network ever outputs
	// something whose decoded value lands exactly on the zero vector (most
	// likely right at initialization, before training has shaped this
	// channel at all) - nudge it off the singularity first, negligible for
	// any real, non-degenerate normal.
	return vectorNode.mul( 2 ).sub( 1 ).add( vec3( 1e-6, 0, 0 ) ).normalize();

}

/**
 * Slices the raw per-channel output array (see evaluateNeuralTextureRaw)
 * into a { [channelKey]: TSL node } map, following the offsets of the
 * *active* (trained) channel layout passed to the constructor - which,
 * unlike the full NeuralMaterialFormat.CHANNELS list, only covers whatever
 * subset of channels this particular material actually varies spatially
 * (see NeuralMaterialSource.classifyMaterialChannels).
 */
function sliceChannels( outputs, activeChannels ) {

	const slices = {};

	for ( const channel of activeChannels ) {

		const values = [];
		for ( let i = 0; i < channel.size; i ++ ) values.push( outputs[ channel.offset + i ] );
		slices[ channel.key ] = channel.size === 1 ? values[ 0 ] : vec3( ...values );

	}

	return slices;

}

function constantToNode( value ) {

	return Array.isArray( value ) ? vec3( ...value ) : float( value );

}

/**
 * A `MeshPhysicalNodeMaterial` driven by one trained `NeuralTextureTrainer`
 * model: channels the source material actually varied spatially (`
 * activeChannels`) read a decoded slice of one shared grid + MLP forward
 * pass - all from the *same* pass, matching the NVIDIA neural texture
 * compression "one decoder, many correlated channels" design - while
 * channels that were just a flat constant on the source material
 * (`constantValues`) are applied directly as ordinary material properties,
 * bypassing the network entirely (see NeuralMaterialSource.
 * classifyMaterialChannels for why: no sense spending network capacity, or
 * training-sample budget, reproducing a value that never varies).
 *
 * @three_import import { NeuralMaterialNodeMaterial } from 'three/addons/neural-parameters/NeuralMaterialNodeMaterial.js';
 */
class NeuralMaterialNodeMaterial extends THREE.MeshPhysicalNodeMaterial {

	constructor( cpuModel, activeChannels, constantValues, options = {} ) {

		super();

		this.cpuModel = cpuModel;
		this.activeChannels = activeChannels;
		this.levelTextures = buildLevelTextures( cpuModel );

		const uvScaleNode = options.uvScaleNode;
		const uvOffsetNode = options.uvOffsetNode;

		let coord = uv();
		if ( uvScaleNode ) coord = coord.mul( uvScaleNode );
		if ( uvOffsetNode ) coord = coord.add( uvOffsetNode );
		const tiledUV = fract( coord );

		const outputs = evaluateNeuralTextureRaw( tiledUV, cpuModel, this.levelTextures );
		const slices = sliceChannels( outputs, activeChannels );
		this._slices = slices;
		this._constantValues = constantValues;

		const isActive = ( key ) => Object.prototype.hasOwnProperty.call( slices, key );

		// --- albedo / opacity ---
		this._shadedColorNode = isActive( 'albedo' ) ? slices.albedo : null;
		if ( isActive( 'albedo' ) ) this.colorNode = this._shadedColorNode;
		else this.color = new THREE.Color( ...constantValues.albedo );

		if ( isActive( 'opacity' ) ) this.opacityNode = slices.opacity.clamp( 0, 1 );
		else this.opacity = constantValues.opacity;

		// --- normal / clearcoat normal (constant = "no bump", leave node unset) ---
		if ( isActive( 'normal' ) ) this.normalNode = decodeSigned( slices.normal );
		if ( isActive( 'clearcoatNormal' ) ) this.clearcoatNormalNode = decodeSigned( slices.clearcoatNormal );

		// --- scalar surface properties ---
		if ( isActive( 'roughness' ) ) this.roughnessNode = slices.roughness.clamp( 0.02, 1 );
		else this.roughness = constantValues.roughness;

		if ( isActive( 'metalness' ) ) this.metalnessNode = slices.metalness.clamp( 0, 1 );
		else this.metalness = constantValues.metalness;

		if ( isActive( 'clearcoat' ) ) this.clearcoatNode = slices.clearcoat.clamp( 0, 1 );
		else this.clearcoat = constantValues.clearcoat;

		if ( isActive( 'clearcoatRoughness' ) ) this.clearcoatRoughnessNode = slices.clearcoatRoughness.clamp( 0.02, 1 );
		else this.clearcoatRoughness = constantValues.clearcoatRoughness;

		// Transmission genuinely requires the renderer's separate screen-space
		// transmission pass to render correctly whenever it's non-zero -
		// that's inherent to reproducing a transmissive material, not
		// something to avoid. It's only skipped here when *constant*, where
		// for every material without a transmission map that constant is 0
		// (opaque), which leaves the expensive pass off exactly when the
		// source material didn't need it either.
		if ( isActive( 'transmission' ) ) this.transmissionNode = slices.transmission.clamp( 0, 1 );
		else this.transmission = constantValues.transmission;

		if ( isActive( 'emissive' ) ) this.emissiveNode = slices.emissive;
		else this.emissive = new THREE.Color( ...constantValues.emissive );

		// --- anisotropy (strength + rotation always classified together, see
		// NeuralMaterialFormat.js's shared anisotropyNode nodeKey) ---
		if ( isActive( 'anisotropyStrength' ) || isActive( 'anisotropyRotation' ) ) {

			const rotationRadians = slices.anisotropyRotation.sub( 0.5 ).mul( TWO_PI );
			const strength = slices.anisotropyStrength.clamp( 0, 1 );
			this.anisotropyNode = vec2( cos( rotationRadians ), sin( rotationRadians ) ).mul( strength );

		} else {

			this.anisotropy = constantValues.anisotropyStrength;
			this.anisotropyRotation = constantValues.anisotropyRotation;

		}

		// --- sheen ---
		if ( isActive( 'sheenColor' ) ) this.sheenNode = slices.sheenColor.clamp( 0, 1 );
		else this.sheenColor = new THREE.Color( ...constantValues.sheenColor );

		if ( isActive( 'sheenRoughness' ) ) this.sheenRoughnessNode = slices.sheenRoughness.clamp( 0.02, 1 );
		else this.sheenRoughness = constantValues.sheenRoughness;

		this.setDebugView( options.debugView || 'shaded' );

	}

	/**
	 * Swaps the visible output between the full physically-shaded material
	 * and an unlit preview of one channel in isolation - trained or
	 * constant alike - matching `NeuralMaterialSource.
	 * buildChannelPreviewMaterials` on the teacher side, so both are
	 * comparable directly.
	 */
	setDebugView( view ) {

		this.userData.debugView = view;

		if ( view === 'shaded' ) {

			this.lights = true;
			if ( this._shadedColorNode ) this.colorNode = this._shadedColorNode;

		} else {

			this.lights = false;
			const channel = getChannel( view );
			const active = Object.prototype.hasOwnProperty.call( this._slices, view );
			const value = active ? this._slices[ view ] : constantToNode( this._constantValues[ view ] );
			this.colorNode = vec4( previewColor( value, channel, active ), 1 );

		}

		this.needsUpdate = true;

	}

	dispose() {

		for ( const levelTexture of this.levelTextures ) levelTexture.dispose();

		super.dispose();

	}

}

export { NeuralMaterialNodeMaterial, sliceChannels, decodeSigned };
