import * as THREE from 'three';
import { atan, cos, fract, sin, uv, vec2, vec3, vec4 } from 'three/tsl';
import { buildLevelTextures, evaluateNeuralTextureRaw } from './NeuralTextureNodeMaterial.js';
import { CHANNELS, getChannel, previewColor } from './NeuralMaterialFormat.js';

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
 * into a { [channelKey]: TSL node } map, following the same offsets defined
 * in NeuralMaterialFormat.js so training and reconstruction never drift out
 * of sync.
 */
function sliceChannels( outputs ) {

	const slices = {};

	for ( const channel of CHANNELS ) {

		const values = [];
		for ( let i = 0; i < channel.size; i ++ ) values.push( outputs[ channel.offset + i ] );
		slices[ channel.key ] = channel.size === 1 ? values[ 0 ] : vec3( ...values );

	}

	return slices;

}

/**
 * A `MeshPhysicalNodeMaterial` driven entirely by one trained
 * `NeuralTextureTrainer` model fit to the full standard PBR channel set
 * (see NeuralMaterialFormat.js): every node slot below reads a decoded slice
 * of the *same* shared grid + MLP forward pass, matching the NVIDIA neural
 * texture compression "one decoder, many correlated channels" design rather
 * than one network per channel.
 *
 * @three_import import { NeuralMaterialNodeMaterial } from 'three/addons/neural-parameters/NeuralMaterialNodeMaterial.js';
 */
class NeuralMaterialNodeMaterial extends THREE.MeshPhysicalNodeMaterial {

	constructor( cpuModel, options = {} ) {

		super();

		this.cpuModel = cpuModel;
		this.levelTextures = buildLevelTextures( cpuModel );

		const uvScaleNode = options.uvScaleNode;
		const uvOffsetNode = options.uvOffsetNode;

		let coord = uv();
		if ( uvScaleNode ) coord = coord.mul( uvScaleNode );
		if ( uvOffsetNode ) coord = coord.add( uvOffsetNode );
		const tiledUV = fract( coord );

		const outputs = evaluateNeuralTextureRaw( tiledUV, cpuModel, this.levelTextures );
		const slices = sliceChannels( outputs );
		this._slices = slices;

		this._shadedColorNode = slices.albedo;
		this.colorNode = this._shadedColorNode;
		this.opacityNode = slices.opacity.clamp( 0, 1 );
		this.normalNode = decodeSigned( slices.normal );
		this.roughnessNode = slices.roughness.clamp( 0.02, 1 );
		this.metalnessNode = slices.metalness.clamp( 0, 1 );
		this.clearcoatNode = slices.clearcoat.clamp( 0, 1 );
		this.clearcoatRoughnessNode = slices.clearcoatRoughness.clamp( 0.02, 1 );
		this.clearcoatNormalNode = decodeSigned( slices.clearcoatNormal );
		this.emissiveNode = slices.emissive;

		const rotationRadians = slices.anisotropyRotation.sub( 0.5 ).mul( TWO_PI );
		const strength = slices.anisotropyStrength.clamp( 0, 1 );
		this.anisotropyNode = vec2( cos( rotationRadians ), sin( rotationRadians ) ).mul( strength );

		// Transmission is trained like every other channel (see
		// NeuralMaterialFormat.js / debug view), but *not* wired into the live
		// shading node by default: MeshPhysicalNodeMaterial treats a non-null
		// transmissionNode as "this object is transmissive" regardless of the
		// value it evaluates to, routing it through the renderer's separate
		// screen-space transmission pass (an extra background capture) - for
		// a demo scene that isn't set up for that, an always-on transmission
		// path is a likely source of rendering corruption for materials that
		// are supposed to be fully opaque. Opt in explicitly once the source
		// material actually uses transmission.
		if ( options.enableTransmission ) this.transmissionNode = slices.transmission.clamp( 0, 1 );

		this.setDebugView( options.debugView || 'shaded' );

	}

	/**
	 * Swaps the visible output between the full physically-shaded material
	 * and an unlit preview of one trained channel in isolation (matching
	 * `NeuralMaterialSource.buildChannelPreviewMaterials` on the teacher
	 * side, so both can be compared directly).
	 */
	setDebugView( view ) {

		this.userData.debugView = view;

		if ( view === 'shaded' ) {

			this.lights = true;
			this.colorNode = this._shadedColorNode;

		} else {

			this.lights = false;
			this.colorNode = vec4( previewColor( this._slices[ view ], getChannel( view ), true ), 1 );

		}

		this.needsUpdate = true;

	}

	dispose() {

		for ( const levelTexture of this.levelTextures ) levelTexture.dispose();

		super.dispose();

	}

}

export { NeuralMaterialNodeMaterial, sliceChannels, decodeSigned };
