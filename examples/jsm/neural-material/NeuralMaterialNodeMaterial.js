import * as THREE from 'three';
import { bitangentWorld, cos, float, fract, normalWorld, sin, tangentWorld, transformNormalToView, uv, vec2, vec3, vec4 } from 'three/tsl';
import { buildLevelTextures, evaluateNeuralTextureRaw } from '../neural-texture/NeuralTextureNodeMaterial.js';
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
 * Turns a trained tangent-space-ish perturbation vector into the mesh's
 * final view-space normal.
 *
 * This is NOT the standard three.js `normalMap(texture, scale)` convention,
 * where the returned vector is left in tangent space and the base
 * `NodeMaterial.setupNormal()` pipeline transforms it downstream using
 * whatever mesh it's applied to - `setupNormal()` actually uses `normalNode`
 * verbatim as the final normal, no further transform. MaterialX's own
 * `<normalmap>` conversion (see examples/jsm/loaders/materialx/
 * MaterialXNodeLibrary.js's mx_normalmap + MaterialXSurfaceMappings.js's
 * transformNormalToView(...) call) does the tangent/bitangent/normal blend
 * *inside* the graph itself, using whatever mesh that graph gets built
 * against - which is why baking that node's output on our flat training
 * quad and reapplying it verbatim to the torus knot was wrong for any real
 * (non-constant) bump detail: the baked value is relative to the quad's own
 * orientation, not a portable tangent-space vector, and setupNormal() never
 * re-blends it per-mesh. This function replicates that same blend here, but
 * against tangentWorld/bitangentWorld/normalWorld as resolved for whatever
 * mesh THIS material is actually applied to (the torus knot) - these are
 * dynamic per-mesh TSL accessors, so this "just works" the same way
 * MaterialX's own conversion does when applied directly to a mesh.
 */
function reconstructFinalNormal( encodedVectorNode ) {

	const tangentSpace = decodeSigned( encodedVectorNode );
	const blended = tangentWorld.mul( tangentSpace.x )
		.add( bitangentWorld.mul( tangentSpace.y ) )
		.add( normalWorld.mul( tangentSpace.z ) )
		.normalize();

	return transformNormalToView( blended );

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
 * @three_import import { NeuralMaterialNodeMaterial } from 'three/addons/neural-material/NeuralMaterialNodeMaterial.js';
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
		if ( isActive( 'normal' ) ) this.normalNode = reconstructFinalNormal( slices.normal );
		if ( isActive( 'clearcoatNormal' ) ) this.clearcoatNormalNode = reconstructFinalNormal( slices.clearcoatNormal );

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
			this.toneMapped = true;
			if ( this._shadedColorNode ) this.colorNode = this._shadedColorNode;

		} else {

			// Tone mapping (ACES etc.) is meant for real lit HDR output, not a
			// flat diagnostic color - left at its default (true) here, it gets
			// applied to this raw channel value anyway, which can skew colors
			// (filmic curves are notorious for pushing bright/uneven values
			// toward warm/red tones) and make this view uncomparable to the
			// teacher's preview material, which already sets toneMapped=false
			// (see NeuralMaterialSource.buildChannelPreviewMaterials).
			this.lights = false;
			this.toneMapped = false;
			const channel = getChannel( view );
			const active = Object.prototype.hasOwnProperty.call( this._slices, view );

			// Normal channels need special handling here: `this._slices[ view ]`
			// is the raw *tangent-space* decoded network output (still encoded
			// [0,1] via decodeSigned inside reconstructFinalNormal), whereas the
			// teacher's preview (NeuralMaterialSource.buildChannelPreviewMaterials)
			// shows `material.normalNode`, which MaterialXLoader already blends
			// through the mesh's own TBN basis into a final view-space normal
			// (see reconstructFinalNormal's doc comment above). Comparing the raw
			// tangent-space slice against that would show two different spaces
			// side by side - route it through the same TBN blend used for actual
			// shading (`this.normalNode`/`this.clearcoatNormalNode` above) so both
			// previews are apples-to-apples, matching how the teacher's channel
			// was already final/blended before previewColor's own signed-encode.
			const isNormalChannel = view === 'normal' || view === 'clearcoatNormal';
			const value = active
				? ( isNormalChannel ? reconstructFinalNormal( this._slices[ view ] ) : this._slices[ view ] )
				: constantToNode( this._constantValues[ view ] );
			const alreadyEncoded = active && isNormalChannel ? false : active;
			this.colorNode = vec4( previewColor( value, channel, alreadyEncoded ), 1 );

		}

		this.needsUpdate = true;

	}

	dispose() {

		for ( const levelTexture of this.levelTextures ) levelTexture.dispose();

		super.dispose();

	}

}

export { NeuralMaterialNodeMaterial, sliceChannels, decodeSigned };
