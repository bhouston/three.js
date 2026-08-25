import * as THREE from 'three';
import { bitangentWorld, fract, log, max, tangentWorld, uv, vec2, vec3 } from 'three/tsl';
import { buildLevelTextures, evaluateNeuralTextureRaw } from './NTCDecoderTSL.js';
import { applyChannelActivation } from './NTCOutputActivations.js';
import { CHANNELS, FRAME_VIEWS, getChannel, buildDebugViewColorNode, buildFrameViewColorNode } from './NTCFormat.js';
import { constantToNode, reconstructFinalNormal } from './NTCOutputTypes.js';

/**
 * Slices the raw per-channel output array (see evaluateNeuralTextureRaw)
 * into a { [channelKey]: TSL node } map, following the offsets of the
 * *active* (trained) channel layout passed to the constructor - which,
 * unlike the full channel vocabulary, only covers whatever subset of
 * channels this particular material actually varies spatially (see
 * NTCSource.classifyMaterialChannels). Each channel's raw
 * (linear-decoder) slice is passed through its own output activation here -
 * see NTCFormat.js / ./NTCOutputActivations.js - so every
 * consumer downstream of `sliceChannels` already sees values in the
 * channel's natural physical range.
 */
function sliceChannels( outputs, activeChannels ) {

	const slices = {};

	for ( const channel of activeChannels ) {

		const values = [];
		for ( let i = 0; i < channel.size; i ++ ) values.push( applyChannelActivation( outputs[ channel.offset + i ], channel.activation ) );

		if ( channel.size === 1 ) slices[ channel.key ] = values[ 0 ];
		else if ( channel.size === 2 ) slices[ channel.key ] = vec2( ...values );
		else slices[ channel.key ] = vec3( ...values );

	}

	return slices;

}

/**
 * Estimates a screen-space LOD (mip index) node from `coord`'s own
 * screen-space derivatives, the same way hardware texture filtering picks a
 * mip level - `max(|d(coord)/dx|, |d(coord)/dy|)` in texel units, log2'd.
 * Deliberately derived from `coord` (the *pre*-`fract()` tiled UV, still
 * continuous across a repeat-wrapped tile boundary) rather than the final
 * wrapped UV fed to `texture()` - `fract()`'s wraparound discontinuity would
 * otherwise show up as a spurious huge derivative (and therefore a spurious
 * max-LOD spike) at every tile seam.
 */
function computeAutoLodNode( coord, textureResolution, maxLod ) {

	const texelCoord = coord.mul( textureResolution );
	const footprint = max( texelCoord.dFdx().length(), texelCoord.dFdy().length() ).max( 1e-6 );

	return log( footprint ).div( Math.LN2 ).clamp( 0, maxLod );

}

/**
 * A `MeshPhysicalNodeMaterial` driven by one trained `NTCTrainer`
 * model: channels the source material actually varied spatially (`
 * activeChannels`) read a decoded slice of one shared grid + MLP forward
 * pass - all from the *same* pass, matching the NVIDIA neural texture
 * compression "one decoder, many correlated channels" design - while
 * channels that were just a flat constant on the source material
 * (`constantValues`) are applied directly as ordinary material properties,
 * bypassing the network entirely (see NTCSource.
 * classifyMaterialChannels for why: no sense spending network capacity, or
 * training-sample budget, reproducing a value that never varies).
 *
 * Applying a trained slice or a resolved constant onto this material is
 * itself channel-defined, not hardcoded here - see each channel descriptor's
 * `applyActive`/`applyConstant` in NTCFormat.js. This constructor
 * is a fully generic loop over whatever `channels` array the classification
 * was built from (`options.channels`, defaulting to the built-in `CHANNELS`
 * vocabulary), so a caller-supplied custom channel array works exactly the
 * same way the built-in one does.
 *
 * @three_import import { NTCNodeMaterial } from 'three/addons/ntc/NTCNodeMaterial.js';
 */
class NTCNodeMaterial extends THREE.MeshPhysicalNodeMaterial {

	/**
	 * `channelClassification` is whatever `NTCSource.
	 * classifyMaterialChannels` returned - `{ activeChannels, constantValues,
	 * totalChannels, packCount }` - passed through as-is rather than
	 * destructured by the caller, since the two fields this constructor
	 * needs are always produced together and never meaningfully used apart.
	 *
	 * `options.channels` is the full channel array (active + constant alike)
	 * `channelClassification` was produced from - defaults to the built-in
	 * `CHANNELS` vocabulary, matching `classifyMaterialChannels`'s own
	 * default, since `setDebugView` needs every channel descriptor (not just
	 * the active ones) to resolve a debug view by key.
	 *
	 * `options.renderer`, when given (already `init()`-ed), lets the decoder
	 * weights use a real fp16 storage buffer instead of an fp32 uniform
	 * array on backends that support it - see evaluateNeuralTextureRaw.
	 */
	// `options.lodNode`, for a `cpuModel` trained with mip-pyramid support
	// (see NTCMipPyramid.js / NTCGridPyramidModel.js - `cpuModel.mipPyramid`,
	// `null` for a model trained with `enableMipPyramid: false` or loaded
	// from a `.ntc` file predating this feature), overrides the LOD (mip
	// index) this material reconstructs at - a TSL float node, useful for an
	// explicit distance-based LOD or for debugging a specific mip level.
	// Defaults to an automatic screen-space-derivative estimate
	// (`computeAutoLodNode` above), the same way hardware mipmapping picks a
	// level - which is what makes a mip-pyramid-trained material anti-alias
	// correctly when viewed from a distance instead of always reconstructing
	// at full (finest-LOD) detail. Ignored entirely for a `cpuModel` without
	// `mipPyramid`.
	constructor( cpuModel, channelClassification, options = {} ) {

		super();

		const { activeChannels, constantValues, renderFlags } = channelClassification;
		const channels = options.channels || CHANNELS;

		// Mirror the source material's `side`/`transparent` (see
		// NTCSource.resolveRenderFlags's doc comment) - most
		// importantly so a transmissive source's `DoubleSide` survives onto
		// this material too, keeping its Beer-Lambert attenuation pass count
		// (and therefore tint strength) consistent with what it was fit
		// against. `renderFlags` is undefined for a manifest saved before this
		// field existed - falls back to this class's inherited default
		// (`FrontSide`/opaque) exactly as before.
		if ( renderFlags ) {

			if ( renderFlags.side !== undefined ) this.side = renderFlags.side;
			if ( renderFlags.transparent !== undefined ) this.transparent = renderFlags.transparent;

		}

		this.cpuModel = cpuModel;
		this.activeChannels = activeChannels;
		this.channels = channels;
		this.levelTextures = buildLevelTextures( cpuModel );

		const uvScaleNode = options.uvScaleNode;
		const uvOffsetNode = options.uvOffsetNode;

		let coord = uv();
		if ( uvScaleNode ) coord = coord.mul( uvScaleNode );
		if ( uvOffsetNode ) coord = coord.add( uvOffsetNode );
		const tiledUV = fract( coord );

		// Auto-LOD (see computeAutoLodNode above) needs `coord` (pre-`fract()`,
		// still continuous across a tile seam) - only actually built when this
		// model is mip-pyramid-aware and the caller didn't already supply an
		// explicit override.
		const mipPyramid = cpuModel.mipPyramid || null;
		const lodNode = mipPyramid ?
			( options.lodNode || computeAutoLodNode( coord, mipPyramid.textureResolution, mipPyramid.maxLod ) ) :
			null;

		const outputs = evaluateNeuralTextureRaw( tiledUV, cpuModel, this.levelTextures, options.renderer, lodNode );
		const slices = sliceChannels( outputs, activeChannels );
		this._slices = slices;
		this._constantValues = constantValues;

		const isActive = ( key ) => Object.prototype.hasOwnProperty.call( slices, key );

		// Every channel in the vocabulary is applied identically here: a
		// trained (active) channel's decoded slice goes through its own
		// `applyActive`, an untrained (constant, or `alwaysConstant`) channel's
		// resolved value goes through its own `applyConstant` - see each
		// descriptor in NTCFormat.js for what that actually does
		// (assign a *Node property, set a plain property, decompose into a
		// legacy strength/rotation pair, leave a property untouched, ...).
		for ( const channel of channels ) {

			if ( isActive( channel.key ) ) channel.applyActive( this, slices[ channel.key ] );
			else channel.applyConstant( this, constantValues[ channel.key ] );

		}

		// `albedo`'s applyActive stashes its trained colorNode as
		// `_shadedColorNode` (see NTCFormat.js) - only set when
		// albedo is actually active; a constant-albedo material has no
		// "trained color node" to swap back in via setDebugView('shaded').
		this._shadedColorNode = this._shadedColorNode || null;

		this.setDebugView( options.debugView || 'shaded' );

	}

	/**
	 * Swaps the visible output between the full physically-shaded material
	 * and an unlit preview of one channel in isolation - trained or
	 * constant alike - matching `NTCSource.
	 * buildChannelPreviewMaterials` on the teacher side, so both are
	 * comparable directly.
	 */
	setDebugView( view ) {

		this.userData.debugView = view;

		if ( view === 'shaded' ) {

			this.lights = true;
			this.toneMapped = true;
			if ( this._shadedColorNode ) this.colorNode = this._shadedColorNode;

		} else if ( FRAME_VIEWS.includes( view ) ) {

			// Debug-only views of the raw tangentWorld/bitangentWorld frame
			// itself, bypassing the trained network entirely - see
			// FRAME_VIEWS's doc comment in NTCFormat.js and
			// NTCSource.buildChannelPreviewMaterials's matching
			// 'tangent'/'bitangent' entries. Both this mesh and the teacher
			// mesh share the same geometry (and therefore the same tangent/
			// bitangent vertex attribute), so this view should be pixel-
			// identical between the two - if it isn't, the mismatch is in
			// how the frame is built/consumed here, not in anything the
			// network learned.
			this.lights = false;
			this.toneMapped = false;
			const frameNode = view === 'tangent' ? tangentWorld : bitangentWorld;
			this.colorNode = buildFrameViewColorNode( frameNode );

		} else {

			// Tone mapping (ACES etc.) is meant for real lit HDR output, not a
			// flat diagnostic color - left at its default (true) here, it gets
			// applied to this raw channel value anyway, which can skew colors
			// (filmic curves are notorious for pushing bright/uneven values
			// toward warm/red tones) and make this view uncomparable to the
			// teacher's preview material, which already sets toneMapped=false
			// (see NTCSource.buildChannelPreviewMaterials).
			this.lights = false;
			this.toneMapped = false;
			const channel = getChannel( view, this.channels );
			const active = Object.prototype.hasOwnProperty.call( this._slices, view );

			// `this._slices[ view ]` already went through this channel's own
			// output activation (see sliceChannels/applyChannelActivation), so
			// it's in the same raw physical range as the teacher's resolved
			// node - always `alreadyEncoded: false` here, exactly like
			// `NTCSource.buildChannelPreviewMaterials`'s own call,
			// so both sides remap into display space identically.
			//
			// A `type: 'normal'` channel needs one extra step: `this._slices[
			// view ]` is still the raw tangent-space (dx, dy) offset, whereas
			// the teacher's preview (NTCSource.
			// buildChannelPreviewMaterials) shows `material.normalNode`, which
			// MaterialXLoader already blends through the mesh's own TBN basis
			// into a final view-space normal (see NTCOutputTypes.
			// reconstructFinalNormal's doc comment). Comparing the raw
			// tangent-space slice against that would show two different spaces
			// side by side - route it through the same TBN blend used for
			// actual shading (via the channel's own `applyActive`, which does
			// exactly this) so both previews are apples-to-apples. Only
			// `type: 'normal'` channels need this; every other channel's
			// slice is already in its final, directly-previewable form.
			const value = active
				? ( channel.type === 'normal' ? reconstructFinalNormal( this._slices[ view ] ) : this._slices[ view ] )
				: constantToNode( this._constantValues[ view ] );

			// `channel.size` (2, for a `type: 'normal'` channel) describes the
			// *trained* (dx, dy) payload, not this preview value - for those
			// channels it's always the TBN-reconstructed, fully 3-component
			// view-space normal above (both when active and, via
			// constantToNode, in the constant-fallback case).
			// `buildDebugViewColorNode` widens the channel descriptor
			// accordingly (see OUTPUT_TYPES.normal.previewSize in
			// NTCOutputTypes.js), so all three components actually reach
			// the display color, matching the familiar blue-dominant
			// tangent-space normal map palette instead of a z-less yellow one.
			this.colorNode = buildDebugViewColorNode( channel, value );

		}

		this.needsUpdate = true;

	}

	dispose() {

		for ( const levelTexture of this.levelTextures ) levelTexture.dispose();

		super.dispose();

	}

}

// The end-to-end classify -> bake -> train -> reconstruct convenience path
// (formerly a `static fit()` here) lives in `training/NTCFit.js` - it pulls
// in `NTCTrainer`/`NTCSource`, which this file deliberately doesn't: every
// other export here only *decodes* an already-trained model (loader- and
// inference-example-facing), with no training-side dependencies at all.

export { NTCNodeMaterial, sliceChannels, reconstructFinalNormal };
