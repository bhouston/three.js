import * as THREE from 'three';
import { bitangentWorld, fract, tangentWorld, uv, vec2, vec3 } from 'three/tsl';
import { buildLevelTextures, evaluateNeuralTextureRaw } from '../neural-texture/NeuralTextureNodeMaterial.js';
import { NeuralTextureTrainer } from '../neural-texture/NeuralTextureTrainer.js';
import { applyChannelActivation } from '../neural/NeuralOutputActivations.js';
import { CHANNELS, FRAME_VIEWS, getChannel, buildDebugViewColorNode, buildFrameViewColorNode, buildChannelActivations } from './NeuralMaterialFormat.js';
import { constantToNode, reconstructFinalNormal } from './NeuralOutputTypes.js';
import { bakeMaterialToTextures, classifyMaterialChannels } from './NeuralMaterialSource.js';

/**
 * Slices the raw per-channel output array (see evaluateNeuralTextureRaw)
 * into a { [channelKey]: TSL node } map, following the offsets of the
 * *active* (trained) channel layout passed to the constructor - which,
 * unlike the full channel vocabulary, only covers whatever subset of
 * channels this particular material actually varies spatially (see
 * NeuralMaterialSource.classifyMaterialChannels). Each channel's raw
 * (linear-decoder) slice is passed through its own output activation here -
 * see NeuralMaterialFormat.js / ../neural/NeuralOutputActivations.js - so every
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
 * Applying a trained slice or a resolved constant onto this material is
 * itself channel-defined, not hardcoded here - see each channel descriptor's
 * `applyActive`/`applyConstant` in NeuralMaterialFormat.js. This constructor
 * is a fully generic loop over whatever `channels` array the classification
 * was built from (`options.channels`, defaulting to the built-in `CHANNELS`
 * vocabulary), so a caller-supplied custom channel array works exactly the
 * same way the built-in one does.
 *
 * @three_import import { NeuralMaterialNodeMaterial } from 'three/addons/neural-material/NeuralMaterialNodeMaterial.js';
 */
class NeuralMaterialNodeMaterial extends THREE.MeshPhysicalNodeMaterial {

	/**
	 * `channelClassification` is whatever `NeuralMaterialSource.
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
	constructor( cpuModel, channelClassification, options = {} ) {

		super();

		const { activeChannels, constantValues, renderFlags } = channelClassification;
		const channels = options.channels || CHANNELS;

		// Mirror the source material's `side`/`transparent` (see
		// NeuralMaterialSource.resolveRenderFlags's doc comment) - most
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

		const outputs = evaluateNeuralTextureRaw( tiledUV, cpuModel, this.levelTextures, options.renderer );
		const slices = sliceChannels( outputs, activeChannels );
		this._slices = slices;
		this._constantValues = constantValues;

		const isActive = ( key ) => Object.prototype.hasOwnProperty.call( slices, key );

		// Every channel in the vocabulary is applied identically here: a
		// trained (active) channel's decoded slice goes through its own
		// `applyActive`, an untrained (constant, or `alwaysConstant`) channel's
		// resolved value goes through its own `applyConstant` - see each
		// descriptor in NeuralMaterialFormat.js for what that actually does
		// (assign a *Node property, set a plain property, decompose into a
		// legacy strength/rotation pair, leave a property untouched, ...).
		for ( const channel of channels ) {

			if ( isActive( channel.key ) ) channel.applyActive( this, slices[ channel.key ] );
			else channel.applyConstant( this, constantValues[ channel.key ] );

		}

		// `albedo`'s applyActive stashes its trained colorNode as
		// `_shadedColorNode` (see NeuralMaterialFormat.js) - only set when
		// albedo is actually active; a constant-albedo material has no
		// "trained color node" to swap back in via setDebugView('shaded').
		this._shadedColorNode = this._shadedColorNode || null;

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

		} else if ( FRAME_VIEWS.includes( view ) ) {

			// Debug-only views of the raw tangentWorld/bitangentWorld frame
			// itself, bypassing the trained network entirely - see
			// FRAME_VIEWS's doc comment in NeuralMaterialFormat.js and
			// NeuralMaterialSource.buildChannelPreviewMaterials's matching
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
			// (see NeuralMaterialSource.buildChannelPreviewMaterials).
			this.lights = false;
			this.toneMapped = false;
			const channel = getChannel( view, this.channels );
			const active = Object.prototype.hasOwnProperty.call( this._slices, view );

			// `this._slices[ view ]` already went through this channel's own
			// output activation (see sliceChannels/applyChannelActivation), so
			// it's in the same raw physical range as the teacher's resolved
			// node - always `alreadyEncoded: false` here, exactly like
			// `NeuralMaterialSource.buildChannelPreviewMaterials`'s own call,
			// so both sides remap into display space identically.
			//
			// A `type: 'normal'` channel needs one extra step: `this._slices[
			// view ]` is still the raw tangent-space (dx, dy) offset, whereas
			// the teacher's preview (NeuralMaterialSource.
			// buildChannelPreviewMaterials) shows `material.normalNode`, which
			// MaterialXLoader already blends through the mesh's own TBN basis
			// into a final view-space normal (see NeuralOutputTypes.
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
			// NeuralOutputTypes.js), so all three components actually reach
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

	/**
	 * End-to-end convenience path covering the sequence every consumer of
	 * this addon otherwise has to hand-assemble from five separate low-level
	 * pieces: classify the material's channels, bake the active ones to
	 * textures, train a `NeuralTextureTrainer` against them, and construct
	 * (and, on every progress tick, re-construct and dispose the previous)
	 * `NeuralMaterialNodeMaterial`.
	 *
	 * `options` is passed straight through to `NeuralTextureTrainer` (so
	 * `levels`, `hiddenSizes`, `iterations`, `learningRate`, etc. all apply),
	 * plus a few fit()-specific fields: `resolution` (bake resolution,
	 * default 512), `debugView` (default 'shaded'), `channels` (the channel
	 * vocabulary to fit against, default the built-in `CHANNELS` - see
	 * NeuralMaterialFormat.js), and `onProgress`, called with the usual
	 * `NeuralTextureTrainer` progress payload plus a `material` field holding
	 * the current (already-disposing-its-predecessor) in-progress material,
	 * suitable for live preview during training.
	 *
	 * Throws if every channel on `material` classifies as constant - see
	 * `NeuralMaterialSource.classifyMaterialChannels` - since there's then
	 * nothing for a network to fit; construct directly from a
	 * classification's `constantValues` in that case instead.
	 */
	static async fit( renderer, material, options = {} ) {

		const { onProgress, resolution = 512, debugView = 'shaded', channels = CHANNELS, ...trainerOptions } = options;

		const channelClassification = classifyMaterialChannels( material, channels );

		if ( channelClassification.activeChannels.length === 0 ) {

			throw new Error( 'THREE.NeuralMaterialNodeMaterial.fit: every channel on this material is constant - there is nothing for a network to fit. Use NeuralMaterialSource.classifyMaterialChannels() directly instead.' );

		}

		const renderTargets = await bakeMaterialToTextures( renderer, material, resolution, channelClassification.activeChannels );

		const trainer = new NeuralTextureTrainer( {
			outputChannels: channelClassification.totalChannels,
			channelActivations: buildChannelActivations( channelClassification.activeChannels ),
			...trainerOptions
		} );

		let current = null;

		const rebuild = ( cpuModel ) => {

			const previous = current;
			current = new NeuralMaterialNodeMaterial( cpuModel, channelClassification, { debugView, channels } );
			if ( previous ) previous.dispose();

			return current;

		};

		try {

			const result = await trainer.train( {
				renderer,
				sourceTextures: renderTargets.map( ( renderTarget ) => renderTarget.texture ),
				onProgress: onProgress ? ( progress ) => onProgress( { ...progress, material: rebuild( progress.cpuModel ) } ) : null
			} );

			rebuild( result.cpuModel );

			return {
				material: current,
				channelClassification,
				cpuModel: result.cpuModel,
				loss: result.loss,
				iteration: result.iteration,
				iterations: result.iterations,
				stoppedEarly: result.stoppedEarly
			};

		} finally {

			for ( const renderTarget of renderTargets ) renderTarget.dispose();

		}

	}

}

export { NeuralMaterialNodeMaterial, sliceChannels, reconstructFinalNormal };
