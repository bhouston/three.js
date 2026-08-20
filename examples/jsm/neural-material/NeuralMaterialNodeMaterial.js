import * as THREE from 'three';
import { bitangentWorld, float, fract, max, normalWorld, sqrt, tangentWorld, transformNormalToView, uv, vec2, vec3 } from 'three/tsl';
import { buildLevelTextures, evaluateNeuralTextureRaw } from '../neural-texture/NeuralTextureNodeMaterial.js';
import { NeuralTextureTrainer } from '../neural-texture/NeuralTextureTrainer.js';
import { applyChannelActivation } from '../neural/NeuralOutputActivations.js';
import { FRAME_VIEWS, SIMPLE_SCALAR_KEYS, getChannel, buildDebugViewColorNode, buildFrameViewColorNode, buildChannelActivations } from './NeuralMaterialFormat.js';
import { bakeMaterialToTextures, classifyMaterialChannels } from './NeuralMaterialSource.js';

/**
 * Turns a trained tangent-space (dx, dy) offset into the mesh's final
 * view-space normal. The network only predicts the 2-component offset (see
 * NeuralMaterialFormat.js's 'tanh'-activated `normal`/`clearcoatNormal`
 * channels) - z is reconstructed here as the positive root
 * `sqrt(1 - dx*dx - dy*dy)`, matching the always-positive-hemisphere
 * tangent-space z that MaterialX's own `normalNode` produces (the same
 * assumption the old full-vector encoding relied on: this z was always ~1,
 * never actually trained as an independent degree of freedom).
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
function reconstructFinalNormal( offsetNode ) {

	const dx = offsetNode.x;
	const dy = offsetNode.y;
	const dz = sqrt( max( float( 1 ).sub( dx.mul( dx ) ).sub( dy.mul( dy ) ), float( 0 ) ) );
	const tangentSpace = vec3( dx, dy, dz );
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
 * (see NeuralMaterialSource.classifyMaterialChannels). Each channel's raw
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

function constantToNode( value ) {

	if ( ! Array.isArray( value ) ) return float( value );

	return value.length === 2 ? vec2( ...value ) : vec3( ...value );

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

	/**
	 * `channelClassification` is whatever `NeuralMaterialSource.
	 * classifyMaterialChannels` returned - `{ activeChannels, constantValues,
	 * totalChannels, packCount }` - passed through as-is rather than
	 * destructured by the caller, since the two fields this constructor
	 * needs are always produced together and never meaningfully used apart.
	 */
	constructor( cpuModel, channelClassification, options = {} ) {

		super();

		const { activeChannels, constantValues } = channelClassification;

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

		// --- albedo (kept separate: this is also `this._shadedColorNode`,
		// swapped back in by setDebugView('shaded')) ---
		this._shadedColorNode = isActive( 'albedo' ) ? slices.albedo : null;
		if ( isActive( 'albedo' ) ) this.colorNode = this._shadedColorNode;
		else this.color = new THREE.Color( ...constantValues.albedo );

		// --- every channel that's just a `${key}Node`/`${key}` pair with a
		// scalar clamp range - see NeuralMaterialFormat.SIMPLE_SCALAR_KEYS ---
		for ( const key of SIMPLE_SCALAR_KEYS ) {

			const channel = getChannel( key );

			if ( isActive( key ) ) this[ key + 'Node' ] = slices[ key ].clamp( ...channel.clampRange );
			else this[ key ] = constantValues[ key ];

		}

		// --- normal / clearcoat normal (constant = "no bump", leave node unset) ---
		if ( isActive( 'normal' ) ) this.normalNode = reconstructFinalNormal( slices.normal );
		if ( isActive( 'clearcoatNormal' ) ) this.clearcoatNormalNode = reconstructFinalNormal( slices.clearcoatNormal );

		if ( isActive( 'emissive' ) ) this.emissiveNode = slices.emissive;
		else this.emissive = new THREE.Color( ...constantValues.emissive );

		// --- anisotropy (trained directly as a signed (ax, ay) direction
		// vector - see NeuralMaterialFormat.js - which is exactly the form
		// MeshPhysicalMaterial.anisotropyNode itself expects, no cos/sin
		// recombination needed) ---
		if ( isActive( 'anisotropy' ) ) {

			this.anisotropyNode = slices.anisotropy;

		} else {

			this.anisotropy = Math.hypot( ...constantValues.anisotropy );
			this.anisotropyRotation = Math.atan2( constantValues.anisotropy[ 1 ], constantValues.anisotropy[ 0 ] );

		}

		// --- sheen (sheenColor is a Color + separate `sheen` intensity
		// property, not a plain scalar, so it's not in SIMPLE_SCALAR_KEYS) ---
		if ( isActive( 'sheenColor' ) ) this.sheenNode = slices.sheenColor.clamp( 0, 1 );
		else this.sheenColor = new THREE.Color( ...constantValues.sheenColor );

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
			const channel = getChannel( view );
			const active = Object.prototype.hasOwnProperty.call( this._slices, view );

			// `this._slices[ view ]` already went through this channel's own
			// output activation (see sliceChannels/applyChannelActivation), so
			// it's in the same raw physical range as the teacher's resolved
			// node - always `alreadyEncoded: false` here, exactly like
			// `NeuralMaterialSource.buildChannelPreviewMaterials`'s own call,
			// so both sides remap into display space identically.
			//
			// Normal channels need one extra step: `this._slices[ view ]` is
			// still the raw tangent-space (dx, dy) offset, whereas the
			// teacher's preview (NeuralMaterialSource.buildChannelPreviewMaterials)
			// shows `material.normalNode`, which MaterialXLoader already blends
			// through the mesh's own TBN basis into a final view-space normal
			// (see reconstructFinalNormal's doc comment above). Comparing the
			// raw tangent-space slice against that would show two different
			// spaces side by side - route it through the same TBN blend used
			// for actual shading (`this.normalNode`/`this.clearcoatNormalNode`
			// above) so both previews are apples-to-apples.
			const isNormalChannel = view === 'normal' || view === 'clearcoatNormal';
			const value = active
				? ( isNormalChannel ? reconstructFinalNormal( this._slices[ view ] ) : this._slices[ view ] )
				: constantToNode( this._constantValues[ view ] );

			// `channel.size` (2) describes the *trained* (dx, dy) payload, not
			// this preview value - for normal/clearcoatNormal it's always the
			// TBN-reconstructed, fully 3-component view-space normal above
			// (both when active and, via constantToNode, in the [0,0,1]
			// constant-fallback case). `buildDebugViewColorNode` widens the
			// channel descriptor to size 3 for exactly these two channels, so
			// all three components actually reach the display color, matching
			// the familiar blue-dominant tangent-space normal map palette
			// instead of a z-less yellow one.
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
	 * default 512), `debugView` (default 'shaded'), and `onProgress`, called
	 * with the usual `NeuralTextureTrainer` progress payload plus a
	 * `material` field holding the current (already-disposing-its-
	 * predecessor) in-progress material, suitable for live preview during
	 * training.
	 *
	 * Throws if every channel on `material` classifies as constant - see
	 * `NeuralMaterialSource.classifyMaterialChannels` - since there's then
	 * nothing for a network to fit; construct directly from a
	 * classification's `constantValues` in that case instead.
	 */
	static async fit( renderer, material, options = {} ) {

		const { onProgress, resolution = 512, debugView = 'shaded', ...trainerOptions } = options;

		const channelClassification = classifyMaterialChannels( material );

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
			current = new NeuralMaterialNodeMaterial( cpuModel, channelClassification, { debugView } );
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
