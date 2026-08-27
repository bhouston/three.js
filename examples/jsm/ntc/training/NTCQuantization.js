import { float, min, max, round } from 'three/tsl';
import { encodeUint8Base64, decodeUint8Base64, computeZeroPoint } from '../NTCBinaryCodec.js';

// Bit depths offered for the 'uint8' scheme (see QUANTIZATION_SCHEMES below
// and NTCBinaryCodec.js's encodeUint8Base64/decodeUint8Base64) - every value
// is still stored as one full byte on disk regardless of `bits` (no
// sub-byte packing, see those functions' doc comments), this only changes
// how many discrete levels that byte range is divided into. 8 (256 levels)
// is this addon's original, still-default behavior; 2/4/6 trade
// reconstruction fidelity for a lower-entropy byte stream, moving toward the
// NVIDIA neural texture compression paper's own 2-4 bit latents (Table 2).
const BITS_OPTIONS = [ 2, 4, 6, 8 ];

// Quantization-Aware Training (QAT) scheme registry, shared by every
// neural-* trainer (texture, material, appearance). Because every trainer's
// gradients are hand-derived (not autodiff - see NeuralGPUComputeTSL.js/
// NeuralTextureGPUComputeTSL.js/NeuralAppearanceGPUComputeTSL.js) and
// already treat a sampled latent as flowing through identity to the loss,
// a Straight-Through Estimator only needs a *forward* quantize step
// inserted where latents are sampled for MLP input - see the call sites in
// NeuralGridModel.js/NeuralTextureGPUComputeTSL.js/
// NeuralAppearanceGPUComputeTSL.js. The backward/gradient-accumulation
// kernels are untouched.
//
// Each scheme has the same shape - `quantizeForwardCPU(x, min, max)` (a
// plain JS function used by the CPU reference model/tests) and
// `quantizeForwardTSL(xNode, minNode, maxNode)` (the GPU-side mirror, built
// from the same 'three/tsl' node functions the rest of this codebase uses -
// see NeuralGPUComputeTSL.js/NeuralMLPTSL.js) - so adding a new scheme later
// is just adding another entry with both functions.
const QUANTIZATION_SCHEMES = {
	none: {
		quantizeForwardCPU: ( x ) => x,
		quantizeForwardTSL: ( xNode ) => xNode
	},
	uint8: {
		// Mirrors `encodeUint8Base64`/`decodeUint8Base64` composed together:
		// clamp to [min, max], quantize to one of `2**bits` levels (256 when
		// `bits` is the default of 8), decode back to float - the exact
		// "simulated quantization" a Straight-Through Estimator forward pass
		// needs. `zeroPreserving` (default false, matching every pre-existing
		// caller's behavior byte-for-byte) switches to the asymmetric/
		// zero-point scheme instead - see NTCBinaryCodec.js's
		// `computeZeroPoint` doc comment for the exact formula and rationale
		// (the NVIDIA neural texture compression paper's Section 4.2).
		quantizeForwardCPU: ( x, lo, hi, bits = 8, zeroPreserving = false ) => {

			const range = hi - lo;
			const levelCount = 2 ** bits;
			const maxLevel = levelCount - 1;

			if ( zeroPreserving ) {

				const scale = range / maxLevel;
				if ( scale === 0 ) return lo;

				const zeroPoint = computeZeroPoint( lo, hi, levelCount );
				const rawLevel = Math.round( x / scale ) + zeroPoint;
				const level = Math.min( maxLevel, Math.max( 0, rawLevel ) );

				return ( level - zeroPoint ) * scale;

			}

			const t = range !== 0 ? Math.min( 1, Math.max( 0, ( x - lo ) / range ) ) : 0;
			const level = Math.round( t * maxLevel );

			return lo + ( level / maxLevel ) * range;

		},
		quantizeForwardTSL: ( xNode, minNode, maxNode, bits = 8, zeroPreserving = false ) => {

			const levelCount = 2 ** bits;
			const maxLevel = float( levelCount - 1 );
			const range = maxNode.sub( minNode );

			if ( zeroPreserving ) {

				// zeroPoint must be a *live* TSL expression derived from
				// minNode/maxNode, not a JS-side constant baked in at kernel-
				// build time (e.g. by reading `minNode.value` once here): with
				// `quantization.range: 'auto'`, these uniforms' `.value` is
				// refreshed periodically (see NTCTrainer.js's
				// QUANTIZATION_RANGE_REFRESH_INTERVAL) *without* rebuilding the
				// compute kernel, so a baked-in zero point would silently go
				// stale the moment the range next changes. Recomputing it here
				// mirrors NTCBinaryCodec.js's computeZeroPoint exactly, just in
				// TSL form.
				const scale = range.div( maxLevel );
				const rawZeroPoint = round( minNode.negate().div( scale ) );
				const zeroPoint = min( maxLevel, max( float( 0.0 ), rawZeroPoint ) );
				const rawLevel = round( xNode.div( scale ) ).add( zeroPoint );
				const level = min( maxLevel, max( float( 0.0 ), rawLevel ) );

				return level.sub( zeroPoint ).mul( scale );

			}

			const t = min( float( 1.0 ), max( float( 0.0 ), xNode.sub( minNode ).div( range ) ) );
			const level = round( t.mul( maxLevel ) );

			return minNode.add( level.div( maxLevel ).mul( range ) );

		}
	}
};

const VALID_MODES = Object.keys( QUANTIZATION_SCHEMES );
const VALID_TARGETS = [ 'latents', 'weights', 'both' ];

const DEFAULT_QUANTIZATION_OPTIONS = {
	mode: 'none',
	target: 'latents',
	range: 'auto',
	perLevel: true,
	// See BITS_OPTIONS/QUANTIZATION_SCHEMES.uint8 above - 8 (256 levels) is
	// this addon's original behavior, kept as the default so an existing
	// `quantization: { mode: 'uint8' }` config (with no `bits` given) is
	// completely unaffected by this option's addition.
	bits: 8,
	// See QUANTIZATION_SCHEMES.uint8's zeroPreserving branch / NTCBinaryCodec.
	// js's computeZeroPoint doc comment - off by default (matching every
	// pre-existing caller's behavior byte-for-byte); most useful turned on
	// together with a low `bits` value.
	zeroPreserving: false
};

/**
 * Validates and defaults a trainer's `quantization` option, mirroring the
 * `NUMERIC_SETTINGS_SCHEMA`/`validateTrainingSettings` validation style in
 * NeuralAppearanceTrainer.js (clear thrown errors naming the bad field).
 * `target` currently only supports `'latents'` - `'weights'`/`'both'` are
 * accepted (reserved for a future weight-quantization scheme) but throw a
 * clear "not yet implemented" if actually selected, rather than silently
 * doing nothing.
 */
function resolveQuantizationConfig( options = {} ) {

	const input = options.quantization || {};
	const mode = input.mode !== undefined ? input.mode : DEFAULT_QUANTIZATION_OPTIONS.mode;
	const target = input.target !== undefined ? input.target : DEFAULT_QUANTIZATION_OPTIONS.target;
	const range = input.range !== undefined ? input.range : DEFAULT_QUANTIZATION_OPTIONS.range;
	const perLevel = input.perLevel !== undefined ? input.perLevel : DEFAULT_QUANTIZATION_OPTIONS.perLevel;
	const bits = input.bits !== undefined ? input.bits : DEFAULT_QUANTIZATION_OPTIONS.bits;
	const zeroPreserving = input.zeroPreserving !== undefined ? input.zeroPreserving : DEFAULT_QUANTIZATION_OPTIONS.zeroPreserving;

	if ( QUANTIZATION_SCHEMES[ mode ] === undefined ) {

		throw new Error( `THREE.NTCQuantization: quantization.mode must be one of [${ VALID_MODES.join( ', ' ) }], got "${ mode }".` );

	}

	if ( BITS_OPTIONS.includes( bits ) === false ) {

		throw new Error( `THREE.NTCQuantization: quantization.bits must be one of [${ BITS_OPTIONS.join( ', ' ) }], got "${ bits }".` );

	}

	if ( typeof zeroPreserving !== 'boolean' ) {

		throw new Error( 'THREE.NTCQuantization: quantization.zeroPreserving must be a boolean.' );

	}

	if ( VALID_TARGETS.includes( target ) === false ) {

		throw new Error( `THREE.NTCQuantization: quantization.target must be one of [${ VALID_TARGETS.join( ', ' ) }], got "${ target }".` );

	}

	if ( target !== 'latents' && mode !== 'none' ) {

		throw new Error( `THREE.NTCQuantization: quantization.target "${ target }" is not yet implemented - only "latents" is currently supported.` );

	}

	if ( range !== 'auto' ) {

		const isRangeTuple = Array.isArray( range ) && range.length === 2 &&
			Number.isFinite( range[ 0 ] ) && Number.isFinite( range[ 1 ] ) && range[ 0 ] <= range[ 1 ];

		if ( isRangeTuple === false ) {

			throw new Error( 'THREE.NTCQuantization: quantization.range must be "auto" or a [min, max] tuple with min <= max.' );

		}

	}

	if ( typeof perLevel !== 'boolean' ) {

		throw new Error( 'THREE.NTCQuantization: quantization.perLevel must be a boolean.' );

	}

	return { mode, target, range, perLevel, bits, zeroPreserving };

}

/**
 * CPU-side per-level (or global) min/max reduction over a flat latent-grid
 * Float32Array, used to implement `quantization.range === 'auto'` (see
 * `resolveQuantizationConfig` above). `gridLevels` is the `layout.gridLevels`
 * array every trainer's GPUModel already builds (`{ offset, floatCount, ... }`
 * - see NTCGPUModel.js/(removed, appearance deleted)'s
 * `computeXXXModelLayout`), so this works identically for both trainers
 * without either one re-deriving level byte ranges. Grids are small (a few
 * thousand texels at most per level), so a plain CPU scan is fast enough -
 * no GPU reduction kernel is needed (see NeuralAppearanceTrainer.js/
 * NTCTrainer.js for how often this actually runs).
 *
 * Returns one `[min, max]` tuple per grid level. When `perLevel` is false,
 * every level gets the *same* tuple - the global min/max across every level
 * - rather than the caller having to special-case a single shared range
 * downstream (the GPU kernels always index one range per level regardless of
 * `perLevel`). A level with zero texels (degenerate, shouldn't normally
 * happen) or a perfectly flat buffer falls back to `[-1, 1]` rather than
 * `[Infinity, -Infinity]`/`[x, x]`, mirroring `encodeUint8Base64`'s
 * `min === max` no-NaN guard.
 */
function computeLatentRanges( data, gridLevels, perLevel = true ) {

	const levelRanges = gridLevels.map( ( level ) => {

		let lo = Infinity;
		let hi = - Infinity;

		for ( let i = level.offset; i < level.offset + level.floatCount; i ++ ) {

			const value = data[ i ];
			if ( value < lo ) lo = value;
			if ( value > hi ) hi = value;

		}

		if ( lo > hi ) return [ - 1, 1 ];

		return [ lo, hi ];

	} );

	if ( perLevel ) return levelRanges;

	let lo = Infinity;
	let hi = - Infinity;

	for ( const [ levelLo, levelHi ] of levelRanges ) {

		if ( levelLo < lo ) lo = levelLo;
		if ( levelHi > hi ) hi = levelHi;

	}

	const globalRange = lo > hi ? [ - 1, 1 ] : [ lo, hi ];

	return gridLevels.map( () => globalRange );

}

/**
 * Reads the current latent-grid buffer back from the GPU, recomputes the
 * per-level `range: 'auto'` min/max (see `computeLatentRanges`), and pushes
 * it into `gpuModel`'s quantization-range uniforms (see
 * `setQuantizationRange` on NTCGPUModel.js/
 * (removed, appearance deleted)) - shared by NTCTrainer.js/
 * NeuralAppearanceTrainer.js so both call the exact same
 * readback-then-rescan sequence. A no-op (skips the readback entirely) when
 * quantization is disabled or the range is a fixed tuple - only `'auto'`
 * ever needs re-measuring.
 */
async function refreshGPUQuantizationRange( gpuModel, renderer ) {

	if ( gpuModel.quantization.mode === 'none' || gpuModel.quantization.range !== 'auto' ) return;

	const buffer = await renderer.getArrayBufferAsync( gpuModel.latentsBuffers.attribute );
	const latents = new Float32Array( buffer );
	const ranges = computeLatentRanges( latents, gpuModel.layout.gridLevels, gpuModel.quantization.perLevel );

	gpuModel.setQuantizationRange( ranges );

}

export {
	QUANTIZATION_SCHEMES,
	DEFAULT_QUANTIZATION_OPTIONS,
	BITS_OPTIONS,
	resolveQuantizationConfig,
	computeLatentRanges,
	refreshGPUQuantizationRange
};
