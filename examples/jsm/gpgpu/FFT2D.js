import { StorageBufferAttribute, MathUtils } from 'three/webgpu';
import {
	Fn, If, instanceIndex, storage, texture, uint, int, ivec2, uvec2, uniform, vec2, vec4,
	storageTexture, textureStore, NodeAccess,
	workgroupArray, workgroupBarrier, workgroupId, invocationLocalIndex, globalId, localId
} from 'three/tsl';

/**
 * Builds the `(cos, sin)` twiddle-factor lookup table shared by every butterfly stage, both row
 * and column, at every size this `FFT2D` instance ever dispatches: for stage span `p` (a power of
 * two, `1..N/2`) and butterfly-local offset `lo` (`0..p-1`), the twiddle factor is
 * `exp(-i*pi*lo/p)`. Substituting `k = lo * (halfMax/p)` -- an integer since every `p` here
 * divides `halfMax` (both are powers of two, and `N` -- `width` or `height` -- always divides
 * `maxN = max(width,height)`, so `p <= N/2` always divides `halfMax = maxN/2` too) -- turns that
 * into `exp(-i*pi*k/halfMax)` for `k` in `0..halfMax-1`: exactly the `halfMax`-point table built
 * here, one lookup replacing a `cos`/`sin` pair per butterfly. Computed once on the CPU since it
 * depends only on `width`/`height`, not on any transform's data. Inspired by the precomputed
 * twiddle-factor texture in Token-Gremlin/natural-disasters' `OceanFFT.js`, adapted from its
 * WebGL fragment-shader texture lookup to a WebGPU storage-buffer one.
 *
 * @param {number} halfMax - `max(width,height) / 2`.
 * @returns {Float32Array} `halfMax` `(cos, sin)` pairs, interleaved.
 */
function buildTwiddleTable( halfMax ) {

	const table = new Float32Array( halfMax * 2 );

	for ( let k = 0; k < halfMax; k ++ ) {

		const angle = - Math.PI * k / halfMax;
		table[ k * 2 ] = Math.cos( angle );
		table[ k * 2 + 1 ] = Math.sin( angle );

	}

	return table;

}

/**
 * Reads a named compute limit off `renderer`'s real WebGPU device, throwing rather than silently
 * falling back to a guessed/hardcoded value if it's unavailable. `_ensureButterfliesBuilt` (and
 * the other call sites below) only run after `renderer.init()` has resolved, so the device is
 * expected to be present; a throw here means this renderer isn't backed by WebGPU.
 *
 * @param {Renderer} renderer
 * @param {string} name - e.g. `'maxComputeInvocationsPerWorkgroup'`.
 * @returns {number}
 */
function requireLimit( renderer, name ) {

	const backend = renderer.backend;

	if ( backend.isWebGPUBackend !== true || backend.device === null ) {

		throw new Error( 'FFT2D requires a WebGPU renderer with an initialized device.' );

	}

	const value = backend.device.limits[ name ];

	if ( typeof value !== 'number' ) {

		throw new Error( `FFT2D: renderer.backend.device.limits.${ name } is not a number (got ${ value }).` );

	}

	return value;

}

/**
 * Reads all of the real `GPUDevice.limits` this module sizes kernels against.
 *
 * @param {Renderer} renderer
 * @returns {{maxComputeInvocationsPerWorkgroup: number, maxComputeWorkgroupSizeX: number, maxComputeWorkgroupSizeY: number, maxComputeWorkgroupStorageSize: number}}
 */
function getComputeLimits( renderer ) {

	return {
		maxComputeInvocationsPerWorkgroup: requireLimit( renderer, 'maxComputeInvocationsPerWorkgroup' ),
		maxComputeWorkgroupSizeX: requireLimit( renderer, 'maxComputeWorkgroupSizeX' ),
		maxComputeWorkgroupSizeY: requireLimit( renderer, 'maxComputeWorkgroupSizeY' ),
		maxComputeWorkgroupStorageSize: requireLimit( renderer, 'maxComputeWorkgroupStorageSize' )
	};

}

/**
 * Picks a workgroup size for elementwise/fallback kernels (the conjugate/load/store passes and
 * `buildMultiDispatchStage`'s per-stage dispatch): the largest power of two, up to `preferred`,
 * that fits the device's actual invocation limits. There's no correctness requirement on this
 * size (unlike the fused/transpose kernels' shared-memory-derived sizes) -- `preferred` is just a
 * reasonable upper bound picked to keep occupancy high on typical hardware without relying on it.
 *
 * @param {Object} limits - A `GPUSupportedLimits`-shaped object, from `getComputeLimits`.
 * @param {number} [preferred=256]
 * @returns {number}
 */
function pickWorkgroupSize( limits, preferred = 256 ) {

	const maxInvocations = Math.min( limits.maxComputeInvocationsPerWorkgroup, limits.maxComputeWorkgroupSizeX );

	let size = 1;

	while ( size * 2 <= maxInvocations && size * 2 <= preferred ) {

		size *= 2;

	}

	return size;

}

/**
 * Largest power-of-two line length that fits `buildFusedLineStage` within one workgroup's
 * invocation and shared-memory budget.
 *
 * @param {Object} limits - A `GPUSupportedLimits`-shaped object.
 * @returns {number}
 */
function computeMaxFusedLineLength( limits ) {

	const maxInvocations = Math.min( limits.maxComputeInvocationsPerWorkgroup, limits.maxComputeWorkgroupSizeX );
	const maxStorageBytes = limits.maxComputeWorkgroupStorageSize;

	let N = 2;

	while ( ( N * 2 ) / 2 <= maxInvocations && 16 * ( N * 2 ) <= maxStorageBytes ) {

		N *= 2;

	}

	return N;

}

/**
 * Largest square tile edge (power of two) `buildTransposeStage` can use within one workgroup's
 * invocation and shared-memory budget.
 *
 * @param {Object} limits - A `GPUSupportedLimits`-shaped object.
 * @returns {number}
 */
function computeTransposeTileSize( limits ) {

	const maxInvocations = Math.min(
		limits.maxComputeInvocationsPerWorkgroup,
		limits.maxComputeWorkgroupSizeX,
		limits.maxComputeWorkgroupSizeY
	);
	const maxStorageBytes = limits.maxComputeWorkgroupStorageSize;

	let T = 2;

	while ( ( T * 2 ) * ( T * 2 ) <= maxInvocations && 8 * ( T * 2 ) * ( T * 2 ) <= maxStorageBytes ) {

		T *= 2;

	}

	return T;

}

/**
 * Fallback radix-2 Stockham FFT butterfly stage: one dispatch per stage, reading/writing through
 * global storage. Generalized to process many contiguous 1D lines of a 2D buffer at once. Used
 * when a line is too long to fuse into a single dispatch (see `computeMaxFusedLineLength`).
 *
 * @tsl
 * @private
 * @param {Object} params
 * @param {number} params.N - Length of each 1D line (power of two).
 * @param {number} params.lineStride - Address increment between lines.
 * @param {number} params.elementStride - Address increment between consecutive elements of a line (always `1` here).
 * @param {number} params.lineCount - Number of lines transformed in parallel.
 * @param {Node<uint>} params.pUniform - Per-stage span uniform (doubles every stage, 1..N/2).
 * @param {number} workgroupSize - Workgroup size to dispatch with (see `pickWorkgroupSize`).
 * @param {StorageBufferNode} params.twiddleNode - `halfMax`-entry twiddle-factor table (see `buildTwiddleTable`).
 * @param {number} params.halfMax - `max(width,height) / 2`; the twiddle table's entry count.
 * @param {StorageBufferNode} readNode - Buffer to read from.
 * @param {StorageBufferNode} writeNode - Buffer to write to.
 * @returns {Function} A parameterless TSL function, `.compute()`-d with `workgroupSize`.
 */
function buildMultiDispatchStage( { N, lineStride, elementStride, lineCount, pUniform, workgroupSize, twiddleNode, halfMax }, readNode, writeNode ) {

	const half = N / 2;
	const dispatchCount = half * lineCount;

	return Fn( () => {

		const t = instanceIndex;
		const line = t.div( uint( half ) );
		const tt = t.mod( uint( half ) );

		const p = pUniform;
		const hi = tt.div( p );
		const lo = tt.mod( p );

		const lineBase = line.mul( uint( lineStride ) );

		const idx1 = lineBase.add( hi.mul( p ).add( lo ).mul( uint( elementStride ) ) );
		const idx2 = idx1.add( uint( half * elementStride ) );

		const v0 = readNode.element( idx1 ).toVar( 'v0' );
		const v1 = readNode.element( idx2 ).toVar( 'v1' );

		// Forward-transform sign convention; the inverse reuses this kernel by conjugating
		// input and output around it (see FFT2D#computeInverse). `p` divides `halfMax` (see
		// `buildTwiddleTable`), so this stride is always an exact integer division.
		const twiddleStride = uint( halfMax ).div( p );
		const tw = twiddleNode.element( lo.mul( twiddleStride ) );
		const c = tw.x;
		const s = tw.y;

		const v1r = v1.x.mul( c ).sub( v1.y.mul( s ) );
		const v1i = v1.x.mul( s ).add( v1.y.mul( c ) );

		const out0 = vec2( v0.x.add( v1r ), v0.y.add( v1i ) );
		const out1 = vec2( v0.x.sub( v1r ), v0.y.sub( v1i ) );

		const j = lineBase.add( hi.mul( p ).mul( 2 ).add( lo ).mul( uint( elementStride ) ) );
		const j2 = j.add( p.mul( uint( elementStride ) ) );

		writeNode.element( j ).assign( out0 );
		writeNode.element( j2 ).assign( out1 );

	} )().compute( dispatchCount, [ workgroupSize ] );

}

/**
 * Whole-line radix-2 Stockham FFT (every stage, one row or column) as a single dispatch, one
 * workgroup per line, entirely in workgroup-shared memory: one global read, `log2(N)` butterfly
 * stages ping-ponging between two shared buffers, one global write. Preferred over
 * `buildMultiDispatchStage` whenever a line fits (see `computeMaxFusedLineLength`). The stage
 * loop is unrolled in JS at build time, so `p` is a compile-time constant per stage.
 *
 * @tsl
 * @private
 * @param {Object} params
 * @param {number} params.N - Length of each 1D line (power of two, `<= computeMaxFusedLineLength(...)`).
 * @param {number} params.lineStride - Address increment between lines.
 * @param {number} params.elementStride - Address increment between consecutive elements of a line (always `1` here).
 * @param {number} params.lineCount - Number of lines (one workgroup each).
 * @param {boolean} [params.conjugateInput=false] - Negate the imaginary part on load, folding the inverse transform's leading conjugate pass in (see `FFT2D#computeInverse`).
 * @param {StorageBufferNode} params.twiddleNode - `halfMax`-entry twiddle-factor table (see `buildTwiddleTable`).
 * @param {number} params.halfMax - `max(width,height) / 2`; the twiddle table's entry count.
 * @param {StorageBufferNode} readNode - Buffer to read from.
 * @param {StorageBufferNode} writeNode - Buffer to write to.
 * @returns {Function} A parameterless TSL function ready to `.compute( dispatchCount, [ half ] )`.
 */
function buildFusedLineStage( { N, lineStride, elementStride, lineCount, conjugateInput = false, twiddleNode, halfMax }, readNode, writeNode ) {

	const half = N / 2;
	const stages = Math.log2( N );
	const dispatchCount = lineCount * half;

	const localA = workgroupArray( 'vec2', N );
	const localB = workgroupArray( 'vec2', N );

	return Fn( () => {

		const line = workgroupId.x;
		const t = invocationLocalIndex;

		const lineBase = line.mul( uint( lineStride ) );
		const t2 = t.add( uint( half ) );

		const load = ( addr ) => {

			const v = readNode.element( addr );
			return conjugateInput ? vec2( v.x, v.y.negate() ) : v;

		};

		localA.element( t ).assign( load( lineBase.add( t.mul( uint( elementStride ) ) ) ) );
		localA.element( t2 ).assign( load( lineBase.add( t2.mul( uint( elementStride ) ) ) ) );

		workgroupBarrier();

		let readBuf = localA;
		let writeBuf = localB;

		for ( let s = 0; s < stages; s ++ ) {

			const p = 1 << s;

			const hi = t.div( uint( p ) );
			const lo = t.mod( uint( p ) );

			const idx1 = hi.mul( uint( p ) ).add( lo );
			const idx2 = idx1.add( uint( half ) );

			// Suffixed per stage since this loop is unrolled in JS into one shader body.
			const v0 = readBuf.element( idx1 ).toVar( `v0_${ s }` );
			const v1 = readBuf.element( idx2 ).toVar( `v1_${ s }` );

			// `p` is a compile-time constant here (the JS loop is unrolled), so the twiddle
			// stride is plain JS arithmetic -- see `buildTwiddleTable`.
			const tw = twiddleNode.element( lo.mul( uint( halfMax / p ) ) );
			const c = tw.x;
			const si = tw.y;

			const v1r = v1.x.mul( c ).sub( v1.y.mul( si ) );
			const v1i = v1.x.mul( si ).add( v1.y.mul( c ) );

			const out0 = vec2( v0.x.add( v1r ), v0.y.add( v1i ) );
			const out1 = vec2( v0.x.sub( v1r ), v0.y.sub( v1i ) );

			const j = hi.mul( uint( p * 2 ) ).add( lo );
			const j2 = j.add( uint( p ) );

			writeBuf.element( j ).assign( out0 );
			writeBuf.element( j2 ).assign( out1 );

			workgroupBarrier();

			[ readBuf, writeBuf ] = [ writeBuf, readBuf ];

		}

		writeNode.element( lineBase.add( t.mul( uint( elementStride ) ) ) ).assign( readBuf.element( t ) );
		writeNode.element( lineBase.add( t2.mul( uint( elementStride ) ) ) ).assign( readBuf.element( t2 ) );

	} )().compute( dispatchCount, [ half ] );

}

/**
 * Tiled matrix-transpose compute pass: reads a `rows x cols` row-major buffer and writes its
 * `cols x rows` transpose, using a shared-memory tile so both the read and the write are
 * coalesced. Used so the column pass, like the row pass, transforms contiguous addresses instead
 * of a strided (`elementStride = width`) one.
 *
 * @tsl
 * @private
 * @param {Object} params
 * @param {number} params.rows - Row count of `readNode`, read as row-major with row length `cols`.
 * @param {number} params.cols - Column count (row length) of `readNode`.
 * @param {number} params.tile - Tile edge length; the workgroup is `tile x tile` (see `computeTransposeTileSize`).
 * @param {boolean} [params.conjugateScaleOutput=false] - Negate the imaginary part and scale both components by `invCount` on write, folding the inverse transform's trailing conjugate-and-scale pass in.
 * @param {number} [params.invCount=1] - `1 / (width * height)`, used only when `conjugateScaleOutput` is `true`.
 * @param {StorageBufferNode} readNode - Buffer to read from, as `rows x cols`.
 * @param {StorageBufferNode} writeNode - Buffer to write to, as `cols x rows`.
 * @returns {Function} A parameterless TSL function ready to `.compute( [ numWorkgroupsX, numWorkgroupsY ], [ tile, tile ] )`.
 */
function buildTransposeStage( { rows, cols, tile, conjugateScaleOutput = false, invCount = 1 }, readNode, writeNode ) {

	const sharedTile = workgroupArray( 'vec2', tile * tile );

	const numWorkgroupsX = Math.ceil( cols / tile );
	const numWorkgroupsY = Math.ceil( rows / tile );

	const fn = Fn( () => {

		const gx = globalId.x;
		const gy = globalId.y;
		const lx = localId.x;
		const ly = localId.y;
		const wx = workgroupId.x;
		const wy = workgroupId.y;

		// `rows`/`cols` aren't generally exact multiples of `tile`, so bounds-check by hand.
		If( gx.lessThan( uint( cols ) ).and( gy.lessThan( uint( rows ) ) ), () => {

			sharedTile.element( ly.mul( uint( tile ) ).add( lx ) ).assign( readNode.element( gy.mul( uint( cols ) ).add( gx ) ) );

		} );

		workgroupBarrier();

		// Local x/y swapped (not global) so the write, like the read, stays coalesced.
		const outX = wy.mul( uint( tile ) ).add( lx );
		const outY = wx.mul( uint( tile ) ).add( ly );

		If( outX.lessThan( uint( rows ) ).and( outY.lessThan( uint( cols ) ) ), () => {

			const v = sharedTile.element( lx.mul( uint( tile ) ).add( ly ) ).toVar();
			const out = conjugateScaleOutput ? vec2( v.x.mul( invCount ), v.y.negate().mul( invCount ) ) : v;

			writeNode.element( outY.mul( uint( rows ) ).add( outX ) ).assign( out );

		} );

	} )().compute( [ numWorkgroupsX, numWorkgroupsY ], [ tile, tile ] );

	return fn;

}

/**
 * A GPU 2D complex-to-complex FFT (WebGPU only), implemented as a row/column decomposition of
 * two iterative radix-2 Stockham autosort 1D FFTs. Both passes run as contiguous, coalesced
 * line transforms: the column pass runs on data transposed by `buildTransposeStage` first, then
 * transposed back afterwards, rather than reading/writing with a strided access pattern. Each
 * line is fused into a single shared-memory dispatch when it fits the device's limits, falling
 * back to a per-stage, global-memory dispatch otherwise (see `buildFusedLineStage` /
 * `buildMultiDispatchStage`).
 *
 * `width` and `height` must both be powers of two.
 *
 * `computeForward`/`computeInverse` each take a source and a destination float texture: read
 * `width * height` complex numbers (`.rg` = `(real, imag)`) out of `sourceTexture`, transform
 * them, and write the result into `destinationTexture`'s `.rg` (other channels, e.g. `.ba` on
 * RGBA, are written as `(0, 1)`). Both textures must be exactly `width` by `height`, and
 * `destinationTexture` must be a `StorageTexture`. A real-valued image is transformed by packing
 * it into a texture with a zero `.g` channel first.
 *
 * The inverse transform reuses the forward butterfly kernels via the standard conjugation
 * identity `ifft(x) = conj( fft( conj(x) ) ) / (width*height)`, rather than shipping a second set
 * of shaders with negated twiddle factors.
 *
 * Anything beyond "transform these complex numbers" -- packing a color channel into a complex
 * source texture, rendering a spectrum/reconstruction as a displayable image, reading back a
 * single bin -- is left to the caller; see `examples/webgpu_fft_2d.html` for worked examples,
 * including running one `FFT2D` instance per color channel for a full-color image.
 *
 * ```js
 * const fft = new FFT2D( 64, 64 );
 * // sourceTexture/spectrumTexture/reconstructedTexture: StorageTexture, 64x64, float, >= 2 channels.
 * await fft.computeForward( renderer, sourceTexture, spectrumTexture );
 * await fft.computeInverse( renderer, spectrumTexture, reconstructedTexture );
 * ```
 *
 * `computeForwardReal`/`computeInverseReal` transform a *real*-valued image (`sourceTexture`'s
 * `.r` channel; no imaginary part needed) for about half the cost of `computeForward`/
 * `computeInverse`, via a row-pair-packing trick reusing the same complex-signal-pair conjugate
 * symmetry as above, applied to one image's own even/odd rows rather than two separate images --
 * see `computeForwardReal`'s docstring for the algorithm. Worth using any time the input (for
 * forward) or the known-real output (for inverse) is real-valued, which includes most of what an
 * `FFT2D` is used for -- images, per-channel filtering, convolution.
 *
 * @three_import import { FFT2D } from 'three/addons/gpgpu/FFT2D.js';
 */
class FFT2D {

	/**
	 * Constructs a new 2D FFT.
	 *
	 * @param {number} width - The width of the transform. Must be a power of two.
	 * @param {number} height - The height of the transform. Must be a power of two.
	 */
	constructor( width, height ) {

		if ( ! MathUtils.isPowerOfTwo( width ) || ! MathUtils.isPowerOfTwo( height ) ) {

			throw new Error( `FFT2D: width (${ width }) and height (${ height }) must both be powers of two.` );

		}

		/**
		 * The width of the transform.
		 *
		 * @type {number}
		 */
		this.width = width;

		/**
		 * The height of the transform.
		 *
		 * @type {number}
		 */
		this.height = height;

		/**
		 * The total number of complex elements (`width * height`).
		 *
		 * @type {number}
		 */
		this.count = width * height;

		const count = this.count;

		this._attributeA = new StorageBufferAttribute( count, 2 );
		this._attributeB = new StorageBufferAttribute( count, 2 );

		this._readNode = storage( this._attributeA, 'vec2', count ).toReadOnly();
		this._writeNode = storage( this._attributeB, 'vec2', count );

		// Shared twiddle-factor table for every butterfly stage, row and column alike -- see
		// `buildTwiddleTable`. Built here (not lazily) since it depends only on width/height, not
		// on the renderer/device.
		this._halfMax = Math.max( width, height ) / 2;
		this._twiddleAttribute = new StorageBufferAttribute( buildTwiddleTable( this._halfMax ), 2 );
		this._twiddleNode = storage( this._twiddleAttribute, 'vec2', this._halfMax ).toReadOnly();

		this._pUniform = uniform( 1, 'uint' );

		this._stagesRow = Math.log2( width );
		this._stagesCol = Math.log2( height );

		// Butterfly kernels (and the elementwise conjugate/load/store ones below) are all built
		// lazily in `_ensureButterfliesBuilt` since sizing any of them -- fused-vs-fallback choice,
		// workgroup size -- needs the real device's compute limits, not known until
		// `renderer.init()`, which the constructor doesn't have access to.
		this._built = false;

		this._current = 'A';

	}

	/**
	 * Repoints the shared read/write storage nodes at the current/opposite ping-pong buffer,
	 * dispatches `kernel`, and flips `_current`.
	 *
	 * @private
	 * @param {Renderer} renderer
	 * @param {Function} kernel
	 */
	_dispatchPingPong( renderer, kernel ) {

		this._readNode.value = this._current === 'A' ? this._attributeA : this._attributeB;
		this._writeNode.value = this._current === 'A' ? this._attributeB : this._attributeA;

		renderer.compute( kernel );

		this._current = this._current === 'A' ? 'B' : 'A';

	}

	/**
	 * Runs one axis's pass: a single dispatch if fused, otherwise one dispatch per stage.
	 *
	 * @private
	 * @param {Renderer} renderer
	 * @param {boolean} fused
	 * @param {number} stages
	 * @param {Function} kernel
	 */
	_runAxisPass( renderer, fused, stages, kernel ) {

		if ( fused ) {

			this._dispatchPingPong( renderer, kernel );

			return;

		}

		for ( let s = 0; s < stages; s ++ ) {

			this._pUniform.value = 1 << s;
			this._dispatchPingPong( renderer, kernel );

		}

	}

	/**
	 * Builds the row/transpose/column/transpose-back kernels (and the standalone elementwise
	 * conjugate kernel) on first use, choosing -- per axis -- between `buildFusedLineStage` and
	 * `buildMultiDispatchStage`, and sizing every workgroup, entirely from the real device's
	 * compute limits (`getComputeLimits`/`pickWorkgroupSize`), with no hardcoded fallback. Also
	 * called (as a no-op after the first time) from `_load`/`_store`, since those may run before
	 * `computeForward` reaches the butterfly passes. Deferred out of the constructor since the
	 * constructor doesn't take a renderer.
	 *
	 * @private
	 * @param {Renderer} renderer
	 */
	_ensureButterfliesBuilt( renderer ) {

		if ( this._built ) return;

		this._built = true;

		const { width, height, count } = this;
		const invCount = 1 / count;

		const limits = getComputeLimits( renderer );
		const maxFusedLineLength = computeMaxFusedLineLength( limits );
		const tile = computeTransposeTileSize( limits );
		const workgroupSize = pickWorkgroupSize( limits );

		this._workgroupSize = workgroupSize;

		this._conjugateKernel = Fn( () => {

			const v = this._readNode.element( instanceIndex ).toVar();
			this._writeNode.element( instanceIndex ).assign( vec2( v.x, v.y.negate() ) );

		} )().compute( count, [ workgroupSize ] );

		this._rowFused = width <= maxFusedLineLength;
		this._colFused = height <= maxFusedLineLength;

		const buildRow = this._rowFused ? buildFusedLineStage : buildMultiDispatchStage;
		const buildCol = this._colFused ? buildFusedLineStage : buildMultiDispatchStage;

		this._rowKernel = buildRow( { N: width, lineStride: width, elementStride: 1, lineCount: height, pUniform: this._pUniform, workgroupSize, twiddleNode: this._twiddleNode, halfMax: this._halfMax }, this._readNode, this._writeNode );

		// Only worth building a conjugate-input row variant when the row axis is fused; the
		// fallback path's kernel is reused across stages via `_pUniform`, so a standalone
		// `_conjugateKernel` pass handles that case instead.
		if ( this._rowFused ) {

			this._rowConjKernel = buildFusedLineStage( { N: width, lineStride: width, elementStride: 1, lineCount: height, conjugateInput: true, twiddleNode: this._twiddleNode, halfMax: this._halfMax }, this._readNode, this._writeNode );

		}

		// After this, the buffer is `width` lines of length `height` (row-major, row length `height`).
		this._transposeFwdKernel = buildTransposeStage( { rows: height, cols: width, tile }, this._readNode, this._writeNode );

		this._colKernel = buildCol( { N: height, lineStride: height, elementStride: 1, lineCount: width, pUniform: this._pUniform, workgroupSize, twiddleNode: this._twiddleNode, halfMax: this._halfMax }, this._readNode, this._writeNode );

		// Transpose back to the original `height` lines of length `width` layout. This stage is
		// always a single dispatch, so the trailing conjugate-and-scale always folds into it.
		this._transposeBackKernel = buildTransposeStage( { rows: width, cols: height, tile }, this._readNode, this._writeNode );
		this._transposeBackConjScaleKernel = buildTransposeStage( { rows: width, cols: height, tile, conjugateScaleOutput: true, invCount }, this._readNode, this._writeNode );

	}

	/**
	 * Runs the butterfly stages -- row pass, transpose, column pass, transpose back -- ping-ponging
	 * between the two buffers throughout. Both `computeForward` and `computeInverse` call this as
	 * their shared core; `inverse` selects the kernel variants that fold the inverse transform's
	 * leading/trailing conjugate (and, for the trailing one, scale) into the existing passes.
	 *
	 * @private
	 * @param {Renderer} renderer
	 * @param {boolean} [inverse=false]
	 */
	_runButterflyPasses( renderer, inverse = false ) {

		this._ensureButterfliesBuilt( renderer );

		// The leading conjugate only folds into the row pass when it's fused; otherwise
		// `computeInverse` has already run it as a standalone pass.
		const foldLeadingConjugate = inverse && this._rowFused;
		const rowKernel = foldLeadingConjugate ? this._rowConjKernel : this._rowKernel;

		this._runAxisPass( renderer, this._rowFused, this._stagesRow, rowKernel );

		this._dispatchPingPong( renderer, this._transposeFwdKernel );

		this._runAxisPass( renderer, this._colFused, this._stagesCol, this._colKernel );

		const transposeBackKernel = inverse ? this._transposeBackConjScaleKernel : this._transposeBackKernel;

		this._dispatchPingPong( renderer, transposeBackKernel );

	}

	/**
	 * Reads `sourceTexture`'s `.rg` channels into whichever ping-pong buffer currently holds the
	 * live data, entirely on the GPU. The texture is sampled with an exact texel fetch, so it
	 * must be exactly `width` by `height`.
	 *
	 * @private
	 * @param {Renderer} renderer
	 * @param {Texture} sourceTexture - A float texture, `width` by `height`, with at least 2 channels.
	 */
	_load( renderer, sourceTexture ) {

		this._ensureButterfliesBuilt( renderer );

		if ( this._loadKernel === undefined ) {

			const width = this.width;
			this._loadTextureNode = texture( sourceTexture );

			this._loadKernel = Fn( () => {

				const x = instanceIndex.mod( uint( width ) );
				const y = instanceIndex.div( uint( width ) );

				this._writeNode.element( instanceIndex ).assign( this._loadTextureNode.load( ivec2( int( x ), int( y ) ) ).rg );

			} )().compute( this.count, [ this._workgroupSize ] );

		}

		this._loadTextureNode.value = sourceTexture;
		this._writeNode.value = this._current === 'A' ? this._attributeA : this._attributeB;

		renderer.compute( this._loadKernel );

	}

	/**
	 * Writes whichever ping-pong buffer currently holds the live data into `destinationTexture`'s
	 * `.rg` channels, leaving other channels as `(0, 1)`.
	 *
	 * @private
	 * @param {Renderer} renderer
	 * @param {StorageTexture} destinationTexture - Must be exactly `width` by `height` in size.
	 */
	_store( renderer, destinationTexture ) {

		this._ensureButterfliesBuilt( renderer );

		if ( this._storeKernel === undefined ) {

			const width = this.width;
			this._storeTextureNode = storageTexture( destinationTexture ).setAccess( NodeAccess.WRITE_ONLY );

			this._storeKernel = Fn( () => {

				const x = instanceIndex.mod( uint( width ) );
				const y = instanceIndex.div( uint( width ) );

				const v = this._readNode.element( instanceIndex );

				textureStore( this._storeTextureNode, uvec2( x, y ), vec4( v.x, v.y, 0, 1 ) );

			} )().compute( this.count, [ this._workgroupSize ] );

		}

		this._storeTextureNode.value = destinationTexture;
		this._readNode.value = this._current === 'A' ? this._attributeA : this._attributeB;

		renderer.compute( this._storeKernel );

	}

	/**
	 * Computes the forward 2D FFT: reads `sourceTexture`'s `.rg` channels as `width * height`
	 * complex numbers, transforms them, and writes the result into `destinationTexture`'s `.rg`
	 * channels.
	 *
	 * @param {Renderer} renderer
	 * @param {Texture} sourceTexture - A float texture, `width` by `height`, with at least 2 channels.
	 * @param {StorageTexture} destinationTexture - A float `StorageTexture`, `width` by `height`, with at least 2 channels.
	 */
	computeForward( renderer, sourceTexture, destinationTexture ) {

		this._load( renderer, sourceTexture );
		this._runButterflyPasses( renderer );
		this._store( renderer, destinationTexture );

	}

	/**
	 * Computes the inverse 2D FFT, via `ifft(x) = conj( fft( conj(x) ) ) / count` -- reuses the
	 * forward butterfly kernels rather than shipping a second set of shaders with negated twiddle
	 * factors.
	 *
	 * @param {Renderer} renderer
	 * @param {Texture} sourceTexture - A float texture, `width` by `height`, with at least 2 channels (typically a spectrum produced by `computeForward`).
	 * @param {StorageTexture} destinationTexture - A float `StorageTexture`, `width` by `height`, with at least 2 channels.
	 */
	computeInverse( renderer, sourceTexture, destinationTexture ) {

		this._ensureButterfliesBuilt( renderer );

		this._load( renderer, sourceTexture );

		if ( ! this._rowFused ) {

			this._dispatchPingPong( renderer, this._conjugateKernel );

		}

		this._runButterflyPasses( renderer, true );

		this._store( renderer, destinationTexture );

	}

	/**
	 * Builds `_halfFFT`, a nested `FFT2D` half this instance's height, on first use -- the engine
	 * behind `computeForwardReal`/`computeInverseReal` (see `computeForwardReal`'s docstring for
	 * the algorithm). No-op after the first call.
	 *
	 * @private
	 */
	_ensureHalfFFT() {

		if ( this._halfFFT !== undefined ) return;

		if ( this.height < 2 ) {

			throw new Error( 'FFT2D: computeForwardReal/computeInverseReal require height >= 2.' );

		}

		this._halfFFT = new FFT2D( this.width, this.height / 2 );

	}

	/**
	 * Packs pairs of rows of a real-valued `sourceTexture` (`.r` channel, `width` by `height`) as
	 * one complex `(width x height/2)` signal, `z[p] = row[2p] + i*row[2p+1]`, into `_halfFFT`'s
	 * ping-pong buffer -- the loading half of `computeForwardReal`'s packing trick.
	 *
	 * @private
	 * @param {Renderer} renderer
	 * @param {Texture} sourceTexture
	 */
	_loadRowPairs( renderer, sourceTexture ) {

		this._ensureHalfFFT();

		const half = this._halfFFT;
		half._ensureButterfliesBuilt( renderer );

		if ( this._loadRowPairsKernel === undefined ) {

			const width = this.width;
			this._loadRowPairsTextureNode = texture( sourceTexture );

			this._loadRowPairsKernel = Fn( () => {

				const x = instanceIndex.mod( uint( width ) );
				const p = instanceIndex.div( uint( width ) );

				const rowA = this._loadRowPairsTextureNode.load( ivec2( int( x ), int( p.mul( 2 ) ) ) ).r;
				const rowB = this._loadRowPairsTextureNode.load( ivec2( int( x ), int( p.mul( 2 ).add( 1 ) ) ) ).r;

				half._writeNode.element( instanceIndex ).assign( vec2( rowA, rowB ) );

			} )().compute( half.count, [ half._workgroupSize ] );

		}

		this._loadRowPairsTextureNode.value = sourceTexture;
		half._writeNode.value = half._current === 'A' ? half._attributeA : half._attributeB;

		renderer.compute( this._loadRowPairsKernel );

	}

	/**
	 * Combines `_halfFFT`'s finished `(width x height/2)` spectrum `Z` -- the transform of the
	 * row-pair-packed signal `z = a + i*b`, where `a`/`b` are this image's even/odd rows -- into
	 * this instance's own `(width x height)` buffer as the true full spectrum `X`, using real-signal
	 * conjugate symmetry plus one radix-2 recombine stage:
	 *
	 * `A[k1,k2] = (Z[k1,k2] + conj(Z[-k1,-k2])) / 2`, `B[k1,k2] = -i * (Z[k1,k2] - conj(Z[-k1,-k2])) / 2`
	 * (`-k1`/`-k2` meaning reflection through the origin in `_halfFFT`'s own `(height/2, width)`
	 * domain) recover `A`/`B`, the 2D spectra of the even/odd row sub-images, from `Z`: this is the
	 * standard real-signal-pair unmixing identity -- for any two real-valued signals `a`/`b` packed
	 * as one complex signal `z = a + i*b`, their spectra are conjugate-symmetric, and this formula
	 * recovers each from `Z = FFT(z)`. Here `a`/`b` are one image's even/odd rows rather than two
	 * separate images, but the math doesn't care which.
	 *
	 * Then the standard radix-2 decimation-in-space combine, using twiddle factors of the *full*
	 * height (from the shared `_twiddleNode`, not `_halfFFT`'s own): `X[k1,k2] = A[k1,k2] +
	 * W_height^k1 * B[k1,k2]`, `X[k1+height/2,k2] = A[k1,k2] - W_height^k1 * B[k1,k2]`, for
	 * `k1` in `0..height/2-1`.
	 *
	 * The store half of `computeForwardReal`'s packing trick.
	 *
	 * @private
	 * @param {Renderer} renderer
	 */
	_recombineRowsForward( renderer ) {

		this._ensureButterfliesBuilt( renderer );

		const half = this._halfFFT;

		if ( this._recombineRowsForwardKernel === undefined ) {

			const width = this.width;
			const halfHeight = this.height / 2;
			const maxN = this._halfMax * 2;
			const twiddleStride = maxN / this.height;
			const dispatchCount = halfHeight * width;

			this._recombineRowsForwardKernel = Fn( () => {

				const t = instanceIndex;
				const k2 = t.mod( uint( width ) );
				const k1 = t.div( uint( width ) );

				const mk1 = uint( halfHeight ).sub( k1 ).mod( uint( halfHeight ) );
				const mk2 = uint( width ).sub( k2 ).mod( uint( width ) );
				const mirrorIndex = mk1.mul( uint( width ) ).add( mk2 );

				const z = half._readNode.element( t ).toVar( 'z' ); // Z[k1,k2]
				const zm = half._readNode.element( mirrorIndex ).toVar( 'zm' ); // Z[-k1,-k2], not yet conjugated

				// a = (z + conj(zm)) / 2; b = -i * (z - conj(zm)) / 2 -- see this method's docstring.
				const a = vec2( z.x.add( zm.x ), z.y.sub( zm.y ) ).mul( 0.5 );
				const b = vec2( z.y.add( zm.y ), zm.x.sub( z.x ) ).mul( 0.5 );

				const tw = this._twiddleNode.element( k1.mul( uint( twiddleStride ) ) );
				const twB = vec2( tw.x.mul( b.x ).sub( tw.y.mul( b.y ) ), tw.x.mul( b.y ).add( tw.y.mul( b.x ) ) );

				const x0 = vec2( a.x.add( twB.x ), a.y.add( twB.y ) );
				const x1 = vec2( a.x.sub( twB.x ), a.y.sub( twB.y ) );

				this._writeNode.element( t ).assign( x0 );
				this._writeNode.element( k1.add( uint( halfHeight ) ).mul( uint( width ) ).add( k2 ) ).assign( x1 );

			} )().compute( dispatchCount, [ this._workgroupSize ] );

		}

		half._readNode.value = half._current === 'A' ? half._attributeA : half._attributeB;
		this._writeNode.value = this._current === 'A' ? this._attributeA : this._attributeB;

		renderer.compute( this._recombineRowsForwardKernel );

	}

	/**
	 * Computes the forward 2D FFT of a real-valued image for about half the cost of
	 * `computeForward`, via row-pair packing: pack pairs of rows of `sourceTexture` (`.r`) as one
	 * complex `(width x height/2)` signal, run one *ordinary* full 2D complex FFT on it (a nested
	 * `FFT2D` instance, `_halfFFT`, reusing every existing kernel unchanged), then recombine that
	 * smaller spectrum into the true full `(width x height)` spectrum with one cheap elementwise
	 * pass (`_recombineRowsForward`). Since FFT cost is `O(n log n)`, halving the element count run
	 * through the row/transpose/column/transpose-back passes is close to, if not quite, a full 2x
	 * -- plus the one small recombine pass. Writes the result into `destinationTexture`'s `.rg`
	 * channels, same format `computeForward` produces (fully usable with `computeInverse`, or with
	 * `computeInverseReal` when the caller knows the spectrum will inverse-transform back to a
	 * purely real image, e.g. after multiplying by another real-signal spectrum).
	 *
	 * @param {Renderer} renderer
	 * @param {Texture} sourceTexture - A float texture, `width` by `height`, real-valued in `.r`.
	 * @param {StorageTexture} destinationTexture - A float `StorageTexture`, `width` by `height`, with at least 2 channels.
	 */
	computeForwardReal( renderer, sourceTexture, destinationTexture ) {

		this._ensureHalfFFT();

		this._loadRowPairs( renderer, sourceTexture );
		this._halfFFT._runButterflyPasses( renderer );
		this._recombineRowsForward( renderer );
		this._store( renderer, destinationTexture );

	}

	/**
	 * Reads a full `(width x height)` complex spectrum `X` directly out of `sourceTexture`'s `.rg`
	 * channels (no intermediate buffer needed) and splits+combines it into `_halfFFT`'s ping-pong
	 * buffer as one composite `(width x height/2)` spectrum `Z = A + i*B`, where `A`/`B` are the
	 * spectra of `X`'s (unknown, to be recovered) even/odd row sub-images -- the inverse of
	 * `_recombineRowsForward`'s combine step: solving `X[k1,k2] = A[k1,k2] + W_height^k1 *
	 * B[k1,k2]`, `X[k1+height/2,k2] = A[k1,k2] - W_height^k1 * B[k1,k2]` for `A`/`B` gives
	 * `A[k1,k2] = (X[k1,k2] + X[k1+height/2,k2]) / 2`, `B[k1,k2] = (X[k1,k2] - X[k1+height/2,k2]) *
	 * conj(W_height^k1) / 2`; then `Z = A + i*B` packs them back into one composite spectrum, ready
	 * for a single ordinary inverse FFT (see `computeInverseReal`'s docstring for why that recovers
	 * both `a` and `b` at once). The loading half of `computeInverseReal`'s trick.
	 *
	 * @private
	 * @param {Renderer} renderer
	 * @param {Texture} sourceTexture - `X`, a float texture, `width` by `height`, with at least 2 channels.
	 */
	_loadSplitRowsFromTexture( renderer, sourceTexture ) {

		this._ensureHalfFFT();

		const half = this._halfFFT;
		half._ensureButterfliesBuilt( renderer );

		if ( this._loadSplitRowsKernel === undefined ) {

			const width = this.width;
			const halfHeight = this.height / 2;
			const maxN = this._halfMax * 2;
			const twiddleStride = maxN / this.height;

			this._loadSplitRowsTextureNode = texture( sourceTexture );

			this._loadSplitRowsKernel = Fn( () => {

				const t = instanceIndex;
				const k2 = t.mod( uint( width ) );
				const k1 = t.div( uint( width ) );

				const x0 = this._loadSplitRowsTextureNode.load( ivec2( int( k2 ), int( k1 ) ) ).rg;
				const x1 = this._loadSplitRowsTextureNode.load( ivec2( int( k2 ), int( k1.add( uint( halfHeight ) ) ) ) ).rg;

				const a = x0.add( x1 ).mul( 0.5 );
				const d = x0.sub( x1 );

				const tw = this._twiddleNode.element( k1.mul( uint( twiddleStride ) ) );

				// b = d * conj(tw) / 2
				const bx = d.x.mul( tw.x ).add( d.y.mul( tw.y ) ).mul( 0.5 );
				const by = d.y.mul( tw.x ).sub( d.x.mul( tw.y ) ).mul( 0.5 );

				// z = a + i*b
				const z = vec2( a.x.sub( by ), a.y.add( bx ) );

				half._writeNode.element( t ).assign( z );

			} )().compute( half.count, [ half._workgroupSize ] );

		}

		this._loadSplitRowsTextureNode.value = sourceTexture;
		half._writeNode.value = half._current === 'A' ? half._attributeA : half._attributeB;

		renderer.compute( this._loadSplitRowsKernel );

	}

	/**
	 * Writes `_halfFFT`'s finished buffer -- `z = a + i*b`, both real-valued (see
	 * `computeInverseReal`'s docstring) -- into a single real-valued `destinationTexture`'s `.r`
	 * channel, interleaved back into `width` by `height` rows: `Re(z[p])` into row `2p`, `Im(z[p])`
	 * into row `2p+1`. The store half of `computeInverseReal`'s trick.
	 *
	 * @private
	 * @param {Renderer} renderer
	 * @param {StorageTexture} destinationTexture - Must be exactly `width` by `height` in size.
	 */
	_storeInterleaveRows( renderer, destinationTexture ) {

		const half = this._halfFFT;

		if ( this._storeInterleaveRowsKernel === undefined ) {

			const width = this.width;
			this._storeInterleaveRowsTextureNode = storageTexture( destinationTexture ).setAccess( NodeAccess.WRITE_ONLY );

			this._storeInterleaveRowsKernel = Fn( () => {

				const t = instanceIndex;
				const x = t.mod( uint( width ) );
				const p = t.div( uint( width ) );

				const z = half._readNode.element( t );

				textureStore( this._storeInterleaveRowsTextureNode, uvec2( x, p.mul( 2 ) ), vec4( z.x, 0, 0, 1 ) );
				textureStore( this._storeInterleaveRowsTextureNode, uvec2( x, p.mul( 2 ).add( 1 ) ), vec4( z.y, 0, 0, 1 ) );

			} )().compute( half.count, [ half._workgroupSize ] );

		}

		this._storeInterleaveRowsTextureNode.value = destinationTexture;
		half._readNode.value = half._current === 'A' ? half._attributeA : half._attributeB;

		renderer.compute( this._storeInterleaveRowsKernel );

	}

	/**
	 * Computes the inverse 2D FFT of a spectrum known to correspond to a real-valued image, for
	 * about half the cost of `computeInverse` -- the counterpart to `computeForwardReal`. Splits
	 * `sourceTexture`'s full spectrum `X` into the composite spectrum `Z = A + i*B` of its even/odd
	 * row sub-images (`_loadSplitRowsFromTexture`), runs one *ordinary* full inverse 2D complex FFT
	 * on it via `_halfFFT` (by linearity, `ifft(A + i*B) = ifft(A) + i*ifft(B) = a + i*b`; both `a`
	 * and `b` come out purely real since `A`/`B` are each, individually, the spectrum of a
	 * real-valued row sub-image), then interleaves `_halfFFT`'s real result back into
	 * `destinationTexture`'s rows
	 * (`_storeInterleaveRows`). Not applicable to a spectrum that doesn't correspond to a real
	 * image -- use `computeInverse` for that.
	 *
	 * @param {Renderer} renderer
	 * @param {Texture} sourceTexture - `X`, a float texture, `width` by `height`, with at least 2 channels (typically a spectrum produced by `computeForwardReal`, or one derived from it, e.g. by spectral multiplication with another real-signal spectrum).
	 * @param {StorageTexture} destinationTexture - A float `StorageTexture`, `width` by `height`, real-valued output written to `.r`.
	 */
	computeInverseReal( renderer, sourceTexture, destinationTexture ) {

		this._ensureHalfFFT();

		const half = this._halfFFT;

		this._loadSplitRowsFromTexture( renderer, sourceTexture );

		if ( ! half._rowFused ) {

			half._dispatchPingPong( renderer, half._conjugateKernel );

		}

		half._runButterflyPasses( renderer, true );

		this._storeInterleaveRows( renderer, destinationTexture );

	}

	/**
	 * Frees the GPU buffers backing this transform.
	 */
	dispose() {

		this._attributeA.dispose?.();
		this._attributeB.dispose?.();
		this._twiddleAttribute.dispose?.();
		this._halfFFT?.dispose();

	}

}

export { FFT2D };
