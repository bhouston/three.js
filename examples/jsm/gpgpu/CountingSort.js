import { StorageBufferAttribute, DynamicDrawUsage, MathUtils } from 'three/webgpu';
import { Fn, atomicAdd, atomicStore, instanceIndex, storage, uint } from 'three/tsl';
import { PrefixSum } from './PrefixSum.js';
import { pickWorkgroupSize } from './GPGPUUtils.js';

// binCount is chosen to target this many elements per bin on average. Every element sharing a bin
// contends on the same atomic counter during the scatter pass, so too few bins serializes scatter
// at large `count`; too many wastes the reset/histogram/prefix-sum passes, which are O(binCount)
// regardless of `count`.
const TARGET_ELEMENTS_PER_BIN = 64;
const MIN_BIN_COUNT = 256; // floor for small `count`, where contention isn't the concern
const MAX_BIN_COUNT = 1 << 20; // safety ceiling; TARGET_ELEMENTS_PER_BIN keeps realistic counts well under this

/**
 * A reusable GPU counting sort.
 *
 * This computes a stable-ish permutation of the integers `[0, count)` that orders them by an
 * arbitrary, user supplied `uint` key ("bin") in the range `[0, binCount)`. It is a good fit for
 * approximate ordering of large element counts (hundreds of thousands to millions) where an exact
 * comparison sort such as a bitonic sort (see {@link BitonicSort}) would be too slow: a counting
 * sort only requires a fixed number of passes (reset, histogram, prefix sum, scatter) regardless of
 * `count`, at the cost of only being accurate to the resolution of `binCount` - elements that land
 * in the same bin end up in an unspecified relative order.
 *
 * This class does not compute the sort key itself. Instead, a TSL function is supplied as the
 * `binNode` constructor argument that maps the current `instanceIndex` to a bin, and an equivalent
 * plain JavaScript function can be supplied to {@link CountingSort#computeCPU} for platforms without
 * compute shader support (e.g. the WebGL backend of {@link WebGPURenderer}).
 *
 * `binCount` and `workgroupSize` are always derived automatically -- from `count` and the
 * renderer's actual device limits, respectively -- so that a `CountingSort` is optimally tuned
 * out of the box; neither is user-configurable.
 *
 * ```js
 * const sort = new CountingSort( count, () => {
 *
 * 	// return a `Node<uint>` bin index for `instanceIndex`, e.g. derived from a depth value.
 *
 * } );
 *
 * sort.compute( renderer );
 *
 * // `sort.orderRead` now holds a storage buffer of `count` indices, ordered by bin.
 * ```
 *
 * @three_import import { CountingSort } from 'three/addons/gpgpu/CountingSort.js';
 */
class CountingSort {

	/**
	 * Constructs a new counting sort.
	 *
	 * @param {number} count - The number of elements to sort.
	 * @param {Function} binNode - A parameterless function returning a `Node<uint>` bin index for `instanceIndex`, in `[0, binCount)`.
	 */
	constructor( count, binNode ) {

		/**
		 * The number of elements to sort.
		 *
		 * @type {number}
		 */
		this.count = count;

		this._binNode = binNode;

		/**
		 * The number of bins/buckets the sort key is quantized into. Scales with `count` to bound
		 * scatter-pass atomic contention (see `TARGET_ELEMENTS_PER_BIN`).
		 *
		 * @type {number}
		 */
		this.binCount = MathUtils.clamp(
			MathUtils.ceilPowerOfTwo( Math.ceil( count / TARGET_ELEMENTS_PER_BIN ) ),
			MIN_BIN_COUNT, MAX_BIN_COUNT
		);

		/**
		 * The workgroup size of the compute shaders executed during the sort. Chosen automatically
		 * from the device's compute limits (see {@link GPGPUUtils#pickWorkgroupSize}). Only available
		 * once {@link CountingSort#compute} has been called for the first time -- see
		 * {@link CountingSort#_ensureBuilt}.
		 *
		 * @type {number|undefined}
		 */
		this.workgroupSize = undefined;

		const orderData = new Uint32Array( count );
		for ( let i = 0; i < count; i ++ ) orderData[ i ] = i;

		/**
		 * The buffer attribute holding the sorted order (a permutation of `[0, count)`). This is
		 * also the attribute that is kept up to date by {@link CountingSort#computeCPU}.
		 *
		 * @type {StorageBufferAttribute}
		 */
		this.orderAttribute = new StorageBufferAttribute( orderData, 1, Uint32Array );

		const binAttribute = new StorageBufferAttribute( new Uint32Array( count ), 1, Uint32Array );
		const histogramAttribute = new StorageBufferAttribute( new Uint32Array( this.binCount ), 1, Uint32Array );
		const offsetAttribute = new StorageBufferAttribute( new Uint32Array( this.binCount ), 1, Uint32Array );

		/**
		 * A read-only storage node for the sorted order buffer.
		 *
		 * @type {StorageBufferNode}
		 */
		this.orderRead = storage( this.orderAttribute, 'uint', count ).toReadOnly();

		/**
		 * A writable storage node for the sorted order buffer.
		 *
		 * @type {StorageBufferNode}
		 */
		this.orderWrite = storage( this.orderAttribute, 'uint', count );

		/**
		 * A read-only storage node holding each element's bin, computed during the histogram pass.
		 *
		 * @type {StorageBufferNode}
		 */
		this.binRead = storage( binAttribute, 'uint', count ).toReadOnly();

		/**
		 * A writable storage node holding each element's bin.
		 *
		 * @type {StorageBufferNode}
		 */
		this.binWrite = storage( binAttribute, 'uint', count );

		/**
		 * An atomic storage node used to accumulate the per-bin histogram.
		 *
		 * @type {StorageBufferNode}
		 */
		this.histogramAtomic = storage( histogramAttribute, 'uint', this.binCount ).toAtomic();

		/**
		 * An atomic storage node used both for the exclusive prefix sum of the histogram and, during
		 * the scatter pass, as a per-bin write cursor.
		 *
		 * @type {StorageBufferNode}
		 */
		this.offsetAtomic = storage( offsetAttribute, 'uint', this.binCount ).toAtomic();

		/**
		 * The prefix sum turning the histogram into the per-bin write offsets. It is handed the raw
		 * attributes rather than the atomic storage nodes above so that it can build its own vectorized
		 * view of them.
		 *
		 * @private
		 * @type {PrefixSum}
		 */
		this._prefixSum = new PrefixSum( histogramAttribute, {
			outputAttribute: offsetAttribute,
			isInclusive: false
		} );

		this._webGLBuffersEnabled = false;

		this._cpuBins = new Uint32Array( count );
		this._cpuCounts = new Uint32Array( this.binCount );
		this._cpuOffsets = new Uint32Array( this.binCount );

		this._resetNode = null;
		this._histogramNode = null;
		this._scatterNode = null;

		this._built = false;

	}

	/**
	 * Resolves {@link CountingSort#workgroupSize} against the renderer's actual compute limits, and
	 * builds the reset/histogram/scatter compute shaders that depend on it and on `binNode`.
	 * Deferred out of the constructor since the constructor doesn't take a renderer; guarded so it
	 * only runs once, the first time {@link CountingSort#compute} is called.
	 *
	 * @private
	 * @param {Renderer} renderer
	 */
	_ensureBuilt( renderer ) {

		if ( this._built ) return;

		this._built = true;

		this.workgroupSize = pickWorkgroupSize( renderer );

		const { binCount, workgroupSize, count } = this;

		this._resetNode = Fn( () => {

			atomicStore( this.histogramAtomic.element( instanceIndex ), uint( 0 ) );
			atomicStore( this.offsetAtomic.element( instanceIndex ), uint( 0 ) );

		} )().compute( binCount, [ workgroupSize ] ).setName( 'CountingSortReset' );

		this._histogramNode = Fn( () => {

			const bin = this._binNode().toVar( 'bin' );

			this.binWrite.element( instanceIndex ).assign( bin );
			atomicAdd( this.histogramAtomic.element( bin ), uint( 1 ) );

		} )().compute( count, [ workgroupSize ] ).setName( 'CountingSortHistogram' );

		this._scatterNode = Fn( () => {

			const bin = this.binRead.element( instanceIndex ).toVar( 'bin' );
			const targetIndex = atomicAdd( this.offsetAtomic.element( bin ), uint( 1 ) ).toVar( 'targetIndex' );

			this.orderWrite.element( targetIndex ).assign( instanceIndex );

		} )().compute( count, [ workgroupSize ] ).setName( 'CountingSortScatter' );

	}

	/**
	 * Executes a complete counting sort on the GPU, updating {@link CountingSort#orderRead}. Builds
	 * the compute shaders (tuned to the renderer's actual device limits) on the first call.
	 *
	 * @param {Renderer} renderer - The current scene's renderer.
	 */
	compute( renderer ) {

		this._ensureBuilt( renderer );

		renderer.compute( this._resetNode );
		renderer.compute( this._histogramNode );
		this._prefixSum.compute( renderer );
		renderer.compute( this._scatterNode );

	}

	/**
	 * Executes a complete counting sort on the CPU, updating {@link CountingSort#orderAttribute}.
	 * Intended as a fallback for backends without compute shader support.
	 *
	 * @param {Function} binFn - A function taking an element index and returning its bin (a plain number in `[0, binCount)`).
	 */
	computeCPU( binFn ) {

		const { count, binCount } = this;
		const order = this.orderAttribute.array;
		const bins = this._cpuBins;
		const counts = this._cpuCounts;
		const offsets = this._cpuOffsets;

		counts.fill( 0 );

		for ( let i = 0; i < count; i ++ ) {

			const bin = binFn( i );

			bins[ i ] = bin;
			counts[ bin ] ++;

		}

		let sum = 0;

		for ( let i = 0; i < binCount; i ++ ) {

			offsets[ i ] = sum;
			sum += counts[ i ];

		}

		for ( let i = 0; i < count; i ++ ) {

			order[ offsets[ bins[ i ] ] ++ ] = i;

		}

		this.orderAttribute.needsUpdate = true;

		if ( this.orderAttribute.pbo !== undefined ) {

			this.orderAttribute.pbo.needsUpdate = true;

		}

	}

	/**
	 * Enables the WebGL-specific storage buffer path (PBO + dynamic draw usage) for the order buffer.
	 * Only needed when {@link CountingSort#computeCPU} is used with the WebGL backend of {@link WebGPURenderer}.
	 */
	enableWebGLBuffers() {

		if ( this._webGLBuffersEnabled === true ) return;

		this.orderAttribute.setUsage( DynamicDrawUsage );
		this.orderRead.setPBO( true );

		this._webGLBuffersEnabled = true;

	}

}

export { CountingSort };
