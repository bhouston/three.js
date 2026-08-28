import { StorageBufferAttribute, DynamicDrawUsage, MathUtils } from 'three/webgpu';
import { Fn, atomicAdd, atomicStore, instanceIndex, storage, uint } from 'three/tsl';
import { PrefixSum } from './PrefixSum.js';
import { pickWorkgroupSize } from './GPGPUUtils.js';

// binCount is chosen (when not given explicitly) to target this many elements per bin on
// average. Every element sharing a bin contends on the same atomic counter during the scatter
// pass, so too few bins serializes scatter at large `count`; too many wastes the reset/histogram/
// prefix-sum passes, which are O(binCount) regardless of `count`.
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
 * `count` can change over the life of a `CountingSort` - e.g. as splats are added to or removed
 * from a {@link GaussianSplatGroup} - via {@link CountingSort#count}. Growth/shrink behavior of
 * the underlying order/bin buffers mirrors {@link GaussianSplatGroup}'s own `autoCompact`/
 * `initialSize` contract: see the `autoCompact` and `initialSize` constructor options.
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
	 * @param {Object} [options={}] - Options that modify the counting sort.
	 * @param {number} [options.binCount] - The number of bins/buckets the sort key is quantized into. Defaults to a value derived from `count`/`initialSize` (see `TARGET_ELEMENTS_PER_BIN`); pass explicitly to override, e.g. when `count` will grow well beyond its initial value.
	 * @param {boolean} [options.autoCompact] - Whether the order/bin buffers are kept sized to exactly fit the current {@link CountingSort#count}. When `true`, every {@link CountingSort#count} assignment that changes the value resizes the buffers, growing or shrinking them to fit. When `false`, the buffers only grow - shrinking `count` never reallocates smaller buffers on its own; call {@link CountingSort#compact} to shrink them to fit. Defaults to `false` when `initialSize` is given, `true` otherwise - matching {@link GaussianSplatGroup}'s own default.
	 * @param {number} [options.initialSize] - Preallocates the order/bin buffers to this many elements up front, so the sort doesn't reallocate as `count` grows until it exceeds this value.
	 */
	constructor( count, binNode, { binCount, autoCompact, initialSize } = {} ) {

		this._binNode = binNode;

		/**
		 * Whether the order/bin buffers are kept sized to exactly fit the current
		 * {@link CountingSort#count} (see the constructor's `autoCompact` option for the full
		 * contract). Safe to change at any time; takes effect the next time
		 * {@link CountingSort#count} is assigned, or on an explicit {@link CountingSort#compact} call.
		 *
		 * @type {boolean}
		 */
		this.autoCompact = autoCompact !== undefined ? autoCompact : ( initialSize === undefined );

		/**
		 * The number of bins/buckets the sort key is quantized into. Fixed for the life of this
		 * `CountingSort` - scales with `count`/`initialSize` at construction to bound scatter-pass
		 * atomic contention (see `TARGET_ELEMENTS_PER_BIN`), unless `binCount` is given explicitly.
		 *
		 * @type {number}
		 */
		this.binCount = binCount !== undefined ? binCount : MathUtils.clamp(
			MathUtils.ceilPowerOfTwo( Math.ceil( Math.max( 1, count, initialSize || 0 ) / TARGET_ELEMENTS_PER_BIN ) ),
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

		this._capacity = 0;

		const histogramAttribute = new StorageBufferAttribute( new Uint32Array( this.binCount ), 1, Uint32Array );
		const offsetAttribute = new StorageBufferAttribute( new Uint32Array( this.binCount ), 1, Uint32Array );

		this._histogramAttribute = histogramAttribute;
		this._offsetAttribute = offsetAttribute;

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
		 * view of them. Independent of `count` - only depends on `binCount`, which never changes - so
		 * it never needs rebuilding when {@link CountingSort#count} changes.
		 *
		 * @private
		 * @type {PrefixSum}
		 */
		this._prefixSum = new PrefixSum( histogramAttribute, {
			outputAttribute: offsetAttribute,
			isInclusive: false
		} );

		// order/bin buffers are sized by `count` - built once against 1-element
		// placeholders and resized in place by `_resizeOrderBuffers` as `count` changes.
		this.orderAttribute = new StorageBufferAttribute( new Uint32Array( 1 ), 1, Uint32Array );
		this._binAttribute = new StorageBufferAttribute( new Uint32Array( 1 ), 1, Uint32Array );

		/**
		 * A read-only storage node for the sorted order buffer.
		 *
		 * @type {StorageBufferNode}
		 */
		this.orderRead = storage( this.orderAttribute, 'uint', 0 ).toReadOnly();

		/**
		 * A writable storage node for the sorted order buffer.
		 *
		 * @type {StorageBufferNode}
		 */
		this.orderWrite = storage( this.orderAttribute, 'uint', 0 );

		/**
		 * A read-only storage node holding each element's bin, computed during the histogram pass.
		 *
		 * @type {StorageBufferNode}
		 */
		this.binRead = storage( this._binAttribute, 'uint', 0 ).toReadOnly();

		/**
		 * A writable storage node holding each element's bin.
		 *
		 * @type {StorageBufferNode}
		 */
		this.binWrite = storage( this._binAttribute, 'uint', 0 );

		this._webGLBuffersEnabled = false;

		this._cpuBins = new Uint32Array( 0 );
		this._cpuCounts = new Uint32Array( this.binCount );
		this._cpuOffsets = new Uint32Array( this.binCount );

		this._resetNode = null;
		this._histogramNode = null;
		this._scatterNode = null;

		this._built = false;

		this._count = 0;

		if ( initialSize !== undefined ) this._resizeOrderBuffers( Math.max( 1, initialSize ) );

		this.count = count;

	}

	/**
	 * The number of elements to sort. Assigning a value different from the current one updates the
	 * dispatch bounds of the compute kernels built by {@link CountingSort#_ensureBuilt}, if any, and -
	 * depending on {@link CountingSort#autoCompact} - may resize the order/bin buffers in place (see
	 * `_resizeOrderBuffers`); unlike constructing a new {@link CountingSort}, this never rebuilds the
	 * compute kernels or the histogram/offset/prefix-sum machinery, which only depend on `binCount`.
	 *
	 * @type {number}
	 */
	get count() {

		return this._count;

	}

	set count( value ) {

		const requiredCapacity = Math.max( 1, value );

		// autoCompact controls whether capacity shrinks automatically or only via compact() -
		// mirrors GaussianSplatGroup's own buffer-resizing contract.
		if ( this.autoCompact === true ? requiredCapacity !== this._capacity : requiredCapacity > this._capacity ) {

			this._resizeOrderBuffers( requiredCapacity );

		}

		this._count = value;

		if ( this._histogramNode !== null ) {

			this._histogramNode.count = value;
			this._scatterNode.count = value;

		}

	}

	/**
	 * The number of elements the order/bin buffers currently have room for. Always `>=`
	 * {@link CountingSort#count}; the two are equal exactly when the buffers are compact - always
	 * true while {@link CountingSort#autoCompact} is `true`, and after an explicit
	 * {@link CountingSort#compact} call otherwise.
	 *
	 * @type {number}
	 * @readonly
	 */
	get capacity() {

		return this._capacity;

	}

	/**
	 * Shrinks the order/bin buffers to exactly fit {@link CountingSort#count}, freeing any slack
	 * accumulated while {@link CountingSort#autoCompact} was `false` (grow-only mode) or while the
	 * sort was preallocated via the constructor's `initialSize` option. A no-op if the buffers
	 * already fit exactly. Not needed when `autoCompact` is `true`, since buffers are already kept
	 * sized to fit.
	 */
	compact() {

		const target = Math.max( 1, this._count );

		if ( target === this._capacity ) return;

		this._resizeOrderBuffers( target );

	}

	// Reallocates the order/bin buffers to exactly fit `capacity` elements and repoints
	// `orderRead`/`orderWrite`/`binRead`/`binWrite` at them, without touching the
	// histogram/offset buffers (sized by `binCount`, which never changes) or any
	// compute kernel.
	_resizeOrderBuffers( capacity ) {

		const oldOrderAttribute = this.orderAttribute;
		const oldBinAttribute = this._binAttribute;

		const orderData = new Uint32Array( capacity );
		for ( let i = 0; i < capacity; i ++ ) orderData[ i ] = i;

		this.orderAttribute = new StorageBufferAttribute( orderData, 1, Uint32Array );
		this._binAttribute = new StorageBufferAttribute( new Uint32Array( capacity ), 1, Uint32Array );

		if ( this._webGLBuffersEnabled === true ) this.orderAttribute.setUsage( DynamicDrawUsage );

		this.orderRead.value = this.orderAttribute;
		this.orderWrite.value = this.orderAttribute;
		this.binRead.value = this._binAttribute;
		this.binWrite.value = this._binAttribute;

		this._cpuBins = new Uint32Array( capacity );
		this._capacity = capacity;

		oldOrderAttribute.dispose();
		oldBinAttribute.dispose();

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

	/**
	 * Frees the GPU buffers backing this counting sort. To change the number of elements
	 * to sort, assign {@link CountingSort#count}; there is no need to construct a new instance.
	 */
	dispose() {

		this.orderAttribute.dispose();
		this._binAttribute.dispose();
		this._histogramAttribute.dispose();
		this._offsetAttribute.dispose();

	}

}

export { CountingSort };
