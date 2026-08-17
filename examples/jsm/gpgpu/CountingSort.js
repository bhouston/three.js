import { StorageBufferAttribute, DynamicDrawUsage } from 'three/webgpu';
import {
	Fn, If, Loop,
	atomicAdd, atomicLoad, atomicStore,
	instanceIndex, invocationLocalIndex, storage, subgroupIndex, uint, workgroupArray, workgroupBarrier,
	subgroupAdd, subgroupElect, subgroupExclusiveAdd
} from 'three/tsl';

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
 * This class does not compute the sort key itself. Instead, a TSL function is supplied via
 * {@link CountingSort#setBinNode} that maps the current `instanceIndex` to a bin, and an equivalent
 * plain JavaScript function can be supplied to {@link CountingSort#computeCPU} for platforms without
 * compute shader support (e.g. the WebGL backend of {@link WebGPURenderer}).
 *
 * ```js
 * const sort = new CountingSort( count, { binCount: 4096 } );
 * sort.setBinNode( () => {
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
 * The prefix sum pass - turning the per-bin histogram into per-bin write offsets - is the one part
 * of this algorithm that cannot be parallelized across `count`; a naive implementation runs it as a
 * single-invocation serial loop over `binCount` bins. When the current device reports the WebGPU
 * `subgroups` feature (see {@link https://web3dsurvey.com/webgpu/features/subgroups} for current
 * support), this class instead computes it with `subgroupAdd`/`subgroupExclusiveAdd`, which is much
 * faster in practice since `workgroupSize` bins are scanned at once instead of one. This is decided
 * automatically per renderer on the first {@link CountingSort#compute} call; when the feature isn't
 * available, the original serial prefix sum is used instead, so no fallback handling is required by
 * users of this class.
 *
 * @three_import import { CountingSort } from 'three/addons/gpgpu/CountingSort.js';
 */
class CountingSort {

	/**
	 * Constructs a new counting sort.
	 *
	 * @param {number} count - The number of elements to sort.
	 * @param {Object} [options={}] - Options that modify the counting sort.
	 * @param {number} [options.binCount=4096] - The number of bins/buckets the sort key is quantized into. Larger values improve sort accuracy at the cost of a longer (but still single-pass) prefix sum.
	 * @param {number} [options.workgroupSize=256] - The workgroup size of the compute shaders executed during the sort.
	 */
	constructor( count, { binCount = 4096, workgroupSize = 256 } = {} ) {

		/**
		 * The number of elements to sort.
		 *
		 * @type {number}
		 */
		this.count = count;

		/**
		 * The number of bins/buckets the sort key is quantized into.
		 *
		 * @type {number}
		 */
		this.binCount = binCount;

		/**
		 * The workgroup size of the compute shaders executed during the sort.
		 *
		 * @type {number}
		 */
		this.workgroupSize = workgroupSize;

		/**
		 * `binCount` rounded up to a multiple of `workgroupSize`. The histogram/offset buffers are
		 * allocated at this size (instead of `binCount`) so that the subgroup-based prefix sum's
		 * `workgroupBarrier()`-using kernel - which, per WGSL uniformity rules, cannot use the
		 * automatic `instanceIndex >= count` bounds check the other passes rely on - can safely let
		 * every invocation of its final tile read/write real, zero-initialized elements instead of
		 * indices past the end of the buffer. See {@link CountingSort#_buildPrefixSubgroupNode}.
		 *
		 * @private
		 * @type {number}
		 */
		this._paddedBinCount = Math.ceil( binCount / workgroupSize ) * workgroupSize;

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
		const histogramAttribute = new StorageBufferAttribute( new Uint32Array( this._paddedBinCount ), 1, Uint32Array );
		const offsetAttribute = new StorageBufferAttribute( new Uint32Array( this._paddedBinCount ), 1, Uint32Array );

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
		this.histogramAtomic = storage( histogramAttribute, 'uint', this._paddedBinCount ).toAtomic();

		/**
		 * An atomic storage node used both for the exclusive prefix sum of the histogram and, during
		 * the scatter pass, as a per-bin write cursor.
		 *
		 * @type {StorageBufferNode}
		 */
		this.offsetAtomic = storage( offsetAttribute, 'uint', this._paddedBinCount ).toAtomic();

		this._webGLBuffersEnabled = false;

		this._cpuBins = new Uint32Array( count );
		this._cpuCounts = new Uint32Array( binCount );
		this._cpuOffsets = new Uint32Array( binCount );

		this._resetNode = null;
		this._histogramNode = null;
		this._prefixNode = null;
		this._scatterNode = null;

		// Lazily built, and only if the renderer reports the WebGPU `subgroups` feature - see
		// `_buildPrefixSubgroupNode()` and `compute()`.
		this._prefixSubgroupNode = null;
		this._subgroupsSupported = null;

	}

	/**
	 * Sets the TSL function used to compute the bin of the element currently referenced by
	 * `instanceIndex`, and (re)builds the compute nodes used by {@link CountingSort#compute}.
	 *
	 * @param {Function} binNode - A parameterless function returning a `Node<uint>` in `[0, binCount)`.
	 */
	setBinNode( binNode ) {

		const { binCount, workgroupSize, count, _paddedBinCount } = this;

		// Dispatched over `_paddedBinCount`, not `binCount`, so the padding bins beyond `binCount`
		// are zeroed too; harmless since `binNode()` never produces a bin outside `[0, binCount)`,
		// but required so that the subgroup-based prefix sum's uniform-control-flow kernel (see
		// `_buildPrefixSubgroupNode()`) reads real, zeroed histogram entries there instead of
		// out-of-bounds memory.
		this._resetNode = Fn( () => {

			atomicStore( this.histogramAtomic.element( instanceIndex ), uint( 0 ) );
			atomicStore( this.offsetAtomic.element( instanceIndex ), uint( 0 ) );

		} )().compute( _paddedBinCount, [ workgroupSize ] ).setName( 'CountingSortReset' );

		this._histogramNode = Fn( () => {

			const bin = binNode().toVar( 'bin' );

			this.binWrite.element( instanceIndex ).assign( bin );
			atomicAdd( this.histogramAtomic.element( bin ), uint( 1 ) );

		} )().compute( count, [ workgroupSize ] ).setName( 'CountingSortHistogram' );

		this._prefixNode = Fn( () => {

			const sum = uint( 0 ).toVar( 'sum' );

			Loop( { start: 0, end: binCount, type: 'uint', name: 'bin', condition: '<' }, ( { bin } ) => {

				const binCountValue = atomicLoad( this.histogramAtomic.element( bin ) ).toVar( 'count' );
				atomicStore( this.offsetAtomic.element( bin ), sum );
				sum.addAssign( binCountValue );

			} );

		} )().compute( 1 ).setName( 'CountingSortPrefix' );

		this._scatterNode = Fn( () => {

			const bin = this.binRead.element( instanceIndex ).toVar( 'bin' );
			const targetIndex = atomicAdd( this.offsetAtomic.element( bin ), uint( 1 ) ).toVar( 'targetIndex' );

			this.orderWrite.element( targetIndex ).assign( instanceIndex );

		} )().compute( count, [ workgroupSize ] ).setName( 'CountingSortScatter' );

	}

	/**
	 * Builds a parallel replacement for {@link CountingSort#_prefixNode} using WebGPU subgroup
	 * intrinsics. Only called once, lazily, the first time {@link CountingSort#compute} is run on a
	 * renderer that reports the `subgroups` feature.
	 *
	 * The naive, serial prefix sum ({@link CountingSort#_prefixNode}) is `O(binCount)` on a single
	 * invocation. This instead runs a single workgroup of `workgroupSize` threads that loops over
	 * `binCount` in `workgroupSize`-wide tiles: each tile is scanned in parallel via
	 * `subgroupExclusiveAdd` (plus a short, subgroup-count-sized loop to combine the workgroup's
	 * subgroups), and a small `carry` value threads the running total from one tile to the next.
	 * `binCount` bins are still processed one tile at a time, but each tile takes one parallel step
	 * instead of `workgroupSize` serial ones.
	 *
	 * @private
	 */
	_buildPrefixSubgroupNode() {

		const { workgroupSize, _paddedBinCount } = this;

		const tileCount = uint( _paddedBinCount / workgroupSize );

		// WGSL requires workgroup array sizes to be compile-time constants, but the actual subgroup
		// size is only known at runtime and varies by device (commonly 32 or 64, but anywhere from 4
		// to 128 per the WebGPU spec). Size for the worst case (smallest legal subgroup size) so the
		// array is always big enough, however many subgroups a workgroup ends up split into.
		const subgroupTotals = workgroupArray( 'uint', Math.max( 1, Math.ceil( workgroupSize / 4 ) ) );
		const carry = workgroupArray( 'uint', 1 );

		this._prefixSubgroupNode = Fn( () => {

			If( invocationLocalIndex.equal( uint( 0 ) ), () => carry.element( 0 ).assign( uint( 0 ) ) );
			workgroupBarrier();

			Loop( { start: uint( 0 ), end: tileCount, type: 'uint', name: 'tile', condition: '<' }, ( { tile } ) => {

				const bin = tile.mul( workgroupSize ).add( invocationLocalIndex ).toVar( 'bin' );
				const binCountValue = atomicLoad( this.histogramAtomic.element( bin ) ).toVar( 'binCountValue' );
				const localExclusive = subgroupExclusiveAdd( binCountValue ).toVar( 'localExclusive' );
				const subgroupTotal = subgroupAdd( binCountValue ).toVar( 'subgroupTotal' );

				If( subgroupElect(), () => subgroupTotals.element( subgroupIndex ).assign( subgroupTotal ) );

				// Reconverge before subgroupTotals is read back below.
				workgroupBarrier();

				const tileOffset = uint( 0 ).toVar( 'tileOffset' );

				Loop( { start: uint( 0 ), end: subgroupIndex, type: 'uint', name: 'i', condition: '<' }, ( { i } ) => {

					tileOffset.addAssign( subgroupTotals.element( i ) );

				} );

				atomicStore( this.offsetAtomic.element( bin ), carry.element( 0 ).add( tileOffset ).add( localExclusive ) );

				// Reconverge before carry is updated: every invocation must have read it above first.
				workgroupBarrier();

				If( invocationLocalIndex.equal( uint( workgroupSize - 1 ) ), () => {

					carry.element( 0 ).addAssign( tileOffset.add( subgroupTotal ) );

				} );

				// Reconverge before the next tile (or the kernel exit) reads carry again.
				workgroupBarrier();

			} );

		} )().compute( workgroupSize, [ workgroupSize ] ).setName( 'CountingSortPrefixSubgroup' );

	}

	/**
	 * Executes a complete counting sort on the GPU, updating {@link CountingSort#orderRead}.
	 *
	 * @param {Renderer} renderer - The current scene's renderer.
	 */
	compute( renderer ) {

		renderer.compute( this._resetNode );
		renderer.compute( this._histogramNode );

		if ( this._subgroupsSupported === null ) {

			this._subgroupsSupported = renderer.hasFeature( 'subgroups' );

			if ( this._subgroupsSupported ) this._buildPrefixSubgroupNode();

		}

		renderer.compute( this._subgroupsSupported ? this._prefixSubgroupNode : this._prefixNode );

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
