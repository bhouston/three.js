import { BufferAttribute } from '../core/BufferAttribute.js';
import { DynamicDrawUsage } from '../constants.js';
import { GaussianSplatMaterial } from '../materials/GaussianSplatMaterial.js';
import { InstancedBufferGeometry } from '../core/InstancedBufferGeometry.js';
import { Matrix4 } from '../math/Matrix4.js';
import { Mesh } from './Mesh.js';
import StorageBufferAttribute from '../renderers/common/StorageBufferAttribute.js';
import { Vector2 } from '../math/Vector2.js';
import { Vector3 } from '../math/Vector3.js';

import {
	Fn,
	Loop,
	atomicAdd,
	atomicLoad,
	atomicStore,
	instanceIndex,
	max,
	storage,
	uint,
	uniform,
	vec4
} from '../nodes/TSL.js';

const BIN_COUNT = 4096;
const WORKGROUP_SIZE = 256;
const SORT_DIRECTION_THRESHOLD = 0.9995;
const SORT_POSITION_THRESHOLD = 0.0025;

const _worldCenter = /*@__PURE__*/ new Vector3();
const _viewCenter = /*@__PURE__*/ new Vector3();
const _worldScale = /*@__PURE__*/ new Vector3();
const _cameraPosition = /*@__PURE__*/ new Vector3();
const _cameraDirection = /*@__PURE__*/ new Vector3();
const _sortDepthRange = /*@__PURE__*/ new Vector2();

/**
 * A first-class scene object for 3D Gaussian splat geometry.
 *
 * @augments Mesh
 */
class GaussianSplat extends Mesh {

	/**
	 * Constructs a new Gaussian splat object.
	 *
	 * @param {BufferGeometry} splatGeometry - The splat geometry to render.
	 * @param {GaussianSplatMaterial|Object} [material] - The splat render-state material or options.
	 */
	constructor( splatGeometry, material = new GaussianSplatMaterial() ) {

		const positionAttribute = splatGeometry.getAttribute( 'position' );
		const covarianceAttribute = splatGeometry.getAttribute( 'covariance' );
		const colorAttribute = splatGeometry.getAttribute( 'color' );

		if ( positionAttribute === undefined || covarianceAttribute === undefined || colorAttribute === undefined ) {

			throw new Error( 'THREE.GaussianSplat: The splat geometry requires position, covariance and color attributes.' );

		}

		const count = positionAttribute.count;

		if ( splatGeometry.boundingBox === null ) splatGeometry.computeBoundingBox();
		if ( splatGeometry.boundingSphere === null ) splatGeometry.computeBoundingSphere();

		const geometry = createGeometry( count );
		const buffers = createStorageBuffers( count, positionAttribute.array, covarianceAttribute.array, colorAttribute.array );

		if ( material.isGaussianSplatMaterial !== true ) material = new GaussianSplatMaterial( material );
		material.setBuffers( buffers );

		super( geometry, material );

		/**
		 * This flag can be used for type testing.
		 *
		 * @type {boolean}
		 * @readonly
		 * @default true
		 */
		this.isGaussianSplat = true;

		this.type = 'GaussianSplat';

		/**
		 * The source splat geometry.
		 *
		 * @type {BufferGeometry}
		 */
		this.splatGeometry = splatGeometry;

		this.frustumCulled = false;

		this._buffers = buffers;
		this._sortMatrix = uniform( new Matrix4() );
		this._sortDepthRange = uniform( new Vector2( 0, 1 ) );
		this._sortInitialized = false;
		this._lastSortPosition = new Vector3( Infinity, Infinity, Infinity );
		this._lastSortDirection = new Vector3( 0, 0, - 1 );
		this._positionAttribute = positionAttribute;
		this._webGLSortBins = new Uint32Array( count );
		this._webGLSortCounts = new Uint32Array( BIN_COUNT );
		this._webGLSortOffsets = new Uint32Array( BIN_COUNT );

		createSortNodes( this );

		this.onBeforeRender = ( renderer, scene, camera ) => {

			if ( this.material.autoSort === true ) {

				this.updateSort( renderer, camera );

			}

		};

	}

	/**
	 * Updates the draw order if the camera has moved enough to need a new sort.
	 *
	 * @param {Renderer} renderer - The renderer.
	 * @param {Camera} camera - The camera used for rendering.
	 * @return {boolean} Whether a sort was dispatched this call.
	 */
	updateSort( renderer, camera ) {

		if ( this._sortInitialized === false || this._needsSort( camera ) === true ) {

			this._updateSortUniforms( camera );

			if ( renderer.backend && renderer.backend.isWebGLBackend === true ) {

				enableWebGLBuffers( this._buffers );
				this._sortCPU();

			} else {

				renderer.compute( this._resetHistogramNode );
				renderer.compute( this._histogramNode );
				renderer.compute( this._prefixNode );
				renderer.compute( this._scatterNode );

			}

			this._sortInitialized = true;

			return true;

		}

		return false;

	}

	_needsSort( camera ) {

		_cameraPosition.setFromMatrixPosition( camera.matrixWorld );

		const e = camera.matrixWorld.elements;
		_cameraDirection.set( - e[ 8 ], - e[ 9 ], - e[ 10 ] ).normalize();

		const positionChanged = _cameraPosition.distanceToSquared( this._lastSortPosition ) > SORT_POSITION_THRESHOLD * SORT_POSITION_THRESHOLD;
		const directionChanged = _cameraDirection.dot( this._lastSortDirection ) < SORT_DIRECTION_THRESHOLD;

		if ( positionChanged === true || directionChanged === true ) {

			this._lastSortPosition.copy( _cameraPosition );
			this._lastSortDirection.copy( _cameraDirection );
			return true;

		}

		return false;

	}

	_updateSortUniforms( camera ) {

		this.updateWorldMatrix( true, false );

		this._sortMatrix.value.multiplyMatrices( camera.matrixWorldInverse, this.matrixWorld );

		_worldCenter.copy( this.splatGeometry.boundingSphere.center ).applyMatrix4( this.matrixWorld );
		_viewCenter.copy( _worldCenter ).applyMatrix4( camera.matrixWorldInverse );
		this.getWorldScale( _worldScale );

		const radius = this.splatGeometry.boundingSphere.radius * Math.max( _worldScale.x, _worldScale.y, _worldScale.z );
		const depth = - _viewCenter.z;
		const nearDepth = Math.max( camera.near, depth - radius );
		const farDepth = Math.max( nearDepth + 0.0001, depth + radius );

		_sortDepthRange.set( nearDepth, farDepth );
		this._sortDepthRange.value.copy( _sortDepthRange );

	}

	_sortCPU() {

		const buffers = this._buffers;
		const centers = this._positionAttribute.array;
		const order = buffers.orderAttribute.array;
		const bins = this._webGLSortBins;
		const counts = this._webGLSortCounts;
		const offsets = this._webGLSortOffsets;
		const matrix = this._sortMatrix.value.elements;
		const nearDepth = this._sortDepthRange.value.x;
		const range = Math.max( this._sortDepthRange.value.y - nearDepth, 0.0001 );
		const scale = ( BIN_COUNT - 1 ) / range;

		counts.fill( 0 );

		for ( let i = 0, l = buffers.count; i < l; i ++ ) {

			const i3 = i * 3;
			const depth = - ( matrix[ 2 ] * centers[ i3 ] + matrix[ 6 ] * centers[ i3 + 1 ] + matrix[ 10 ] * centers[ i3 + 2 ] + matrix[ 14 ] );
			const depthBin = Math.min( BIN_COUNT - 1, Math.max( 0, Math.floor( ( depth - nearDepth ) * scale ) ) );
			const bin = BIN_COUNT - 1 - depthBin;

			bins[ i ] = bin;
			counts[ bin ] ++;

		}

		let sum = 0;

		for ( let i = 0; i < BIN_COUNT; i ++ ) {

			offsets[ i ] = sum;
			sum += counts[ i ];

		}

		for ( let i = 0, l = buffers.count; i < l; i ++ ) {

			order[ offsets[ bins[ i ] ] ++ ] = i;

		}

		buffers.orderAttribute.needsUpdate = true;

		if ( buffers.orderAttribute.pbo !== undefined ) {

			buffers.orderAttribute.pbo.needsUpdate = true;

		}

	}

	copy( source, recursive ) {

		super.copy( source, recursive );

		this.splatGeometry = source.splatGeometry;

		return this;

	}

}

function createGeometry( count ) {

	const geometry = new InstancedBufferGeometry();
	geometry.setAttribute( 'position', new BufferAttribute( new Float32Array( [
		- 2, - 2, 0,
		2, - 2, 0,
		2, 2, 0,
		- 2, 2, 0
	] ), 3 ) );
	geometry.setIndex( [ 0, 1, 2, 0, 2, 3 ] );
	geometry.instanceCount = count;

	return geometry;

}

function createStorageBuffers( count, centers, covariances, colors ) {

	const centerData = new Float32Array( count * 4 );
	const covarianceAData = new Float32Array( count * 4 );
	const covarianceBData = new Float32Array( count * 4 );
	const colorData = new Float32Array( count * 4 );
	const orderData = new Uint32Array( count );
	const binData = new Uint32Array( count );

	for ( let i = 0; i < count; i ++ ) {

		const i3 = i * 3;
		const i4 = i * 4;
		const i6 = i * 6;

		centerData[ i4 ] = centers[ i3 ];
		centerData[ i4 + 1 ] = centers[ i3 + 1 ];
		centerData[ i4 + 2 ] = centers[ i3 + 2 ];

		covarianceAData[ i4 ] = covariances[ i6 ];
		covarianceAData[ i4 + 1 ] = covariances[ i6 + 1 ];
		covarianceAData[ i4 + 2 ] = covariances[ i6 + 2 ];
		covarianceAData[ i4 + 3 ] = covariances[ i6 + 3 ];

		covarianceBData[ i4 ] = covariances[ i6 + 4 ];
		covarianceBData[ i4 + 1 ] = covariances[ i6 + 5 ];

		colorData[ i4 ] = colors[ i4 ] / 255;
		colorData[ i4 + 1 ] = colors[ i4 + 1 ] / 255;
		colorData[ i4 + 2 ] = colors[ i4 + 2 ] / 255;
		colorData[ i4 + 3 ] = colors[ i4 + 3 ] / 255;

		orderData[ i ] = i;

	}

	const centerAttribute = new StorageBufferAttribute( centerData, 4 );
	const covarianceAAttribute = new StorageBufferAttribute( covarianceAData, 4 );
	const covarianceBAttribute = new StorageBufferAttribute( covarianceBData, 4 );
	const colorAttribute = new StorageBufferAttribute( colorData, 4 );
	const orderAttribute = new StorageBufferAttribute( orderData, 1, Uint32Array );
	const binAttribute = new StorageBufferAttribute( binData, 1, Uint32Array );
	const histogramAttribute = new StorageBufferAttribute( new Uint32Array( BIN_COUNT ), 1, Uint32Array );
	const offsetAttribute = new StorageBufferAttribute( new Uint32Array( BIN_COUNT ), 1, Uint32Array );

	return {
		count,
		orderAttribute,
		webGLBuffersEnabled: false,
		centerRead: storage( centerAttribute, 'vec4', count ).toReadOnly(),
		covarianceARead: storage( covarianceAAttribute, 'vec4', count ).toReadOnly(),
		covarianceBRead: storage( covarianceBAttribute, 'vec4', count ).toReadOnly(),
		colorRead: storage( colorAttribute, 'vec4', count ).toReadOnly(),
		orderRead: storage( orderAttribute, 'uint', count ).toReadOnly(),
		orderWrite: storage( orderAttribute, 'uint', count ),
		binRead: storage( binAttribute, 'uint', count ).toReadOnly(),
		binWrite: storage( binAttribute, 'uint', count ),
		histogramAtomic: storage( histogramAttribute, 'uint', BIN_COUNT ).toAtomic(),
		offsetAtomic: storage( offsetAttribute, 'uint', BIN_COUNT ).toAtomic()
	};

}

function enableWebGLBuffers( buffers ) {

	if ( buffers.webGLBuffersEnabled === true ) return;

	buffers.orderAttribute.setUsage( DynamicDrawUsage );
	buffers.centerRead.setPBO( true );
	buffers.covarianceARead.setPBO( true );
	buffers.covarianceBRead.setPBO( true );
	buffers.colorRead.setPBO( true );
	buffers.orderRead.setPBO( true );
	buffers.webGLBuffersEnabled = true;

}

function createSortNodes( splat ) {

	const { _buffers: buffers } = splat;
	const sortMatrix = splat._sortMatrix;
	const sortDepthRange = splat._sortDepthRange;

	splat._resetHistogramNode = Fn( () => {

		atomicStore( buffers.histogramAtomic.element( instanceIndex ), uint( 0 ) );
		atomicStore( buffers.offsetAtomic.element( instanceIndex ), uint( 0 ) );

	} )().compute( BIN_COUNT, [ WORKGROUP_SIZE ] ).setName( 'GaussianSplatSortReset' );

	splat._histogramNode = Fn( () => {

		const center = buffers.centerRead.element( instanceIndex ).xyz.toVar( 'center' );
		const viewCenter = sortMatrix.mul( vec4( center, 1 ) ).xyz.toVar( 'viewCenter' );
		const depth = viewCenter.z.negate().toVar( 'depth' );
		const range = max( sortDepthRange.y.sub( sortDepthRange.x ), 0.0001 ).toVar( 'range' );
		const normalized = depth.sub( sortDepthRange.x ).div( range ).clamp( 0, 1 ).toVar( 'normalized' );
		const depthBin = uint( normalized.mul( BIN_COUNT - 1 ) ).toVar( 'depthBin' );
		const bin = uint( BIN_COUNT - 1 ).sub( depthBin ).toVar( 'bin' );

		buffers.binWrite.element( instanceIndex ).assign( bin );
		atomicAdd( buffers.histogramAtomic.element( bin ), uint( 1 ) );

	} )().compute( buffers.count, [ WORKGROUP_SIZE ] ).setName( 'GaussianSplatSortHistogram' );

	splat._prefixNode = Fn( () => {

		const sum = uint( 0 ).toVar( 'sum' );

		Loop( { start: 0, end: BIN_COUNT, type: 'uint', name: 'bin', condition: '<' }, ( { bin } ) => {

			const count = atomicLoad( buffers.histogramAtomic.element( bin ) ).toVar( 'count' );
			atomicStore( buffers.offsetAtomic.element( bin ), sum );
			sum.addAssign( count );

		} );

	} )().compute( 1 ).setName( 'GaussianSplatSortPrefix' );

	splat._scatterNode = Fn( () => {

		const bin = buffers.binRead.element( instanceIndex ).toVar( 'bin' );
		const targetIndex = atomicAdd( buffers.offsetAtomic.element( bin ), uint( 1 ) ).toVar( 'targetIndex' );
		buffers.orderWrite.element( targetIndex ).assign( instanceIndex );

	} )().compute( buffers.count, [ WORKGROUP_SIZE ] ).setName( 'GaussianSplatSortScatter' );

}

export { GaussianSplat };
