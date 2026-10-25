import { Box3, Float32BufferAttribute, Group, Material, Matrix4, Mesh, Object3D, PropertyBinding, Vector3 } from 'three';
import { mergeGeometries } from './BufferGeometryUtils.js';

const batchGroups = /*@__PURE__*/ new WeakSet();

/**
 * Merges compatible static opaque meshes within spatial cells. Original meshes
 * and their geometries/materials are retained and restored by `dispose()`.
 * Vertex attributes, including light-map UVs, are preserved. Batches are attached
 * to `root` in its local space, so moving the root remains supported.
 *
 * Descendant transforms and geometry must remain static until disposal. Supply
 * all animation clips that can affect descendants, including glTF clips stored
 * separately from the scene. A filter can exclude other externally driven meshes.
 * Skinned, morphed, instanced, transparent, custom-shader, custom-render-hook,
 * displacement, LOD and clipping-group meshes are left untouched.
 *
 * @param {Object3D} root - Root containing the static scene geometry.
 * @param {Object} [options={}] - Batching settings.
 * @param {?number} [options.cellSize=null] - Cell size in root-local units; defaults to one third of the largest eligible scene dimension.
 * @param {Array<AnimationClip>} [options.animations=[]] - Clips whose animated descendants must be excluded.
 * @param {?Function} [options.filter=null] - Optional additional predicate for eligible meshes.
 * @return {{group: Group, sourceMeshes: Array<Mesh>, batchCount: number, dispose: Function}} Batches and idempotent cleanup. Materials and original geometries remain owned by the caller.
 */
function batchStaticMeshes( root, { cellSize = null, animations = [], filter = null } = {} ) {

	if ( cellSize !== null && ( ! Number.isFinite( cellSize ) || cellSize <= 0 ) ) throw new RangeError( 'batchStaticMeshes: cellSize must be positive and finite.' );
	root.updateWorldMatrix( true, true );
	const inverse = root.matrixWorld.clone().invert();
	const animated = new Set(), clips = new Set( animations );
	root.traverse( object => {

		for ( const clip of object.animations ) clips.add( clip );

	} );
	for ( const clip of clips ) for ( const track of clip.tracks ) {

		const target = PropertyBinding.findNode( root, PropertyBinding.parseTrackName( track.name ).nodeName );
		if ( target && target !== root ) target.traverse( object => animated.add( object ) );

	}

	let orderedRoot = false;
	for ( let parent = root; parent; parent = parent.parent ) if ( parent.isGroup && parent.renderOrder !== 0 ) orderedRoot = true;
	const bounds = new Box3(), records = [];
	root.traverseVisible( object => {

		if ( root.isLOD || batchGroups.has( root ) || object === root || ! object.isMesh || object.isSkinnedMesh || object.isInstancedMesh || object.children.length || animated.has( object ) ) return;
		const material = object.material, geometry = object.geometry;
		if ( ! material || ! geometry?.attributes.position || Array.isArray( material ) || material.transparent || material.isShaderMaterial || material.isNodeMaterial || material.displacementMap || material.onBeforeCompile !== Material.prototype.onBeforeCompile ) return;
		if ( geometry.attributes.position.itemSize !== 3 || ( geometry.attributes.normal && geometry.attributes.normal.itemSize !== 3 ) || ( geometry.attributes.tangent && geometry.attributes.tangent.itemSize !== 4 ) ) return;
		if ( object.onBeforeRender !== Object3D.prototype.onBeforeRender || object.onAfterRender !== Object3D.prototype.onAfterRender || object.customDepthMaterial || object.customDistanceMaterial ) return;
		if ( geometry.isInstancedBufferGeometry || geometry.groups.length || Object.keys( geometry.morphAttributes ).length || geometry.drawRange.start !== 0 || geometry.drawRange.count !== Infinity ) return;
		if ( filter && ! filter( object ) ) return;
		let groupPath = [];
		for ( let parent = object.parent; parent && parent !== root; parent = parent.parent ) {

			if ( parent.isLOD || parent.isClippingGroup || batchGroups.has( parent ) ) return;
			if ( parent.isGroup ) groupPath.unshift( [ parent.renderOrder, parent.layers.mask ] );

		}

		if ( ! orderedRoot && groupPath.every( ( [ order ] ) => order === 0 ) ) groupPath = [];

		const matrix = new Matrix4().multiplyMatrices( inverse, object.matrixWorld );
		if ( matrix.determinant() <= 0 ) return;
		const names = Object.keys( geometry.attributes ).sort();
		if ( names.some( name => geometry.attributes[ name ].isGLBufferAttribute || ( geometry.attributes[ name ].isFloat16BufferAttribute && geometry.attributes[ name ].itemSize > 4 ) || geometry.attributes[ name ].isInstancedBufferAttribute || geometry.attributes[ name ].data?.isInstancedInterleavedBuffer ) ) return;
		const attributes = names.map( name => {

			const attribute = geometry.attributes[ name ];
			const array = attribute.isInterleavedBufferAttribute ? attribute.data.array : attribute.array;
			return `${ name }:${ array.constructor.name }:${ attribute.itemSize }:${ attribute.normalized }:${ attribute.gpuType }`;

		} ).join( ',' );
		geometry.computeBoundingBox();
		const box = geometry.boundingBox.clone().applyMatrix4( matrix );
		bounds.union( box );
		records.push( { object, matrix, center: box.getCenter( new Vector3() ), groupPath, attributes } );

	} );
	const size = bounds.getSize( new Vector3() );
	cellSize ??= Math.max( size.x, size.y, size.z, 0.001 ) / 3;
	const groups = new Map();
	for ( const record of records ) {

		const { object, groupPath, attributes, center } = record;
		const cell = center.clone().sub( bounds.min ).divideScalar( cellSize ).floor().toArray();
		const key = `${ cell }:${ object.material.uuid }:${ object.castShadow }:${ object.receiveShadow }:${ object.layers.mask }:${ object.renderOrder }:${ JSON.stringify( groupPath ) }:${ object.frustumCulled }:${ Boolean( object.geometry.index ) }:${ attributes }`;
		if ( ! groups.has( key ) ) groups.set( key, [] );
		groups.get( key ).push( record );

	}

	const group = new Group(), sourceMeshes = [], batchMeshes = [], orderGroups = new Map();
	group.name = 'Static Mesh Batches';
	group.layers.mask = 0; // Preserve the root's inherited group order for every camera layer mask.
	batchGroups.add( group );
	for ( const records of groups.values() ) {

		if ( records.length < 2 ) continue;
		const geometries = records.map( record => {

			const geometry = record.object.geometry.clone();
			for ( const [ name, attribute ] of Object.entries( geometry.attributes ) ) {

				// Transform decoded values at float precision. Writing transformed
				// positions back into a quantized attribute would truncate them.
				// Float16 attributes also need promotion because mergeGeometries
				// produces ordinary BufferAttributes, losing the half-float marker.
				if ( attribute.isFloat16BufferAttribute || ( [ 'position', 'normal', 'tangent' ].includes( name ) && ( attribute.normalized || ! ( attribute.array instanceof Float32Array ) ) ) ) {

					const values = new Float32Array( attribute.count * attribute.itemSize );
					const getters = [ 'getX', 'getY', 'getZ', 'getW' ];
					for ( let i = 0; i < attribute.count; i ++ ) for ( let c = 0; c < attribute.itemSize; c ++ ) values[ i * attribute.itemSize + c ] = attribute[ getters[ c ] ]( i );
					geometry.setAttribute( name, new Float32BufferAttribute( values, attribute.itemSize ) );

				}

			}

			return geometry.applyMatrix4( record.matrix );

		} );
		const geometry = mergeGeometries( geometries );
		for ( const source of geometries ) source.dispose();
		if ( geometry === null ) continue;
		geometry.computeBoundingSphere();
		const { object: source, groupPath } = records[ 0 ];
		const mesh = new Mesh( geometry, source.material );
		mesh.castShadow = source.castShadow;
		mesh.receiveShadow = source.receiveShadow;
		mesh.layers.mask = source.layers.mask;
		mesh.renderOrder = source.renderOrder;
		mesh.frustumCulled = source.frustumCulled;
		let parent = group;
		if ( groupPath.length ) {

			const key = JSON.stringify( groupPath );
			if ( ! orderGroups.has( key ) ) {

				let current = group;
				for ( const [ order, mask ] of groupPath ) {

					const ordered = new Group(); ordered.renderOrder = order; ordered.layers.mask = mask; current.add( ordered ); current = ordered;

				}

				orderGroups.set( key, current );

			}

			parent = orderGroups.get( key );

		}

		parent.add( mesh ); batchMeshes.push( mesh );
		for ( const { object } of records ) {

			object.visible = false; sourceMeshes.push( object );

		}

	}

	if ( batchMeshes.length ) root.add( group );
	let disposed = false;
	return {
		group, sourceMeshes, batchCount: batchMeshes.length,
		dispose() {

			if ( disposed ) return;
			disposed = true;
			group.removeFromParent();
			for ( const mesh of batchMeshes ) mesh.geometry.dispose();
			for ( const source of sourceMeshes ) source.visible = true;

		}
	};

}

export { batchStaticMeshes };
