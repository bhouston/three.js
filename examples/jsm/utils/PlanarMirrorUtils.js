import { BackSide, Plane, Vector3 } from 'three/webgpu';

/**
 * Finds the world-space reflecting plane of a static metallic mesh. Accepts
 * planes and thin slabs, but rejects curved objects and narrow solid beams.
 * Color and roughness are deliberately unrestricted. Textured metalness is
 * qualified by its scalar factor; the material still applies its texture.
 *
 * @param {Mesh} mesh - Mesh with a single standard or physical material.
 * @param {Vector3} sceneCenter - Used to select the inward face of a thin slab.
 * @return {?Plane} The reflecting plane, or null if the mesh is not a candidate.
 */
export function getPlanarMirrorPlane( mesh, sceneCenter ) {

	const material = mesh.material;
	if ( ! mesh.isMesh || mesh.isSkinnedMesh || mesh.isInstancedMesh || Array.isArray( material ) || ! material || ! ( material.metalness >= 0.9 ) || material.transparent || material.side === BackSide ) return null;

	const geometry = mesh.geometry;
	const position = geometry.getAttribute( 'position' );
	if ( ! position || geometry.morphAttributes.position?.length ) return null;
	const index = geometry.index;
	const start = geometry.drawRange.start;
	const end = Math.min( index ? index.count : position.count, start + geometry.drawRange.count );
	const a = new Vector3(), b = new Vector3(), c = new Vector3();
	const edge = new Vector3(), cross = new Vector3();
	const normal = new Vector3(), tangent = new Vector3();
	let largestArea = 0;

	const read = ( target, i ) => target.fromBufferAttribute( position, index ? index.getX( i ) : i ).applyMatrix4( mesh.matrixWorld );
	for ( let i = start; i + 2 < end; i += 3 ) {

		read( a, i ); read( b, i + 1 ); read( c, i + 2 );
		edge.subVectors( b, a );
		cross.subVectors( c, a ).cross( edge );
		const area = cross.lengthSq();
		if ( area > largestArea ) {

			largestArea = area;
			normal.copy( cross ).negate().normalize();
			tangent.copy( edge ).normalize();

		}

	}

	if ( largestArea === 0 ) return null;
	const bitangent = new Vector3().crossVectors( normal, tangent );
	const min = new Vector3( Infinity, Infinity, Infinity );
	const max = new Vector3( - Infinity, - Infinity, - Infinity );
	for ( let i = start; i < end; i ++ ) {

		read( a, i );
		b.set( a.dot( tangent ), a.dot( bitangent ), a.dot( normal ) );
		min.min( b ); max.max( b );

	}

	const extent = max.clone().sub( min );
	if ( ! Number.isFinite( extent.lengthSq() ) || Math.min( extent.x, extent.y ) <= 0 || extent.z > Math.min( extent.x, extent.y ) * 0.02 ) return null;

	// Prefer the slab face facing into the scene, rather than its hidden back.
	if ( sceneCenter.dot( normal ) < ( min.z + max.z ) * 0.5 ) {

		normal.negate();
		return new Plane( normal, min.z );

	}

	return new Plane( normal, - max.z );

}
