import { Box3, Mesh, Sphere, Vector3 } from 'three';

const _point = new Vector3();
const _empty = [];

/**
 * Experimental static visibility index for the common renderer's getRenderChildren hook.
 * Keeps the authored hierarchy intact. Rebuild after changing transforms, geometry,
 * frustumCulled or child membership. Plain static Mesh bounds are indexed; other
 * renderables and dynamic/bundled subtrees retain ordinary traversal. Shader
 * displacement requires conservative geometry bounds. Do not overlap index roots.
 * WebGLRenderer does not consume the traversal hook.
 *
 * @three_import import { SceneSpatialIndex } from 'three/addons/utils/SceneSpatialIndex.js';
 */
class SceneSpatialIndex {

	/**
	 * @param {Object3D} root - Root of the static subtree.
	 */
	constructor( root ) {

		this.root = root;
		this.enabled = true;
		this.hierarchical = true;
		this.stats = { nodes: 0, spheres: 0, candidates: 0 };
		this._hooks = [];
		this._entries = [];
		this._always = [];
		this._order = new Map();
		this._lists = new Map();
		this._included = new Set();
		this._all = false;
		this._tree = null;
		this.rebuild();

	}

	/** Recomputes world bounds and the independent spatial hierarchy. */
	rebuild() {

		this.dispose();
		this._entries = [];
		this._always = [];
		this._order.clear();
		this.root.updateWorldMatrix( true, true );
		this._visit( this.root );
		this._tree = this._entries.length > 0 ? this._build( this._entries.slice() ) : null;

	}

	_visit( object ) {

		if ( object.isLOD || object.isBundleGroup || object.isSkinnedMesh || object.getRenderChildren !== undefined ) {

			if ( object !== this.root ) this._always.push( object );
			return;

		}

		if ( object !== this.root ) {

			if ( object.constructor === Mesh && object.frustumCulled && object.intersectsFrustum === Mesh.prototype.intersectsFrustum ) {

				const geometry = object.geometry;
				if ( geometry.boundingSphere === null ) geometry.computeBoundingSphere();
				if ( object.boundingSphere === null ) object.computeBoundingSphere();
				const sphere = new Sphere().copy( object.boundingSphere || geometry.boundingSphere ).applyMatrix4( object.matrixWorld );
				const box = sphere.getBoundingBox( new Box3() );
				this._entries.push( { object, sphere, box } );

			} else if ( ! object.isGroup ) {

				this._always.push( object );

			}

		}

		if ( object.children.length > 0 || object === this.root ) {

			const previous = object.getRenderChildren;
			const callback = ( camera, frustum ) => {

				if ( ! this.enabled ) return object.children;
				if ( object === this.root ) this._prepare( frustum );
				return this._all ? object.children : ( this._lists.get( object ) || _empty );

			};

			this._hooks.push( { object, previous, callback } );
			object.getRenderChildren = callback;

		}

		for ( let i = 0; i < object.children.length; i ++ ) {

			const child = object.children[ i ];
			this._order.set( child, i );
			this._visit( child );

		}

	}

	_build( entries ) {

		const box = new Box3();
		for ( const entry of entries ) box.union( entry.box );
		if ( entries.length <= 8 ) return { box, entries };
		const size = box.getSize( new Vector3() );
		const axis = size.x >= size.y && size.x >= size.z ? 'x' : ( size.y >= size.z ? 'y' : 'z' );
		entries.sort( ( a, b ) => a.sphere.center[ axis ] - b.sphere.center[ axis ] );
		const middle = entries.length >> 1;
		return { box, children: [ this._build( entries.slice( 0, middle ) ), this._build( entries.slice( middle ) ) ] };

	}

	_classify( box, frustum, mask ) {

		this.stats.nodes ++;
		if ( frustum.planes === undefined ) return frustum.intersectsBox( box ) ? mask : - 1;

		for ( let i = 0; i < 6; i ++ ) {

			if ( ( mask & ( 1 << i ) ) === 0 ) continue;
			const plane = frustum.planes[ i ];
			const n = plane.normal;
			_point.set( n.x >= 0 ? box.max.x : box.min.x, n.y >= 0 ? box.max.y : box.min.y, n.z >= 0 ? box.max.z : box.min.z );
			if ( plane.distanceToPoint( _point ) < 0 ) return - 1;
			_point.set( n.x >= 0 ? box.min.x : box.max.x, n.y >= 0 ? box.min.y : box.max.y, n.z >= 0 ? box.min.z : box.max.z );
			if ( plane.distanceToPoint( _point ) >= 0 ) mask &= ~ ( 1 << i );

		}

		return mask;

	}

	_include( object ) {

		while ( object !== this.root && ! this._included.has( object ) ) {

			this._included.add( object );
			const parent = object.parent;
			let list = this._lists.get( parent );
			if ( list === undefined ) {

				list = [];
				this._lists.set( parent, list );

			}

			list.push( object );
			object = parent;

		}

	}

	_query( node, frustum, mask ) {

		mask = this._classify( node.box, frustum, mask );
		if ( mask === - 1 ) return;
		if ( node.entries ) {

			for ( const entry of node.entries ) {

				if ( mask !== 0 ) this.stats.spheres ++;
				if ( mask === 0 || frustum.intersectsSphere( entry.sphere ) ) {

					this.stats.candidates ++;
					this._include( entry.object );

				}

			}

		} else {

			for ( const child of node.children ) this._query( child, frustum, mask );

		}

	}

	_prepare( frustum ) {

		this.stats.nodes = 0;
		this.stats.spheres = 0;
		this.stats.candidates = 0;
		this._lists.clear();
		this._included.clear();
		this._all = false;

		if ( this.hierarchical ) {

			if ( this._tree !== null && this._classify( this._tree.box, frustum, 63 ) === 0 ) {

				this._all = true;
				this.stats.candidates = this._entries.length;
				return;

			}

			if ( this._tree !== null ) this._query( this._tree, frustum, 63 );

		} else {

			for ( const entry of this._entries ) {

				this.stats.spheres ++;
				if ( frustum.intersectsSphere( entry.sphere ) ) {

					this.stats.candidates ++;
					this._include( entry.object );

				}

			}

		}

		for ( const object of this._always ) this._include( object );
		for ( const list of this._lists.values() ) list.sort( ( a, b ) => this._order.get( a ) - this._order.get( b ) );

	}

	/** Restores ordinary child traversal without reparenting any objects. */
	dispose() {

		for ( const { object, previous, callback } of this._hooks ) {

			if ( object.getRenderChildren === callback ) {

				if ( previous === undefined ) delete object.getRenderChildren;
				else object.getRenderChildren = previous;

			}

		}

		this._hooks.length = 0;
		this._lists.clear();
		this._included.clear();
		this._entries.length = 0;
		this._always.length = 0;
		this._order.clear();
		this._tree = null;

	}

}

export { SceneSpatialIndex };
