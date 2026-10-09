import { BufferAttribute, BufferGeometry, MathUtils, Vector3 } from 'three';
import { potpack } from '../libs/potpack.module.js';

const _a = new Vector3();
const _b = new Vector3();
const _c = new Vector3();
const _normal = new Vector3();

/**
 * Generates light map UVs for a set of meshes and packs them into a single atlas.
 * The result is written to the `uv1` attribute.
 *
 * Charts are built from connected triangles facing the same principal axis and are
 * projected onto that axis plane. This is fast and works well for architectural scenes,
 * but curved surfaces produce many small charts.
 *
 * Every mesh receives a new geometry, so meshes sharing a geometry get distinct atlas
 * regions. Chart sizes take the world scale into account, so unwrap after positioning
 * the meshes. Morph attributes are not supported.
 *
 * ```js
 * const unwrapper = new UVUnwrapper();
 * const size = unwrapper.unwrap( meshes );
 * ```
 *
 * @three_import import { UVUnwrapper } from 'three/addons/utils/UVUnwrapper.js';
 */
class UVUnwrapper {

	/**
	 * Constructs a new UV unwrapper.
	 */
	constructor() {

		/**
		 * The target number of texels per world unit. It is reduced
		 * if the atlas would exceed the maximum size.
		 *
		 * @type {number}
		 * @default 16
		 */
		this.texelsPerUnit = 16;

		/**
		 * The padding around each chart in texels.
		 *
		 * @type {number}
		 * @default 3
		 */
		this.padding = 3;

		/**
		 * The maximum atlas size in texels.
		 *
		 * @type {number}
		 * @default 2048
		 */
		this.maxSize = 2048;

	}

	/**
	 * Unwraps the given meshes into a shared atlas.
	 *
	 * @param {Array<Mesh>} meshes - The meshes to unwrap.
	 * @return {number} The atlas size in texels.
	 */
	unwrap( meshes ) {

		const charts = [];
		const results = [];

		for ( const mesh of meshes ) {

			mesh.updateWorldMatrix( true, false );

			results.push( this._buildCharts( mesh, charts ) );

		}

		// pack, lowering the density until the atlas fits

		const padding = this.padding;

		let density = this.texelsPerUnit;
		let size;

		while ( true ) {

			for ( const chart of charts ) {

				chart.w = Math.ceil( ( chart.maxU - chart.minU ) * density ) + 1 + 2 * padding;
				chart.h = Math.ceil( ( chart.maxV - chart.minV ) * density ) + 1 + 2 * padding;

			}

			const { w, h } = potpack( charts );

			size = MathUtils.ceilPowerOfTwo( Math.max( w, h ) );

			if ( size <= this.maxSize ) break;

			density *= 0.75;

		}

		for ( const result of results ) {

			this._writeGeometry( result, density, size );

		}

		return size;

	}

	/**
	 * Groups the triangles of the given mesh into charts.
	 *
	 * @private
	 * @param {Mesh} mesh - The mesh.
	 * @param {Array<Object>} charts - The list of all charts.
	 * @return {Object} The per-mesh unwrap data.
	 */
	_buildCharts( mesh, charts ) {

		const geometry = mesh.geometry;
		const position = geometry.attributes.position;
		const index = geometry.index;
		const triangleCount = ( index !== null ? index.count : position.count ) / 3;

		const getVertex = ( corner ) => index !== null ? index.getX( corner ) : corner;

		const scale = [ 0, 1, 2 ].map( ( axis ) => _a.setFromMatrixColumn( mesh.matrixWorld, axis ).length() );

		// classify triangles by principal axis and connect them via shared positions

		const directions = new Uint8Array( triangleCount );
		const keys = [];
		const trianglesByKey = new Map();

		for ( let t = 0; t < triangleCount; t ++ ) {

			_a.fromBufferAttribute( position, getVertex( t * 3 ) );
			_b.fromBufferAttribute( position, getVertex( t * 3 + 1 ) );
			_c.fromBufferAttribute( position, getVertex( t * 3 + 2 ) );

			_normal.subVectors( _b, _a ).cross( _c.sub( _a ) );

			const n = [ _normal.x, _normal.y, _normal.z ];
			const abs = n.map( Math.abs );
			const axis = abs.indexOf( Math.max( ...abs ) );

			directions[ t ] = axis * 2 + ( n[ axis ] < 0 ? 1 : 0 );

			for ( let k = 0; k < 3; k ++ ) {

				const v = getVertex( t * 3 + k );
				const key = Math.round( position.getX( v ) * 1e5 ) + ',' + Math.round( position.getY( v ) * 1e5 ) + ',' + Math.round( position.getZ( v ) * 1e5 );

				keys.push( key );

				if ( trianglesByKey.has( key ) === false ) trianglesByKey.set( key, [] );
				trianglesByKey.get( key ).push( t );

			}

		}

		// grow charts, rejecting triangles that would overlap in the projection

		const chartIndices = new Int32Array( triangleCount ).fill( - 1 );
		const meshCharts = [];

		for ( let seed = 0; seed < triangleCount; seed ++ ) {

			if ( chartIndices[ seed ] !== - 1 ) continue;

			const axis = directions[ seed ] >> 1;
			const u = ( axis + 1 ) % 3;
			const v = ( axis + 2 ) % 3;

			const chart = {
				u, v, scaleU: scale[ u ], scaleV: scale[ v ],
				minU: Infinity, minV: Infinity, maxU: - Infinity, maxV: - Infinity,
				occupied: new Set(),
				vertices: new Map()
			};

			const queue = [ seed ];

			while ( queue.length > 0 ) {

				const t = queue.pop();

				if ( chartIndices[ t ] !== - 1 || this._place( chart, t, position, getVertex ) === false ) continue;

				chartIndices[ t ] = meshCharts.length;

				for ( let k = 0; k < 3; k ++ ) {

					for ( const neighbor of trianglesByKey.get( keys[ t * 3 + k ] ) ) {

						if ( chartIndices[ neighbor ] === - 1 && directions[ neighbor ] === directions[ seed ] ) queue.push( neighbor );

					}

				}

			}

			chart.occupied = null;

			meshCharts.push( chart );
			charts.push( chart );

		}

		return { mesh, triangleCount, getVertex, chartIndices, meshCharts };

	}

	/**
	 * Adds a triangle to a chart unless its projection overlaps the chart.
	 *
	 * @private
	 * @param {Object} chart - The chart.
	 * @param {number} t - The triangle index.
	 * @param {BufferAttribute} position - The position attribute.
	 * @param {Function} getVertex - Maps a triangle corner to a vertex index.
	 * @return {boolean} Whether the triangle was added.
	 */
	_place( chart, t, position, getVertex ) {

		// rasterize texel centers at the target density

		const density = this.texelsPerUnit;
		const p = [];

		for ( let k = 0; k < 3; k ++ ) {

			const vertex = getVertex( t * 3 + k );

			p.push( position.getComponent( vertex, chart.u ) * chart.scaleU, position.getComponent( vertex, chart.v ) * chart.scaleV );

		}

		const x0 = p[ 0 ] * density, y0 = p[ 1 ] * density;
		const x1 = p[ 2 ] * density, y1 = p[ 3 ] * density;
		const x2 = p[ 4 ] * density, y2 = p[ 5 ] * density;

		const area = ( x1 - x0 ) * ( y2 - y0 ) - ( x2 - x0 ) * ( y1 - y0 );
		const sign = Math.sign( area );

		const cells = [];

		if ( sign !== 0 ) {

			const minX = Math.floor( Math.min( x0, x1, x2 ) ), maxX = Math.ceil( Math.max( x0, x1, x2 ) );
			const minY = Math.floor( Math.min( y0, y1, y2 ) ), maxY = Math.ceil( Math.max( y0, y1, y2 ) );

			for ( let i = minX; i < maxX; i ++ ) {

				for ( let j = minY; j < maxY; j ++ ) {

					const x = i + 0.5, y = j + 0.5;

					const w0 = ( ( x1 - x ) * ( y2 - y ) - ( x2 - x ) * ( y1 - y ) ) * sign;
					const w1 = ( ( x2 - x ) * ( y0 - y ) - ( x0 - x ) * ( y2 - y ) ) * sign;
					const w2 = ( ( x0 - x ) * ( y1 - y ) - ( x1 - x ) * ( y0 - y ) ) * sign;

					if ( w0 < 0 || w1 < 0 || w2 < 0 ) continue;

					const cell = i + ',' + j;

					if ( chart.occupied.has( cell ) ) return false;

					cells.push( cell );

				}

			}

		}

		for ( const cell of cells ) chart.occupied.add( cell );

		for ( let k = 0; k < 3; k ++ ) {

			chart.minU = Math.min( chart.minU, p[ k * 2 ] );
			chart.maxU = Math.max( chart.maxU, p[ k * 2 ] );
			chart.minV = Math.min( chart.minV, p[ k * 2 + 1 ] );
			chart.maxV = Math.max( chart.maxV, p[ k * 2 + 1 ] );

		}

		return true;

	}

	/**
	 * Creates the unwrapped geometry of a mesh. Vertices shared by several charts are duplicated.
	 *
	 * @private
	 * @param {Object} result - The per-mesh unwrap data.
	 * @param {number} density - The final texels per world unit.
	 * @param {number} size - The atlas size.
	 */
	_writeGeometry( result, density, size ) {

		const { mesh, triangleCount, getVertex, chartIndices, meshCharts } = result;

		const source = mesh.geometry;
		const position = source.attributes.position;
		const offset = this.padding + 0.5;

		const sources = [];
		const indices = [];
		const uvs = [];

		for ( let t = 0; t < triangleCount; t ++ ) {

			const chart = meshCharts[ chartIndices[ t ] ];

			for ( let k = 0; k < 3; k ++ ) {

				const vertex = getVertex( t * 3 + k );

				if ( chart.vertices.has( vertex ) === false ) {

					chart.vertices.set( vertex, sources.length );
					sources.push( vertex );

					const u = position.getComponent( vertex, chart.u ) * chart.scaleU;
					const v = position.getComponent( vertex, chart.v ) * chart.scaleV;

					uvs.push(
						( chart.x + offset + ( u - chart.minU ) * density ) / size,
						( chart.y + offset + ( v - chart.minV ) * density ) / size
					);

				}

				indices.push( chart.vertices.get( vertex ) );

			}

		}

		const geometry = new BufferGeometry();

		for ( const name in source.attributes ) {

			const attribute = source.attributes[ name ];
			const itemSize = attribute.itemSize;
			const copy = new BufferAttribute( new attribute.array.constructor( sources.length * itemSize ), itemSize, attribute.normalized );

			for ( let i = 0; i < sources.length; i ++ ) {

				for ( let c = 0; c < itemSize; c ++ ) {

					copy.setComponent( i, c, attribute.getComponent( sources[ i ], c ) );

				}

			}

			geometry.setAttribute( name, copy );

		}

		geometry.setAttribute( 'uv1', new BufferAttribute( new Float32Array( uvs ), 2 ) );
		geometry.setIndex( indices );

		for ( const group of source.groups ) {

			geometry.addGroup( group.start, group.count, group.materialIndex );

		}

		for ( const chart of meshCharts ) chart.vertices = null;

		mesh.geometry = geometry;

	}

}

export { UVUnwrapper };
