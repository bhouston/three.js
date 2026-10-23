// CPU-only research experiment. Run: node test/benchmarks/frustum-coherence.js
// This measures candidate selection, not occlusion or rendering performance.
import assert from 'node:assert/strict';
import { performance } from 'node:perf_hooks';
import { Box3 } from '../../src/math/Box3.js';
import { Frustum } from '../../src/math/Frustum.js';
import { Matrix4 } from '../../src/math/Matrix4.js';
import { Vector3 } from '../../src/math/Vector3.js';
import { PerspectiveCamera } from '../../src/cameras/PerspectiveCamera.js';

const boxes = [];

// A deterministic, spatially distributed static world. IDs remain stable.
for ( let x = 0; x < 32; x ++ ) {

	for ( let z = 0; z < 32; z ++ ) {

		for ( let y = 0; y < 4; y ++ ) {

			const center = new Vector3( ( x - 16 ) * 12, y * 8, ( z - 16 ) * 12 );
			boxes.push( new Box3().setFromCenterAndSize( center, new Vector3( 4, 4, 4 ) ) );

		}

	}

}

function build( ids ) {

	const bounds = new Box3();
	for ( const id of ids ) bounds.union( boxes[ id ] );
	const node = { bounds, ids, children: null, rejectPlane: 0 };
	if ( ids.length <= 8 ) return node;
	const size = bounds.getSize( new Vector3() );
	const axis = size.x >= size.z ? 'x' : 'z';
	ids.sort( ( a, b ) => boxes[ a ].min[ axis ] - boxes[ b ].min[ axis ] );
	const middle = Math.floor( ids.length / 2 );
	node.children = [ build( ids.slice( 0, middle ) ), build( ids.slice( middle ) ) ];
	return node;

}

const root = build( boxes.map( ( _, id ) => id ) );
const camera = new PerspectiveCamera( 65, 16 / 9, 0.1, 180 );
const matrix = new Matrix4();
const frustum = new Frustum();
const target = new Vector3();

function setView( frame, path ) {

	const angle = frame * ( path === 'turn' ? 0.075 : 0.003 );
	camera.position.set( Math.sin( angle ) * 95, 10, Math.cos( angle ) * 95 );
	target.set( 0, 10, 0 );
	if ( path === 'teleport' && frame % 17 === 0 ) camera.position.multiplyScalar( - 1 );
	if ( path === 'projection' ) {

		camera.fov = 35 + frame % 70;
		camera.aspect = frame % 2 ? 0.8 : 16 / 9;

	} else {

		camera.fov = 65;
		camera.aspect = 16 / 9;

	}

	camera.updateProjectionMatrix();
	camera.lookAt( target );
	camera.updateMatrixWorld();
	matrix.multiplyMatrices( camera.projectionMatrix, camera.matrixWorldInverse );
	frustum.setFromProjectionMatrix( matrix );

}

// Test the positive vertex for exclusion, and the negative vertex for full inclusion.
// A plane bit can be cleared only for descendants of a fully enclosed bound.
function classify( bounds, mask, first, stats ) {

	for ( let step = 0; step < 6; step ++ ) {

		const i = ( first + step ) % 6;
		if ( ( mask & ( 1 << i ) ) === 0 ) continue;
		const plane = frustum.planes[ i ];
		const n = plane.normal;
		stats.planes ++;
		const maximum = n.x * ( n.x >= 0 ? bounds.max.x : bounds.min.x ) +
			n.y * ( n.y >= 0 ? bounds.max.y : bounds.min.y ) +
			n.z * ( n.z >= 0 ? bounds.max.z : bounds.min.z ) + plane.constant;
		if ( maximum < 0 ) return { mask: - 1, plane: i };
		const minimum = n.x * ( n.x >= 0 ? bounds.min.x : bounds.max.x ) +
			n.y * ( n.y >= 0 ? bounds.min.y : bounds.max.y ) +
			n.z * ( n.z >= 0 ? bounds.min.z : bounds.max.z ) + plane.constant;
		if ( minimum >= 0 ) mask &= ~ ( 1 << i );

	}

	return { mask, plane: first };

}

function select( mode ) {

	const stats = { planes: 0, nodes: 0, leaves: 0 };
	const visible = [];
	if ( mode === 'flat' ) {

		for ( let id = 0; id < boxes.length; id ++ ) {

			// Match Three.js intersectsBox's early-out cost, without inclusion tests.
			let inside = true;
			for ( const plane of frustum.planes ) {

				stats.planes ++;
				const n = plane.normal;
				const b = boxes[ id ];
				if ( n.x * ( n.x >= 0 ? b.max.x : b.min.x ) +
					n.y * ( n.y >= 0 ? b.max.y : b.min.y ) +
					n.z * ( n.z >= 0 ? b.max.z : b.min.z ) + plane.constant < 0 ) {

					inside = false;
					break;

				}

			}

			if ( inside ) visible.push( id );

		}

	} else {

		visit( root, 63 );

	}

	return { visible, stats };

	function visit( node, mask ) {

		stats.nodes ++;
		const result = classify( node.bounds, mask, mode === 'coherent' ? node.rejectPlane : 0, stats );
		if ( result.mask === - 1 ) {

			if ( mode === 'coherent' ) node.rejectPlane = result.plane;
			return;

		}

		if ( result.mask === 0 ) {

			for ( const id of node.ids ) visible.push( id );

		} else if ( node.children ) {

			for ( const child of node.children ) visit( child, result.mask );

		} else {

			for ( const id of node.ids ) {

				stats.leaves ++;
				if ( classify( boxes[ id ], result.mask, 0, stats ).mask !== - 1 ) visible.push( id );

			}

		}

	}

}

// Validate the complete candidate set, including newly revealed IDs. A previous
// visible list alone cannot find these. Check projection changes and camera cuts.
for ( const path of [ 'walk', 'turn', 'teleport', 'projection' ] ) {

	for ( let frame = 0; frame < 180; frame ++ ) {

		setView( frame, path );
		const expected = boxes.map( ( box, id ) => frustum.intersectsBox( box ) ? id : - 1 ).filter( id => id !== - 1 );
		for ( const mode of [ 'flat', 'hierarchy', 'coherent' ] ) {

			assert.deepEqual( select( mode ).visible.sort( ( a, b ) => a - b ), expected, `${ path }/${ frame }/${ mode }` );

		}

	}

}

// Sequential CPU runs, warmed separately. Timings include output enumeration;
// exclude camera updates, correctness assertions, tree construction and sorting.
const rows = [];
for ( const mode of [ 'flat', 'hierarchy', 'coherent' ] ) {

	let elapsed = 0, planes = 0, visible = 0;
	for ( let frame = 0; frame < 1500; frame ++ ) {

		setView( frame, 'walk' );
		const start = performance.now();
		const result = select( mode );
		const duration = performance.now() - start;
		if ( frame < 300 ) continue;
		elapsed += duration;
		planes += result.stats.planes;
		visible += result.visible.length;

	}

	rows.push( { mode, 'CPU ms/update': ( elapsed / 1200 ).toFixed( 4 ),
		'planes/update': Math.round( planes / 1200 ), 'candidates/update': Math.round( visible / 1200 ) } );

}

console.log( 'PASS: 720 views × 3 modes match Frustum.intersectsBox (4096 static AABBs).' );
console.table( rows );
console.log( 'Plane counts compare algorithmic work, not equal-cost operations. No GPU/rendering speedup is claimed.' );
