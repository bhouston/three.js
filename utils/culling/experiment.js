// CPU candidate-selection experiment. No rendering or frame-rate measurements.
import assert from 'node:assert/strict';
import { Box3, Frustum, Matrix4, PerspectiveCamera, Plane, Vector3 } from '../../src/Three.Core.js';

const objects = [];
let seed = 123456789;
function random() {

	seed = ( Math.imul( seed, 1664525 ) + 1013904223 ) >>> 0;
	return seed / 4294967296;

}

for ( let id = 0; id < 32768; id ++ ) {

	const center = new Vector3( random() * 2000 - 1000, random() * 40, random() * 2000 - 1000 );
	const size = new Vector3().setScalar( 1 + random() * 15 );
	objects.push( { id, box: new Box3().setFromCenterAndSize( center, size ) } );

}

function build( items ) {

	const box = new Box3();
	for ( const item of items ) box.union( item.box );
	if ( items.length <= 8 ) return { box, items };
	const size = box.getSize( new Vector3() );
	const axis = size.x >= size.z ? 'x' : 'z';
	items.sort( ( a, b ) => ( a.box.min[ axis ] + a.box.max[ axis ] ) - ( b.box.min[ axis ] + b.box.max[ axis ] ) );
	const middle = items.length >> 1;
	return { box, children: [ build( items.slice( 0, middle ) ), build( items.slice( middle ) ) ] };

}

const root = build( objects.slice() );
const point = new Vector3();

// A cleared bit is a containment proof for this query only, never for the next camera.
function classify( box, frustum, mask, stats ) {

	for ( let i = 0; i < 6; i ++ ) {

		if ( ( mask & ( 1 << i ) ) === 0 ) continue;
		stats.planes ++;
		const plane = frustum.planes[ i ];
		const n = plane.normal;
		point.set( n.x >= 0 ? box.max.x : box.min.x, n.y >= 0 ? box.max.y : box.min.y, n.z >= 0 ? box.max.z : box.min.z );
		if ( plane.distanceToPoint( point ) < 0 ) return - 1;
		point.set( n.x >= 0 ? box.min.x : box.max.x, n.y >= 0 ? box.min.y : box.max.y, n.z >= 0 ? box.min.z : box.max.z );
		if ( plane.distanceToPoint( point ) >= 0 ) mask &= ~ ( 1 << i );

	}

	return mask;

}

function query( node, frustum, mask, stats, result ) {

	stats.nodes ++;
	mask = classify( node.box, frustum, mask, stats );
	if ( mask === - 1 ) return;
	if ( node.items ) {

		for ( const item of node.items ) {

			stats.leaves ++;
			if ( classify( item.box, frustum, mask, stats ) !== - 1 ) result.push( item.id );

		}

	} else {

		for ( const child of node.children ) query( child, frustum, mask, stats, result );

	}

}

// Deterministic containment, rejection and touching-plane checks.
const enclosing = new Frustum( ...[[ 1, 0, 0 ], [ - 1, 0, 0 ], [ 0, 1, 0 ], [ 0, - 1, 0 ], [ 0, 0, 1 ], [ 0, 0, - 1 ]].map( n => new Plane( new Vector3( ...n ), 2000 ) ) );
assert.equal( classify( root.box, enclosing, 63, { planes: 0 } ), 0 );
const all = [];
query( root, enclosing, 63, { nodes: 0, leaves: 0, planes: 0 }, all );
assert.equal( all.length, objects.length );
assert.equal( new Set( all ).size, objects.length );
enclosing.planes[ 0 ].constant = - 5000;
const none = [];
query( root, enclosing, 63, { nodes: 0, leaves: 0, planes: 0 }, none );
assert.deepEqual( none, [] );
enclosing.planes[ 0 ].constant = - 1;
const touching = new Box3( new Vector3(), new Vector3( 1, 1, 1 ) );
assert.equal( enclosing.intersectsBox( touching ), true );
assert.notEqual( classify( touching, enclosing, 63, { planes: 0 } ), - 1 );

function run( name, views ) {

	const stats = { nodes: 0, leaves: 0, planes: 0 };
	let flatPlanes = 0;
	let candidates = 0;
	for ( const camera of views ) {

		camera.updateMatrixWorld();
		const frustum = new Frustum().setFromProjectionMatrix( new Matrix4().multiplyMatrices( camera.projectionMatrix, camera.matrixWorldInverse ) );
		const expected = objects.filter( item => frustum.intersectsBox( item.box ) ).map( item => item.id );
		const flatStats = { planes: 0 };
		const flat = objects.filter( item => classify( item.box, frustum, 63, flatStats ) !== - 1 ).map( item => item.id );
		const actual = [];
		query( root, frustum, 63, stats, actual );
		actual.sort( ( a, b ) => a - b );
		assert.deepEqual( flat, expected );
		assert.deepEqual( actual, expected );
		flatPlanes += flatStats.planes;
		candidates += actual.length;

	}

	return { name, views: views.length, objects: objects.length, candidates, flatBoxTests: objects.length * views.length, flatPlaneTests: flatPlanes, hierarchyNodeTests: stats.nodes, hierarchyLeafVisits: stats.leaves, hierarchyPlaneTests: stats.planes };

}

function cameraAt( position, direction, fov = 75, far = 350 ) {

	const camera = new PerspectiveCamera( fov, 1, 0.1, far );
	camera.position.copy( position );
	if ( Math.abs( direction.y ) === 1 ) camera.up.set( 0, 0, direction.y );
	camera.lookAt( position.clone().add( direction ) );
	return camera;

}

const walk = Array.from( { length: 120 }, ( _, i ) => cameraAt( new Vector3( i * 2, 4, 0 ), new Vector3( Math.sin( i * 0.01 ), 0, - Math.cos( i * 0.01 ) ) ) );
const cube = [[ 1, 0, 0 ], [ - 1, 0, 0 ], [ 0, 1, 0 ], [ 0, - 1, 0 ], [ 0, 0, 1 ], [ 0, 0, - 1 ]].map( d => cameraAt( new Vector3( 0, 4, 0 ), new Vector3( ...d ), 90 ) );
const stress = Array.from( { length: 80 }, () => cameraAt( new Vector3( random() * 4000 - 2000, random() * 200 - 100, random() * 4000 - 2000 ), new Vector3( random() - 0.5, random() - 0.5, random() - 0.5 ), 20 + random() * 140, 50 + random() * 4000 ) );
console.log( JSON.stringify( [ run( 'walk', walk ), run( 'cubemap', cube ), run( 'teleports and wide views', stress ) ], null, 2 ) );
