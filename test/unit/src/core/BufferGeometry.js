import { describe, test, expect } from 'vitest';
import { BufferGeometry } from '@src/core/BufferGeometry.js';

import {
	BufferAttribute,
	Float16BufferAttribute,
	Uint16BufferAttribute,
	Uint32BufferAttribute
} from '@src/core/BufferAttribute.js';
import { Vector3 } from '@src/math/Vector3.js';
import { Matrix4 } from '@src/math/Matrix4.js';
import { Quaternion } from '@src/math/Quaternion.js';
import { Sphere } from '@src/math/Sphere.js';
import { x, y, z } from '@test-utils/math-constants.js';
import { EventDispatcher } from '@src/core/EventDispatcher.js';
import { toHalfFloat } from '@src/extras/DataUtils.js';
import { StaticDrawUsage, FloatType } from '@src/constants.js';

const DegToRad = Math.PI / 180;

function bufferAttributeEquals( a, b, tolerance ) {

	tolerance = tolerance || 0.0001;

	if ( a.count !== b.count || a.itemSize !== b.itemSize ) {

		return false;

	}

	for ( let i = 0, il = a.count * a.itemSize; i < il; i ++ ) {

		const delta = Math.abs( a.array[ i ] - b.array[ i ] );
		if ( delta > tolerance ) {

			return false;

		}

	}

	return true;

}

function getBBForVertices( vertices ) {

	const geometry = new BufferGeometry();

	geometry.setAttribute( 'position', new BufferAttribute( new Float32Array( vertices ), 3 ) );
	geometry.computeBoundingBox();

	return geometry.boundingBox;

}

function getBSForVertices( vertices ) {

	const geometry = new BufferGeometry();

	geometry.setAttribute( 'position', new BufferAttribute( new Float32Array( vertices ), 3 ) );
	geometry.computeBoundingSphere();

	return geometry.boundingSphere;

}

function getNormalsForVertices( vertices ) {

	const geometry = new BufferGeometry();

	geometry.setAttribute( 'position', new BufferAttribute( new Float32Array( vertices ), 3 ) );

	geometry.computeVertexNormals();

	expect( geometry.attributes.normal !== undefined ).toBeTruthy();

	return geometry.attributes.normal.array;

}

describe( 'Core', () => {

	describe( 'BufferGeometry', () => {

		test( 'Extending', () => {

			const object = new BufferGeometry();
			expect( object instanceof EventDispatcher ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new BufferGeometry();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new BufferGeometry();
			expect( object.type === 'BufferGeometry' ).toBeTruthy();

		} );

		test( 'isBufferGeometry', () => {

			const object = new BufferGeometry();
			expect( object.isBufferGeometry ).toBeTruthy();

		} );

		test( 'setIndex/getIndex', () => {

			const a = new BufferGeometry();
			const uint16 = [ 1, 2, 3 ];
			const uint32 = [ 65535, 65536, 65537 ];
			const str = 'foo';

			a.setIndex( uint16 );
			expect( a.getIndex() instanceof Uint16BufferAttribute ).toBeTruthy();
			expect( a.getIndex().array ).toEqual( new Uint16Array( uint16 ) );

			a.setIndex( uint32 );
			expect( a.getIndex() instanceof Uint32BufferAttribute ).toBeTruthy();
			expect( a.getIndex().array ).toEqual( new Uint32Array( uint32 ) );

			a.setIndex( str );
			expect( a.getIndex() ).toBe( str );

		} );

		test( 'set / delete Attribute', () => {

			const geometry = new BufferGeometry();
			const attributeName = 'position';

			expect( geometry.attributes[ attributeName ] === undefined ).toBeTruthy();

			geometry.setAttribute( attributeName, new BufferAttribute( new Float32Array( [ 1, 2, 3 ], 1 ) ) );

			expect( geometry.attributes[ attributeName ] !== undefined ).toBeTruthy();

			geometry.deleteAttribute( attributeName );

			expect( geometry.attributes[ attributeName ] === undefined ).toBeTruthy();

		} );

		test( 'addGroup', () => {

			const a = new BufferGeometry();
			const expected = [
				{
					start: 0,
					count: 1,
					materialIndex: 0
				},
				{
					start: 1,
					count: 2,
					materialIndex: 2
				}
			];

			a.addGroup( 0, 1, 0 );
			a.addGroup( 1, 2, 2 );

			expect( a.groups ).toEqual( expected );

			a.clearGroups();
			expect( a.groups.length ).toBe( 0 );

		} );

		test( 'setDrawRange', () => {

			const a = new BufferGeometry();

			a.setDrawRange( 1.0, 7 );

			expect( a.drawRange ).toEqual( {
				start: 1,
				count: 7
			} );

		} );

		test( 'applyMatrix4', () => {

			const geometry = new BufferGeometry();
			geometry.setAttribute( 'position', new BufferAttribute( new Float32Array( 6 ), 3 ) );

			const matrix = new Matrix4().set(
				1, 0, 0, 1.5,
				0, 1, 0, - 2,
				0, 0, 1, 3,
				0, 0, 0, 1
			);
			geometry.applyMatrix4( matrix );

			const position = geometry.attributes.position.array;
			const m = matrix.elements;
			expect( position[ 0 ] === m[ 12 ] && position[ 1 ] === m[ 13 ] && position[ 2 ] === m[ 14 ] ).toBeTruthy();
			expect( position[ 3 ] === m[ 12 ] && position[ 4 ] === m[ 13 ] && position[ 5 ] === m[ 14 ] ).toBeTruthy();
			expect( geometry.attributes.position.version === 1 ).toBeTruthy();

		} );

		test( 'applyQuaternion', () => {

			const geometry = new BufferGeometry();
			geometry.setAttribute( 'position', new BufferAttribute( new Float32Array( [ 1, 2, 3, 4, 5, 6 ] ), 3 ) );

			const q = new Quaternion( 0.5, 0.5, 0.5, 0.5 );
			geometry.applyQuaternion( q );

			const pos = geometry.attributes.position.array;

			// geometry was rotated around the (1, 1, 1) axis.
			expect( pos[ 0 ] === 3 && pos[ 1 ] === 1 && pos[ 2 ] === 2 &&
				pos[ 3 ] === 6 && pos[ 4 ] === 4 && pos[ 5 ] === 5 ).toBeTruthy();

		} );

		test( 'rotateX/Y/Z', () => {

			const geometry = new BufferGeometry();
			geometry.setAttribute( 'position', new BufferAttribute( new Float32Array( [ 1, 2, 3, 4, 5, 6 ] ), 3 ) );

			const pos = geometry.attributes.position.array;

			geometry.rotateX( 180 * DegToRad );

			// object was rotated around x so all items should be flipped but the x ones
			expect( pos[ 0 ] === 1 && pos[ 1 ] === - 2 && pos[ 2 ] === - 3 &&
				pos[ 3 ] === 4 && pos[ 4 ] === - 5 && pos[ 5 ] === - 6 ).toBeTruthy();

			geometry.rotateY( 180 * DegToRad );

			// vertices were rotated around y so all items should be flipped again but the y ones
			expect( pos[ 0 ] === - 1 && pos[ 1 ] === - 2 && pos[ 2 ] === 3 &&
				pos[ 3 ] === - 4 && pos[ 4 ] === - 5 && pos[ 5 ] === 6 ).toBeTruthy();

			geometry.rotateZ( 180 * DegToRad );

			// vertices were rotated around z so all items should be flipped again but the z ones
			expect( pos[ 0 ] === 1 && pos[ 1 ] === 2 && pos[ 2 ] === 3 &&
				pos[ 3 ] === 4 && pos[ 4 ] === 5 && pos[ 5 ] === 6 ).toBeTruthy();

		} );

		test( 'translate', () => {

			const geometry = new BufferGeometry();
			geometry.setAttribute( 'position', new BufferAttribute( new Float32Array( [ 1, 2, 3, 4, 5, 6 ] ), 3 ) );

			const pos = geometry.attributes.position.array;

			geometry.translate( 10, 20, 30 );

			expect( pos[ 0 ] === 11 && pos[ 1 ] === 22 && pos[ 2 ] === 33 &&
				pos[ 3 ] === 14 && pos[ 4 ] === 25 && pos[ 5 ] === 36 ).toBeTruthy();

		} );

		test( 'scale', () => {

			const geometry = new BufferGeometry();
			geometry.setAttribute( 'position', new BufferAttribute( new Float32Array( [ - 1, - 1, - 1, 2, 2, 2 ] ), 3 ) );

			const pos = geometry.attributes.position.array;

			geometry.scale( 1, 2, 3 );

			expect( pos[ 0 ] === - 1 && pos[ 1 ] === - 2 && pos[ 2 ] === - 3 &&
				pos[ 3 ] === 2 && pos[ 4 ] === 4 && pos[ 5 ] === 6 ).toBeTruthy();

		} );

		test( 'lookAt', () => {

			const a = new BufferGeometry();
			const vertices = new Float32Array( [
				- 1.0, - 1.0, 1.0,
				1.0, - 1.0, 1.0,
				1.0, 1.0, 1.0,

				1.0, 1.0, 1.0,
				- 1.0, 1.0, 1.0,
				- 1.0, - 1.0, 1.0
			] );
			a.setAttribute( 'position', new BufferAttribute( vertices, 3 ) );

			const sqrt = Math.sqrt( 2 );
			const expected = new BufferAttribute( new Float32Array( [
				1, 0, - sqrt,
				- 1, 0, - sqrt,
				- 1, sqrt, 0,

				- 1, sqrt, 0,
				1, sqrt, 0,
				1, 0, - sqrt
			] ), 3 );

			a.lookAt( new Vector3( 0, 1, - 1 ) );

			expect( bufferAttributeEquals( a.attributes.position, expected ) ).toBeTruthy();

		} );

		test( 'center', () => {

			const geometry = new BufferGeometry();
			geometry.setAttribute( 'position', new BufferAttribute( new Float32Array( [
				- 1, - 1, - 1,
				1, 1, 1,
				4, 4, 4
			] ), 3 ) );

			geometry.center();

			const pos = geometry.attributes.position.array;

			// the boundingBox should go from (-1, -1, -1) to (4, 4, 4) so it has a size of (5, 5, 5)
			// after centering it the vertices should be placed between (-2.5, -2.5, -2.5) and (2.5, 2.5, 2.5)
			expect( pos[ 0 ] === - 2.5 && pos[ 1 ] === - 2.5 && pos[ 2 ] === - 2.5 &&
				pos[ 3 ] === - 0.5 && pos[ 4 ] === - 0.5 && pos[ 5 ] === - 0.5 &&
				pos[ 6 ] === 2.5 && pos[ 7 ] === 2.5 && pos[ 8 ] === 2.5 ).toBeTruthy();

		} );

		test( 'computeBoundingBox', () => {

			let bb = getBBForVertices( [ - 1, - 2, - 3, 13, - 2, - 3.5, - 1, - 20, 0, - 4, 5, 6 ] );

			expect( bb.min.x === - 4 && bb.min.y === - 20 && bb.min.z === - 3.5 ).toBeTruthy();
			expect( bb.max.x === 13 && bb.max.y === 5 && bb.max.z === 6 ).toBeTruthy();

			bb = getBBForVertices( [ - 1, - 1, - 1 ] );

			expect( bb.min.x === bb.max.x && bb.min.y === bb.max.y && bb.min.z === bb.max.z ).toBeTruthy();
			expect( bb.min.x === - 1 && bb.min.y === - 1 && bb.min.z === - 1 ).toBeTruthy();

		} );

		test( 'computeBoundingSphere', () => {

			let bs = getBSForVertices( [ - 10, 0, 0, 10, 0, 0 ] );

			expect( bs.radius === 10 ).toBeTruthy();
			expect( bs.center.x === 0 && bs.center.y === 0 && bs.center.y === 0 ).toBeTruthy();

			bs = getBSForVertices( [ - 5, 11, - 3, 5, - 11, 3 ] );
			const radius = new Vector3( 5, 11, 3 ).length();

			expect( bs.radius === radius ).toBeTruthy();
			expect( bs.center.x === 0 && bs.center.y === 0 && bs.center.y === 0 ).toBeTruthy();

		} );

		const toHalfFloatArray = ( f32Array ) => {

			const f16Array = new Uint16Array( f32Array.length );
			for ( let i = 0, n = f32Array.length; i < n; ++ i ) {

				f16Array[ i ] = toHalfFloat( f32Array[ i ] );

			}

			return f16Array;

		};

		test( 'computeBoundingBox - Float16', () => {

			const vertices = [ - 1, - 2, - 3, 13, - 2, - 3.5, - 1, - 20, 0, - 4, 5, 6 ];
			const geometry = new BufferGeometry();

			geometry.setAttribute( 'position', new Float16BufferAttribute( toHalfFloatArray( vertices ), 3 ) );
			geometry.computeBoundingBox();

			let bb = geometry.boundingBox;

			expect( bb.min.x === - 4 && bb.min.y === - 20 && bb.min.z === - 3.5 ).toBeTruthy();
			expect( bb.max.x === 13 && bb.max.y === 5 && bb.max.z === 6 ).toBeTruthy();

			bb = getBBForVertices( [ - 1, - 1, - 1 ] );

			expect( bb.min.x === bb.max.x && bb.min.y === bb.max.y && bb.min.z === bb.max.z ).toBeTruthy();
			expect( bb.min.x === - 1 && bb.min.y === - 1 && bb.min.z === - 1 ).toBeTruthy();

		} );

		test( 'computeBoundingSphere - Float16', () => {

			const vertices = [ - 10, 0, 0, 10, 0, 0 ];
			const geometry = new BufferGeometry();

			geometry.setAttribute( 'position', new Float16BufferAttribute( toHalfFloatArray( vertices ), 3 ) );
			geometry.computeBoundingSphere();

			let bs = geometry.boundingSphere;

			expect( bs.radius === 10 ).toBeTruthy();
			expect( bs.center.x === 0 && bs.center.y === 0 && bs.center.y === 0 ).toBeTruthy();

			bs = getBSForVertices( [ - 5, 11, - 3, 5, - 11, 3 ] );
			const radius = new Vector3( 5, 11, 3 ).length();

			expect( bs.radius === radius ).toBeTruthy();
			expect( bs.center.x === 0 && bs.center.y === 0 && bs.center.y === 0 ).toBeTruthy();

		} );

		test( 'computeVertexNormals', () => {

			// get normals for a counter clockwise created triangle
			let normals = getNormalsForVertices( [ - 1, 0, 0, 1, 0, 0, 0, 1, 0 ] );

			expect( normals[ 0 ] === 0 && normals[ 1 ] === 0 && normals[ 2 ] === 1 ).toBeTruthy();

			expect( normals[ 3 ] === 0 && normals[ 4 ] === 0 && normals[ 5 ] === 1 ).toBeTruthy();

			expect( normals[ 6 ] === 0 && normals[ 7 ] === 0 && normals[ 8 ] === 1 ).toBeTruthy();

			// get normals for a clockwise created triangle
			normals = getNormalsForVertices( [ 1, 0, 0, - 1, 0, 0, 0, 1, 0 ] );

			expect( normals[ 0 ] === 0 && normals[ 1 ] === 0 && normals[ 2 ] === - 1 ).toBeTruthy();

			expect( normals[ 3 ] === 0 && normals[ 4 ] === 0 && normals[ 5 ] === - 1 ).toBeTruthy();

			expect( normals[ 6 ] === 0 && normals[ 7 ] === 0 && normals[ 8 ] === - 1 ).toBeTruthy();

			normals = getNormalsForVertices( [ 0, 0, 1, 0, 0, - 1, 1, 1, 0 ] );

			// the triangle is rotated by 45 degrees to the right so the normals of the three vertices
			// should point to (1, -1, 0).normalized(). The simplest solution is to check against a normalized
			// vector (1, -1, 0) but you will get calculation errors because of floating calculations so another
			// valid technique is to create a vector which stands in 90 degrees to the normals and calculate the
			// dot product which is the cos of the angle between them. This should be < floating calculation error
			// which can be taken from Number.EPSILON
			const direction = new Vector3( 1, 1, 0 ).normalize(); // a vector which should have 90 degrees difference to normals
			const difference = direction.dot( new Vector3( normals[ 0 ], normals[ 1 ], normals[ 2 ] ) );
			expect( difference < Number.EPSILON ).toBeTruthy();

			// get normals for a line should be NAN because you need min a triangle to calculate normals
			normals = getNormalsForVertices( [ 1, 0, 0, - 1, 0, 0 ] );
			for ( let i = 0; i < normals.length; i ++ ) {

				expect( ! normals[ i ] ).toBeTruthy();

			}

		} );

		test( 'computeVertexNormals (indexed)', () => {

			const sqrt = 0.5 * Math.sqrt( 2 );
			const normal = new BufferAttribute( new Float32Array( [
				- 1, 0, 0, - 1, 0, 0, - 1, 0, 0,
				sqrt, sqrt, 0, sqrt, sqrt, 0, sqrt, sqrt, 0
			] ), 3 );
			const position = new BufferAttribute( new Float32Array( [
				0.5, 0.5, 0.5, 0.5, 0.5, - 0.5, 0.5, - 0.5, 0.5,
				0.5, - 0.5, - 0.5, - 0.5, 0.5, - 0.5, - 0.5, 0.5, 0.5
			] ), 3 );
			// the index buffer defines the same two triangles but in counter-clockwise order,
			// which should result in flipped normals.
			const flippedNormals = normal.clone().applyMatrix4( new Matrix4().makeScale( - 1, - 1, - 1 ) );
			const index = new BufferAttribute( new Uint16Array( [
				0, 2, 1, 3, 5, 4
			] ), 1 );

			let a = new BufferGeometry();
			a.setAttribute( 'position', position );
			a.computeVertexNormals();
			expect( bufferAttributeEquals( normal, a.getAttribute( 'normal' ) ) ).toBeTruthy();

			// a second time to see if the existing normals get properly deleted
			a.computeVertexNormals();
			expect( bufferAttributeEquals( normal, a.getAttribute( 'normal' ) ) ).toBeTruthy();

			// indexed geometry
			a = new BufferGeometry();
			a.setAttribute( 'position', position );
			a.setIndex( index );
			a.computeVertexNormals();
			expect( bufferAttributeEquals( flippedNormals, a.getAttribute( 'normal' ) ) ).toBeTruthy();

		} );

		test( 'toNonIndexed', () => {

			const geometry = new BufferGeometry();
			const vertices = new Float32Array( [
				0.5, 0.5, 0.5, 0.5, 0.5, - 0.5, 0.5, - 0.5, 0.5, 0.5, - 0.5, - 0.5
			] );
			const index = new BufferAttribute( new Uint16Array( [ 0, 2, 1, 2, 3, 1 ] ) );
			const expected = new Float32Array( [
				0.5, 0.5, 0.5, 0.5, - 0.5, 0.5, 0.5, 0.5, - 0.5,
				0.5, - 0.5, 0.5, 0.5, - 0.5, - 0.5, 0.5, 0.5, - 0.5
			] );

			geometry.setAttribute( 'position', new BufferAttribute( vertices, 3 ) );
			geometry.setIndex( index );

			const nonIndexed = geometry.toNonIndexed();

			expect( nonIndexed.getAttribute( 'position' ).array ).toEqual( expected );

		} );

		test( 'toJSON', () => {

			const index = new BufferAttribute( new Uint16Array( [ 0, 1, 2, 3 ] ), 1 );
			const attribute1 = new BufferAttribute( new Uint16Array( [ 1, 3, 5, 7 ] ), 1 );
			attribute1.name = 'attribute1';
			const a = new BufferGeometry();
			a.name = 'JSONQUnit.test';
			// a.parameters = { "placeholder": 0 };
			a.setAttribute( 'attribute1', attribute1 );
			a.setIndex( index );
			a.addGroup( 0, 1, 2 );
			a.boundingSphere = new Sphere( new Vector3( x, y, z ), 0.5 );
			let j = a.toJSON();
			const gold = {
				'metadata': {
					'version': 4.7,
					'type': 'BufferGeometry',
					'generator': 'BufferGeometry.toJSON'
				},
				'uuid': a.uuid,
				'type': 'BufferGeometry',
				'name': 'JSONQUnit.test',
				'data': {
					'attributes': {
						'attribute1': {
							'itemSize': 1,
							'type': 'Uint16Array',
							'array': [ 1, 3, 5, 7 ],
							'normalized': false,
							'name': 'attribute1',
							'usage': StaticDrawUsage,
							'gpuType': FloatType
						}
					},
					'index': {
						'type': 'Uint16Array',
						'array': [ 0, 1, 2, 3 ]
					},
					'groups': [
						{
							'start': 0,
							'count': 1,
							'materialIndex': 2
						}
					],
					'boundingSphere': {
						'center': [ 2, 3, 4 ],
						'radius': 0.5
					}
				}
			};

			expect( j ).toEqual( gold );

			// add morphAttributes
			a.morphAttributes.attribute1 = [];
			a.morphAttributes.attribute1.push( attribute1.clone() );
			j = a.toJSON();
			gold.data.morphAttributes = {
				'attribute1': [ {
					'itemSize': 1,
					'type': 'Uint16Array',
					'array': [ 1, 3, 5, 7 ],
					'normalized': false,
					'name': 'attribute1',
					'usage': StaticDrawUsage,
					'gpuType': FloatType
				} ]
			};
			gold.data.morphTargetsRelative = false;

			expect( j ).toEqual( gold );

		} );

		test( 'clone', () => {

			const a = new BufferGeometry();
			a.setAttribute( 'attribute1', new BufferAttribute( new Float32Array( [ 1, 2, 3, 4, 5, 6 ] ), 3 ) );
			a.setAttribute( 'attribute2', new BufferAttribute( new Float32Array( [ 0, 1, 3, 5, 6 ] ), 1 ) );
			a.addGroup( 0, 1, 2 );
			a.computeBoundingBox();
			a.computeBoundingSphere();
			a.setDrawRange( 0, 1 );
			const b = a.clone();

			expect( a ).not.toBe( b );
			expect( a.id ).not.toBe( b.id );

			expect( Object.keys( a.attributes ).count ).toBe( Object.keys( b.attributes ).count );
			expect( bufferAttributeEquals( a.getAttribute( 'attribute1' ), b.getAttribute( 'attribute1' ) ) ).toBeTruthy();
			expect( bufferAttributeEquals( a.getAttribute( 'attribute2' ), b.getAttribute( 'attribute2' ) ) ).toBeTruthy();

			expect( a.groups ).toEqual( b.groups );

			expect( a.boundingBox.equals( b.boundingBox ) ).toBeTruthy();
			expect( a.boundingSphere.equals( b.boundingSphere ) ).toBeTruthy();

			expect( a.drawRange.start ).toBe( b.drawRange.start );
			expect( a.drawRange.count ).toBe( b.drawRange.count );

		} );

		test( 'copy', () => {

			const geometry = new BufferGeometry();
			geometry.setAttribute( 'attrName', new BufferAttribute( new Float32Array( [ 1, 2, 3, 4, 5, 6 ] ), 3 ) );
			geometry.setAttribute( 'attrName2', new BufferAttribute( new Float32Array( [ 0, 1, 3, 5, 6 ] ), 1 ) );

			const copy = new BufferGeometry().copy( geometry );

			expect( copy !== geometry && geometry.id !== copy.id ).toBeTruthy();

			Object.keys( geometry.attributes ).forEach( function ( key ) {

				const attribute = geometry.attributes[ key ];
				expect( attribute !== undefined ).toBeTruthy();

				for ( let i = 0; i < attribute.array.length; i ++ ) {

					expect( attribute.array[ i ] === copy.attributes[ key ].array[ i ] ).toBeTruthy();

				}

			} );

		} );

		test( 'dispose', () => {

			const object = new BufferGeometry();
			object.dispose();

		} );

	} );

} );
