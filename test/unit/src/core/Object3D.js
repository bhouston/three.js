import { describe, test, expect } from 'vitest';
import { Object3D } from '@src/core/Object3D.js';

import { Vector3 } from '@src/math/Vector3.js';
import { Euler } from '@src/math/Euler.js';
import { Quaternion } from '@src/math/Quaternion.js';
import { Matrix4 } from '@src/math/Matrix4.js';
import {
	x,
	y,
	z,
	w,
	eps
} from '@test-utils/math-constants.js';
import { EventDispatcher } from '@src/core/EventDispatcher.js';

const matrixEquals4 = ( a, b ) => {

	for ( let i = 0; i < 16; i ++ ) {

		if ( Math.abs( a.elements[ i ] - b.elements[ i ] ) >= eps ) {

			return false;

		}

	}

	return true;

};

describe( 'Core', () => {

	describe( 'Object3D', () => {

		const RadToDeg = 180 / Math.PI;

		const eulerEquals = function ( a, b, tolerance ) {

			tolerance = tolerance || 0.0001;

			if ( a.order != b.order ) {

				return false;

			}

			return (
				Math.abs( a.x - b.x ) <= tolerance &&
				Math.abs( a.y - b.y ) <= tolerance &&
				Math.abs( a.z - b.z ) <= tolerance
			);

		};

		test( 'Extending', () => {

			const object = new Object3D();
			expect( object instanceof EventDispatcher ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new Object3D();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new Object3D();
			expect( object.type === 'Object3D' ).toBeTruthy();

		} );

		test( 'DEFAULT_UP', () => {

			const currentDefaultUp = new Vector3().copy( Object3D.DEFAULT_UP );
			const v = new Vector3();

			try {

				expect( Object3D.DEFAULT_UP ).toEqual( v.set( 0, 1, 0 ) );

				const object = new Object3D();

				expect( object.up ).toEqual( v.set( 0, 1, 0 ) );

				Object3D.DEFAULT_UP.set( 0, 0, 1 );

				const object2 = new Object3D();

				expect( object2.up ).toEqual( v.set( 0, 0, 1 ) );

			} finally {

				Object3D.DEFAULT_UP.copy( currentDefaultUp );

			}

		} );

		test( 'DEFAULT_MATRIX_AUTO_UPDATE', () => {

			const currentDefaultMatrixAutoUpdate = Object3D.DEFAULT_MATRIX_AUTO_UPDATE;

			try {

				expect( currentDefaultMatrixAutoUpdate ).toBe( true );

				const object = new Object3D();

				expect( object.matrixAutoUpdate ).toBe( true );

				Object3D.DEFAULT_MATRIX_AUTO_UPDATE = false;

				const object2 = new Object3D();

				expect( object2.matrixAutoUpdate ).toBe( false );

			} finally {

				Object3D.DEFAULT_MATRIX_AUTO_UPDATE = currentDefaultMatrixAutoUpdate;

			}

		} );

		test( 'isObject3D', () => {

			const object = new Object3D();
			expect( object.isObject3D ).toBeTruthy();

			const object2 = {};
			expect( object2.isObject3D === undefined ).toBeTruthy();

		} );

		test( 'applyMatrix4', () => {

			const a = new Object3D();
			const m = new Matrix4();
			const expectedPos = new Vector3( x, y, z );
			const expectedQuat = new Quaternion( 0.5 * Math.sqrt( 2 ), 0, 0, 0.5 * Math.sqrt( 2 ) );

			m.makeRotationX( Math.PI / 2 );
			m.setPosition( new Vector3( x, y, z ) );

			a.applyMatrix4( m );

			expect( a.position ).toEqual( expectedPos );
			expect(
				Math.abs( a.quaternion.x - expectedQuat.x ) <= eps &&
				Math.abs( a.quaternion.y - expectedQuat.y ) <= eps &&
				Math.abs( a.quaternion.z - expectedQuat.z ) <= eps
			).toBeTruthy();

		} );

		test( 'applyQuaternion', () => {

			const a = new Object3D();
			const sqrt = 0.5 * Math.sqrt( 2 );
			const quat = new Quaternion( 0, sqrt, 0, sqrt );
			const expected = new Quaternion( sqrt / 2, sqrt / 2, 0, 0 );

			a.quaternion.set( 0.25, 0.25, 0.25, 0.25 );
			a.applyQuaternion( quat );

			expect(
				Math.abs( a.quaternion.x - expected.x ) <= eps &&
				Math.abs( a.quaternion.y - expected.y ) <= eps &&
				Math.abs( a.quaternion.z - expected.z ) <= eps
			).toBeTruthy();

		} );

		test( 'setRotationFromAxisAngle', () => {

			const a = new Object3D();
			const axis = new Vector3( 0, 1, 0 );
			let angle = Math.PI;
			const expected = new Euler( - Math.PI, 0, - Math.PI );
			const euler = new Euler();

			a.setRotationFromAxisAngle( axis, angle );
			euler.setFromQuaternion( a.getWorldQuaternion( new Quaternion() ) );
			expect( eulerEquals( euler, expected ) ).toBeTruthy();

			axis.set( 1, 0, 0 );
			angle = 0;
			expected.set( 0, 0, 0 );

			a.setRotationFromAxisAngle( axis, angle );
			euler.setFromQuaternion( a.getWorldQuaternion( new Quaternion() ) );
			expect( eulerEquals( euler, expected ) ).toBeTruthy();

		} );

		test( 'setRotationFromEuler', () => {

			const a = new Object3D();
			const rotation = new Euler( ( 45 / RadToDeg ), 0, Math.PI );
			const expected = rotation.clone(); // bit obvious
			const euler = new Euler();

			a.setRotationFromEuler( rotation );
			euler.setFromQuaternion( a.getWorldQuaternion( new Quaternion() ) );
			expect( eulerEquals( euler, expected ) ).toBeTruthy();

		} );

		test( 'setRotationFromMatrix', () => {

			const a = new Object3D();
			const m = new Matrix4();
			const eye = new Vector3( 0, 0, 0 );
			const target = new Vector3( 0, 1, - 1 );
			const up = new Vector3( 0, 1, 0 );
			const euler = new Euler();

			m.lookAt( eye, target, up );
			a.setRotationFromMatrix( m );
			euler.setFromQuaternion( a.getWorldQuaternion( new Quaternion() ) );
			expect( euler.x * RadToDeg ).toNumEqual( 45 );

		} );

		test( 'setRotationFromQuaternion', () => {

			const a = new Object3D();
			const rotation = new Quaternion().setFromEuler( new Euler( Math.PI, 0, - Math.PI ) );
			const euler = new Euler();

			a.setRotationFromQuaternion( rotation );
			euler.setFromQuaternion( a.getWorldQuaternion( new Quaternion() ) );
			expect( eulerEquals( euler, new Euler( Math.PI, 0, - Math.PI ) ) ).toBeTruthy();

		} );

		test( 'rotateX', () => {

			const obj = new Object3D();
			const angleInRad = 1.562;
			obj.rotateX( angleInRad );

			expect( obj.rotation.x ).toNumEqual( angleInRad );

		} );

		test( 'rotateY', () => {

			const obj = new Object3D();
			const angleInRad = - 0.346;
			obj.rotateY( angleInRad );

			expect( obj.rotation.y ).toNumEqual( angleInRad );

		} );

		test( 'rotateZ', () => {

			const obj = new Object3D();
			const angleInRad = 1;
			obj.rotateZ( angleInRad );

			expect( obj.rotation.z ).toNumEqual( angleInRad );

		} );

		test( 'translateOnAxis', () => {

			const obj = new Object3D();
			obj.translateOnAxis( new Vector3( 1, 0, 0 ), 1 );
			obj.translateOnAxis( new Vector3( 0, 1, 0 ), 1.23 );
			obj.translateOnAxis( new Vector3( 0, 0, 1 ), - 4.56 );

			expect( obj.position ).toEqual( {
				x: 1,
				y: 1.23,
				z: - 4.56,
			} );

		} );

		test( 'translateX', () => {

			const obj = new Object3D();
			obj.translateX( 1.234 );

			expect( obj.position.x ).toNumEqual( 1.234 );

		} );

		test( 'translateY', () => {

			const obj = new Object3D();
			obj.translateY( 1.234 );

			expect( obj.position.y ).toNumEqual( 1.234 );

		} );

		test( 'translateZ', () => {

			const obj = new Object3D();
			obj.translateZ( 1.234 );

			expect( obj.position.z ).toNumEqual( 1.234 );

		} );

		test( 'localToWorld', () => {

			const v = new Vector3();
			const expectedPosition = new Vector3( 5, - 1, - 4 );

			const parent = new Object3D();
			const child = new Object3D();

			parent.position.set( 1, 0, 0 );
			parent.rotation.set( 0, Math.PI / 2, 0 );
			parent.scale.set( 2, 1, 1 );

			child.position.set( 0, 1, 0 );
			child.rotation.set( Math.PI / 2, 0, 0 );
			child.scale.set( 1, 2, 1 );

			parent.add( child );
			parent.updateMatrixWorld();

			child.localToWorld( v.set( 2, 2, 2 ) );

			expect(
				Math.abs( v.x - expectedPosition.x ) <= eps &&
				Math.abs( v.y - expectedPosition.y ) <= eps &&
				Math.abs( v.z - expectedPosition.z ) <= eps
			).toBeTruthy();

		} );

		test( 'worldToLocal', () => {

			const v = new Vector3();
			const expectedPosition = new Vector3( - 1, 0.5, - 1 );

			const parent = new Object3D();
			const child = new Object3D();

			parent.position.set( 1, 0, 0 );
			parent.rotation.set( 0, Math.PI / 2, 0 );
			parent.scale.set( 2, 1, 1 );

			child.position.set( 0, 1, 0 );
			child.rotation.set( Math.PI / 2, 0, 0 );
			child.scale.set( 1, 2, 1 );

			parent.add( child );
			parent.updateMatrixWorld();

			child.worldToLocal( v.set( 2, 2, 2 ) );

			expect(
				Math.abs( v.x - expectedPosition.x ) <= eps &&
				Math.abs( v.y - expectedPosition.y ) <= eps &&
				Math.abs( v.z - expectedPosition.z ) <= eps
			).toBeTruthy();

		} );

		test( 'lookAt', () => {

			const obj = new Object3D();
			obj.lookAt( new Vector3( 0, - 1, 1 ) );

			expect( obj.rotation.x * RadToDeg ).toNumEqual( 45 );

		} );

		test( 'add/remove/removeFromParent/clear', () => {

			const a = new Object3D();
			const child1 = new Object3D();
			const child2 = new Object3D();

			expect( a.children.length ).toBe( 0 );

			a.add( child1 );
			expect( a.children.length ).toBe( 1 );
			expect( a.children[ 0 ] ).toBe( child1 );

			a.add( child2 );
			expect( a.children.length ).toBe( 2 );
			expect( a.children[ 1 ] ).toBe( child2 );
			expect( a.children[ 0 ] ).toBe( child1 );

			a.remove( child1 );
			expect( a.children.length ).toBe( 1 );
			expect( a.children[ 0 ] ).toBe( child2 );

			a.add( child1 );
			a.remove( child1, child2 );
			expect( a.children.length ).toBe( 0 );

			child1.add( child2 );
			expect( child1.children.length ).toBe( 1 );
			a.add( child2 );
			expect( a.children.length ).toBe( 1 );
			expect( a.children[ 0 ] ).toBe( child2 );
			expect( child1.children.length ).toBe( 0 );

			a.add( child1 );
			expect( a.children.length ).toBe( 2 );
			a.clear();
			expect( a.children.length ).toBe( 0 );
			expect( child1.parent ).toBe( null );
			expect( child2.parent ).toBe( null );

			a.add( child1 );
			expect( a.children.length ).toBe( 1 );
			child1.removeFromParent();
			expect( a.children.length ).toBe( 0 );
			expect( child1.parent ).toBe( null );

		} );

		test( 'attach', () => {

			const object = new Object3D();
			const oldParent = new Object3D();
			const newParent = new Object3D();
			const expectedMatrixWorld = new Matrix4();

			// Attach to a parent

			object.position.set( 1, 2, 3 );
			object.rotation.set( Math.PI / 2, Math.PI / 3, Math.PI / 4 );
			object.scale.set( 2, 3, 4 );
			newParent.position.set( 4, 5, 6 );
			newParent.rotation.set( Math.PI / 5, Math.PI / 6, Math.PI / 7 );
			newParent.scale.set( 5, 5, 5 );

			object.updateMatrixWorld();
			newParent.updateMatrixWorld();
			expectedMatrixWorld.copy( object.matrixWorld );

			newParent.attach( object );

			expect( object.parent && object.parent == newParent &&
				oldParent.children.indexOf( object ) === - 1 ).toBeTruthy();

			expect( matrixEquals4( expectedMatrixWorld, object.matrixWorld ) ).toBeTruthy();

			// Attach to a new parent from an old parent

			object.position.set( 1, 2, 3 );
			object.rotation.set( Math.PI / 2, Math.PI / 3, Math.PI / 4 );
			object.scale.set( 2, 3, 4 );
			oldParent.position.set( 4, 5, 6 );
			oldParent.rotation.set( Math.PI / 5, Math.PI / 6, Math.PI / 7 );
			oldParent.scale.set( 5, 5, 5 );
			newParent.position.set( 7, 8, 9 );
			newParent.rotation.set( Math.PI / 8, Math.PI / 9, Math.PI / 10 );
			newParent.scale.set( 6, 6, 6 );

			oldParent.add( object );
			oldParent.updateMatrixWorld();
			newParent.updateMatrixWorld();
			expectedMatrixWorld.copy( object.matrixWorld );

			newParent.attach( object );

			expect( object.parent && object.parent == newParent &&
				newParent.children.indexOf( object ) !== - 1 &&
				oldParent.children.indexOf( object ) === - 1 ).toBeTruthy();

			expect( matrixEquals4( expectedMatrixWorld, object.matrixWorld ) ).toBeTruthy();

		} );

		test( 'getObjectById/getObjectByName/getObjectByProperty', () => {

			const parent = new Object3D();
			const childName = new Object3D();
			const childId = new Object3D(); // id = parent.id + 2
			const childNothing = new Object3D();

			parent.prop = true;
			childName.name = 'foo';
			parent.add( childName, childId, childNothing );

			expect( parent.getObjectByProperty( 'prop', true ) ).toBe( parent );
			expect( parent.getObjectByName( 'foo' ) ).toBe( childName );
			expect( parent.getObjectById( parent.id + 2 ) ).toBe( childId );
			expect( parent.getObjectByProperty( 'no-property', 'no-value' ) ).toBe( undefined );

		} );

		test( 'getObjectsByProperty', () => {

			const parent = new Object3D();
			const childName = new Object3D();
			const childNothing = new Object3D();
			const childName2 = new Object3D();
			const childName3 = new Object3D();

			parent.prop = true;
			childName.name = 'foo';
			childName2.name = 'foo';
			childName3.name = 'foo';
			childName2.add( childName3 );
			childName.add( childName2 );
			parent.add( childName, childNothing );

			expect( parent.getObjectsByProperty( 'name', 'foo' ).length ).toBe( 3 );
			expect( parent.getObjectsByProperty( 'name', 'foo' ).some( obj => obj.name !== 'foo' ) ).toBe( false );

		} );

		test( 'getWorldPosition', () => {

			const a = new Object3D();
			const b = new Object3D();
			const expectedSingle = new Vector3( x, y, z );
			const expectedParent = new Vector3( x, y, 0 );
			const expectedChild = new Vector3( x, y, 7 );
			const position = new Vector3();

			a.translateX( x );
			a.translateY( y );
			a.translateZ( z );

			expect( a.getWorldPosition( position ) ).toEqual( expectedSingle );

			// translate child and then parent
			b.translateZ( 7 );
			a.add( b );
			a.translateZ( - z );

			expect( a.getWorldPosition( position ) ).toEqual( expectedParent );
			expect( b.getWorldPosition( position ) ).toEqual( expectedChild );

		} );

		test( 'getWorldScale', () => {

			const a = new Object3D();
			const m = new Matrix4().makeScale( x, y, z );
			const expected = new Vector3( x, y, z );

			a.applyMatrix4( m );

			expect( a.getWorldScale( new Vector3() ) ).toEqual( expected );

		} );

		test( 'getWorldDirection', () => {

			const a = new Object3D();
			const expected = new Vector3( 0, - 0.5 * Math.sqrt( 2 ), 0.5 * Math.sqrt( 2 ) );
			const direction = new Vector3();

			a.lookAt( new Vector3( 0, - 1, 1 ) );
			a.getWorldDirection( direction );

			expect(
				Math.abs( direction.x - expected.x ) <= eps &&
				Math.abs( direction.y - expected.y ) <= eps &&
				Math.abs( direction.z - expected.z ) <= eps
			).toBeTruthy();

		} );

		test( 'localTransformVariableInstantiation', () => {

			const a = new Object3D();
			const b = new Object3D();
			const c = new Object3D();
			const d = new Object3D();

			a.getWorldDirection( new Vector3() );
			a.lookAt( new Vector3( 0, - 1, 1 ) );

			expect( true ).toBeTruthy();

			b.getWorldPosition( new Vector3() );
			b.lookAt( new Vector3( 0, - 1, 1 ) );

			expect( true ).toBeTruthy();

			c.getWorldQuaternion( new Quaternion() );
			c.lookAt( new Vector3( 0, - 1, 1 ) );

			expect( true ).toBeTruthy();

			d.getWorldScale( new Vector3() );
			d.lookAt( new Vector3( 0, - 1, 1 ) );

			expect( true ).toBeTruthy();

		} );

		test( 'traverse/traverseVisible/traverseAncestors', () => {

			const a = new Object3D();
			const b = new Object3D();
			const c = new Object3D();
			const d = new Object3D();
			let names = [];
			const expectedNormal = [ 'parent', 'child', 'childchild 1', 'childchild 2' ];
			const expectedVisible = [ 'parent', 'child', 'childchild 2' ];
			const expectedAncestors = [ 'child', 'parent' ];

			a.name = 'parent';
			b.name = 'child';
			c.name = 'childchild 1';
			c.visible = false;
			d.name = 'childchild 2';

			b.add( c );
			b.add( d );
			a.add( b );

			a.traverse( function ( obj ) {

				names.push( obj.name );

			} );
			expect( names ).toEqual( expectedNormal );

			names = [];
			a.traverseVisible( function ( obj ) {

				names.push( obj.name );

			} );
			expect( names ).toEqual( expectedVisible );

			names = [];
			c.traverseAncestors( function ( obj ) {

				names.push( obj.name );

			} );
			expect( names ).toEqual( expectedAncestors );

		} );

		test( 'updateMatrix', () => {

			const a = new Object3D();
			a.position.set( 2, 3, 4 );
			a.quaternion.set( 5, 6, 7, 8 );
			a.scale.set( 9, 10, 11 );

			expect( a.matrix.elements ).toEqual( [
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				0, 0, 0, 1
			] );

			a.updateMatrix();

			expect( a.matrix.elements ).toEqual( [
				- 1521, 1548, - 234, 0,
				- 520, - 1470, 1640, 0,
				1826, 44, - 1331, 0,
				2, 3, 4, 1
			] );

			expect( a.matrixWorldNeedsUpdate ).toBe( true );

		} );

		test( 'updateMatrixWorld', () => {

			const parent = new Object3D();
			const child = new Object3D();

			// -- Standard usage test

			parent.position.set( 1, 2, 3 );
			child.position.set( 4, 5, 6 );
			parent.add( child );

			parent.updateMatrixWorld();

			expect( parent.matrix.elements ).toEqual( [
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				1, 2, 3, 1
			] );

			expect( parent.matrixWorld.elements ).toEqual( [
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				1, 2, 3, 1
			] );

			expect( child.matrix.elements ).toEqual( [
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				4, 5, 6, 1
			] );

			expect( child.matrixWorld.elements ).toEqual( [
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				5, 7, 9, 1
			] );

			expect( parent.matrixWorldNeedsUpdate || child.matrixWorldNeedsUpdate ).toBe( false );

			// -- No sync between local position/quaternion/scale/matrix and world matrix test

			parent.position.set( 0, 0, 0 );
			parent.updateMatrix();

			expect( parent.matrixWorld.elements ).toEqual( [
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				1, 2, 3, 1
			] );

			// -- matrixAutoUpdate = false test

			// Resetting local and world matrices to the origin
			child.position.set( 0, 0, 0 );
			parent.updateMatrixWorld();

			parent.position.set( 1, 2, 3 );
			parent.matrixAutoUpdate = false;
			child.matrixAutoUpdate = false;
			parent.updateMatrixWorld();

			expect( parent.matrix.elements ).toEqual( [
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				0, 0, 0, 1
			] );

			expect( parent.matrixWorld.elements ).toEqual( [
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				0, 0, 0, 1
			] );

			expect( child.matrixWorld.elements ).toEqual( [
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				0, 0, 0, 1
			] );

			// -- matrixWorldAutoUpdate = false test

			parent.position.set( 3, 2, 1 );
			parent.updateMatrix();

			parent.matrixAutoUpdate = true;
			child.matrixAutoUpdate = true;
			parent.matrixWorldNeedsUpdate = true;
			child.matrixWorldAutoUpdate = false;
			parent.updateMatrixWorld();

			expect( child.matrixWorld.elements ).toEqual( [
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				0, 0, 0, 1
			] );

			// -- Propagation to children world matrices test

			child.position.set( 0, 0, 0 );
			parent.position.set( 1, 2, 3 );
			child.matrixWorldAutoUpdate = true;
			parent.updateMatrixWorld();

			expect( child.matrixWorld.elements ).toEqual( [
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				1, 2, 3, 1
			] );

			// -- force argument test

			// Resetting the local and world matrices to the origin
			child.position.set( 0, 0, 0 );
			child.matrixAutoUpdate = true;
			parent.updateMatrixWorld();

			parent.position.set( 1, 2, 3 );
			parent.updateMatrix();
			parent.matrixAutoUpdate = false;
			parent.matrixWorldNeedsUpdate = false;

			parent.updateMatrixWorld( true );

			expect( parent.matrixWorld.elements ).toEqual( [
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				1, 2, 3, 1
			] );

			// -- Restriction test: No effect to parent matrices

			// Resetting the local and world matrices to the origin
			parent.position.set( 0, 0, 0 );
			child.position.set( 0, 0, 0 );
			parent.matrixAutoUpdate = true;
			child.matrixAutoUpdate = true;
			parent.updateMatrixWorld();

			parent.position.set( 1, 2, 3 );
			child.position.set( 4, 5, 6 );

			child.updateMatrixWorld();

			expect( parent.matrix.elements ).toEqual( [
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				0, 0, 0, 1
			] );

			expect( parent.matrixWorld.elements ).toEqual( [
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				0, 0, 0, 1
			] );

			expect( child.matrixWorld.elements ).toEqual( [
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				4, 5, 6, 1
			] );

		} );

		test( 'updateWorldMatrix', () => {

			const object = new Object3D();
			const parent = new Object3D();
			const child = new Object3D();

			const m = new Matrix4();
			const v = new Vector3();

			parent.add( object );
			object.add( child );

			parent.position.set( 1, 2, 3 );
			object.position.set( 4, 5, 6 );
			child.position.set( 7, 8, 9 );

			// Update the world matrix of an object

			object.updateWorldMatrix();

			expect( parent.matrix.elements ).toEqual( m.elements );

			expect( parent.matrixWorld.elements ).toEqual( m.elements );

			expect( object.matrix.elements ).toEqual( m.setPosition( object.position ).elements );

			expect( object.matrixWorld.elements ).toEqual( m.setPosition( object.position ).elements );

			expect( child.matrix.elements ).toEqual( m.identity().elements );

			expect( child.matrixWorld.elements ).toEqual( m.elements );

			// Update the world matrices of an object and its parents

			object.matrix.identity();
			object.matrixWorld.identity();

			object.updateWorldMatrix( true, false );

			expect( parent.matrix.elements ).toEqual( m.setPosition( parent.position ).elements );

			expect( parent.matrixWorld.elements ).toEqual( m.setPosition( parent.position ).elements );

			expect( object.matrix.elements ).toEqual( m.setPosition( object.position ).elements );

			expect( object.matrixWorld.elements ).toEqual( m.setPosition( v.copy( parent.position ).add( object.position ) ).elements );

			expect( child.matrix.elements ).toEqual( m.identity().elements );

			expect( child.matrixWorld.elements ).toEqual( m.identity().elements );

			// Update the world matrices of an object and its children

			parent.matrix.identity();
			parent.matrixWorld.identity();
			object.matrix.identity();
			object.matrixWorld.identity();

			object.updateWorldMatrix( false, true );

			expect( parent.matrix.elements ).toEqual( m.elements );

			expect( parent.matrixWorld.elements ).toEqual( m.elements );

			expect( object.matrix.elements ).toEqual( m.setPosition( object.position ).elements );

			expect( object.matrixWorld.elements ).toEqual( m.setPosition( object.position ).elements );

			expect( child.matrix.elements ).toEqual( m.setPosition( child.position ).elements );

			expect( child.matrixWorld.elements ).toEqual( m.setPosition( v.copy( object.position ).add( child.position ) ).elements );

			// Update the world matrices of an object and its parents and children

			object.matrix.identity();
			object.matrixWorld.identity();
			child.matrix.identity();
			child.matrixWorld.identity();

			object.updateWorldMatrix( true, true );

			expect( parent.matrix.elements ).toEqual( m.setPosition( parent.position ).elements );

			expect( parent.matrixWorld.elements ).toEqual( m.setPosition( parent.position ).elements );

			expect( object.matrix.elements ).toEqual( m.setPosition( object.position ).elements );

			expect( object.matrixWorld.elements ).toEqual( m.setPosition( v.copy( parent.position ).add( object.position ) ).elements );

			expect( child.matrix.elements ).toEqual( m.setPosition( child.position ).elements );

			expect( child.matrixWorld.elements ).toEqual( m.setPosition( v.copy( parent.position ).add( object.position ).add( child.position ) ).elements );

			// object.matrixAutoUpdate = false test

			object.matrix.identity();
			object.matrixWorld.identity();

			object.matrixAutoUpdate = false;
			object.matrixWorldNeedsUpdate = true;
			object.updateWorldMatrix( true, false );

			expect( object.matrix.elements ).toEqual( m.identity().elements );

			expect( object.matrixWorld.elements ).toEqual( m.setPosition( parent.position ).elements );

			// object.matrixWorldAutoUpdate = false test

			parent.matrixWorldAutoUpdate = false;
			child.matrixWorldAutoUpdate = false;

			child.matrixWorld.identity();
			parent.matrixWorld.identity();

			child.updateWorldMatrix( true, true );

			expect( child.matrixWorld.elements ).toEqual( m.identity().elements );

			expect( parent.matrixWorld.elements ).toEqual( m.identity().elements );

		} );

		test( 'toJSON', () => {

			const a = new Object3D();
			const child = new Object3D();
			const childChild = new Object3D();

			a.name = 'a\'s name';
			a.matrix.set( 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 );
			a.visible = false;
			a.castShadow = true;
			a.receiveShadow = true;
			a.userData[ 'foo' ] = 'bar';
			a.up.set( 1, 0, 0 );

			child.uuid = '5D4E9AE8-DA61-4912-A575-71A5BE3D72CD';
			childChild.uuid = 'B43854B3-E970-4E85-BD41-AAF8D7BFA189';
			child.add( childChild );
			a.add( child );

			const gold = {
				'metadata': {
					'version': 4.7,
					'type': 'Object',
					'generator': 'Object3D.toJSON'
				},
				'object': {
					'uuid': '0A1E4F43-CB5B-4097-8F82-DC2969C0B8C2',
					'type': 'Object3D',
					'name': 'a\'s name',
					'castShadow': true,
					'receiveShadow': true,
					'visible': false,
					'frustumCulled': true,
					'renderOrder': 0,
					'static': false,
					'matrixAutoUpdate': true,
					'userData': { 'foo': 'bar' },
					'layers': 1,
					'matrix': [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
					'children': [
						{
							'uuid': '5D4E9AE8-DA61-4912-A575-71A5BE3D72CD',
							'type': 'Object3D',
							'name': '',
							'castShadow': false,
							'receiveShadow': false,
							'visible': true,
							'frustumCulled': true,
							'renderOrder': 0,
							'static': false,
							'matrixAutoUpdate': true,
							'layers': 1,
							'matrix': [ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 ],
							'children': [
								{
									'uuid': 'B43854B3-E970-4E85-BD41-AAF8D7BFA189',
									'type': 'Object3D',
									'name': '',
									'castShadow': false,
									'receiveShadow': false,
									'visible': true,
									'frustumCulled': true,
									'renderOrder': 0,
									'static': false,
									'matrixAutoUpdate': true,
									'layers': 1,
									'matrix': [ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 ],
									'up': [ 0, 1, 0 ]
								}
							],
							'up': [ 0, 1, 0 ]
						}
					],
					'up': [ 1, 0, 0 ]
				}
			};

			// hacks
			const out = a.toJSON();
			out.object.uuid = '0A1E4F43-CB5B-4097-8F82-DC2969C0B8C2';

			expect( out ).toEqual( gold );

		} );

		test( 'clone', () => {

			/* eslint-disable prefer-const*/
			let a;
			const b = new Object3D();

			expect( a ).toBe( undefined );

			a = b.clone();
			expect( a ).not.toBe( b );

			a.uuid = b.uuid;
			expect( a ).toEqualLikeQUnit( b );

		} );

		test( 'copy', () => {

			const a = new Object3D();
			const b = new Object3D();
			const child = new Object3D();
			const childChild = new Object3D();

			a.name = 'original';
			b.name = 'to-be-copied';

			b.position.set( x, y, z );
			b.quaternion.set( x, y, z, w );
			b.scale.set( 2, 3, 4 );

			// bogus test values
			b.matrix.set( 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 );
			b.matrixWorld.set( 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 );

			b.matrixAutoUpdate = false;
			b.matrixWorldNeedsUpdate = true;

			b.layers.mask = 2;
			b.visible = false;

			b.castShadow = true;
			b.receiveShadow = true;

			b.frustumCulled = false;
			b.renderOrder = 1;

			b.userData[ 'foo' ] = 'bar';

			child.add( childChild );
			b.add( child );

			expect( a ).not.toEqual( b );
			a.copy( b, true );

			// check they're all unique instances
			expect(
				a.uuid !== b.uuid &&
				a.children[ 0 ].uuid !== b.children[ 0 ].uuid &&
				a.children[ 0 ].children[ 0 ].uuid !== b.children[ 0 ].children[ 0 ].uuid
			).toBeTruthy();

			// and now fix that
			a.uuid = b.uuid;
			a.children[ 0 ].uuid = b.children[ 0 ].uuid;
			a.children[ 0 ].children[ 0 ].uuid = b.children[ 0 ].children[ 0 ].uuid;

			expect( a ).toEqualLikeQUnit( b );

		} );

	} );

} );
