import { describe, test, expect } from 'vitest';
import { BufferAttribute, BufferGeometry } from 'three';
import * as BufferGeometryUtils from '../../../../examples/jsm/utils/BufferGeometryUtils.js';

const getGeometry = () => {

	const geometry = new BufferGeometry();

	// square
	const vertices = new Float32Array( [
		- 1.0, - 1.0, 0.0, // Bottom left
		1.0, - 1.0, 0.0, // Bottom right
		1.0, 1.0, 0.0, // Top right
		- 1.0, 1.0, 0.0 // Top left
	] );

	const morphVertices = new Float32Array( [
		0.0, - 1.0, 0.0, // Bottom
		1.0, 0.0, 0.0, // Right
		0.0, 1.0, 0.0, // Top
		- 1.0, 0.0, 0.0 // Left
	] );

	geometry.setAttribute( 'position', new BufferAttribute( vertices, 3 ) );

	geometry.morphAttributes.position = [
		new BufferAttribute( morphVertices, 3 )
	];

	return geometry;

};

describe( 'Addons', () => {

	describe( 'Utils', () => {

		describe( 'BufferGeometryUtils', () => {

			describe( 'mergeVertices', () => {

				test( 'can handle morphAttributes without crashing', () => {

					const geometry = getGeometry();

					const indexedGeometry = BufferGeometryUtils.mergeVertices( geometry );

					expect( geometry.morphAttributes.position[ 0 ] ).toEqual( indexedGeometry.morphAttributes.position[ 0 ] );
					expect( indexedGeometry.index ).toBeTruthy();

				} );

			} );

		} );

	} );


} );
