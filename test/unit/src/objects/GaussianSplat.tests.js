import { BufferAttribute } from '../../../../src/core/BufferAttribute.js';
import { BufferGeometry } from '../../../../src/core/BufferGeometry.js';
import { GaussianSplat } from '../../../../src/objects/GaussianSplat.js';
import { Mesh } from '../../../../src/objects/Mesh.js';

function createSplatGeometry() {

	const geometry = new BufferGeometry();
	geometry.setAttribute( 'position', new BufferAttribute( new Float32Array( [ 1, 2, 3 ] ), 3 ) );
	geometry.setAttribute( 'covariance', new BufferAttribute( new Float32Array( [ 4, 0, 0, 9, 0, 16 ] ), 6 ) );
	geometry.setAttribute( 'color', new BufferAttribute( new Uint8Array( [ 128, 128, 128, 128 ] ), 4, true ) );

	return geometry;

}

export default QUnit.module( 'Objects', () => {

	QUnit.module( 'GaussianSplat', () => {

		// INHERITANCE
		QUnit.test( 'Extending', ( assert ) => {

			const object = new GaussianSplat( createSplatGeometry() );
			assert.strictEqual(
				object instanceof Mesh, true,
				'GaussianSplat extends from Mesh'
			);

		} );

		// INSTANCING
		QUnit.test( 'Instancing', ( assert ) => {

			const object = new GaussianSplat( createSplatGeometry() );
			assert.ok( object, 'Can instantiate a GaussianSplat.' );

		} );

		// PROPERTIES
		QUnit.test( 'type', ( assert ) => {

			const object = new GaussianSplat( createSplatGeometry() );
			assert.ok(
				object.type === 'GaussianSplat',
				'GaussianSplat.type should be GaussianSplat'
			);

		} );

		// PUBLIC
		QUnit.test( 'isGaussianSplat', ( assert ) => {

			const object = new GaussianSplat( createSplatGeometry() );
			assert.ok(
				object.isGaussianSplat,
				'GaussianSplat.isGaussianSplat should be true'
			);

		} );

		QUnit.test( 'splatGeometry', ( assert ) => {

			const geometry = createSplatGeometry();
			const object = new GaussianSplat( geometry );

			assert.strictEqual( object.splatGeometry, geometry, 'Stores source splat geometry.' );
			assert.strictEqual( object.frustumCulled, false, 'Disables frustum culling by default.' );
			assert.strictEqual( object.geometry.instanceCount, 1, 'Creates one rendered quad instance per splat.' );

		} );

	} );

} );
