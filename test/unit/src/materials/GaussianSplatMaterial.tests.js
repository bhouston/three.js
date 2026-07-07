import { GaussianSplatMaterial } from '../../../../src/materials/GaussianSplatMaterial.js';
import { Material } from '../../../../src/materials/Material.js';

export default QUnit.module( 'Materials', () => {

	QUnit.module( 'GaussianSplatMaterial', () => {

		// INHERITANCE
		QUnit.test( 'Extending', ( assert ) => {

			const object = new GaussianSplatMaterial();
			assert.strictEqual(
				object instanceof Material, true,
				'GaussianSplatMaterial extends from Material'
			);

		} );

		// INSTANCING
		QUnit.test( 'Instancing', ( assert ) => {

			const object = new GaussianSplatMaterial();
			assert.ok( object, 'Can instantiate a GaussianSplatMaterial.' );

		} );

		// PROPERTIES
		QUnit.test( 'type', ( assert ) => {

			const object = new GaussianSplatMaterial();
			assert.ok(
				object.type === 'GaussianSplatMaterial',
				'GaussianSplatMaterial.type should be GaussianSplatMaterial'
			);

		} );

		// PUBLIC
		QUnit.test( 'isGaussianSplatMaterial', ( assert ) => {

			const object = new GaussianSplatMaterial();
			assert.ok(
				object.isGaussianSplatMaterial,
				'GaussianSplatMaterial.isGaussianSplatMaterial should be true'
			);

		} );

		QUnit.test( 'render state defaults', ( assert ) => {

			const object = new GaussianSplatMaterial();

			assert.strictEqual( object.transparent, true, 'Transparent by default.' );
			assert.strictEqual( object.depthWrite, false, 'Depth write disabled by default.' );
			assert.strictEqual( object.depthTest, true, 'Depth test enabled by default.' );
			assert.strictEqual( object.autoSort, true, 'Auto sort enabled by default.' );

		} );

	} );

} );
