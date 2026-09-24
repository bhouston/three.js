import { builtinRadianceContext } from '../../../../../src/nodes/core/ContextNode.js';
import { vec3 } from '../../../../../src/nodes/tsl/TSLBase.js';

export default QUnit.module( 'Nodes', () => {

	QUnit.module( 'Core', () => {

		QUnit.module( 'ContextNode', () => {

			QUnit.test( 'builtinRadianceContext() provides radiance to opaque materials only', ( assert ) => {

				const radiance = vec3( 1 );
				const { getRadiance } = builtinRadianceContext( radiance ).value;

				assert.strictEqual( getRadiance( null, { material: { transparent: false } } ), radiance, 'opaque materials receive the radiance' );
				assert.strictEqual( getRadiance( null, { material: { transparent: true } } ), null, 'transparent materials are left untouched' );

			} );

		} );

	} );

} );
