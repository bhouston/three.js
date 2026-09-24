import NodeBuilder from '../../../../../src/nodes/core/NodeBuilder.js';

export default QUnit.module( 'Nodes', () => {

	QUnit.module( 'Core', () => {

		QUnit.module( 'NodeBuilder', () => {

			QUnit.test( 'getSharedContext() does not leak control-flow state', ( assert ) => {

				const builder = new NodeBuilder( null, null, null );
				builder.context = { nodeBlock: {}, nodeLoop: {}, shared: true };

				const context = builder.getSharedContext();

				assert.false( 'nodeBlock' in context, 'nodeBlock is removed' );
				assert.false( 'nodeLoop' in context, 'nodeLoop is removed' );
				assert.true( context.shared, 'other entries are kept' );

			} );

		} );

	} );

} );
