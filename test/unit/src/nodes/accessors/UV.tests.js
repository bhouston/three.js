import { semanticUV } from '../../../../../src/nodes/accessors/UV.js';
import NodeBuilder from '../../../../../src/nodes/core/NodeBuilder.js';
import { replaceDefaultUV, replaceUV } from '../../../../../src/nodes/utils/UVUtils.js';
import { vec2 } from '../../../../../src/nodes/tsl/TSLBase.js';

export default QUnit.module( 'Nodes', () => {

	QUnit.module( 'Accessors', () => {

		QUnit.module( 'UV', () => {

			QUnit.test( 'semantic UV falls back to the indexed geometry attribute', ( assert ) => {

				const uv0 = semanticUV( 0 ).setup( { context: {} } );
				const uv1 = semanticUV( 1 ).setup( { context: {} } );

				assert.strictEqual( uv0.getAttributeName(), 'uv', 'UV0 resolves to the uv attribute' );
				assert.strictEqual( uv1.getAttributeName(), 'uv1', 'UV1 resolves to the uv1 attribute' );

			} );

			QUnit.test( 'semantic UV passes its index and source to the context callback', ( assert ) => {

				const replacement = vec2( 0.25, 0.75 );
				const builder = {
					context: {
						getUVAttribute( index, sourceNode, callbackBuilder ) {

							assert.strictEqual( index, 2, 'passes the requested UV index' );
							assert.ok( sourceNode.isSemanticUVNode, 'passes the semantic source node' );
							assert.strictEqual( callbackBuilder, builder, 'passes the current builder' );
							return replacement;

						}
					}
				};
				const resolved = semanticUV( 2 ).setup( builder );

				assert.ok( resolved.isContextNode, 'wraps the replacement in a context node' );
				assert.strictEqual( resolved.node, replacement, 'uses the callback replacement' );
				assert.strictEqual( resolved.value.getUVAttribute, null, 'disables recursive semantic replacement while building the override' );

			} );

			QUnit.test( 'replaceUV adapts texture and indexed semantic UV sources', ( assert ) => {

				const calls = [];
				const replacement = vec2( 0.5 );
				const contextNode = replaceUV( ( source, builder ) => {

					calls.push( { source, builder } );
					return replacement;

				} );
				const textureNode = { isTextureNode: true };
				const sourceNode = semanticUV( 3 );
				const builder = {};

				assert.strictEqual( contextNode.value.getUV( textureNode, builder ), replacement, 'replaces default texture UVs' );
				assert.strictEqual( contextNode.value.getUVAttribute( 3, sourceNode, builder ), replacement, 'replaces semantic UVs' );
				assert.deepEqual( calls[ 0 ].source, {
					index: 0,
					sourceNode: null,
					textureNode
				}, 'describes the texture source' );
				assert.deepEqual( calls[ 1 ].source, {
					index: 3,
					sourceNode,
					textureNode: null
				}, 'describes the indexed semantic source' );

			} );

			QUnit.test( 'replaceDefaultUV keeps its existing callback contract', ( assert ) => {

				const replacement = vec2( 0.5 );
				const textureNode = { isTextureNode: true };
				const contextNode = replaceDefaultUV( ( receivedTextureNode ) => {

					assert.strictEqual( receivedTextureNode, textureNode, 'passes the texture node directly' );
					return replacement;

				} );

				assert.strictEqual( contextNode.value.getUV( textureNode ), replacement, 'uses the callback result' );
				assert.notOk( 'getUVAttribute' in contextNode.value, 'does not install a semantic UV callback' );

			} );

			QUnit.test( 'semantic UV serializes its index', ( assert ) => {

				const data = {};
				semanticUV( 4 ).serialize( data );

				assert.strictEqual( data.index, 4, 'preserves the UV index' );

			} );

			QUnit.test( 'semantic UV callbacks are excluded from shared shader context', ( assert ) => {

				const builder = new NodeBuilder( null, null, null );
				builder.context.getUVAttribute = () => vec2( 0.5 );

				assert.notOk( 'getUVAttribute' in builder.getSharedContext(), 'does not share material-specific semantic UV callbacks' );

			} );

		} );

	} );

} );
