import { describe, test, expect } from 'vitest';
import { semanticUV } from '@src/nodes/accessors/UV.js';
import NodeBuilder from '@src/nodes/core/NodeBuilder.js';
import { replaceDefaultUV, replaceUV } from '@src/nodes/utils/UVUtils.js';
import { vec2 } from '@src/nodes/tsl/TSLBase.js';

describe( 'Nodes', () => {

	describe( 'Accessors', () => {

		describe( 'UV', () => {

			test( 'semantic UV falls back to the indexed geometry attribute', () => {

				const uv0 = semanticUV( 0 ).setup( { context: {} } );
				const uv1 = semanticUV( 1 ).setup( { context: {} } );

				expect( uv0.getAttributeName() ).toBe( 'uv' );
				expect( uv1.getAttributeName() ).toBe( 'uv1' );

			} );

			test( 'semantic UV passes its index and source to the context callback', () => {

				const replacement = vec2( 0.25, 0.75 );
				const builder = {
					context: {
						getUVAttribute( index, sourceNode, callbackBuilder ) {

							expect( index ).toBe( 2 );
							expect( sourceNode.isSemanticUVNode ).toBeTruthy();
							expect( callbackBuilder ).toBe( builder );
							return replacement;

						}
					}
				};
				const resolved = semanticUV( 2 ).setup( builder );

				expect( resolved.isContextNode ).toBeTruthy();
				expect( resolved.node ).toBe( replacement );
				expect( resolved.value.getUVAttribute ).toBe( null );

			} );

			test( 'replaceUV adapts texture and indexed semantic UV sources', () => {

				const calls = [];
				const replacement = vec2( 0.5 );
				const contextNode = replaceUV( ( source, builder ) => {

					calls.push( { source, builder } );
					return replacement;

				} );
				const textureNode = { isTextureNode: true };
				const sourceNode = semanticUV( 3 );
				const builder = {};

				expect( contextNode.value.getUV( textureNode, builder ) ).toBe( replacement );
				expect( contextNode.value.getUVAttribute( 3, sourceNode, builder ) ).toBe( replacement );
				expect( calls[ 0 ].source ).toEqual( {
					index: 0,
					sourceNode: null,
					textureNode
				} );
				expect( calls[ 1 ].source ).toEqual( {
					index: 3,
					sourceNode,
					textureNode: null
				} );

			} );

			test( 'replaceDefaultUV keeps its existing callback contract', () => {

				const replacement = vec2( 0.5 );
				const textureNode = { isTextureNode: true };
				const contextNode = replaceDefaultUV( ( receivedTextureNode ) => {

					expect( receivedTextureNode ).toBe( textureNode );
					return replacement;

				} );

				expect( contextNode.value.getUV( textureNode ) ).toBe( replacement );
				expect( 'getUVAttribute' in contextNode.value ).toBeFalsy();

			} );

			test( 'semantic UV serializes its index', () => {

				const data = {};
				semanticUV( 4 ).serialize( data );

				expect( data.index ).toBe( 4 );

			} );

			test( 'semantic UV callbacks are excluded from shared shader context', () => {

				const builder = new NodeBuilder( null, null, null );
				builder.context.getUVAttribute = () => vec2( 0.5 );

				expect( 'getUVAttribute' in builder.getSharedContext() ).toBeFalsy();

			} );

		} );

	} );

} );
