import { describe, test, expect } from 'vitest';
import { MaterialXLoader } from '../../../../examples/jsm/loaders/MaterialXLoader.js';
import { MaterialXDocument } from '../../../../examples/jsm/loaders/materialx/MaterialXDocument.js';
import { MaterialXLog } from '../../../../examples/jsm/loaders/materialx/MaterialXLog.js';

function parseNodeGraph( body, options = {} ) {

	const source = `<?xml version="1.0"?>
<materialx version="1.39">
	<nodegraph name="test_graph">
		${body}
	</nodegraph>
</materialx>`;

	return new MaterialXLoader().parse( source, options ).materials.test_graph;

}

function collectSemanticUVNodes( material ) {

	const nodes = new Set();

	for ( const value of Object.values( material ) ) {

		if ( value && value.isNode === true ) {

			value.traverse( ( node ) => {

				if ( node.isSemanticUVNode === true ) nodes.add( node );

			} );

		}

	}

	return [ ...nodes ];

}

describe( 'Addons', () => {

	describe( 'Loaders', () => {

		describe( 'MaterialXLoader', () => {

			test( 'procedural fallback coordinates use semantic UV0', () => {

				const material = parseNodeGraph( `
					<checkerboard name="checker" type="color3" />
					<output name="out" type="color3" nodename="checker" />
				` );
				const uvNodes = collectSemanticUVNodes( material );

				expect( uvNodes.length ).toBe( 1 );
				expect( uvNodes[ 0 ].index ).toBe( 0 );

			} );

			test( 'defaultgeomprop preserves the requested UV set', () => {

				const material = parseNodeGraph( `
					<checkerboard name="checker" type="color3">
						<input name="texcoord" type="vector2" defaultgeomprop="UV1" />
					</checkerboard>
					<output name="out" type="color3" nodename="checker" />
				` );
				const uvNodes = collectSemanticUVNodes( material );

				expect( uvNodes.length ).toBe( 1 );
				expect( uvNodes[ 0 ].index ).toBe( 1 );

			} );

			test( 'explicit texcoord nodes use the semantic resolver', () => {

				const material = parseNodeGraph( `
					<texcoord name="coords" type="vector2">
						<input name="index" type="integer" value="2" />
					</texcoord>
					<checkerboard name="checker" type="color3">
						<input name="texcoord" type="vector2" nodename="coords" />
					</checkerboard>
					<output name="out" type="color3" nodename="checker" />
				` );
				const uvNodes = collectSemanticUVNodes( material );

				expect( uvNodes.length ).toBe( 1 );
				expect( uvNodes[ 0 ].index ).toBe( 2 );

			} );

			test( 'UV-space conversion remains downstream of semantic resolution', () => {

				const loader = new MaterialXLoader();
				const bottomLeftDocument = new MaterialXDocument( loader.manager, '', new MaterialXLog(), null, 'bottom-left' );
				const topLeftDocument = new MaterialXDocument( loader.manager, '', new MaterialXLog(), null, 'top-left' );
				const bottomLeftUv = bottomLeftDocument.compileContext.getTexcoordNode( 0 );
				const topLeftUv = topLeftDocument.compileContext.getTexcoordNode( 0 );
				const topLeftSemanticNodes = new Set();

				topLeftUv.traverse( ( node ) => {

					if ( node.isSemanticUVNode === true ) topLeftSemanticNodes.add( node );

				} );

				expect( bottomLeftUv.isSemanticUVNode ).toBeTruthy();
				expect( topLeftUv.isSemanticUVNode ).toBeFalsy();
				expect( topLeftSemanticNodes.size ).toBe( 1 );

			} );

		} );

	} );

} );
