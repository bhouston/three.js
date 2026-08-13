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

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'Loaders', () => {

		QUnit.module( 'MaterialXLoader', () => {

			QUnit.test( 'procedural fallback coordinates use semantic UV0', ( assert ) => {

				const material = parseNodeGraph( `
					<checkerboard name="checker" type="color3" />
					<output name="out" type="color3" nodename="checker" />
				` );
				const uvNodes = collectSemanticUVNodes( material );

				assert.strictEqual( uvNodes.length, 1, 'creates one semantic UV source' );
				assert.strictEqual( uvNodes[ 0 ].index, 0, 'uses UV0 by default' );

			} );

			QUnit.test( 'defaultgeomprop preserves the requested UV set', ( assert ) => {

				const material = parseNodeGraph( `
					<checkerboard name="checker" type="color3">
						<input name="texcoord" type="vector2" defaultgeomprop="UV1" />
					</checkerboard>
					<output name="out" type="color3" nodename="checker" />
				` );
				const uvNodes = collectSemanticUVNodes( material );

				assert.strictEqual( uvNodes.length, 1, 'creates one semantic UV source' );
				assert.strictEqual( uvNodes[ 0 ].index, 1, 'preserves UV1' );

			} );

			QUnit.test( 'explicit texcoord nodes use the semantic resolver', ( assert ) => {

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

				assert.strictEqual( uvNodes.length, 1, 'creates one semantic UV source' );
				assert.strictEqual( uvNodes[ 0 ].index, 2, 'preserves the explicit texcoord index' );

			} );

			QUnit.test( 'UV-space conversion remains downstream of semantic resolution', ( assert ) => {

				const loader = new MaterialXLoader();
				const bottomLeftDocument = new MaterialXDocument( loader.manager, '', new MaterialXLog(), null, 'bottom-left' );
				const topLeftDocument = new MaterialXDocument( loader.manager, '', new MaterialXLog(), null, 'top-left' );
				const bottomLeftUv = bottomLeftDocument.compileContext.getTexcoordNode( 0 );
				const topLeftUv = topLeftDocument.compileContext.getTexcoordNode( 0 );
				const topLeftSemanticNodes = new Set();

				topLeftUv.traverse( ( node ) => {

					if ( node.isSemanticUVNode === true ) topLeftSemanticNodes.add( node );

				} );

				assert.ok( bottomLeftUv.isSemanticUVNode, 'bottom-left input uses the semantic source directly' );
				assert.notOk( topLeftUv.isSemanticUVNode, 'top-left input wraps the semantic source in a conversion graph' );
				assert.strictEqual( topLeftSemanticNodes.size, 1, 'the conversion graph contains one semantic source' );

			} );

		} );

	} );

} );
