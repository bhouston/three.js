// Pure draft semantics checks; no browser, DOM, image decoder or GPU context is used.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { cloneGLTFProceduralSource, compileGLTFProceduralGraph, validateGLTFProceduralGraph } from '../../../../examples/jsm/loaders/materialx/GLTFProceduralGraph.js';

const directory = new URL( '../../../../examples/models/gltf/TextureProcedurals/', import.meta.url );
const read = name => JSON.parse( readFileSync( new URL( `${ name }.gltf`, directory ), 'utf8' ) );

for ( const name of [ 'checkerboard', 'noise', 'texture', 'texture-external' ] ) {

	test( `${ name }: ordinary fallback references and experimental graph semantics`, () => {

		const json = read( name );
		const graph = json.extensions.KHR_texture_procedurals.procedurals[ 0 ];
		const binding = json.materials[ 0 ].pbrMetallicRoughness.baseColorTexture;
		assert.ok( json.images[ json.textures[ binding.index ].source ].uri );
		const validated = validateGLTFProceduralGraph( graph, { textureCount: json.textures.length } );
		assert.deepEqual( validated.source, graph );
		assert.deepEqual( validated.textureIndices, name.startsWith( 'texture' ) ? [ 1 ] : [] );
		if ( name === 'noise' ) {

			assert.deepEqual( validated.streams, [], 'Object-space noise needs no UV stream.' );

		} else {

			assert.equal( validated.streams[ 0 ].semantic, 'TEXCOORD_0' );

		}

	} );

}

test( 'graph output names cannot collide with normalized processing-node symbols', async () => {

	const graph = read( 'checkerboard' ).extensions.KHR_texture_procedurals.procedurals[ 0 ];
	graph.outputs.node_0 = graph.outputs.pattern;
	delete graph.outputs.pattern;
	assert.ok( ( await compileGLTFProceduralGraph( graph ) ).outputs.node_0.isNode );

} );

for ( const [ name, mutate ] of Object.entries( {
	cycle: graph => {

		graph.nodes[ 0 ].inputs.color1 = { nodetype: 'input', type: 'color3', node: 0, output: 'out' };

	},
	index: graph => {

		graph.outputs.pattern.node = - 1;

	},
	port: graph => {

		graph.outputs.pattern.output = 'missing';

	},
	type: graph => {

		graph.nodes[ 0 ].inputs.color1.value = [ 1, 2 ];

	},
	unknown: graph => {

		graph.nodes[ 0 ].nodetype = 'arbitrary_shader';

	},
	legacy: graph => {

		graph.outputs = [ graph.outputs.pattern ];

	},
	deadSampling: graph => {

		graph.nodes.push( { nodetype: 'image', type: 'color3', inputs: { file: { nodetype: 'input', type: 'filename', value: [ 0 ] }, filtertype: { nodetype: 'input', type: 'string', value: 'cubic' } }, outputs: { out: { nodetype: 'output', type: 'color3' } } } );

	}
} ) ) {

	test( `strict graph validation rejects ${ name }`, () => {

		const graph = read( 'checkerboard' ).extensions.KHR_texture_procedurals.procedurals[ 0 ];
		mutate( graph );
		assert.throws( () => validateGLTFProceduralGraph( graph, { textureCount: 1 } ), /KHR_texture_procedurals/ );

	} );

}

test( 'authored sources must be plain, finite, serializable JSON', () => {

	const cyclic = {};
	cyclic.self = cyclic;
	let getterExecuted = false;
	const accessor = Object.defineProperty( {}, 'value', { enumerable: true, get() {

		getterExecuted = true;
		return 1;

	} } );
	for ( const metadata of [ { callback() {} }, { missing: undefined }, { value: Infinity }, { value: new Date() }, { value: new Array( 2 ) }, cyclic, accessor, JSON.parse( '{"__proto__":{}}' ) ] ) {

		assert.throws( () => cloneGLTFProceduralSource( metadata ), /JSON|serializ|plain|finite|unsafe|circular|cycle|property|key/i );

	}

	assert.equal( getterExecuted, false, 'Asset property getters are never executed.' );

} );
