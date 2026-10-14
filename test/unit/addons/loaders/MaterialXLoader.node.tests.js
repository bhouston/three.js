// JSON document parsing and compilation do not require DOMParser or a GPU.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Texture } from 'three/webgpu';
import { texture } from 'three/tsl';
import { MaterialXLoader } from '../../../../examples/jsm/loaders/MaterialXLoader.js';
import { json } from './fixtures/MaterialXJSON.js';

test( 'addon JSON objects, text and buffers compile without browser globals', () => {

	assert.equal( typeof globalThis.DOMParser, 'undefined' );
	const loader = new MaterialXLoader();
	const text = '\n\t' + JSON.stringify( json );
	const bytes = new TextEncoder().encode( text );
	for ( const result of [ loader.parse( json ), loader.parse( text ), loader.parseBuffer( bytes ), loader.parseBuffer( bytes.buffer ) ] ) {

		assert.ok( result.materials.test_graph.isNodeMaterial );
		assert.deepEqual( result.errors, [] );

	}

} );

test( 'parseDocument exposes reusable normalized nodes without compiling materials', () => {

	let resolverCalls = 0;
	const source = structuredClone( json );
	const document = new MaterialXLoader().parseDocument( source, { nodeResolver: () => {

		resolverCalls ++;
		return null;

	} } );
	assert.equal( document.rootNode.element, 'materialx' );
	assert.equal( document.rootNode.getAttribute( 'version' ), '1.39' );
	const graph = document.rootNode.getChildByName( 'test_graph' );
	assert.equal( graph.element, 'nodegraph' );
	assert.equal( graph.getChildByName( 'scaled' ).getChildByName( 'in1' ).value, '0.25, 0.5, 0.75' );
	assert.equal( resolverCalls, 0, 'Reading a document does not evaluate processing nodes.' );
	assert.ok( graph.getChildByName( 'out' ).getNode().isNode );
	assert.ok( resolverCalls > 0, 'An addon evaluates outputs with the existing MaterialX compiler.' );
	assert.deepEqual( source, json, 'Parsing does not mutate the source.' );

} );

test( 'JSON scalar attributes normalize to the same strings as XML', () => {

	const source = structuredClone( json );
	source.attributes.version = 1.39;
	source.children[ 0 ].children[ 0 ].children[ 1 ].attributes.value = 2;
	const document = new MaterialXLoader().parseDocument( source );
	assert.equal( document.rootNode.getAttribute( 'version' ), '1.39' );
	assert.equal( document.rootNode.getChildByName( 'test_graph' ).getChildByName( 'scaled' ).getChildByName( 'in2' ).value, '2' );

} );

test( 'node policy rejects excluded processing categories before host resolution', () => {

	const loader = new MaterialXLoader();
	assert.deepEqual( loader.parse( json, { allowedNodes: [ 'multiply' ] } ).errors, [] );
	assert.deepEqual( loader.parse( json, { allowedNodes: new Set( [ 'multiply' ] ) } ).errors, [] );
	for ( const policy of [ { allowedNodes: [] }, { blockedNodes: [ 'multiply' ] }, { allowedNodes: [ 'multiply' ], blockedNodes: new Set( [ 'multiply' ] ) } ] ) {

		let resolverCalls = 0;
		assert.throws( () => loader.parse( json, { ...policy, nodeResolver: () => {

			resolverCalls ++;
			return null;

		} } ), /not allowed|blocked|policy|disallowed/i );
		assert.equal( resolverCalls, 0 );

	}

} );

test( 'textureResolver supplies ready TSL textures through the common filename path', () => {

	const source = structuredClone( json );
	source.children[ 0 ].children[ 0 ] = {
		element: 'image', attributes: { name: 'scaled', type: 'color3' },
		children: [ { element: 'input', attributes: { name: 'file', type: 'filename', value: 'host:ready-image' } } ]
	};
	const readyTexture = new Texture();
	const supplied = texture( readyTexture );
	const filenames = [];
	const document = new MaterialXLoader().parseDocument( source, { textureResolver: filename => {

		filenames.push( filename.value );
		return supplied;

	} } );
	const graph = document.rootNode.getChildByName( 'test_graph' );
	assert.ok( graph.getChildByName( 'out' ).getNode().isNode );
	assert.deepEqual( filenames, [ 'host:ready-image' ] );
	assert.equal( graph.getChildByName( 'scaled' ).getChildByName( 'file' ).getTexture().value, readyTexture );
	assert.deepEqual( document.log.errors, [] );

} );

test( 'malformed addon JSON has useful format and shape diagnostics', () => {

	const loader = new MaterialXLoader();
	for ( const source of [ '{"element":', {}, [], { element: 'materialx', children: {} }, { element: 'materialx', attributes: { version: {} } }, { element: 'materialx', attributes: { version: Infinity } }, { element: 'materialx', children: [ null ] } ] ) {

		assert.throws( () => loader.parse( source ), /MaterialX|JSON|document|attribute|children/i );

	}

} );

test( 'addon JSON rejects executable properties, sparse arrays and cyclic trees', () => {

	let accessorCalled = false;
	const accessor = { element: 'materialx', get attributes() {

		accessorCalled = true;
		return {};

	} };
	const cyclic = { element: 'materialx', children: [] };
	cyclic.children.push( cyclic );
	const loader = new MaterialXLoader();
	for ( const source of [ accessor, cyclic, { element: 'materialx', children: new Array( 1 ) }, { element: 'materialx', attributes: { callback() {} } }, { element: 'materialx', evaluate() {} } ] ) {

		assert.throws( () => loader.parse( source ), /MaterialX|JSON|document/i );

	}

	assert.equal( accessorCalled, false, 'Asset getters are rejected without executing them.' );

} );
