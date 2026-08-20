// Vitest port of the QUnit `runStdGeometryTests` helper in qunit-utils.js.
// Kept as a separate module (rather than reusing qunit-utils.js) because that
// file extends the QUnit.assert global and calls QUnit.assert.* directly,
// which doesn't exist under vitest.
import { expect } from 'vitest';
import { ObjectLoader } from '@src/loaders/ObjectLoader.js';

function getDifferingProp( geometryA, geometryB ) {

	const geometryKeys = Object.keys( geometryA );
	const cloneKeys = Object.keys( geometryB );

	let differingProp = undefined;

	for ( let i = 0, l = geometryKeys.length; i < l; i ++ ) {

		const key = geometryKeys[ i ];

		if ( cloneKeys.indexOf( key ) < 0 ) {

			differingProp = key;
			break;

		}

	}

	return differingProp;

}

function checkGeometryClone( geom ) {

	const copy = geom.clone();
	expect( copy.uuid ).not.toBe( geom.uuid );
	expect( copy.id ).not.toBe( geom.id );

	let differingProp = getDifferingProp( geom, copy );
	expect( differingProp === undefined ).toBeTruthy();

	differingProp = getDifferingProp( copy, geom );
	expect( differingProp === undefined ).toBeTruthy();

	// json round trip with clone
	checkGeometryJsonRoundtrip( copy );

}

// Compare json file with its source geometry.
function checkGeometryJsonWriting( geom, json ) {

	expect( json.metadata.version ).toBe( 4.7 );
	expect( geom ).toEqualKey( json, 'type' );
	expect( geom ).toEqualKey( json, 'uuid' );
	expect( json.id ).toBe( undefined );

	const params = geom.parameters;
	if ( ! params ) {

		return;

	}

	// All parameters from geometry should be persisted.
	let keys = Object.keys( params );
	for ( let i = 0, l = keys.length; i < l; i ++ ) {

		expect( params ).toEqualKey( json, keys[ i ] );

	}

	// All parameters from json should be transferred to the geometry.
	// json is flat. Ignore first level json properties that are not parameters.
	const notParameters = [ 'metadata', 'uuid', 'type', 'name' ];
	keys = Object.keys( json );
	for ( let i = 0, l = keys.length; i < l; i ++ ) {

		const key = keys[ i ];
		if ( notParameters.indexOf( key ) === - 1 ) expect( params ).toEqualKey( json, key );

	}

}

// Check parsing and reconstruction of json geometry
function checkGeometryJsonReading( json, geom ) {

	const wrap = [ json ];

	const loader = new ObjectLoader();
	const output = loader.parseGeometries( wrap );

	expect( output[ geom.uuid ] ).toBeTruthy();

	let differingProp = getDifferingProp( output[ geom.uuid ], geom );
	expect( differingProp === undefined ).toBeTruthy();

	differingProp = getDifferingProp( geom, output[ geom.uuid ] );
	expect( differingProp === undefined ).toBeTruthy();

}

// Verify geom -> json -> geom
function checkGeometryJsonRoundtrip( geom ) {

	const json = geom.toJSON();
	checkGeometryJsonWriting( geom, json );
	checkGeometryJsonReading( json, geom );

}

// Run common geometry tests.
function runStdGeometryTests( geometries ) {

	for ( let i = 0, l = geometries.length; i < l; i ++ ) {

		const geom = geometries[ i ];

		// Clone
		checkGeometryClone( geom );

		// json round trip
		checkGeometryJsonRoundtrip( geom );

	}

}

export { runStdGeometryTests };
