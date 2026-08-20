// Vitest-native equivalent of the "LIGHT TEST HELPERS" section of
// qunit-utils.js (which relies on the global QUnit object and can't be
// imported into a vitest test file). Used only by test/unit/src/lights/*.js.
import { expect } from 'vitest';
import { ObjectLoader } from '@src/loaders/ObjectLoader.js';

// Run common light tests.
function runStdLightTests( lights ) {

	for ( let i = 0, l = lights.length; i < l; i ++ ) {

		const light = lights[ i ];

		// copy and clone
		checkLightCopyClone( light );

		// THREE.Light doesn't get parsed by ObjectLoader as it's only
		// used as an abstract base class - so we skip the JSON tests
		if ( light.type !== 'Light' ) {

			// json round trip
			checkLightJsonRoundtrip( light );

		}

	}

}

function checkLightCopyClone( light ) {

	// copy
	const newLight = new light.constructor( 0xc0ffee );
	newLight.copy( light );

	expect( newLight.uuid ).not.toBe( light.uuid );
	expect( newLight.id ).not.toBe( light.id );
	expect( newLight ).toSmartEqual( light );

	// real copy?
	newLight.color.setHex( 0xc0ffee );
	expect( newLight.color.getHex() ).not.toBe( light.color.getHex() );

	// Clone
	const clone = light.clone(); // better get a new clone
	expect( clone.uuid ).not.toBe( light.uuid );
	expect( clone.id ).not.toBe( light.id );
	expect( clone ).toSmartEqual( light );

	// real clone?
	clone.color.setHex( 0xc0ffee );
	expect( clone.color.getHex() ).not.toBe( light.color.getHex() );

	if ( light.type !== 'Light' ) {

		// json round trip with clone
		checkLightJsonRoundtrip( clone );

	}

}

// Compare json file with its source Light.
function checkLightJsonWriting( light, json ) {

	expect( json.metadata.version == '4.7' ).toBe( true );

	const object = json.object;
	expect( light ).toEqualKey( object, 'type' );
	expect( light ).toEqualKey( object, 'uuid' );
	expect( object.id ).toBeUndefined();

}

// Check parsing and reconstruction of json Light
function checkLightJsonReading( json, light ) {

	const loader = new ObjectLoader();
	const outputLight = loader.parse( json );

	expect( outputLight ).toSmartEqual( light );

}

// Verify light -> json -> light
function checkLightJsonRoundtrip( light ) {

	const json = light.toJSON();
	checkLightJsonWriting( light, json );
	checkLightJsonReading( json, light );

}

export { runStdLightTests };
