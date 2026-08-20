import { describe, test, expect } from 'vitest';
import { FBXLoader } from '../../../../examples/jsm/loaders/FBXLoader.js';

describe( 'Addons', () => {

	describe( 'Loaders', () => {

		describe( 'FBXLoader', () => {

			test( 'morphAttributes length match geometry position length', async () => {

				const fbxLoader = new FBXLoader();
				const fbx = await new Promise( ( resolve, reject ) => {

					fbxLoader.load( '/examples/models/fbx/morph_test.fbx', resolve, undefined, reject );

				} );

				const mesh = fbx.children[ 0 ];
				const baseGeometryLength = mesh.geometry.attributes.position.count;
				const morphAttributesLength = mesh.geometry.morphAttributes.position[ 0 ].count;

				expect( baseGeometryLength === morphAttributesLength ).toBeTruthy();

			} );

		} );

	} );

} );
