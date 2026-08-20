import { describe, test, expect } from 'vitest';
import { PropertyBinding } from '@src/animation/PropertyBinding.js';

import { BoxGeometry } from '@src/geometries/BoxGeometry.js';
import { Mesh } from '@src/objects/Mesh.js';
import { MeshBasicMaterial } from '@src/materials/MeshBasicMaterial.js';

describe( 'Animation', () => {

	describe( 'PropertyBinding', () => {

		test( 'Instancing', () => {

			const geometry = new BoxGeometry();
			const material = new MeshBasicMaterial();
			const mesh = new Mesh( geometry, material );
			const path = '.material.opacity';
			const parsedPath = {
				nodeName: '',
				objectName: 'material',
				objectIndex: undefined,
				propertyName: 'opacity',
				propertyIndex: undefined
			  };

			// mesh, path
			const object = new PropertyBinding( mesh, path );
			expect( object ).toBeTruthy();

			// mesh, path, parsedPath
			const object_all = new PropertyBinding( mesh, path, parsedPath );
			expect( object_all ).toBeTruthy();

		} );

		test( 'sanitizeNodeName', () => {

			expect(
				PropertyBinding.sanitizeNodeName( 'valid-name-123_' )
			).toBe( 'valid-name-123_' );

			expect(
				PropertyBinding.sanitizeNodeName( '急須' )
			).toBe( '急須' );

			expect(
				PropertyBinding.sanitizeNodeName( 'space separated name 123_ -' )
			).toBe( 'space_separated_name_123__-' );

			expect(
				PropertyBinding.sanitizeNodeName( '"Mátyás" %_* 😇' )
			).toBe( '"Mátyás"_%_*_😇' );

			expect(
				PropertyBinding.sanitizeNodeName( '/invalid: name ^123.[_]' )
			).toBe( 'invalid_name_^123_' );

		} );

		test( 'parseTrackName', () => {

			const paths = [

				[
					'.property',
					{
						nodeName: undefined,
						objectName: undefined,
						objectIndex: undefined,
						propertyName: 'property',
						propertyIndex: undefined
					}
				],

				[
					'nodeName.property',
					{
						nodeName: 'nodeName',
						objectName: undefined,
						objectIndex: undefined,
						propertyName: 'property',
						propertyIndex: undefined
					}
				],

				[
					'a.property',
					{
						nodeName: 'a',
						objectName: undefined,
						objectIndex: undefined,
						propertyName: 'property',
						propertyIndex: undefined
					}
				],

				[
					'no.de.Name.property',
					{
						nodeName: 'no.de.Name',
						objectName: undefined,
						objectIndex: undefined,
						propertyName: 'property',
						propertyIndex: undefined
					}
				],

				[
					'no.d-e.Name.property',
					{
						nodeName: 'no.d-e.Name',
						objectName: undefined,
						objectIndex: undefined,
						propertyName: 'property',
						propertyIndex: undefined
					}
				],

				[
					'nodeName.property[accessor]',
					{
						nodeName: 'nodeName',
						objectName: undefined,
						objectIndex: undefined,
						propertyName: 'property',
						propertyIndex: 'accessor'
					}
				],

				[
					'nodeName.material.property[accessor]',
					{
						nodeName: 'nodeName',
						objectName: 'material',
						objectIndex: undefined,
						propertyName: 'property',
						propertyIndex: 'accessor'
					}
				],

				[
					'no.de.Name.material.property',
					{
						nodeName: 'no.de.Name',
						objectName: 'material',
						objectIndex: undefined,
						propertyName: 'property',
						propertyIndex: undefined
					}
				],

				[
					'no.de.Name.material[materialIndex].property',
					{
						nodeName: 'no.de.Name',
						objectName: 'material',
						objectIndex: 'materialIndex',
						propertyName: 'property',
						propertyIndex: undefined
					}
				],

				[
					'uuid.property[accessor]',
					{
						nodeName: 'uuid',
						objectName: undefined,
						objectIndex: undefined,
						propertyName: 'property',
						propertyIndex: 'accessor'
					}
				],

				[
					'uuid.objectName[objectIndex].propertyName[propertyIndex]',
					{
						nodeName: 'uuid',
						objectName: 'objectName',
						objectIndex: 'objectIndex',
						propertyName: 'propertyName',
						propertyIndex: 'propertyIndex'
					}
				],

				[
					'parentName/nodeName.property',
					{
						// directoryName is currently unused.
						nodeName: 'nodeName',
						objectName: undefined,
						objectIndex: undefined,
						propertyName: 'property',
						propertyIndex: undefined
					}
				],

				[
					'parentName/no.de.Name.property',
					{
						// directoryName is currently unused.
						nodeName: 'no.de.Name',
						objectName: undefined,
						objectIndex: undefined,
						propertyName: 'property',
						propertyIndex: undefined
					}
				],

				[
					'parentName/parentName/nodeName.property[index]',
					{
						// directoryName is currently unused.
						nodeName: 'nodeName',
						objectName: undefined,
						objectIndex: undefined,
						propertyName: 'property',
						propertyIndex: 'index'
					}
				],

				[
					'.bone[Armature.DEF_cog].position',
					{
						nodeName: undefined,
						objectName: 'bone',
						objectIndex: 'Armature.DEF_cog',
						propertyName: 'position',
						propertyIndex: undefined
					}
				],

				[
					'scene:helium_balloon_model:helium_balloon_model.position',
					{
						nodeName: 'helium_balloon_model',
						objectName: undefined,
						objectIndex: undefined,
						propertyName: 'position',
						propertyIndex: undefined
					}
				],

				[
					'急須.材料[零]',
					{
						nodeName: '急須',
						objectName: undefined,
						objectIndex: undefined,
						propertyName: '材料',
						propertyIndex: '零'
					}
				],

				[
					'📦.🎨[🔴]',
					{
						nodeName: '📦',
						objectName: undefined,
						objectIndex: undefined,
						propertyName: '🎨',
						propertyIndex: '🔴'
					}
				]

			];

			paths.forEach( function ( path ) {

				expect(
					PropertyBinding.parseTrackName( path[ 0 ] )
				).toSmartEqual( path[ 1 ] );

			} );

		} );

		test( 'setValue', () => {

			const paths = [
				'.material.opacity',
				'.material[opacity]'
			];

			paths.forEach( function ( path ) {

				const originalValue = 0;
				const expectedValue = 1;

				const geometry = new BoxGeometry();
				const material = new MeshBasicMaterial();
				material.opacity = originalValue;
				const mesh = new Mesh( geometry, material );

				const binding = new PropertyBinding( mesh, path, null );
				binding.bind();

				expect( material.opacity ).toBe( originalValue );

				binding.setValue( [ expectedValue ], 0 );
				expect( material.opacity ).toBe( expectedValue );

			} );

		} );

	} );

} );
