import { describe, test, expect } from 'vitest';
import { BoxGeometry, Mesh, MeshStandardMaterial, Scene, SphereGeometry } from 'three';
import { USDZExporter } from '../../../../examples/jsm/exporters/USDZExporter.js';
import { USDLoader } from '../../../../examples/jsm/loaders/USDLoader.js';
import { unzipSync, strFromU8 } from '../../../../examples/jsm/libs/fflate.module.js';

function isValidUSDA( usda ) {

	const header = usda.split( '\n' )[ 0 ];
	if ( header !== '#usda 1.0' ) return false;

	return true;

}

function closeTo( actual, expected, tolerance ) {

	expect( Math.abs( actual - expected ) ).toBeLessThanOrEqual( tolerance );

}

describe( 'Addons', () => {

	describe( 'Exporters', () => {

		describe( 'USDZExporter', () => {

			test( 'methods', () => {

				const exporter = new USDZExporter();
				expect( exporter instanceof USDZExporter ).toBeTruthy();
				expect( typeof exporter.parseAsync === 'function' ).toBeTruthy();
				expect( typeof exporter.parse === 'function' ).toBeTruthy();
				expect( typeof exporter.setTextureUtils === 'function' ).toBeTruthy();

			} );

			test( 'export basic scene', async () => {

				const exporter = new USDZExporter();

				const scene = new Scene();
				const geometry = new BoxGeometry( 1, 1, 1 );
				const material = new MeshStandardMaterial( {
					color: 0x00ff00,
					roughness: 0.5,
					metalness: 0.8,
				} );
				const mesh = new Mesh( geometry, material );
				mesh.name = 'box';
				scene.add( mesh );

				const result = await exporter.parseAsync( scene );

				expect( result.buffer instanceof ArrayBuffer ).toBeTruthy();
				expect( result.buffer.byteLength > 0 ).toBeTruthy();

				const unzipped = unzipSync( result );
				const fileNames = Object.keys( unzipped );

				const modelFileName = 'model.usda';

				expect( fileNames.length > 0 ).toBeTruthy();
				expect( fileNames[ 0 ] ).toBe( modelFileName );
				expect( isValidUSDA( strFromU8( unzipped[ modelFileName ] ) ) ).toBeTruthy();

			} );

			test( 'export scene with onlyVisible option', async () => {

				const exporter = new USDZExporter( );

				const scene = new Scene();

				const geometry = new BoxGeometry( 1, 1, 1 );
				const material1 = new MeshStandardMaterial( { color: 0xff0000 } );
				const material2 = new MeshStandardMaterial( { color: 0x00ff00 } );

				const box1 = new Mesh( geometry, material1 );
				box1.name = 'box1';
				box1.position.set( - 1, 0, 0 );

				const box2 = new Mesh( geometry, material2 );
				box2.name = 'box2';
				box2.position.set( 1, 0, 0 );
				box2.visible = false;

				scene.add( box1 );
				scene.add( box2 );

				// onlyVisible = true

				const options = {
					onlyVisible: true,
				};
				const exportResult = await exporter.parseAsync( scene, options );

				expect( exportResult.buffer instanceof ArrayBuffer ).toBeTruthy();
				expect( exportResult.buffer.byteLength > 0 ).toBeTruthy();

				const unzipped = unzipSync( exportResult );
				const fileNames = Object.keys( unzipped );
				const modelFileName = 'model.usda';

				expect( fileNames.includes( modelFileName ) ).toBeTruthy();

				const usdaContent = strFromU8( unzipped[ modelFileName ] );
				expect( isValidUSDA( usdaContent ) ).toBeTruthy();

				expect( usdaContent.includes( 'box1' ) ).toBeTruthy();
				expect( ! usdaContent.includes( 'box2' ) ).toBeTruthy();

				// onlyVisible = false

				options.onlyVisible = false;
				const exportResult2 = await exporter.parseAsync( scene, options );

				expect( exportResult2.buffer instanceof ArrayBuffer ).toBeTruthy();
				expect( exportResult2.buffer.byteLength > 0 ).toBeTruthy();

				const unzipped2 = unzipSync( exportResult2 );
				const fileNames2 = Object.keys( unzipped2 );

				expect( fileNames2.includes( modelFileName ) ).toBeTruthy();

				const usdaContent2 = strFromU8( unzipped2[ modelFileName ] );
				expect( isValidUSDA( usdaContent2 ) ).toBeTruthy();

				expect( usdaContent2.includes( 'box1' ) ).toBeTruthy();
				expect( usdaContent2.includes( 'box2' ) ).toBeTruthy();

			} );

			test( 'export and import', async () => {

				const exporter = new USDZExporter();

				const originalScene = new Scene();
				const boxGeometry = new BoxGeometry( 1, 1, 1 );
				const boxMaterial = new MeshStandardMaterial( {
					color: 0x00ff00,
					roughness: 0.5,
					metalness: 0.8,
				} );
				const box = new Mesh( boxGeometry, boxMaterial );
				box.name = 'box1';
				box.position.set( 1, 2, 3 );
				box.scale.set( 0.5, 1.5, 2.0 );
				box.rotation.set( Math.PI / 4, Math.PI / 3, Math.PI / 2 );
				originalScene.add( box );

				const sphereGeometry = new SphereGeometry( 1, 8, 6 );
				const sphereMaterial = new MeshStandardMaterial( {
					color: 0x0000ff,
					roughness: 0.9,
					metalness: 0.1,
				} );
				const sphere = new Mesh( sphereGeometry, sphereMaterial );
				sphere.name = 'sphere1';
				sphere.position.set( 0, 0, 0 );
				originalScene.add( sphere );

				const meshes = [ box, sphere ];

				originalScene.updateMatrixWorld( true );

				const exportResult = await exporter.parseAsync( originalScene );

				expect( exportResult.buffer instanceof ArrayBuffer ).toBeTruthy();

				const loader = new USDLoader();
				const importedScene = loader.parse( exportResult.buffer );

				expect( importedScene ).toBeTruthy();

				for ( const mesh of meshes ) {

					const name = mesh.name;

					const importedMesh = importedScene.getObjectByName( name );

					expect( importedMesh ).toBeTruthy();
					expect( importedMesh.name ).toBe( name );

					const tolerance = 0.0000001;
					const vectorCloseTo = ( a, b, tolerance ) => {

						closeTo( a.x, b.x, tolerance );
						closeTo( a.y, b.y, tolerance );
						closeTo( a.z, b.z, tolerance );

					};

					vectorCloseTo( importedMesh.position, mesh.position, tolerance );
					vectorCloseTo( importedMesh.scale, mesh.scale, tolerance );
					vectorCloseTo( importedMesh.rotation, mesh.rotation, tolerance );

					expect( importedMesh.geometry ).toBeTruthy();
					expect( importedMesh.geometry.attributes.position ).toBeTruthy();

					expect( importedMesh.material ).toBeTruthy();
					expect( importedMesh.material.isMeshStandardMaterial ).toBeTruthy();
					closeTo( importedMesh.material.color.r, mesh.material.color.r, tolerance );
					closeTo( importedMesh.material.color.g, mesh.material.color.g, tolerance );
					closeTo( importedMesh.material.color.b, mesh.material.color.b, tolerance );
					closeTo( importedMesh.material.roughness, mesh.material.roughness, tolerance );
					closeTo( importedMesh.material.metalness, mesh.material.metalness, tolerance );

				}

			} );

		} );

	} );

} );
