import { ClampToEdgeWrapping, Color, MeshPhysicalMaterial, MeshStandardMaterial, Texture } from 'three';
import { MaterialXLoader } from '../../../../examples/jsm/loaders/MaterialXLoader.js';
import { MaterialXExporter } from '../../../../examples/jsm/exporters/MaterialXExporter.js';

// 1x1 red pixel. A data URI isn't valid MaterialX (`.mtlx` has no such convention - it's checked
// here purely as a URL the exporter can pass through as-is and our own loader can fetch back).
const PIXEL_PNG = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=';

function createTexture() {

	const texture = new Texture();
	texture.image = { src: PIXEL_PNG };
	return texture;

}

const MTLX_TEXT = `<?xml version="1.0"?>
<materialx version="1.39" colorspace="lin_rec709">
  <standard_surface name="SR_test" type="surfaceshader">
    <input name="base_color" type="color3" value="0.8, 0.2, 0.1" />
    <input name="specular_roughness" type="float" value="0.4" />
    <input name="metalness" type="float" value="0.6" />
  </standard_surface>
  <surfacematerial name="test_material" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="SR_test" />
  </surfacematerial>
</materialx>
`;

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'Exporters', () => {

		QUnit.module( 'MaterialXExporter', () => {

			QUnit.test( 'methods', ( assert ) => {

				const exporter = new MaterialXExporter();
				assert.ok( exporter instanceof MaterialXExporter, 'MaterialXExporter can be instantiated' );
				assert.ok( typeof exporter.parse === 'function', 'parse method exists' );

			} );

			QUnit.test( 'export and re-import', ( assert ) => {

				const loader = new MaterialXLoader();
				const { materials, document } = loader.parse( MTLX_TEXT );

				assert.ok( materials.test_material, 'Loader parses the original document' );
				assert.ok( materials.test_material.isMeshPhysicalNodeMaterial, 'Material is a MeshPhysicalNodeMaterial' );

				const exporter = new MaterialXExporter();
				const exportedText = exporter.parse( document );

				assert.ok( typeof exportedText === 'string', 'Export returns a string' );
				assert.ok( exportedText.includes( 'standard_surface' ), 'Exported text contains the standard_surface node' );
				assert.ok( exportedText.includes( 'value="0.4"' ), 'Exported text preserves input values' );

				const reimported = loader.parse( exportedText );

				assert.ok( reimported.materials.test_material, 'Re-imported document contains the same material' );
				assert.ok( reimported.materials.test_material.isMeshPhysicalNodeMaterial, 'Re-imported material is a MeshPhysicalNodeMaterial' );
				assert.equal( reimported.errors.length, 0, 'Re-import produces no errors' );

			} );

			QUnit.test( 'export a MeshStandardMaterial', ( assert ) => {

				const texture = createTexture();

				const material = new MeshStandardMaterial( {
					name: 'flat_material',
					color: 0x8020ff,
					roughness: 0.4,
					metalness: 0.6,
					map: texture,
				} );

				const exporter = new MaterialXExporter();
				const exportedText = exporter.parse( material );

				assert.ok( typeof exportedText === 'string', 'Export returns a string' );
				assert.ok( exportedText.includes( 'standard_surface' ), 'Exported text contains a standard_surface node' );
				assert.ok( exportedText.includes( PIXEL_PNG ), 'Exported text references the base_color texture' );
				assert.ok( exportedText.includes( 'value="0.4"' ), 'Exported text contains the roughness value' );
				assert.ok( exportedText.includes( 'value="0.6"' ), 'Exported text contains the metalness value' );

				const loader = new MaterialXLoader();
				const reimported = loader.parse( exportedText );

				assert.ok( reimported.materials.flat_material, 'Re-imported document contains the exported material' );
				assert.ok( reimported.materials.flat_material.isMeshPhysicalNodeMaterial, 'Re-imported material is a MeshPhysicalNodeMaterial' );
				assert.equal( reimported.errors.length, 0, 'Re-import produces no errors' );

			} );

			QUnit.test( 'export a MeshPhysicalMaterial\'s extra attributes', ( assert ) => {

				const material = new MeshPhysicalMaterial( {
					name: 'physical_material',
					clearcoat: 0.5,
					clearcoatRoughness: 0.2,
					sheen: 1,
					sheenColor: new Color( 0x224488 ),
					sheenRoughness: 0.3,
					transmission: 0.7,
					ior: 1.4,
					specularIntensity: 0.8,
					specularColor: new Color( 0xffddcc ),
					emissive: new Color( 0x112233 ),
					emissiveIntensity: 2,
				} );

				const exportedText = new MaterialXExporter().parse( material );

				for ( const expected of [
					'name="coat" type="float" value="0.5"',
					'name="coat_roughness" type="float" value="0.2"',
					'name="sheen" type="float" value="1"',
					'name="sheen_roughness" type="float" value="0.3"',
					'name="transmission" type="float" value="0.7"',
					'name="specular_IOR" type="float" value="1.4"',
					'name="specular" type="float" value="0.8"',
					'name="emission" type="float" value="2"',
				] ) {

					assert.ok( exportedText.includes( expected ), `Exported text contains ${expected}` );

				}

				const reimported = new MaterialXLoader().parse( exportedText );

				assert.ok( reimported.materials.physical_material, 'Re-imported document contains the exported material' );
				assert.equal( reimported.errors.length, 0, 'Re-import produces no errors' );

			} );

			QUnit.test( 'export texture transform, wrapping and UV index', ( assert ) => {

				const texture = createTexture();
				texture.offset.set( 0.25, 0.1 );
				texture.repeat.set( 2, 3 );
				texture.rotation = Math.PI / 4;
				texture.wrapS = texture.wrapT = ClampToEdgeWrapping;
				texture.channel = 1;

				const material = new MeshStandardMaterial( { name: 'uv_material', map: texture } );

				const exportedText = new MaterialXExporter().parse( material );

				assert.ok( exportedText.includes( '<place2d' ), 'Exported text wires a place2d node for the texture transform' );
				assert.ok( exportedText.includes( 'name="index" type="integer" value="1"' ), 'Exported text uses the texture\'s UV channel' );
				assert.ok( exportedText.includes( 'uaddressmode" type="string" value="clamp"' ), 'Exported text preserves wrapS' );
				assert.ok( exportedText.includes( 'vaddressmode" type="string" value="clamp"' ), 'Exported text preserves wrapT' );

				const reimported = new MaterialXLoader().parse( exportedText );

				assert.ok( reimported.materials.uv_material, 'Re-imported document contains the exported material' );
				assert.equal( reimported.errors.length, 0, 'Re-import produces no errors' );

			} );

			QUnit.test( 'export a shared roughness/metalness texture via separate channels', ( assert ) => {

				const ormTexture = createTexture();

				const material = new MeshStandardMaterial( {
					name: 'orm_material',
					roughnessMap: ormTexture,
					metalnessMap: ormTexture,
				} );

				const exportedText = new MaterialXExporter().parse( material );

				const imageNodeCount = ( exportedText.match( /<image /g ) || [] ).length;
				assert.equal( imageNodeCount, 1, 'The shared texture is only exported as a single image node' );
				assert.ok( exportedText.includes( 'name="index" type="integer" value="1"' ), 'Roughness reads the green channel' );
				assert.ok( exportedText.includes( 'name="index" type="integer" value="2"' ), 'Metalness reads the blue channel' );

				const reimported = new MaterialXLoader().parse( exportedText );

				assert.ok( reimported.materials.orm_material, 'Re-imported document contains the exported material' );
				assert.equal( reimported.errors.length, 0, 'Re-import produces no errors' );

			} );

			QUnit.test( 'references a texture\'s existing relative path as-is', ( assert ) => {

				// The portable, spec-conformant case: .mtlx `filename` inputs are plain
				// filesystem paths, matching every sample file in examples/materialx/.
				const texture = new Texture();
				texture.image = { src: 'textures/diffuse.png' };

				const material = new MeshStandardMaterial( { name: 'path_material', map: texture } );

				const exportedText = new MaterialXExporter().parse( material );

				assert.ok( exportedText.includes( 'value="textures/diffuse.png"' ), 'Exported text references the texture by its own relative path, unmodified' );
				assert.notOk( exportedText.includes( 'data:' ), 'Exported text does not invent a data URI' );

			} );

			QUnit.test( 'skips a texture with no resolvable URL', ( assert ) => {

				const texture = new Texture(); // no .image set

				const material = new MeshStandardMaterial( { name: 'no_url_material', color: 0xff0000, map: texture } );

				const exportedText = new MaterialXExporter().parse( material );

				assert.notOk( exportedText.includes( '<image' ), 'No image node is emitted for the unresolvable texture' );
				assert.ok( exportedText.includes( 'name="base_color" type="color3" value=' ), 'base_color falls back to the constant color value' );

			} );

		} );

	} );

} );
