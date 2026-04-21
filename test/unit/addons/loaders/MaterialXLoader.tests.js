import { MaterialXLoader } from '../../../../examples/jsm/loaders/MaterialXLoader.js';
import { readFileSync } from 'node:fs';

const curatedMaterialXSamples = [
	{ filename: 'open_pbr_pearl_thinfilm.mtlx', surface: 'open_pbr_surface', usesTexture: false },
	{ filename: 'open_pbr_soapbubble_transmission.mtlx', surface: 'open_pbr_surface', usesTexture: false },
	{ filename: 'open_pbr_brushed_metal_anisotropy.mtlx', surface: 'open_pbr_surface', usesTexture: false },
	{ filename: 'open_pbr_velvet_fuzz.mtlx', surface: 'open_pbr_surface', usesTexture: false },
	{ filename: 'gltf_pbr_carpaint_clearcoat.mtlx', surface: 'gltf_pbr', usesTexture: false },
	{ filename: 'gltf_pbr_glass_dispersion.mtlx', surface: 'gltf_pbr', usesTexture: false },
	{ filename: 'gltf_pbr_gold_metal.mtlx', surface: 'gltf_pbr', usesTexture: false },
	{ filename: 'gltf_pbr_default_feature_sweep.mtlx', surface: 'gltf_pbr', usesTexture: false },
	{ filename: 'standard_surface_marble_veins.mtlx', surface: 'standard_surface', usesTexture: false },
	{ filename: 'standard_surface_jade_translucent.mtlx', surface: 'standard_surface', usesTexture: false },
	{ filename: 'standard_surface_wood_grain.mtlx', surface: 'standard_surface', usesTexture: true },
	{ filename: 'standard_surface_logic_composite_nodes.mtlx', surface: 'standard_surface', usesTexture: false }
];

function readCuratedSample( filename ) {

	const samplePath = new URL( `../../../../examples/materialx/${ filename }`, import.meta.url );
	return readFileSync( samplePath, 'utf8' );

}

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'Loaders', () => {

		QUnit.module( 'MaterialXLoader', () => {

			QUnit.test( 'Instancing', ( assert ) => {

				const loader = new MaterialXLoader();
				assert.ok( loader instanceof MaterialXLoader, 'Can instantiate a MaterialXLoader.' );

			} );

			QUnit.test( 'parse returns materials and report', ( assert ) => {

				const xml = [
					'<materialx version="1.38">',
					'  <standard_surface name="test_surface" type="surfaceshader">',
					'    <input name="base_color" type="color3" value="1.0,0.0,0.0" />',
					'  </standard_surface>',
					'  <surfacematerial name="test_material" type="material">',
					'    <input name="surfaceshader" type="surfaceshader" nodename="test_surface" />',
					'  </surfacematerial>',
					'</materialx>'
				].join( '\n' );

				const loader = new MaterialXLoader().setUnsupportedPolicy( 'ignore' );
				const result = loader.parse( xml );

				assert.ok( result.materials.test_material, 'Compiled surfacematerial exists.' );
				assert.ok( result.report, 'Includes diagnostics report.' );
				assert.ok( Array.isArray( result.report.issues ), 'Issues are reported as array.' );
				assert.ok( Array.isArray( result.report.ignoredSurfaceInputs ), 'Ignored surface input listing exists.' );
				assert.ok( Array.isArray( result.report.missingReferences ), 'Missing reference listing exists.' );
				assert.ok( Array.isArray( result.report.invalidValues ), 'Invalid value listing exists.' );

			} );

			QUnit.test( 'unsupported policy "error" throws when unsupported nodes are present', ( assert ) => {

				const xml = [
					'<materialx version="1.38">',
					'  <nodegraph name="ng1">',
					'    <mysterynode name="m1" type="float" />',
					'    <output name="out" type="float" nodename="m1" />',
					'  </nodegraph>',
					'</materialx>'
				].join( '\n' );

				const loader = new MaterialXLoader().setUnsupportedPolicy( 'error' );

				assert.throws( () => {

					loader.parse( xml );

				}, /MaterialX translation reported/, 'Error policy fails parse when issues exist.' );

			} );

			QUnit.test( 'curated local MaterialX suite parses and provides 4/4/4 surface coverage', ( assert ) => {

				const loader = new MaterialXLoader().setUnsupportedPolicy( 'ignore' );
				const surfaceCounts = {
					open_pbr_surface: 0,
					gltf_pbr: 0,
					standard_surface: 0
				};
				let texturedSamples = 0;

				for ( const sample of curatedMaterialXSamples ) {

					const xml = readCuratedSample( sample.filename );
					const result = loader.parse( xml );

					assert.ok( Object.keys( result.materials ).length > 0, `${ sample.filename }: has at least one compiled surfacematerial.` );
					assert.ok( xml.includes( `<${ sample.surface }` ), `${ sample.filename }: includes expected ${ sample.surface } node.` );

					surfaceCounts[ sample.surface ] ++;

					const hasTextureNode = /<(image|tiledimage)\b/.test( xml );
					if ( hasTextureNode ) texturedSamples ++;

					if ( sample.usesTexture ) {

						assert.ok( hasTextureNode, `${ sample.filename }: texture usage expected by manifest.` );

					}

				}

				assert.strictEqual( surfaceCounts.open_pbr_surface, 4, 'Has 4 OpenPBR samples.' );
				assert.strictEqual( surfaceCounts.gltf_pbr, 4, 'Has 4 glTF PBR samples.' );
				assert.strictEqual( surfaceCounts.standard_surface, 4, 'Has 4 Standard Surface samples.' );
				assert.ok( texturedSamples >= 1, 'At least one curated sample exercises image/tiledimage nodes.' );

			} );

			QUnit.test( 'open_pbr geometry_thin_walled maps to thickness and is not ignored', ( assert ) => {

				const xml = [
					'<materialx version="1.39">',
					'  <surfacematerial name="test_material" type="material">',
					'    <input name="surfaceshader" type="surfaceshader" nodename="test_surface" />',
					'  </surfacematerial>',
					'  <open_pbr_surface name="test_surface" type="surfaceshader">',
					'    <input name="transmission_depth" type="float" value="0.7" />',
					'    <input name="geometry_thin_walled" type="boolean" value="true" />',
					'  </open_pbr_surface>',
					'</materialx>'
				].join( '\n' );

				const loader = new MaterialXLoader().setUnsupportedPolicy( 'ignore' );
				const result = loader.parse( xml );
				const material = result.materials.test_material;

				assert.ok( material, 'Compiled material exists.' );
				assert.notOk(
					result.report.ignoredSurfaceInputs.some( ( issue ) => issue.message.includes( '"geometry_thin_walled"' ) ),
					'geometry_thin_walled is recognized and not reported as ignored.'
				);
				assert.strictEqual( material.thicknessNode.constructor.name, 'ConditionalNode', 'geometry_thin_walled drives conditional thickness logic.' );
				assert.strictEqual( material.thicknessNode.ifNode.value, 0, 'Thin-walled=true branch forces zero thickness.' );
				assert.strictEqual( material.thicknessNode.elseNode.value, 0.7, 'Non-thin-walled branch uses transmission_depth.' );

			} );

		} );

	} );

} );
