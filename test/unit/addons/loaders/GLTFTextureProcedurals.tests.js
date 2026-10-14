import { ClampToEdgeWrapping, DataTexture, Group, Mesh, MeshBasicMaterial, MeshPhysicalMaterial, MeshStandardMaterial, NearestFilter, NoColorSpace, PlaneGeometry, RepeatWrapping, SRGBColorSpace } from 'three';
import { GLTFLoader } from '../../../../examples/jsm/loaders/GLTFLoader.js';
import { GLTFExporter } from '../../../../examples/jsm/exporters/GLTFExporter.js';
import { GLTFTextureProceduralsLoaderExtension } from '../../../../examples/jsm/loaders/GLTFTextureProceduralsLoaderExtension.js';
import { GLTFTextureProceduralsExporterExtension } from '../../../../examples/jsm/exporters/GLTFTextureProceduralsExporterExtension.js';
import { createTextureProceduralMaterial, getTextureProcedurals, packTextureProceduralChannels, updateTextureProceduralMaterial } from '../../../../examples/jsm/utils/GLTFTextureProcedurals.js';
import { compileGLTFProceduralGraph, validateGLTFProceduralGraph } from '../../../../examples/jsm/loaders/materialx/GLTFProceduralGraph.js';

const EXTENSION = 'KHR_texture_procedurals';
const LIBRARY = 'EXT_texture_procedurals_mx_1_39';
const FIXTURES = '/examples/models/gltf/TextureProcedurals/';

function loader( registered = true ) {

	const result = new GLTFLoader();
	if ( registered ) result.register( parser => new GLTFTextureProceduralsLoaderExtension( parser ) );
	return result;

}

function exporter( registered = true ) {

	const result = new GLTFExporter();
	if ( registered ) result.register( writer => new GLTFTextureProceduralsExporterExtension( writer ) );
	return result;

}

async function fixture( name = 'checkerboard' ) {

	return ( await fetch( `${ FIXTURES }${ name }.gltf` ) ).json();

}

function parse( json, registered = true ) {

	return loader( registered ).parseAsync( JSON.stringify( json ), FIXTURES );

}

function material( result ) {

	let found;
	result.scene.traverse( object => {

		if ( object.isMesh ) found = object.material;

	} );
	return found;

}

function constant( type, value ) {

	return {
		nodetype: 'constant', type,
		inputs: { value: { nodetype: 'input', type, value } },
		outputs: { out: { nodetype: 'output', type } }
	};

}

function packedGraph() {

	return {
		name: 'packed channels', nodetype: 'nodegraph', type: 'multioutput', inputs: {},
		nodes: [ constant( 'color3', [ 0.4, 0.3, 0.7 ] ), constant( 'color4', [ 0.2, 0.5, 0.8, 0.6 ] ), constant( 'vector3', [ 0.5, 0.5, 1 ] ) ],
		outputs: {
			orm: { nodetype: 'output', type: 'color3', node: 0, output: 'out' },
			color: { nodetype: 'output', type: 'color4', node: 1, output: 'out' },
			normal: { nodetype: 'output', type: 'vector3', node: 2, output: 'out' }
		}
	};

}

function findNode( root, predicate, seen = new WeakSet() ) {

	if ( ! root || typeof root !== 'object' || seen.has( root ) ) return null;
	seen.add( root );
	if ( predicate( root ) ) return root;
	for ( const value of Object.values( root ) ) {

		const found = findNode( value, predicate, seen );
		if ( found ) return found;

	}

	return null;

}

QUnit.module( 'Addons', () => {

	QUnit.module( 'GLTFTextureProcedurals', () => {

		for ( const name of [ 'checkerboard', 'noise', 'texture', 'texture-external' ] ) {

			QUnit.test( `${ name } fixture loads, exports JSON/GLB and reloads`, async ( assert ) => {

				const json = await fixture( name );
				const loaded = await parse( json );
				assert.true( material( loaded ).isNodeMaterial, 'Graph compiles to a node material.' );
				assert.ok( material( loaded ).colorNode, 'Procedural color participates in shading.' );

				for ( const binary of [ false, true ] ) {

					const exported = await exporter().parseAsync( loaded.scene, { binary } );
					const reloaded = await loader().parseAsync( binary ? exported : JSON.stringify( exported ), FIXTURES );
					const state = getTextureProcedurals( material( reloaded ) );
					const expected = structuredClone( json.extensions[ EXTENSION ].procedurals );
					// Exported texture indices may differ, while every other graph property remains stable.
					for ( let i = 0; i < expected[ 0 ].nodes.length; i ++ ) {

						if ( expected[ 0 ].nodes[ i ].inputs.file ) expected[ 0 ].nodes[ i ].inputs.file.value = state.source.procedurals[ 0 ].nodes[ i ].inputs.file.value;

					}

					assert.deepEqual( state.source.procedurals, expected, `Graph survives ${ binary ? 'GLB' : 'JSON' }.` );
					assert.true( material( reloaded ).isNodeMaterial, 'Reload evaluates procedural content.' );
					if ( ! binary ) {

						assert.ok( exported.extensionsUsed.includes( EXTENSION ) && exported.extensionsUsed.includes( LIBRARY ), 'Both identifiers declared.' );
						assert.notOk( exported.extensionsRequired?.includes( EXTENSION ), 'Optional by default.' );
						assert.ok( exported.textures[ exported.materials[ 0 ].pbrMetallicRoughness.baseColorTexture.index ], 'Standard fallback remains valid.' );

					}

				}

			} );

		}

		QUnit.test( 'ordinary loader/exporter preserve fallback behavior without registration', async ( assert ) => {

			const loaded = await parse( await fixture(), false );
			assert.notOk( material( loaded ).isNodeMaterial, 'Unregistered loader uses ordinary material.' );
			assert.ok( material( loaded ).map, 'Fallback texture is present.' );
			const exported = await exporter( false ).parseAsync( loaded.scene );
			assert.notOk( exported.extensionsUsed?.includes( EXTENSION ), 'Ordinary exporter does not emit procedural content.' );

		} );

		QUnit.test( 'graph interfaces, named multioutput processing nodes and functional custom definitions', async ( assert ) => {

			const definitions = [ {
				nodetype: 'nodedef', name: 'ND_tint', node: 'custom_tint',
				inputs: { tint: { nodetype: 'input', type: 'color3', value: [ 0.2, 0.4, 0.6 ] } },
				outputs: { out: { nodetype: 'output', type: 'color3' } }
			}, {
				nodetype: 'nodegraph', nodedef: 0, type: 'color3',
				nodes: [ { ...constant( 'color3', [ 1, 1, 1 ] ), inputs: { value: { nodetype: 'input', type: 'color3', input: 'tint' } } } ],
				outputs: { out: { nodetype: 'output', type: 'color3', node: 0, output: 'out' } }
			} ];
			const graph = {
				nodetype: 'nodegraph', type: 'multioutput',
				inputs: { tint: { nodetype: 'input', type: 'color3', value: [ 0.1, 0.3, 0.7 ] } },
				nodes: [ {
					nodetype: 'custom_tint', type: 'color3',
					inputs: { tint: { nodetype: 'input', type: 'color3', input: 'tint' } },
					outputs: { out: { nodetype: 'output', type: 'color3' } }
				}, {
					nodetype: 'separate3', type: 'multioutput',
					inputs: { in: { nodetype: 'input', type: 'color3', node: 0, output: 'out' } },
					outputs: { outr: { nodetype: 'output', type: 'float' }, outg: { nodetype: 'output', type: 'float' }, outb: { nodetype: 'output', type: 'float' } }
				} ],
				outputs: { color: { nodetype: 'output', type: 'color3', node: 0, output: 'out' }, roughness: { nodetype: 'output', type: 'float', node: 1, output: 'outg' } }
			};
			const compiled = await compileGLTFProceduralGraph( graph, { definitions } );
			assert.ok( compiled.outputs.color?.isNode && compiled.outputs.roughness?.isNode, 'Custom functional graph and named node outputs compile.' );
			const authored = await createTextureProceduralMaterial( new MeshBasicMaterial(), { procedurals: [ graph ], procedural_definitions: definitions }, { baseColorTexture: { index: 0, output: 'color' } } );
			const exported = await exporter().parseAsync( new Mesh( new PlaneGeometry(), authored ) );
			assert.deepEqual( exported.extensions[ EXTENSION ].procedural_definitions, definitions, 'Definitions preserved.' );
			const reloaded = await parse( exported );
			assert.true( material( reloaded ).isNodeMaterial, 'Functional custom graph evaluates after reload.' );

		} );

		QUnit.test( 'explicit numbered streams and strict profile validation', async ( assert ) => {

			const graph = {
				nodetype: 'nodegraph', type: 'vector2', inputs: {},
				nodes: [ { nodetype: 'texcoord', type: 'vector2', inputs: { index: { nodetype: 'input', type: 'integer', value: [ 1 ] } }, outputs: { out: { nodetype: 'output', type: 'vector2' } } } ],
				outputs: { uv: { nodetype: 'output', type: 'vector2', node: 0, output: 'out' } }
			};
			const result = await compileGLTFProceduralGraph( graph );
			assert.ok( result.streams.some( stream => stream.semantic === 'TEXCOORD_1' ), 'Explicit UV1 required.' );
			const legacy = structuredClone( graph );
			legacy.inputs = [];
			assert.throws( () => validateGLTFProceduralGraph( legacy ), /named|map/i, 'Legacy port arrays rejected.' );
			const uv3 = structuredClone( graph );
			uv3.type = uv3.nodes[ 0 ].type = uv3.nodes[ 0 ].outputs.out.type = uv3.outputs.uv.type = 'vector3';
			assert.throws( () => validateGLTFProceduralGraph( uv3 ), /UV|texcoord|vector2|component/i, 'Three-component UVs rejected.' );
			const scalar = structuredClone( graph );
			scalar.nodes[ 0 ].inputs.index.value = 1;
			assert.throws( () => validateGLTFProceduralGraph( scalar ), /array/i, 'Legacy numeric scalar rejected.' );

		} );

		QUnit.test( 'shared named outputs, packed channels, factors, alpha and normals', async ( assert ) => {

			const source = { procedurals: [ packedGraph() ] };
			const bindings = {
				baseColorTexture: { index: 0, output: 'color' },
				metallicRoughnessTexture: { index: 0, output: 'orm' },
				occlusionTexture: { index: 0, output: 'orm' },
				normalTexture: { index: 0, output: 'normal' }
			};
			const fallback = new MeshStandardMaterial( { color: 0x80c0ff, metalness: 0.8, roughness: 0.6, opacity: 0.7, transparent: true, alphaTest: 0.2 } );
			fallback.normalScale.set( 0.4, 0.4 );
			fallback.aoMapIntensity = 0.5;
			const first = await createTextureProceduralMaterial( fallback, source, bindings );
			const second = await createTextureProceduralMaterial( fallback, source, { baseColorTexture: bindings.baseColorTexture } );
			assert.strictEqual( first.metalness, 0.8, 'Metallic factor retained.' );
			assert.strictEqual( first.roughness, 0.6, 'Roughness factor retained.' );
			assert.strictEqual( first.opacity, 0.7, 'Alpha factor retained.' );
			assert.true( first.transparent, 'Blend retained.' );
			assert.strictEqual( first.alphaTest, 0.2, 'Alpha cutoff retained.' );
			assert.deepEqual( first.normalScale.toArray(), [ 0.4, 0.4 ], 'Normal scale retained.' );
			assert.ok( first.metalnessNode && first.roughnessNode && first.aoNode && first.normalNode, 'Packed and normal channels compile.' );
			assert.strictEqual( findNode( first.roughnessNode, node => node.isSplitNode )?.components, 'y', 'Roughness reads G.' );
			assert.strictEqual( findNode( first.metalnessNode, node => node.isSplitNode )?.components, 'z', 'Metallic reads B.' );
			assert.strictEqual( findNode( first.aoNode, node => node.isSplitNode )?.components, 'x', 'Occlusion reads R.' );
			assert.ok( findNode( first.colorNode, node => node.property === 'color' ), 'Base color factor is applied to graph output.' );
			assert.ok( findNode( first.colorNode, node => node.isSplitNode && node.components === 'w' ), 'Graph alpha channel remains part of base color.' );
			const scene = new Group();
			scene.add( new Mesh( new PlaneGeometry(), first ), new Mesh( new PlaneGeometry(), second ) );
			const json = await exporter().parseAsync( scene );
			assert.strictEqual( json.extensions[ EXTENSION ].procedurals.length, 1, 'Shared graph exported once.' );
			assert.deepEqual( json.extensions[ EXTENSION ].procedurals[ 0 ].outputs, source.procedurals[ 0 ].outputs, 'Named outputs survive.' );
			const pbr = json.materials[ 0 ].pbrMetallicRoughness;
			assert.ok( pbr.metallicRoughnessTexture, 'Packed slot exported.' );
			assert.notOk( pbr.metallicTexture || pbr.roughnessTexture, 'No invented unpacked slots.' );
			assert.strictEqual( json.materials[ 0 ].normalTexture.scale, 0.4, 'glTF normal scale exported.' );
			assert.strictEqual( json.materials[ 0 ].occlusionTexture.strength, 0.5, 'Occlusion strength exported.' );

		} );

		QUnit.test( 'physical material extension channels preserve factors and packed components', async ( assert ) => {

			const graph = { nodetype: 'nodegraph', type: 'color4', inputs: {}, nodes: [ constant( 'color4', [ 0.2, 0.4, 0.6, 0.8 ] ) ], outputs: { out: { nodetype: 'output', type: 'color4', node: 0, output: 'out' } } };
			const channels = {
				clearcoatTexture: [ 'KHR_materials_clearcoat', 'clearcoatNode', 'r' ],
				clearcoatRoughnessTexture: [ 'KHR_materials_clearcoat', 'clearcoatRoughnessNode', 'g' ],
				clearcoatNormalTexture: [ 'KHR_materials_clearcoat', 'clearcoatNormalNode', 'rgb' ],
				transmissionTexture: [ 'KHR_materials_transmission', 'transmissionNode', 'r' ],
				thicknessTexture: [ 'KHR_materials_volume', 'thicknessNode', 'g' ],
				specularTexture: [ 'KHR_materials_specular', 'specularIntensityNode', 'a' ],
				specularColorTexture: [ 'KHR_materials_specular', 'specularColorNode', 'rgb' ],
				sheenColorTexture: [ 'KHR_materials_sheen', 'sheenNode', 'rgb' ],
				sheenRoughnessTexture: [ 'KHR_materials_sheen', 'sheenRoughnessNode', 'a' ],
				iridescenceTexture: [ 'KHR_materials_iridescence', 'iridescenceNode', 'r' ],
				iridescenceThicknessTexture: [ 'KHR_materials_iridescence', 'iridescenceThicknessNode', 'g' ]
			};
			const bindings = Object.fromEntries( Object.keys( channels ).map( channel => [ channel, { index: 0, output: 'out' } ] ) );
			const fallback = new MeshPhysicalMaterial( { clearcoat: 0.7, clearcoatRoughness: 0.3, transmission: 0.6, thickness: 0.4, sheen: 0.5, sheenRoughness: 0.8, iridescence: 0.2, specularIntensity: 0.9 } );
			fallback.clearcoatNormalScale.setScalar( 0.6 );
			fallback.sheenColor.setRGB( 0.2, 0.3, 0.4 );
			const authored = await createTextureProceduralMaterial( fallback, { procedurals: [ graph ] }, bindings );
			const json = await exporter().parseAsync( new Mesh( new PlaneGeometry(), authored ) );
			for ( const [ channel, [ extension, property, component ]] of Object.entries( channels ) ) {

				assert.ok( authored[ property ], `${ channel } compiles.` );
				const canonical = { r: 'x', g: 'y', b: 'z', a: 'w', rgb: 'xyz' }[ component ];
				assert.ok( findNode( authored[ property ], node => node.isSplitNode && node.components === canonical ), `${ channel } reads ${ component }.` );
				assert.ok( json.materials[ 0 ].extensions[ extension ][ channel ].extensions[ EXTENSION ], `${ channel } uses its standard texture-info slot.` );

			}

			assert.strictEqual( json.materials[ 0 ].extensions.KHR_materials_clearcoat.clearcoatNormalTexture.scale, 0.6, 'Clearcoat normal scale retained with dummy fallback.' );
			const loaded = await parse( json );
			assert.strictEqual( material( loaded ).clearcoat, 0.7, 'Clearcoat factor survives reload.' );
			assert.strictEqual( material( loaded ).thickness, 0.4, 'Thickness factor survives reload.' );

		} );

		QUnit.test( 'required authoring policy, nested runtime edits and invalid source metadata', async ( assert ) => {

			const source = ( await fixture() ).extensions[ EXTENSION ];
			const bindings = { baseColorTexture: { index: 0, output: 'pattern' } };
			const authored = await createTextureProceduralMaterial( new MeshBasicMaterial(), source, bindings, { required: true } );
			const mesh = new Mesh( new PlaneGeometry(), authored );
			const exported = await exporter().parseAsync( mesh );
			assert.ok( exported.extensionsRequired.includes( EXTENSION ) && exported.extensionsUsed.includes( LIBRARY ), 'Required content is an explicit policy; library declaration is asset-wide.' );
			const nested = findNode( authored.colorNode, node => node.isConstNode && node.value?.isColor );
			assert.ok( nested, 'Compiled graph contains a nested color constant.' );
			if ( nested ) {

				nested.value.r = 0.33;
				await assert.rejects( exporter().parseAsync( mesh ), /runtime|edit|serializ/i, 'Nested shader mutation cannot silently export stale source.' );

			}

			for ( const metadata of [ { callback: () => {} }, { value: undefined }, { value: Infinity } ] ) {

				await assert.rejects( createTextureProceduralMaterial( new MeshBasicMaterial(), { ...source, metadata }, bindings ), /JSON|serializ|finite/i, 'Invalid source metadata is diagnosed before copying.' );

			}

		} );

		QUnit.test( 'packing separate authored channels and anisotropy/bump mappings', async ( assert ) => {

			const graph = {
				nodetype: 'nodegraph', type: 'multioutput', inputs: {},
				nodes: [ constant( 'float', [ 0.3 ] ), constant( 'float', [ 0.7 ] ), constant( 'float', [ 0.4 ] ) ],
				outputs: {
					rough: { nodetype: 'output', type: 'float', node: 0, output: 'out' },
					metal: { nodetype: 'output', type: 'float', node: 1, output: 'out' },
					ao: { nodetype: 'output', type: 'float', node: 2, output: 'out' }
				}
			};
			const packed = packTextureProceduralChannels( graph, { roughness: 'rough', metallic: 'metal', occlusion: 'ao' } );
			assert.strictEqual( packed.nodes.at( - 1 ).nodetype, 'combine3', 'Standard combine3 packs channels.' );
			assert.strictEqual( packed.nodes.at( - 1 ).inputs.in1.node, 2, 'AO packed in R.' );
			assert.strictEqual( packed.nodes.at( - 1 ).inputs.in2.node, 0, 'Roughness packed in G.' );
			assert.strictEqual( packed.nodes.at( - 1 ).inputs.in3.node, 1, 'Metallic packed in B.' );
			const bindings = { metallicRoughnessTexture: { index: 0, output: 'orm' }, occlusionTexture: { index: 0, output: 'orm' } };
			const authored = await createTextureProceduralMaterial( new MeshStandardMaterial(), { procedurals: [ packed ] }, bindings );
			const scene = new Group();
			scene.add( new Mesh( new PlaneGeometry(), authored ), new Mesh( new PlaneGeometry(), await createTextureProceduralMaterial( new MeshStandardMaterial(), { procedurals: [ packed ] }, bindings ) ) );
			for ( const binary of [ false, true ] ) {

				const exported = await exporter().parseAsync( scene, { binary } );
				const loaded = await loader().parseAsync( binary ? exported : JSON.stringify( exported ), FIXTURES );
				assert.deepEqual( getTextureProcedurals( material( loaded ) ).source.procedurals, [ packed ], 'Packed authored shared graph survives JSON/GLB.' );

			}

			const fallback = new MeshPhysicalMaterial( { anisotropy: 0.6, anisotropyRotation: 0.3, bumpScale: 0.2 } );
			const direction = packedGraph();
			const physical = await createTextureProceduralMaterial( fallback, { procedurals: [ direction ] }, { anisotropyTexture: { index: 0, output: 'orm' }, bumpTexture: { index: 0, output: 'orm' } } );
			assert.ok( findNode( physical.anisotropyNode, node => node.property === 'anisotropyRotation' ), 'Anisotropy rotation remains active.' );
			assert.ok( findNode( physical.anisotropyNode, node => node.isSplitNode && node.components === 'z' ), 'Anisotropy strength reads B.' );
			assert.ok( findNode( physical.normalNode, node => node.property === 'bumpScale' ), 'Bump scale remains active.' );
			const json = await exporter().parseAsync( new Mesh( new PlaneGeometry(), physical ) );
			assert.ok( json.materials[ 0 ].extensions.KHR_materials_anisotropy.anisotropyTexture, 'Anisotropy standard texture-info exported.' );
			assert.ok( json.materials[ 0 ].extensions.EXT_materials_bump.bumpTexture, 'Bump standard texture-info exported.' );
			await assert.rejects( createTextureProceduralMaterial( fallback, { procedurals: [ direction ] }, { normalTexture: { index: 0, output: 'normal' }, bumpTexture: { index: 0, output: 'orm' } } ), /normal.*bump|bump.*normal/i, 'Simultaneous procedural bump/normal diagnosed.' );

		} );

		QUnit.test( 'missing stream fallback preserves final geometry material flags', async ( assert ) => {

			const json = await fixture();
			const graph = json.extensions[ EXTENSION ].procedurals[ 0 ];
			graph.nodes.push( { nodetype: 'texcoord', type: 'vector2', inputs: { index: { nodetype: 'input', type: 'integer', value: [ 1 ] } }, outputs: { out: { nodetype: 'output', type: 'vector2' } } } );
			graph.nodes[ 0 ].inputs.texcoord = { nodetype: 'input', type: 'vector2', node: 1, output: 'out' };
			json.meshes[ 0 ].primitives[ 0 ].attributes.COLOR_0 = 1;
			delete json.meshes[ 0 ].primitives[ 0 ].attributes.NORMAL;
			const loaded = await parse( json );
			const fallback = material( loaded );
			assert.notOk( fallback.isNodeMaterial, 'Missing UV1 uses ordinary fallback.' );
			assert.true( fallback.vertexColors, 'Geometry vertex color flag survives fallback.' );
			assert.true( fallback.flatShading, 'Missing-normal flat shading flag survives fallback.' );
			assert.ok( loaded.userData.textureProceduralsDiagnostics.length, 'Missing stream diagnostic returned.' );
			json.extensionsRequired = [ EXTENSION ];
			await assert.rejects( parse( json ), /TEXCOORD_1|stream/i, 'Required missing stream fails.' );

		} );

		QUnit.test( 'required missing root and unsupported graph interface animation fail', async ( assert ) => {

			const json = await fixture();
			delete json.extensions[ EXTENSION ];
			json.extensionsRequired = [ EXTENSION ];
			await assert.rejects( parse( json ), /root|procedural/i, 'Required declaration needs a root payload.' );
			const animated = await fixture();
			animated.extensionsUsed.push( 'KHR_animation_pointer' );
			animated.extensionsRequired = [ EXTENSION ];
			animated.animations = [ { channels: [ { sampler: 0, target: { extensions: { KHR_animation_pointer: { pointer: '/extensions/KHR_texture_procedurals/procedurals/0/inputs/color/value' } } } } ], samplers: [ { input: 0, output: 0 } ] } ];
			await assert.rejects( parse( animated ), /animat|pointer/i, 'Required procedural animation is explicitly rejected.' );
			delete animated.extensionsRequired;
			const optional = await parse( animated );
			assert.ok( optional.userData.textureProceduralsDiagnostics.some( diagnostic => /animation/i.test( diagnostic.message ) ), 'Optional graph-interface animation emits a diagnostic.' );
			assert.strictEqual( optional.animations[ 0 ].tracks.length, 0, 'Metadata preservation is not claimed as animation support.' );
			animated.extensionsRequired = [ 'KHR_animation_pointer' ];
			await assert.rejects( parse( animated ), /animat|pointer/i, 'Required animation pointer fails even if procedural extension is optional.' );

		} );

		QUnit.test( 'authored source edits serialize and uncompiled edits fail', async ( assert ) => {

			const json = await fixture();
			const authored = await createTextureProceduralMaterial( new MeshBasicMaterial(), json.extensions[ EXTENSION ], { baseColorTexture: { index: 0, output: 'pattern' } } );
			const mesh = new Mesh( new PlaneGeometry(), authored );
			getTextureProcedurals( authored ).source.procedurals[ 0 ].nodes[ 0 ].inputs.color1.value = [ 0.1, 0.2, 0.3 ];
			await assert.rejects( exporter().parseAsync( mesh ), /edit|compil|stale|changed/i, 'Uncompiled edit is diagnosed.' );
			await updateTextureProceduralMaterial( authored );
			const exported = await exporter().parseAsync( mesh );
			assert.deepEqual( exported.extensions[ EXTENSION ].procedurals[ 0 ].nodes[ 0 ].inputs.color1.value, [ 0.1, 0.2, 0.3 ], 'Supported edit exported.' );
			authored.colorNode = null;
			await assert.rejects( exporter().parseAsync( mesh ), /node|edit|changed|serializ/i, 'Runtime node replacement is diagnosed.' );

		} );

		QUnit.test( 'texture-backed authored graph remaps resource indices deterministically', async ( assert ) => {

			const json = await fixture( 'texture' );
			json.extensions[ EXTENSION ].procedurals[ 0 ].nodes[ 0 ].inputs.file.value = [ 0 ];
			const resource = new DataTexture( new Uint8Array( [ 255, 0, 0, 255 ] ), 1, 1 );
			resource.colorSpace = SRGBColorSpace;
			resource.needsUpdate = true;
			const procedural = await createTextureProceduralMaterial( new MeshBasicMaterial(), json.extensions[ EXTENSION ], { baseColorTexture: { index: 0, output: 'pattern' } }, { textures: [ resource ] } );
			const mesh = new Mesh( new PlaneGeometry(), procedural );
			const first = await exporter().parseAsync( mesh );
			const second = await exporter().parseAsync( mesh );
			assert.deepEqual( first, second, 'Repeated export is deterministic.' );
			const index = first.extensions[ EXTENSION ].procedurals[ 0 ].nodes[ 0 ].inputs.file.value[ 0 ];
			assert.ok( first.textures[ index ] && first.images[ first.textures[ index ].source ], 'Filename addresses exported resource.' );
			const loaded = await parse( first );
			assert.ok( getTextureProcedurals( material( loaded ) ).textures[ index ].image, 'Resource ready after async load.' );

		} );

		QUnit.test( 'graph image color interpretation, coordinates, sampling and async readiness', async ( assert ) => {

			const graph = {
				nodetype: 'nodegraph', type: 'multioutput',
				inputs: { resource: { nodetype: 'input', type: 'filename', value: [ 0 ] } },
				nodes: [ {
					nodetype: 'image', type: 'color3', colorspace: 'srgb_texture',
					inputs: { file: { nodetype: 'input', type: 'filename', input: 'resource' }, uaddressmode: { nodetype: 'input', type: 'string', value: 'clamp' }, filtertype: { nodetype: 'input', type: 'string', value: 'closest' } },
					outputs: { out: { nodetype: 'output', type: 'color3' } }
				}, {
					nodetype: 'image', type: 'color3',
					inputs: { file: { nodetype: 'input', type: 'filename', input: 'resource' } },
					outputs: { out: { nodetype: 'output', type: 'color3' } }
				} ],
				outputs: { decoded: { nodetype: 'output', type: 'color3', node: 0, output: 'out' }, raw: { nodetype: 'output', type: 'color3', node: 1, output: 'out' } }
			};
			const resource = new DataTexture( new Uint8Array( [ 128, 128, 128, 255 ] ), 1, 1 );
			resource.colorSpace = SRGBColorSpace;
			resource.offset.set( 0.3, 0.4 );
			resource.repeat.set( 2, 3 );
			resource.rotation = 0.7;
			resource.channel = 3;
			resource.wrapS = RepeatWrapping;
			resource.updateMatrix();
			const originalMatrix = resource.matrix.toArray();
			let release;
			let started;
			const ready = new Promise( resolve => {

				started = resolve;

			} );
			const pending = new Promise( resolve => {

				release = resolve;

			} );
			let resolved = false;
			const promise = compileGLTFProceduralGraph( graph, { textureCount: 1, resolveTexture: async () => {

				started();
				return pending;

			} } ).then( result => {

				resolved = true;
				return result;

			} );
			await ready;
			assert.false( resolved, 'Graph compilation waits for async resource readiness.' );
			release( resource );
			const result = await promise;
			const sample = findNode( result.outputs.decoded, node => node.isTextureNode );
			assert.notStrictEqual( sample.value, resource, 'Graph sampling uses its own texture state.' );
			assert.strictEqual( sample.value.colorSpace, NoColorSpace, 'Color decoding belongs to the graph, rather than glTF fallback texture state.' );
			assert.strictEqual( sample.value.wrapS, ClampToEdgeWrapping, 'Image node clamp overrides fallback wrap.' );
			assert.strictEqual( sample.value.minFilter, NearestFilter, 'Image node closest overrides fallback filtering.' );
			assert.deepEqual( sample.value.offset.toArray(), [ 0, 0 ], 'Fallback transform ignored.' );
			assert.deepEqual( sample.value.repeat.toArray(), [ 1, 1 ], 'Fallback scale ignored.' );
			assert.strictEqual( sample.value.channel, 0, 'Fallback texCoord ignored.' );
			assert.false( sample.value.flipY, 'glTF image orientation retained.' );
			assert.deepEqual( resource.matrix.toArray(), originalMatrix, 'Fallback texture transform remains untouched.' );
			const files = [ ...result.document.nodesXLib.values() ].filter( node => node.type === 'filename' && node.parent.element === 'image' );
			assert.strictEqual( files.filter( file => file.getAttribute( 'colorspace' ) === 'srgb_texture' ).length, 1, 'sRGB decoding is explicit on one image input.' );
			assert.strictEqual( files.filter( file => ! file.getAttribute( 'colorspace' ) ).length, 1, 'Shared filename does not leak decoding to raw image input.' );
			const colors = { nodetype: 'nodegraph', type: 'color3', inputs: {}, nodes: [ { nodetype: 'geomcolor', type: 'color3', inputs: { index: { nodetype: 'input', type: 'integer', value: [ 1 ] } }, outputs: { out: { nodetype: 'output', type: 'color3' } } } ], outputs: { out: { nodetype: 'output', type: 'color3', node: 0, output: 'out' } } };
			const compiledColors = await compileGLTFProceduralGraph( colors );
			assert.deepEqual( compiledColors.streams[ 0 ], { semantic: 'COLOR_1', attribute: 'color_1', itemSize: 3 }, 'Numbered COLOR_1 is explicit.' );

		} );

		QUnit.test( 'imported ready resource replacement invalidates compilation and exports the edit', async ( assert ) => {

			const loaded = await parse( await fixture( 'texture' ) );
			const imported = material( loaded );
			const state = getTextureProcedurals( imported );
			const index = state.source.procedurals[ 0 ].nodes[ 0 ].inputs.file.value[ 0 ];
			const replacement = new DataTexture( new Uint8Array( [ 220, 80, 40, 255 ] ), 1, 1 );
			replacement.needsUpdate = true;
			state.textures[ index ] = replacement;
			await assert.rejects( exporter().parseAsync( loaded.scene ), /resource|compil|changed/i, 'Resource edit needs recompilation.' );
			await updateTextureProceduralMaterial( imported );
			const sample = findNode( imported.colorNode, node => node.isTextureNode );
			assert.strictEqual( sample.value.image, replacement.image, 'Loader graph cache uses edited ready image.' );
			const json = await exporter().parseAsync( loaded.scene );
			const reloaded = await parse( json );
			const image = findNode( material( reloaded ).colorNode, node => node.isTextureNode ).value.image;
			assert.strictEqual( image.width, 1, 'Edited resource survives export/reload.' );

		} );

		QUnit.test( 'UV4 and COLOR1 geometry streams survive loader/exporter round trips', async ( assert ) => {

			const json = await fixture();
			json.meshes[ 0 ].primitives[ 0 ].attributes.TEXCOORD_4 = 2;
			json.meshes[ 0 ].primitives[ 0 ].attributes.COLOR_1 = 1;
			const graph = json.extensions[ EXTENSION ].procedurals[ 0 ];
			graph.nodes.push( {
				nodetype: 'texcoord', type: 'vector2',
				inputs: { index: { nodetype: 'input', type: 'integer', value: [ 4 ] } },
				outputs: { out: { nodetype: 'output', type: 'vector2' } }
			}, {
				nodetype: 'geomcolor', type: 'color3',
				inputs: { index: { nodetype: 'input', type: 'integer', value: [ 1 ] } },
				outputs: { out: { nodetype: 'output', type: 'color3' } }
			} );
			graph.nodes[ 0 ].inputs.texcoord = { nodetype: 'input', type: 'vector2', node: 1, output: 'out' };
			graph.nodes[ 0 ].inputs.color1 = { nodetype: 'input', type: 'color3', node: 2, output: 'out' };
			json.extensionsRequired = [ EXTENSION ];
			const loaded = await parse( json );
			assert.true( material( loaded ).isNodeMaterial, 'Numbered geometry graph compiles.' );
			for ( const binary of [ false, true ] ) {

				const exported = await exporter().parseAsync( loaded.scene, { binary } );
				if ( ! binary ) {

					assert.ok( exported.meshes[ 0 ].primitives[ 0 ].attributes.TEXCOORD_4 !== undefined, 'UV4 exports with its glTF semantic.' );
					assert.ok( exported.meshes[ 0 ].primitives[ 0 ].attributes.COLOR_1 !== undefined, 'COLOR1 exports with its glTF semantic.' );

				}

				const reloaded = await loader().parseAsync( binary ? exported : JSON.stringify( exported ), FIXTURES );
				assert.true( material( reloaded ).isNodeMaterial, 'Numbered streams evaluate after JSON/GLB reload.' );

			}

		} );

		QUnit.test( 'nonserializable normal scales and unsupported normal UV frame fail clearly', async ( assert ) => {

			const source = { procedurals: [ packedGraph() ] };
			const bindings = { normalTexture: { index: 0, output: 'normal' } };
			for ( const scale of [[ - 1, 1 ], [ 0.2, 0.6 ]] ) {

				const fallback = new MeshStandardMaterial();
				fallback.normalScale.fromArray( scale );
				const authored = await createTextureProceduralMaterial( fallback, source, bindings );
				await assert.rejects( exporter().parseAsync( new Mesh( new PlaneGeometry(), authored ) ), /normal.*scale|scale.*normal/i, 'Nonserializable normal scale diagnosed.' );

			}

			const uv = structuredClone( source );
			uv.procedurals[ 0 ].nodes.push( { nodetype: 'texcoord', type: 'vector2', inputs: { index: { nodetype: 'input', type: 'integer', value: [ 1 ] } }, outputs: { out: { nodetype: 'output', type: 'vector2' } } } );
			await assert.rejects( createTextureProceduralMaterial( new MeshStandardMaterial(), uv, bindings ), /TEXCOORD_0|UV0/i, 'Procedural normal currently needs UV0 tangent frame.' );

		} );

		const invalidGraphs = {
			'unknown node': graph => {

				graph.nodes[ 0 ].nodetype = 'execute_arbitrary_source';

			},
			'bad node index': graph => {

				graph.outputs.pattern.node = 99;

			},
			'bad output port': graph => {

				graph.outputs.pattern.output = 'missing';

			},
			'invalid value type': graph => {

				graph.nodes[ 0 ].inputs.color1.value = [ 'red', 0, 0 ];

			},
			'incompatible type': graph => {

				graph.nodes[ 0 ].inputs.color1.type = 'vector2'; graph.nodes[ 0 ].inputs.color1.value = [ 1, 0 ];

			},
			'cycle': graph => {

				graph.nodes[ 0 ].inputs.color1 = { nodetype: 'input', type: 'color3', node: 0, output: 'out' };

			},
			'bad graph input': graph => {

				graph.nodes[ 0 ].inputs.color1 = { nodetype: 'input', type: 'color3', input: 'missing' };

			}
		};

		for ( const [ name, mutate ] of Object.entries( invalidGraphs ) ) {

			QUnit.test( `${ name }: optional fallback and required failure`, async ( assert ) => {

				const json = await fixture();
				mutate( json.extensions[ EXTENSION ].procedurals[ 0 ] );
				const optional = await parse( json );
				assert.notOk( material( optional ).colorNode, 'Optional unsupported graph uses ordinary fallback.' );
				assert.ok( material( optional ).map, 'Fallback resource retained.' );
				json.extensionsRequired = [ EXTENSION, LIBRARY ];
				await assert.rejects( parse( json ), /procedural|graph|node|input|output|type|cycle/i, 'Required graph fails with a useful diagnostic.' );

			} );

		}

		QUnit.test( 'invalid graph/texture references and unsupported library declarations', async ( assert ) => {

			for ( const mutate of [
				json => {

					json.materials[ 0 ].pbrMetallicRoughness.baseColorTexture.extensions[ EXTENSION ].index = 9;

				},
				json => {

					json.extensions[ EXTENSION ].procedurals[ 0 ].nodes = [ { ...constant( 'color3', [ 1, 1, 1 ] ), nodetype: 'image', inputs: { file: { nodetype: 'input', type: 'filename', value: [ 99 ] } } } ];

				},
				json => {

					json.extensionsUsed.push( 'EXT_texture_procedurals_mx_1_38' );

				},
				json => {

					json.extensionsUsed = json.extensionsUsed.filter( name => name !== LIBRARY );

				}
			] ) {

				const json = await fixture();
				mutate( json );
				json.extensionsRequired = [ EXTENSION ];
				await assert.rejects( parse( json ), /procedural|texture|library|MaterialX|version|index/i, 'Invalid required declaration/reference fails.' );

			}

		} );

		QUnit.test( 'optional library conflicts and unexpected companion payload use diagnostics and fallback', async ( assert ) => {

			for ( const mutate of [
				json => {

					json.extensionsUsed.push( 'EXT_texture_procedurals_mx_1_38' );

				},
				json => {

					json.extensions[ LIBRARY ] = {};

				}
			] ) {

				const json = await fixture();
				mutate( json );
				const loaded = await parse( json );
				assert.notOk( material( loaded ).isNodeMaterial, 'Unsupported optional declaration retains fallback.' );
				assert.ok( material( loaded ).map, 'Standard texture remains usable.' );
				assert.ok( loaded.userData.textureProceduralsDiagnostics.length, 'Optional failure emits diagnostics.' );

			}

		} );

	} );

} );
