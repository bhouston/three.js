import { MeshBasicNodeMaterial, MeshPhysicalNodeMaterial } from 'three/webgpu';
import { Fn, materialReference, vec2, vec4, normalMap, bumpMap, mix } from 'three/tsl';
import { cloneGLTFProceduralSource, collectGLTFProceduralTextureIndices, compileGLTFProceduralGraph } from '../loaders/materialx/GLTFProceduralGraph.js';

const EXTENSION_NAME = 'KHR_texture_procedurals';
const LIBRARY_NAME = 'EXT_texture_procedurals_mx_1_39';
const states = new WeakMap();
const functions = new WeakMap();
let functionSerial = 0;

// Paths are glTF texture-info paths, not additional material slots.
const PROCEDURAL_CHANNELS = {

	baseColorTexture: { path: [ 'pbrMetallicRoughness', 'baseColorTexture' ], map: 'map', properties: [ 'colorNode' ], width: 3 },
	metallicRoughnessTexture: { path: [ 'pbrMetallicRoughness', 'metallicRoughnessTexture' ], map: 'roughnessMap', properties: [ 'roughnessNode', 'metalnessNode' ], width: 3 },
	normalTexture: { path: [ 'normalTexture' ], map: 'normalMap', properties: [ 'normalNode' ], width: 3 },
	occlusionTexture: { path: [ 'occlusionTexture' ], map: 'aoMap', properties: [ 'aoNode' ], width: 1 },
	emissiveTexture: { path: [ 'emissiveTexture' ], map: 'emissiveMap', properties: [ 'emissiveNode' ], width: 3 },
	clearcoatTexture: { path: [ 'extensions', 'KHR_materials_clearcoat', 'clearcoatTexture' ], map: 'clearcoatMap', properties: [ 'clearcoatNode' ], factor: 'clearcoat', component: 'r', width: 1 },
	clearcoatRoughnessTexture: { path: [ 'extensions', 'KHR_materials_clearcoat', 'clearcoatRoughnessTexture' ], map: 'clearcoatRoughnessMap', properties: [ 'clearcoatRoughnessNode' ], factor: 'clearcoatRoughness', component: 'g', width: 3 },
	clearcoatNormalTexture: { path: [ 'extensions', 'KHR_materials_clearcoat', 'clearcoatNormalTexture' ], map: 'clearcoatNormalMap', properties: [ 'clearcoatNormalNode' ], width: 3 },
	transmissionTexture: { path: [ 'extensions', 'KHR_materials_transmission', 'transmissionTexture' ], map: 'transmissionMap', properties: [ 'transmissionNode' ], factor: 'transmission', component: 'r', width: 1 },
	thicknessTexture: { path: [ 'extensions', 'KHR_materials_volume', 'thicknessTexture' ], map: 'thicknessMap', properties: [ 'thicknessNode' ], factor: 'thickness', component: 'g', width: 3 },
	specularTexture: { path: [ 'extensions', 'KHR_materials_specular', 'specularTexture' ], map: 'specularIntensityMap', properties: [ 'specularIntensityNode' ], factor: 'specularIntensity', component: 'a', width: 4 },
	specularColorTexture: { path: [ 'extensions', 'KHR_materials_specular', 'specularColorTexture' ], map: 'specularColorMap', properties: [ 'specularColorNode' ], factor: 'specularColor', component: 'rgb', width: 3 },
	sheenColorTexture: { path: [ 'extensions', 'KHR_materials_sheen', 'sheenColorTexture' ], map: 'sheenColorMap', properties: [ 'sheenNode' ], width: 3 },
	sheenRoughnessTexture: { path: [ 'extensions', 'KHR_materials_sheen', 'sheenRoughnessTexture' ], map: 'sheenRoughnessMap', properties: [ 'sheenRoughnessNode' ], factor: 'sheenRoughness', component: 'a', width: 4 },
	iridescenceTexture: { path: [ 'extensions', 'KHR_materials_iridescence', 'iridescenceTexture' ], map: 'iridescenceMap', properties: [ 'iridescenceNode' ], factor: 'iridescence', component: 'r', width: 1 },
	iridescenceThicknessTexture: { path: [ 'extensions', 'KHR_materials_iridescence', 'iridescenceThicknessTexture' ], map: 'iridescenceThicknessMap', properties: [ 'iridescenceThicknessNode' ], width: 3 },
	anisotropyTexture: { path: [ 'extensions', 'KHR_materials_anisotropy', 'anisotropyTexture' ], map: 'anisotropyMap', properties: [ 'anisotropyNode' ], width: 3 },
	bumpTexture: { path: [ 'extensions', 'EXT_materials_bump', 'bumpTexture' ], map: 'bumpMap', properties: [ 'normalNode' ], width: 1 }

};

function getTextureInfo( materialDef, channel, create = false ) {

	const path = PROCEDURAL_CHANNELS[ channel ].path;
	let object = materialDef;
	for ( const key of path.slice( 0, - 1 ) ) {

		if ( object[ key ] === undefined ) {

			if ( ! create ) return undefined;
			object[ key ] = {};

		}

		object = object[ key ];

	}

	const key = path[ path.length - 1 ];
	if ( create && object[ key ] === undefined ) object[ key ] = {};
	return object[ key ];

}

function proceduralError( message ) {

	return new Error( `THREE.GLTFTextureProcedurals: ${ message }` );

}

function setMaterialFlags( material ) {

	if ( material.isMeshBasicNodeMaterial ) {

		material.isMeshBasicMaterial = true;

	} else {

		material.isMeshStandardMaterial = true;
		material.isMeshPhysicalMaterial = true;

	}

}

/** Returns the editable source and bindings. Call updateTextureProceduralMaterial after editing. */
function getTextureProcedurals( material ) {

	return states.get( material );

}

/** Adds a packed glTF ORM output to separate authored float graph outputs. */
function packTextureProceduralChannels( graph, { roughness, metallic, occlusion }, output = 'orm' ) {

	const packed = cloneGLTFProceduralSource( graph );
	if ( typeof output !== 'string' || Object.hasOwn( packed.outputs, output ) ) throw proceduralError( 'Packed output must have a new unique name.' );
	const input = name => {

		const port = packed.outputs[ name ];
		if ( ! Object.hasOwn( packed.outputs, name ) || port.type !== 'float' ) throw proceduralError( `Packing requires a named float output: ${ name }.` );
		return { ...port, nodetype: 'input' };

	};

	const index = packed.nodes.length;
	packed.nodes.push( {
		nodetype: 'combine3', type: 'color3',
		inputs: { in1: occlusion === undefined ? { nodetype: 'input', type: 'float', value: [ 1 ] } : input( occlusion ), in2: input( roughness ), in3: input( metallic ) },
		outputs: { out: { nodetype: 'output', type: 'color3' } }
	} );
	packed.outputs[ output ] = { nodetype: 'output', type: 'color3', node: index, output: 'out' };
	packed.type = 'multioutput';
	return packed;

}

// Record graph structure and literal values, excluding renderer caches and live
// material uniforms. This also detects edits below the material's root nodes.
function nodeFingerprint( roots ) {

	const visited = new Set();
	const entries = [];
	function visit( node ) {

		if ( visited.has( node ) ) return;
		visited.add( node );
		// MaterialReferenceNode creates its internal uniform lazily during
		// rendering. Its property/type define the serializable factor binding.
		const children = node.isMaterialReferenceNode ? [] : node.getChildren();
		const entry = { uuid: node.uuid, children: children.map( child => child.uuid ) };
		for ( const key of [ 'nodeType', 'uniformType', 'method', 'scope', 'property', 'attributeName', 'components', 'op', 'convertTo', 'sampler', 'updateMatrix' ] ) {

			if ( node[ key ] !== undefined ) entry[ key ] = node[ key ];

		}

		if ( node.isMaterialReferenceNode ) entry.material = node.material && node.material.uuid;

		if ( typeof node.jsFunc === 'function' ) {

			if ( ! functions.has( node.jsFunc ) ) functions.set( node.jsFunc, functionSerial ++ );
			entry.function = functions.get( node.jsFunc );

		}

		if ( node.isConstNode ) entry.value = node.value && node.value.toArray ? node.value.toArray() : node.value;
		if ( node.isTextureNode ) {

			const texture = node.value;
			entry.texture = [ texture.uuid, texture.wrapS, texture.wrapT, texture.minFilter, texture.magFilter, texture.colorSpace, texture.flipY, ...texture.offset.toArray(), ...texture.repeat.toArray(), texture.rotation, ...texture.center.toArray() ];

		}

		entries.push( entry );
		for ( const child of children ) visit( child );

	}

	for ( const node of Object.values( roots ) ) visit( node );
	return JSON.stringify( entries );

}

function attachTextureProcedurals( material, state ) {

	states.set( material, state );
	setMaterialFlags( material );
	material.userData.textureProcedurals = { profile: 'b13ced8e5dda85ad6b982b1b27bc5fe570306d38' };

}

/** Creates a procedural node material, retaining the supplied ordinary material as its fallback. */
async function createTextureProceduralMaterial( fallbackMaterial, source, bindings, options = {} ) {

	const material = fallbackMaterial.isMeshBasicMaterial || fallbackMaterial.isMeshBasicNodeMaterial ?
		new MeshBasicNodeMaterial() : new MeshPhysicalNodeMaterial();
	material.copy( fallbackMaterial );
	attachTextureProcedurals( material, {
		source: cloneGLTFProceduralSource( source ),
		bindings: cloneGLTFProceduralSource( bindings ),
		textures: options.textures || [],
		required: options.required === true,
		compiled: new Map(),
		nodes: {},
		compileGraph: options.compileGraph
	} );
	await updateTextureProceduralMaterial( material );
	return material;

}

/** Recompiles serializable graph edits. Arbitrary replacement TSL cannot be exported. */
async function updateTextureProceduralMaterial( material ) {

	const state = states.get( material );
	if ( ! state ) throw proceduralError( 'Material has no procedural authoring state.' );
	const source = state.source;
	cloneGLTFProceduralSource( source );
	cloneGLTFProceduralSource( state.bindings );
	if ( ! source || ! Array.isArray( source.procedurals ) ) throw proceduralError( 'Expected source.procedurals array.' );
	const compiled = new Map();
	const nodes = {};
	const ref = ( property, type = 'float' ) => materialReference( property, type );

	for ( const channel of Object.keys( state.bindings ) ) {

		const config = PROCEDURAL_CHANNELS[ channel ];
		if ( ! Object.hasOwn( PROCEDURAL_CHANNELS, channel ) ) throw proceduralError( `Unsupported texture-info channel ${ channel }.` );
		if ( material.isMeshBasicNodeMaterial && channel !== 'baseColorTexture' ) throw proceduralError( `Unlit materials only support baseColorTexture (${ channel }).` );
		const binding = state.bindings[ channel ];
		const graph = source.procedurals[ binding.index ];
		if ( ! Number.isInteger( binding.index ) || ! graph ) throw proceduralError( `Invalid graph index for ${ channel }.` );
		if ( ! compiled.has( binding.index ) ) {

			const options = {
				definitions: source.procedural_definitions || [],
				textures: state.textures,
				textureCount: state.textures.length,
				resolveTexture: async index => {

					const texture = state.textures[ index ];
					if ( ! texture || ! texture.isTexture ) throw proceduralError( `Missing texture ${ index }.` );
					return texture;

				}
			};
			compiled.set( binding.index, await ( state.compileGraph ? state.compileGraph( binding.index, graph, options ) : compileGLTFProceduralGraph( graph, options ) ) );

		}

		const result = compiled.get( binding.index );
		const outputNames = Object.keys( graph.outputs );
		const output = binding.output === undefined && outputNames.length === 1 ? outputNames[ 0 ] : binding.output;
		if ( typeof output !== 'string' || ! Object.hasOwn( graph.outputs, output ) || ! result.outputs[ output ] ) throw proceduralError( `Invalid or missing named output for ${ channel }.` );
		const type = graph.outputs[ output ].type;
		const width = { float: 1, color3: 3, color4: 4, vector2: 2, vector3: 3, vector4: 4 }[ type ];
		if ( width === undefined || width < config.width ) throw proceduralError( `Incompatible ${ type } output for ${ channel }.` );
		const value = result.outputs[ output ];
		const scalar = width === 1 ? value : value[ config.component || 'r' ];

		if ( channel === 'baseColorTexture' ) {

			nodes.colorNode = vec4( value.rgb.mul( ref( 'color', 'color' ) ), width === 4 ? value.a : 1 );

		} else if ( channel === 'metallicRoughnessTexture' ) {

			nodes.roughnessNode = value.g.mul( ref( 'roughness' ) );
			nodes.metalnessNode = value.b.mul( ref( 'metalness' ) );

		} else if ( channel === 'normalTexture' || channel === 'clearcoatNormalTexture' ) {

			// The current renderer's derivative tangent frame uses UV0. Other frames
			// need an explicit future mapping, rather than the fallback texture's texCoord.
			if ( ( result.streams || [] ).some( stream => /^TEXCOORD_/.test( stream.semantic ) && stream.semantic !== 'TEXCOORD_0' ) ) throw proceduralError( 'Procedural tangent-space normals currently require TEXCOORD_0.' );
			if ( material[ channel === 'normalTexture' ? 'normalScale' : 'clearcoatNormalScale' ].x < 0 ) throw proceduralError( 'Procedural normals require a nonnegative glTF normal scale.' );
			const scale = ref( channel === 'normalTexture' ? 'normalScale' : 'clearcoatNormalScale', 'vec2' );
			// glTF stores OpenGL tangent normals. Match GLTFLoader's derivative
			// green-channel convention, including newly authored meshes and clones.
			const scaleNode = Fn( ( builder ) => vec2( scale.x.abs(), builder.geometry.hasAttribute( 'tangent' ) ? scale.y.abs() : scale.y.abs().negate() ) )();
			nodes[ config.properties[ 0 ] ] = normalMap( value.rgb, scaleNode );

		} else if ( channel === 'occlusionTexture' ) {

			nodes.aoNode = scalar.sub( 1 ).mul( ref( 'aoMapIntensity' ) ).add( 1 );

		} else if ( channel === 'emissiveTexture' ) {

			nodes.emissiveNode = value.rgb.mul( ref( 'emissive', 'color' ) ).mul( ref( 'emissiveIntensity' ) );

		} else if ( channel === 'sheenColorTexture' ) {

			nodes.sheenNode = value.rgb.mul( ref( 'sheenColor', 'color' ) ).mul( ref( 'sheen' ) );

		} else if ( channel === 'iridescenceThicknessTexture' ) {

			nodes.iridescenceThicknessNode = mix( ref( 'iridescenceThicknessRange.0' ), ref( 'iridescenceThicknessRange.1' ), value.g );

		} else if ( channel === 'anisotropyTexture' ) {

			const direction = value.rg.mul( 2 ).sub( 1 ).normalize().mul( value.b ).mul( ref( 'anisotropy' ) );
			const rotation = ref( 'anisotropyRotation' );
			nodes.anisotropyNode = vec2( direction.x.mul( rotation.cos() ).sub( direction.y.mul( rotation.sin() ) ), direction.x.mul( rotation.sin() ).add( direction.y.mul( rotation.cos() ) ) );

		} else if ( channel === 'bumpTexture' ) {

			if ( state.bindings.normalTexture ) throw proceduralError( 'Simultaneous procedural normal and bump bindings are unsupported.' );
			nodes.normalNode = bumpMap( scalar, ref( 'bumpScale' ) );

		} else {

			nodes[ config.properties[ 0 ] ] = scalar.mul( ref( config.factor, config.component === 'rgb' ? 'color' : 'float' ) );
			if ( channel === 'sheenRoughnessTexture' ) nodes.sheenRoughnessNode = nodes.sheenRoughnessNode.clamp( 0.0001, 1 );

		}

	}

	// Apply atomically so failed edits leave the last working material intact.
	for ( const property of Object.keys( state.nodes ) ) material[ property ] = null;
	Object.assign( material, nodes );
	state.nodes = nodes;
	state.compiled = compiled;
	state.snapshot = JSON.stringify( { source, bindings: state.bindings } );
	state.fingerprint = nodeFingerprint( nodes );
	const resources = new Set();
	for ( const index of compiled.keys() ) {

		for ( const textureIndex of collectGLTFProceduralTextureIndices( source.procedurals[ index ], { definitions: source.procedural_definitions || [], textureCount: state.textures.length } ) ) resources.add( textureIndex );

	}

	state.resourceSnapshot = [ ...resources ].sort( ( a, b ) => a - b ).map( index => [ index, state.textures[ index ] && state.textures[ index ].uuid ] );
	material.needsUpdate = true;
	return material;

}

function assertTextureProceduralsSerializable( material ) {

	const state = states.get( material );
	if ( ! state ) {

		if ( material.userData.textureProcedurals ) throw proceduralError( 'Procedural state was lost. Recreate cloned materials through the authoring API.' );
		return null;

	}

	cloneGLTFProceduralSource( state.source );
	cloneGLTFProceduralSource( state.bindings );
	if ( material.transparent && material.alphaTest > 0 ) throw proceduralError( 'Simultaneous alpha blending and alpha cutoff cannot be represented by one glTF alphaMode.' );

	if ( state.snapshot !== JSON.stringify( { source: state.source, bindings: state.bindings } ) ) throw proceduralError( 'Graph edits must be compiled with updateTextureProceduralMaterial before export.' );
	for ( const property of Object.keys( state.nodes ) ) {

		if ( material[ property ] !== state.nodes[ property ] ) throw proceduralError( `Runtime edit to ${ property } cannot be serialized. Edit the source graph and recompile.` );

	}

	for ( const property of Object.keys( material ) ) {

		if ( property.endsWith( 'Node' ) && material[ property ] !== null && material[ property ] !== undefined && ! Object.hasOwn( state.nodes, property ) ) throw proceduralError( `Additional runtime ${ property } cannot be serialized.` );

	}

	if ( state.fingerprint !== nodeFingerprint( state.nodes ) ) throw proceduralError( 'Edits to compiled graph nodes cannot be serialized. Edit the source graph and recompile.' );
	if ( state.resourceSnapshot.some( ( [ index, uuid ] ) => ! state.textures[ index ] || state.textures[ index ].uuid !== uuid ) ) throw proceduralError( 'Texture resources changed; recompile the graph before export.' );
	return state;

}

export { EXTENSION_NAME, LIBRARY_NAME, PROCEDURAL_CHANNELS, getTextureInfo, proceduralError, attachTextureProcedurals, assertTextureProceduralsSerializable, createTextureProceduralMaterial, getTextureProcedurals, updateTextureProceduralMaterial, packTextureProceduralChannels };
