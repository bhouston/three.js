import { DefaultLoadingManager, LinearFilter, NearestFilter, NoColorSpace } from 'three/webgpu';
import { attribute, texture } from 'three/tsl';
import { MaterialXLoader } from '../MaterialXLoader.js';
import { MtlXLibrary } from './MaterialXNodeLibrary.js';
import registryData from './MaterialXNodeInterfaceRegistry.js';
import { createMaterialXCompileRegistry } from './compile/MaterialXCompileRegistry.js';
import { TEXTURE_ADDRESS_MODE_WRAPPING } from './MaterialXUtils.js';

const COMPONENTS = { boolean: 1, float: 1, integer: 1, vector2: 2, vector3: 3, vector4: 4, color3: 3, color4: 4, matrix3x3: 9, matrix4x4: 16 };
const TYPE_MAP = { matrix3x3: 'matrix33', matrix4x4: 'matrix44' };
const COMPILE_REGISTRY = createMaterialXCompileRegistry();
const STD_DEFS = Object.entries( registryData.nodedefs );
const PATTERN_NODES = new Set( STD_DEFS.filter( ( [ , definition ] ) => Object.values( definition.outputs ).every( type => Object.hasOwn( COMPONENTS, type ) || [ 'string', 'filename', 'matrix33', 'matrix44' ].includes( type ) ) ).map( ( [ , definition ] ) => definition.node ) );

function texcoordAttribute( index ) {

	return index <= 3 ? `uv${index || ''}` : `texcoord_${index}`;

}

function fail( path, message ) {

	throw new Error( `KHR_texture_procedurals: ${path}: ${message}` );

}

function objectMap( value, path ) {

	if ( value === undefined ) return {};
	if ( value === null || typeof value !== 'object' || Array.isArray( value ) ) fail( path, 'Expected a named object map.' );
	return value;

}

function normalizedType( type, path ) {

	if ( type !== 'string' && type !== 'filename' && ! Object.hasOwn( COMPONENTS, type ) ) fail( path, `Unsupported pattern type "${type}".` );
	return TYPE_MAP[ type ] || type;

}

/**
 * Copies plain JSON without evaluating getters or toJSON hooks. Runtime values,
 * cyclic metadata, sparse arrays, and prototype-related keys are rejected.
 *
 * @param {*} source - A serializable graph or definition representation.
 * @return {*} An independent plain JSON representation.
 */
function cloneGLTFProceduralSource( source ) {

	const ancestors = new Set();
	function copy( value, path ) {

		if ( value === null || typeof value === 'string' || typeof value === 'boolean' ) return value;
		if ( typeof value === 'number' && Number.isFinite( value ) ) return value;
		if ( typeof value !== 'object' ) fail( path, 'Source must contain only serializable JSON values.' );
		if ( ancestors.has( value ) ) fail( path, 'Cyclic source metadata cannot be serialized.' );
		const isArray = Array.isArray( value );
		if ( ! isArray && Object.getPrototypeOf( value ) !== Object.prototype && Object.getPrototypeOf( value ) !== null ) fail( path, 'Expected plain JSON objects, not runtime instances.' );
		ancestors.add( value );
		const result = isArray ? [] : {};
		const descriptors = Object.getOwnPropertyDescriptors( value );
		for ( const key of Reflect.ownKeys( descriptors ) ) {

			if ( isArray && key === 'length' ) continue;
			if ( typeof key !== 'string' || key === '__proto__' || key === 'constructor' || key === 'prototype' ) fail( path, 'Prototype-related and symbol keys are unsupported.' );
			const descriptor = descriptors[ key ];
			if ( ! Object.hasOwn( descriptor, 'value' ) || ! descriptor.enumerable ) fail( `${path}.${key}`, 'Accessors and non-enumerable metadata are not plain JSON.' );
			if ( isArray && ( ! /^(0|[1-9]\d*)$/.test( key ) || Number( key ) >= value.length ) ) fail( path, 'Array metadata cannot be serialized.' );
			result[ key ] = copy( descriptor.value, `${path}.${key}` );

		}

		if ( isArray && Object.keys( descriptors ).length - 1 !== value.length ) fail( path, 'Sparse arrays cannot be serialized.' );
		ancestors.delete( value );
		return result;

	}

	return copy( source, 'source' );

}

function validateValue( port, path, textureCount ) {

	const value = port.value;
	if ( port.type === 'string' ) {

		if ( typeof value !== 'string' ) fail( path, 'String values must be strings.' );
		return value;

	}

	if ( port.type === 'boolean' ) {

		if ( typeof value !== 'boolean' ) fail( path, 'Boolean values must be true or false.' );
		return String( value );

	}

	const length = port.type === 'filename' ? 1 : COMPONENTS[ port.type ];
	if ( ! Array.isArray( value ) || value.length !== length || value.some( ( component ) => typeof component !== 'number' || ! Number.isFinite( component ) ) ) {

		fail( path, `Expected a finite numeric array of length ${length}.` );

	}

	if ( port.type === 'integer' && ! Number.isSafeInteger( value[ 0 ] ) ) fail( path, 'Integer values must be safe integers.' );
	if ( port.type === 'filename' ) {

		if ( ! Number.isSafeInteger( value[ 0 ] ) || value[ 0 ] < 0 || ( textureCount !== undefined && value[ 0 ] >= textureCount ) ) fail( path, `Invalid glTF texture index ${value[ 0 ]}.` );
		return `gltf-texture:${value[ 0 ]}`;

	}

	// The draft declares row-major matrices; MaterialX uses column-major values.
	if ( port.type === 'matrix3x3' || port.type === 'matrix4x4' ) {

		const size = port.type === 'matrix3x3' ? 3 : 4;
		const transposed = [];
		for ( let column = 0; column < size; column ++ ) {

			for ( let row = 0; row < size; row ++ ) transposed.push( value[ row * size + column ] );

		}

		return transposed.join( ', ' );

	}

	return value.join( ', ' );

}

function jsonElement( nodeName, attributes, children = [] ) {

	return { element: nodeName, attributes, children };

}

function checkPort( port, kind, path ) {

	objectMap( port, path );
	if ( port.nodetype !== kind ) fail( path, `Expected nodetype "${kind}".` );
	normalizedType( port.type, path );
	if ( port.unit !== undefined || port.unittype !== undefined ) fail( path, 'Unit metadata and conversions are unsupported by the experimental profile.' );
	if ( port.colorspace !== undefined && port.type !== 'filename' ) fail( path, 'Only filename image resources support explicit color-space interpretation.' );

}

function checkOutputs( outputs, declared, path ) {

	const names = Object.keys( outputs );
	if ( names.length === 0 ) fail( path, 'At least one named output is required.' );
	for ( const [ name, port ] of Object.entries( outputs ) ) {

		checkPort( port, 'output', `${path}.${name}` );
		if ( declared && declared[ name ] !== normalizedType( port.type, path ) ) fail( `${path}.${name}`, 'Output name or type does not match the node definition.' );

	}

	if ( declared && Object.keys( declared ).length !== names.length ) fail( path, 'Every node definition output must be declared.' );

}

function checkGraphType( graph, outputs, path ) {

	const types = Object.values( outputs ).map( ( port ) => port.type );
	if ( graph.type !== ( types.length > 1 ? 'multioutput' : types[ 0 ] ) ) fail( `${path}.type`, 'Graph/node type must match its single output, or be multioutput for multiple outputs.' );

}

/**
 * Validates the experimental named-port profile and translates pattern graphs into
 * the existing MaterialX parser's node tree. Functional definitions are expanded,
 * never interpreted as source code. The returned source remains suitable for export.
 *
 * @param {Object} graph - A KHR_texture_procedurals graph.
 * @param {Object} [options] - Definitions and optional glTF texture count.
 * @return {Object} Validated source, normalized tree, resources, and geometry streams.
 */
function validateGLTFProceduralGraph( graph, { definitions = [], textureCount } = {} ) {

	const source = cloneGLTFProceduralSource( graph );
	definitions = cloneGLTFProceduralSource( definitions );
	objectMap( source, 'graph' );
	const custom = new Map();
	const implementations = new Map();
	const textureIndices = new Set();
	const streams = new Map();
	const children = [];
	let serial = 0;
	if ( ! Array.isArray( definitions ) ) fail( 'procedural_definitions', 'Expected an array.' );

	for ( let i = 0; i < definitions.length; i ++ ) {

		const definition = definitions[ i ];
		const path = `procedural_definitions[${i}]`;
		objectMap( definition, path );
		if ( definition.nodetype === 'nodedef' ) {

			if ( typeof definition.node !== 'string' || definition.node === '' ) fail( path, 'A custom node category is required.' );
			if ( STD_DEFS.some( ( [ , def ] ) => def.node === definition.node ) || custom.has( definition.node ) ) fail( path, 'Conflicting custom node category; stdlib overrides and multiple custom versions are unsupported.' );
			const inputs = objectMap( definition.inputs, `${path}.inputs` );
			const outputs = objectMap( definition.outputs, `${path}.outputs` );
			checkOutputs( outputs, null, `${path}.outputs` );
			for ( const [ name, input ] of Object.entries( inputs ) ) {

				checkPort( input, 'input', `${path}.inputs.${name}` );
				if ( input.defaultgeomprop !== undefined || input.unit !== undefined || input.colorspace !== undefined ) fail( path, 'Custom definitions cannot declare implicit geometry, unit, or color-space bindings.' );
				if ( input.node !== undefined || input.input !== undefined || input.output !== undefined ) fail( path, 'Custom definition inputs cannot have connections.' );
				if ( input.value !== undefined ) {

					validateValue( input, `${path}.inputs.${name}.value`, textureCount );
					if ( input.type === 'filename' ) textureIndices.add( input.value[ 0 ] );

				}

			}

			for ( const output of Object.values( outputs ) ) {

				if ( output.value !== undefined || output.defaultvalue !== undefined || output.defaultinput !== undefined ) fail( path, 'Custom definition outputs cannot have defaults.' );

			}

			custom.set( definition.node, { definition, index: i } );

		} else if ( definition.nodetype === 'nodegraph' ) {

			if ( ! Number.isSafeInteger( definition.nodedef ) || definition.nodedef < 0 || definitions[ definition.nodedef ]?.nodetype !== 'nodedef' ) fail( path, 'Invalid functional graph nodedef index.' );
			if ( definition.inputs !== undefined ) fail( path, 'Functional graphs use the nodedef interface and cannot declare inputs.' );
			if ( implementations.has( definition.nodedef ) ) fail( path, 'Multiple implementations of a custom definition are unsupported.' );
			implementations.set( definition.nodedef, definition );

		} else {

			fail( path, 'Only nodedef and functional nodegraph definitions are supported; shader/source code is never executed.' );

		}

	}

	function valueExpression( port, path ) {

		const value = validateValue( port, path, textureCount );
		if ( port.type === 'filename' ) textureIndices.add( port.value[ 0 ] );
		if ( port.colorspace !== undefined && port.colorspace !== 'lin_rec709' && port.colorspace !== 'srgb_texture' ) fail( path, 'Unsupported image color space; use lin_rec709 or srgb_texture.' );
		return { type: normalizedType( port.type, path ), value, ...( port.colorspace !== undefined ? { colorspace: port.colorspace } : {} ) };

	}

	function expand( current, interfaceValues, definitionStack, path ) {

		if ( current.nodetype !== 'nodegraph' ) fail( path, 'Expected nodetype "nodegraph".' );
		if ( ! Array.isArray( current.nodes ) ) fail( `${path}.nodes`, 'Expected a flattened processing-node array.' );
		const inputs = objectMap( current.inputs, `${path}.inputs` );
		const outputs = objectMap( current.outputs, `${path}.outputs` );
		checkOutputs( outputs, null, `${path}.outputs` );
		checkGraphType( current, outputs, path );
		const interfaces = new Map( interfaceValues );
		const resolved = new Map();
		const visiting = new Set();

		for ( const [ name, input ] of Object.entries( inputs ) ) {

			checkPort( input, 'input', `${path}.inputs.${name}` );
			if ( input.node !== undefined || input.input !== undefined || input.output !== undefined ) fail( `${path}.inputs.${name}`, 'Graph interface inputs must provide constant values.' );
			const expression = valueExpression( input, `${path}.inputs.${name}.value` );
			const inputName = `interface_${serial ++}`;
			children.push( jsonElement( 'input', { name: inputName, ...expression } ) );
			interfaces.set( name, expression );

		}

		function resolvePort( port, portPath ) {

			const type = normalizedType( port.type, portPath );
			const modes = [ 'value', 'node', 'input' ].filter( ( key ) => port[ key ] !== undefined );
			if ( modes.length !== 1 ) fail( portPath, 'Expected exactly one value, processing node, or graph interface input.' );
			let expression;
			if ( modes[ 0 ] === 'value' ) {

				if ( port.output !== undefined ) fail( portPath, 'An output selector requires a processing-node connection.' );
				expression = valueExpression( port, `${portPath}.value` );

			} else if ( modes[ 0 ] === 'input' ) {

				if ( typeof port.input !== 'string' || ! interfaces.has( port.input ) ) fail( portPath, `Unknown graph interface input "${port.input}".` );
				if ( port.output !== undefined ) fail( portPath, 'Graph interface input connections do not have output selectors.' );
				expression = interfaces.get( port.input );

			} else {

				if ( ! Number.isSafeInteger( port.node ) || port.node < 0 || port.node >= current.nodes.length ) fail( portPath, `Invalid processing-node index ${port.node}.` );
				const nodeOutputs = resolveNode( port.node );
				const names = Object.keys( nodeOutputs );
				if ( port.output !== undefined && typeof port.output !== 'string' ) fail( portPath, 'Output selectors must be strings.' );
				if ( port.output === undefined && names.length > 1 ) fail( portPath, 'A named output selector is required for a multioutput node.' );
				const name = port.output ?? names[ 0 ];
				if ( ! Object.hasOwn( nodeOutputs, name ) ) fail( portPath, `Unknown processing-node output "${name}".` );
				expression = nodeOutputs[ name ];

			}

			if ( expression.type !== type ) fail( portPath, `Type mismatch: expected ${type}, connected ${expression.type}. Use an explicit convert node.` );
			return { ...expression };

		}

		function resolveNode( index ) {

			if ( resolved.has( index ) ) return resolved.get( index );
			if ( visiting.has( index ) ) fail( `${path}.nodes[${index}]`, 'Cycle detected in processing-node connections.' );
			visiting.add( index );
			const node = current.nodes[ index ];
			const nodePath = `${path}.nodes[${index}]`;
			objectMap( node, nodePath );
			const nodeInputs = objectMap( node.inputs, `${nodePath}.inputs` );
			const nodeOutputs = objectMap( node.outputs, `${nodePath}.outputs` );
			if ( node.unit !== undefined || node.unittype !== undefined ) fail( nodePath, 'Unit metadata and conversions are unsupported by the experimental profile.' );
			if ( node.colorspace !== undefined && ! Object.values( nodeInputs ).some( ( port ) => port.type === 'filename' ) ) fail( nodePath, 'Only image nodes with filename inputs support explicit color-space interpretation.' );
			checkOutputs( nodeOutputs, null, `${nodePath}.outputs` );
			checkGraphType( node, nodeOutputs, nodePath );
			for ( const [ name, port ] of Object.entries( nodeInputs ) ) checkPort( port, 'input', `${nodePath}.inputs.${name}` );
			for ( const output of Object.values( nodeOutputs ) ) {

				if ( output.node !== undefined || output.input !== undefined || output.value !== undefined || output.output !== undefined ) fail( nodePath, 'Processing-node outputs declare ports, not connections.' );

			}

			const customDefinition = custom.get( node.nodetype );
			let result;
			if ( customDefinition ) {

				const { definition, index: definitionIndex } = customDefinition;
				if ( definitionStack.includes( definitionIndex ) ) fail( nodePath, 'Recursive custom functional definition detected.' );
				if ( node.version !== undefined && node.version !== ( definition.version || '' ) ) fail( nodePath, 'Unsupported custom definition version.' );
				const implementation = implementations.get( definitionIndex );
				if ( ! implementation ) fail( nodePath, 'Custom node has no functional graph implementation.' );
				const declaredOutputs = Object.fromEntries( Object.entries( definition.outputs ).map( ( [ name, port ] ) => [ name, normalizedType( port.type, nodePath ) ] ) );
				checkOutputs( nodeOutputs, declaredOutputs, `${nodePath}.outputs` );
				checkOutputs( objectMap( implementation.outputs, nodePath ), declaredOutputs, `${nodePath}.implementation.outputs` );
				const values = new Map();
				for ( const [ name, port ] of Object.entries( nodeInputs ) ) {

					if ( ! Object.hasOwn( definition.inputs || {}, name ) ) fail( `${nodePath}.inputs.${name}`, 'Unknown custom node input.' );
					if ( port.type !== definition.inputs[ name ].type ) fail( `${nodePath}.inputs.${name}`, 'Custom node input type mismatch.' );

				}

				for ( const [ name, port ] of Object.entries( definition.inputs || {} ) ) {

					if ( ! nodeInputs[ name ] && port.value === undefined ) fail( `${nodePath}.inputs.${name}`, 'Missing required custom input with no definition default.' );
					values.set( name, nodeInputs[ name ] ? resolvePort( nodeInputs[ name ], `${nodePath}.inputs.${name}` ) : valueExpression( port, `${nodePath}.defaults.${name}` ) );

				}

				result = expand( implementation, values, [ ...definitionStack, definitionIndex ], `${nodePath}.implementation` );

			} else {

				if ( ! COMPILE_REGISTRY.has( node.nodetype ) && ! Object.hasOwn( MtlXLibrary, node.nodetype ) ) fail( nodePath, `Unsupported pattern node "${node.nodetype}".` );
				const candidates = STD_DEFS.filter( ( [ , def ] ) => def.node === node.nodetype && ( def.version ? def.version === ( node.version || '' ) || ( ! node.version && def.isdefaultversion === true ) : ! node.version ) );
				const match = candidates.find( ( [ , def ] ) => {

					if ( Object.entries( nodeInputs ).some( ( [ name, port ] ) => def.inputs[ name ]?.type !== normalizedType( port.type, nodePath ) ) ) return false;
					return Object.keys( def.outputs ).length === Object.keys( nodeOutputs ).length && Object.entries( nodeOutputs ).every( ( [ name, port ] ) => def.outputs[ name ] === normalizedType( port.type, nodePath ) );

				} );
				if ( ! match ) fail( nodePath, `Inputs, outputs, types, or version do not match a supported MaterialX definition for "${node.nodetype}".` );
				const [ nodedef, definition ] = match;
				const inputChildren = [];
				const expressions = new Map();
				for ( const [ name, port ] of Object.entries( nodeInputs ) ) {

					const expression = resolvePort( port, `${nodePath}.inputs.${name}` );
					if ( expression.type === 'filename' && node.colorspace !== undefined ) {

						if ( node.colorspace !== 'lin_rec709' && node.colorspace !== 'srgb_texture' ) fail( nodePath, 'Unsupported image color space; use lin_rec709 or srgb_texture.' );
						expression.colorspace = node.colorspace;

					}

					expressions.set( name, expression );
					inputChildren.push( jsonElement( 'input', { name, ...expression } ) );

				}

				for ( const [ name, port ] of Object.entries( definition.inputs ) ) {

					if ( ! Object.hasOwn( nodeInputs, name ) && port.value === undefined && ! port.defaultgeomprop ) fail( nodePath, `Missing required input "${name}".` );
					if ( ! Object.hasOwn( nodeInputs, name ) && /^UV\d+$/.test( port.defaultgeomprop || '' ) ) {

						const index = Number( port.defaultgeomprop.slice( 2 ) );
						const semantic = `TEXCOORD_${index}`;
						streams.set( semantic, { semantic, attribute: texcoordAttribute( index ), itemSize: 2 } );

					}

					if ( ! Object.hasOwn( nodeInputs, name ) && ( port.defaultgeomprop === 'Pobject' || port.defaultgeomprop === 'Pworld' ) ) {

						streams.set( 'POSITION', { semantic: 'POSITION', attribute: 'position', itemSize: 3 } );

					}

					if ( ! Object.hasOwn( nodeInputs, name ) && /^[TB](object|world)$/.test( port.defaultgeomprop || '' ) ) {

						streams.set( 'TANGENT', { semantic: 'TANGENT', attribute: 'tangent', itemSize: 4 } );

					}

				}

				for ( const modeName of [ 'uaddressmode', 'vaddressmode', 'filtertype' ] ) {

					const expression = expressions.get( modeName );
					if ( ! expression ) continue;
					if ( typeof expression.value !== 'string' ) fail( nodePath, 'Image addressing and filtering parameters must be constant strings.' );
					const allowed = modeName === 'filtertype' ? [ 'linear', 'closest' ] : [ 'constant', 'clamp', 'periodic', 'mirror' ];
					if ( ! allowed.includes( expression.value ) ) fail( nodePath, `Unsupported ${modeName} "${expression.value}".` );

				}

				if ( node.nodetype === 'texcoord' || node.nodetype === 'geomcolor' ) {

					const stream = expressions.get( 'index' );
					if ( stream && stream.value === undefined ) fail( nodePath, 'Geometry stream indices must be constants.' );
					const streamIndex = stream ? Number( stream.value ) : 0;
					if ( ! Number.isSafeInteger( streamIndex ) || streamIndex < 0 ) fail( nodePath, 'Geometry stream indices must be nonnegative safe integers.' );
					if ( node.nodetype === 'texcoord' && node.type !== 'vector2' ) fail( nodePath, 'Three-component UVs are outside the experimental profile.' );
					const semantic = `${node.nodetype === 'texcoord' ? 'TEXCOORD' : 'COLOR'}_${streamIndex}`;
					const attributeName = node.nodetype === 'texcoord' ? texcoordAttribute( streamIndex ) : streamIndex === 0 ? 'color' : `color_${streamIndex}`;
					streams.set( semantic, { semantic, attribute: attributeName, itemSize: node.nodetype === 'texcoord' ? 2 : COMPONENTS[ node.type ] } );

				}

				if ( node.nodetype === 'position' ) streams.set( 'POSITION', { semantic: 'POSITION', attribute: 'position', itemSize: 3 } );
				if ( node.nodetype === 'tangent' || node.nodetype === 'bitangent' ) streams.set( 'TANGENT', { semantic: 'TANGENT', attribute: 'tangent', itemSize: 4 } );

				const name = `node_${serial ++}`;
				children.push( jsonElement( node.nodetype, { name, type: node.type === 'multioutput' ? 'multioutput' : normalizedType( node.type, nodePath ), nodedef }, inputChildren ) );
				result = Object.fromEntries( Object.entries( nodeOutputs ).map( ( [ outputName, port ] ) => [ outputName, { type: normalizedType( port.type, nodePath ), nodename: name, output: outputName } ] ) );

			}

			visiting.delete( index );
			resolved.set( index, result );
			return result;

		}

		for ( let i = 0; i < current.nodes.length; i ++ ) resolveNode( i );
		return Object.fromEntries( Object.entries( outputs ).map( ( [ name, port ] ) => [ name, resolvePort( port, `${path}.outputs.${name}` ) ] ) );

	}

	// Validate all functional implementations, including unused ones, without
	// adding their geometry requirements or compiler symbols to the bound graph.
	for ( const [ index, implementation ] of implementations ) {

		const definition = definitions[ index ];
		const values = new Map();
		for ( const [ name, port ] of Object.entries( definition.inputs || {} ) ) {

			values.set( name, port.value !== undefined ? valueExpression( port, `procedural_definitions[${index}].inputs.${name}` ) : { type: normalizedType( port.type, 'definition' ), value: port.type === 'string' ? '' : port.type === 'filename' ? 'gltf-texture:0' : '0' } );

		}

		const declared = Object.fromEntries( Object.entries( definition.outputs ).map( ( [ name, port ] ) => [ name, normalizedType( port.type, 'definition' ) ] ) );
		checkOutputs( objectMap( implementation.outputs, 'implementation' ), declared, 'implementation.outputs' );
		expand( implementation, values, [ index ], `procedural_definitions[${index}].implementation` );
		children.length = 0;
		streams.clear();
		serial = 0;

	}

	const outputExpressions = expand( source, [], [], 'graph' );
	const outputNames = Object.create( null );
	for ( const [ name, expression ] of Object.entries( outputExpressions ) ) {

		const normalizedName = `output_${serial ++}`;
		outputNames[ name ] = normalizedName;
		children.push( jsonElement( 'output', { name: normalizedName, ...expression } ) );

	}

	const normalized = jsonElement( 'materialx', { name: '', colorspace: 'lin_rec709' }, [ jsonElement( 'nodegraph', { name: 'procedural' }, children ) ] );
	return { source, normalized, outputNames, textureIndices: Array.from( textureIndices ).sort( ( a, b ) => a - b ), streams: Array.from( streams.values() ) };

}

/**
 * Compiles a validated draft pattern graph with the existing MaterialX compiler.
 * Texture resolution is awaited before any output is returned. glTF texture-info
 * transforms, coordinates, and sampler state are deliberately not applied.
 *
 * @param {Object} graph - A procedural graph.
 * @param {Object} [options] - Validation options, loading manager, and async texture resolver.
 * @return {Promise<Object>} Named TSL outputs and serializable source representation.
 */
async function compileGLTFProceduralGraph( graph, options = {} ) {

	const validated = validateGLTFProceduralGraph( graph, options );
	const textures = new Map();
	const resources = await Promise.all( validated.textureIndices.map( async ( index ) => {

		if ( typeof options.resolveTexture !== 'function' ) fail( 'resources', 'Texture-backed graphs require an async glTF texture resolver.' );
		const resource = await options.resolveTexture( index );
		if ( resource?.isTexture !== true ) fail( 'resources', `Missing texture ${index}.` );
		return resource;

	} ) );
	for ( let i = 0; i < resources.length; i ++ ) textures.set( validated.textureIndices[ i ], resources[ i ] );
	const samples = new Map();
	const textureResolver = ( nodeX ) => {

		if ( ! nodeX.value?.startsWith( 'gltf-texture:' ) ) return null;
		const index = Number( nodeX.value.slice( 'gltf-texture:'.length ) );

		const modes = nodeX.parent.getTextureAddressModes();
		const filter = nodeX.parent.getChildByName( 'filtertype' );
		const filtertype = filter ? filter.getNode() : 'linear';
		const key = `${index}|${modes.u}|${modes.v}|${filtertype}`;
		if ( samples.has( key ) ) return samples.get( key );
		const resource = textures.get( index ).clone();
		resource.wrapS = TEXTURE_ADDRESS_MODE_WRAPPING[ modes.u ];
		resource.wrapT = TEXTURE_ADDRESS_MODE_WRAPPING[ modes.v ];
		if ( filtertype !== 'linear' && filtertype !== 'closest' ) fail( 'sampling', `Unsupported MaterialX image filter "${filtertype}"; only linear and closest are supported.` );
		resource.magFilter = filtertype === 'closest' ? NearestFilter : LinearFilter;
		resource.minFilter = resource.magFilter;
		resource.colorSpace = NoColorSpace;
		resource.offset.set( 0, 0 );
		resource.repeat.set( 1, 1 );
		resource.center.set( 0, 0 );
		resource.rotation = 0;
		resource.matrix.identity();
		resource.matrixAutoUpdate = false;
		resource.channel = 0;
		resource.anisotropy = 1;
		resource.flipY = false;
		resource.needsUpdate = true;
		const sample = texture( resource );
		samples.set( key, sample );
		return sample;

	};

	// GLTFLoader keeps COLOR_n as color_n for n > 0, whereas MaterialX's
	// vertexColor accessor uses colorN. Resolve numbered glTF streams explicitly.
	const document = new MaterialXLoader( options.manager || DefaultLoadingManager ).parseDocument( validated.normalized, {
		allowedNodes: PATTERN_NODES,
		texcoordResolver: ( index = 0 ) => attribute( texcoordAttribute( index ), 'vec2' ),
		textureResolver,
		nodeResolver: ( nodeX ) => {

			if ( nodeX.element !== 'geomcolor' ) return null;
			const index = Number( nodeX.getChildByName( 'index' )?.value || 0 );
			return attribute( index === 0 ? 'color' : `color_${index}`, nodeX.type === 'color4' ? 'vec4' : 'vec3' );

		}
	} );
	const log = document.log;

	const outputs = Object.create( null );
	for ( const [ name, normalizedName ] of Object.entries( validated.outputNames ) ) outputs[ name ] = document.getMaterialXNode( 'procedural', normalizedName ).getNode();
	if ( log.errors.length ) fail( 'compile', log.errors.map( ( entry ) => entry.message ).join( ' ' ) );
	return { ...validated, outputs, textures, document, diagnostics: log.entries };

}

/**
 * Returns deterministic glTF texture references from a validated source graph.
 *
 * @param {Object} graph - A procedural graph.
 * @param {Object} [options] - Validation options.
 * @return {Array<number>} Sorted texture indices.
 */
function collectGLTFProceduralTextureIndices( graph, options = {} ) {

	return validateGLTFProceduralGraph( graph, options ).textureIndices;

}

export { validateGLTFProceduralGraph, compileGLTFProceduralGraph, collectGLTFProceduralTextureIndices, cloneGLTFProceduralSource };
