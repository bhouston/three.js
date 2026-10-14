import { validateGLTFProceduralGraph } from '../loaders/materialx/GLTFProceduralGraph.js';
import { EXTENSION_NAME, LIBRARY_NAME, PROCEDURAL_CHANNELS, getTextureInfo, proceduralError, assertTextureProceduralsSerializable } from '../utils/GLTFTextureProcedurals.js';

// Valid opaque white 1x1 PNG. This keeps standard glTF valid, not the procedural appearance.
const DUMMY_IMAGE = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4AWP4DwQACfsD/c8LaHIAAAAASUVORK5CYII=';

function canonicalJSON( value ) {

	return JSON.stringify( value, ( key, item ) => {

		if ( item && typeof item === 'object' && ! Array.isArray( item ) ) {

			return Object.fromEntries( Object.keys( item ).sort().map( name => [ name, item[ name ] ] ) );

		}

		return item;

	} );

}

function preserveExtensionFactors( name, material, extension ) {

	// Ordinary hooks may omit inactive lobes. Procedural bindings still need
	// their sibling factors/ranges for faithful round trips and later edits.
	switch ( name ) {

		case 'KHR_materials_clearcoat':
			extension.clearcoatFactor = material.clearcoat;
			extension.clearcoatRoughnessFactor = material.clearcoatRoughness;
			break;
		case 'KHR_materials_transmission':
			extension.transmissionFactor = material.transmission;
			break;
		case 'KHR_materials_volume':
			extension.thicknessFactor = material.thickness;
			extension.attenuationColor = material.attenuationColor.toArray();
			if ( material.attenuationDistance !== Infinity ) extension.attenuationDistance = material.attenuationDistance;
			break;
		case 'KHR_materials_specular':
			extension.specularFactor = material.specularIntensity;
			extension.specularColorFactor = material.specularColor.toArray();
			break;
		case 'KHR_materials_sheen':
			// glTF has a single sheen color factor. Combine Three.js's separate
			// strength and color instead of adding an invented scalar slot.
			extension.sheenColorFactor = material.sheenColor.toArray().map( component => component * material.sheen );
			extension.sheenRoughnessFactor = material.sheenRoughness;
			break;
		case 'KHR_materials_iridescence':
			extension.iridescenceFactor = material.iridescence;
			extension.iridescenceIor = material.iridescenceIOR;
			extension.iridescenceThicknessMinimum = material.iridescenceThicknessRange[ 0 ];
			extension.iridescenceThicknessMaximum = material.iridescenceThicknessRange[ 1 ];
			break;
		case 'KHR_materials_anisotropy':
			extension.anisotropyStrength = material.anisotropy;
			extension.anisotropyRotation = material.anisotropyRotation;
			break;
		case 'EXT_materials_bump':
			extension.bumpFactor = material.bumpScale;
			break;

	}

}

/**
 * Opt-in exporter for serializable procedural authoring state, never reverse-compiling TSL.
 * Register with GLTFExporter.register( writer => new GLTFTextureProceduralsExporterExtension( writer ) ).
 */
class GLTFTextureProceduralsExporterExtension {

	constructor( writer ) {

		this.name = EXTENSION_NAME;
		this.writer = writer;
		this.graphIndices = new Map();
		this.definitions = new Map();
		this.resources = new WeakMap();
		this.dummyIndex = undefined;

	}

	dummyTexture() {

		if ( this.dummyIndex !== undefined ) return this.dummyIndex;
		const json = this.writer.json;
		json.images = json.images || [];
		json.textures = json.textures || [];
		const source = json.images.push( { uri: DUMMY_IMAGE, mimeType: 'image/png', name: 'Procedural placeholder (not baked)' } ) - 1;
		this.dummyIndex = json.textures.push( { source } ) - 1;
		return this.dummyIndex;

	}

	async remapTextures( object, textures ) {

		if ( ! object || typeof object !== 'object' ) return;
		if ( ( object.nodetype === 'input' || object.nodetype === 'output' ) && object.type === 'filename' && object.value !== undefined ) {

			const index = object.value[ 0 ];
			if ( ! textures[ index ] || ! textures[ index ].isTexture ) throw proceduralError( `Missing procedural texture resource ${ index } during export.` );
			const source = textures[ index ];
			if ( ! this.resources.has( source ) ) {

				// Graph images use glTF's top-first orientation. The ordinary writer
				// bakes flipY, so exporting an authored Texture with its default true
				// would otherwise change procedural sampling after reload.
				const resource = source.flipY ? source.clone() : source;
				if ( source.flipY ) resource.flipY = false;
				this.resources.set( source, resource );

			}

			object.value = [ await this.writer.processTextureAsync( this.resources.get( source ) ) ];
			return;

		}

		for ( const key of Object.keys( object ) ) await this.remapTextures( object[ key ], textures );

	}

	async writeMaterialAsync( material, materialDef ) {

		const state = assertTextureProceduralsSerializable( material );
		if ( ! state ) return;
		const writer = this.writer;
		const source = state.source;
		const definitions = source.procedural_definitions || [];
		const graphIndices = Object.values( state.bindings ).map( binding => binding.index );
		for ( const index of new Set( graphIndices ) ) validateGLTFProceduralGraph( source.procedurals[ index ], { definitions, textureCount: state.textures.length } );
		writer.json.extensions = writer.json.extensions || {};
		const root = writer.json.extensions[ this.name ] = writer.json.extensions[ this.name ] || { procedurals: [] };
		// Preserve serializable root metadata in addition to executable graphs.
		for ( const key of Object.keys( source ) ) {

			if ( key === 'procedurals' || key === 'procedural_definitions' ) continue;
			if ( root[ key ] !== undefined && canonicalJSON( root[ key ] ) !== canonicalJSON( source[ key ] ) ) throw proceduralError( `Conflicting root metadata ${ key }.` );
			root[ key ] = JSON.parse( JSON.stringify( source[ key ] ) );

		}

		const remappedDefinitions = JSON.parse( JSON.stringify( definitions ) );
		await this.remapTextures( remappedDefinitions, state.textures );
		const definitionRemap = new Map();
		for ( let i = 0; i < remappedDefinitions.length; i ++ ) {

			const definition = remappedDefinitions[ i ];
			if ( definition.nodetype !== 'nodedef' ) continue;
			const key = definition.node;
			const existing = this.definitions.get( key );
			if ( existing !== undefined && canonicalJSON( root.procedural_definitions[ existing ] ) !== canonicalJSON( definition ) ) throw proceduralError( `Conflicting custom definition ${ definition.node }.` );
			root.procedural_definitions = root.procedural_definitions || [];
			const index = existing === undefined ? root.procedural_definitions.push( definition ) - 1 : existing;
			this.definitions.set( key, index );
			definitionRemap.set( i, index );

		}

		for ( const definition of remappedDefinitions ) {

			if ( definition.nodetype !== 'nodegraph' ) continue;
			definition.nodedef = definitionRemap.get( definition.nodedef );
			const existing = root.procedural_definitions.find( candidate => candidate.nodetype === 'nodegraph' && candidate.nodedef === definition.nodedef );
			if ( existing && canonicalJSON( existing ) !== canonicalJSON( definition ) ) throw proceduralError( 'Conflicting functional custom implementation.' );
			if ( ! existing ) root.procedural_definitions.push( definition );

		}

		const localRemap = new Map();
		for ( const index of [ ...new Set( graphIndices ) ].sort( ( a, b ) => a - b ) ) {

			const graph = JSON.parse( JSON.stringify( source.procedurals[ index ] ) );
			await this.remapTextures( graph, state.textures );
			const key = canonicalJSON( graph );
			if ( ! this.graphIndices.has( key ) ) this.graphIndices.set( key, root.procedurals.push( graph ) - 1 );
			localRemap.set( index, this.graphIndices.get( key ) );

		}

		for ( const channel of Object.keys( PROCEDURAL_CHANNELS ) ) {

			const binding = state.bindings[ channel ];
			if ( ! binding ) continue;
			const info = getTextureInfo( materialDef, channel, true );
			if ( channel === 'normalTexture' || channel === 'clearcoatNormalTexture' ) {

				const scale = material[ channel === 'normalTexture' ? 'normalScale' : 'clearcoatNormalScale' ];
				if ( ! Number.isFinite( scale.x ) || ! Number.isFinite( scale.y ) || scale.x < 0 || Math.abs( scale.x ) !== Math.abs( scale.y ) ) throw proceduralError( 'Procedural normal scales must be finite, nonnegative and univariate for glTF export.' );
				info.scale = Math.abs( scale.x );

			}

			if ( channel === 'occlusionTexture' ) info.strength = material.aoMapIntensity;
			if ( info.index === undefined ) {

				const fallback = material[ PROCEDURAL_CHANNELS[ channel ].map ];
				info.index = fallback ? await writer.processTextureAsync( fallback ) : this.dummyTexture();
				if ( fallback ) {

					info.texCoord = fallback.channel;
					writer.applyTextureTransform( info, fallback );

				}

			}

			info.extensions = info.extensions || {};
			info.extensions[ this.name ] = { ...binding, index: localRemap.get( binding.index ) };
			const path = PROCEDURAL_CHANNELS[ channel ].path;
			if ( path[ 0 ] === 'extensions' ) {

				writer.extensionsUsed[ path[ 1 ] ] = true;
				preserveExtensionFactors( path[ 1 ], material, materialDef.extensions[ path[ 1 ] ] );

			}

		}

		writer.extensionsUsed[ EXTENSION_NAME ] = true;
		writer.extensionsUsed[ LIBRARY_NAME ] = true;
		if ( state.required ) writer.extensionsRequired[ EXTENSION_NAME ] = true;
		if ( materialDef.extras ) {

			delete materialDef.extras.textureProcedurals;
			if ( Object.keys( materialDef.extras ).length === 0 ) delete materialDef.extras;

		}

	}

}

export { GLTFTextureProceduralsExporterExtension };
