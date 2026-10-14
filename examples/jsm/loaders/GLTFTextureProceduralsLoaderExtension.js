import { compileGLTFProceduralGraph, validateGLTFProceduralGraph } from './materialx/GLTFProceduralGraph.js';
import { EXTENSION_NAME, LIBRARY_NAME, PROCEDURAL_CHANNELS, getTextureInfo, proceduralError, attachTextureProcedurals, createTextureProceduralMaterial, getTextureProcedurals } from '../utils/GLTFTextureProcedurals.js';

/**
 * Experimental opt-in loader for the b13ced8e KHR_texture_procedurals draft.
 * Register with GLTFLoader.register( parser => new GLTFTextureProceduralsLoaderExtension( parser ) ).
 * Procedural node materials require WebGPURenderer. See docs/TextureProcedurals.md.
 */
class GLTFTextureProceduralsLoaderExtension {

	constructor( parser, options = {} ) {

		this.name = EXTENSION_NAME;
		this.parser = parser;
		this.onDiagnostic = options.onDiagnostic || ( diagnostic => console.warn( diagnostic.message ) );
		this.diagnostics = [];
		this.materials = new Map();
		this.graphs = new Map();
		this.textures = new Array( ( parser.json.textures || [] ).length );
		this.required = ( parser.json.extensionsRequired || [] ).some( name => name === EXTENSION_NAME || /^EXT_texture_procedurals_mx_/.test( name ) );
		this.disabled = false;

	}

	diagnose( error, context = {} ) {

		if ( this.required ) throw error;
		const diagnostic = { message: error.message, ...context };
		this.diagnostics.push( diagnostic );
		this.onDiagnostic( diagnostic );

	}

	async beforeRoot() {

		const json = this.parser.json;
		this.source = json.extensions && json.extensions[ EXTENSION_NAME ];
		if ( ! this.source ) {

			if ( this.required || ( json.extensionsUsed || [] ).includes( EXTENSION_NAME ) ) this.diagnose( proceduralError( 'Missing root KHR_texture_procedurals data.' ) );
			return;

		}

		try {

			const used = json.extensionsUsed || [];
			const libraries = used.filter( name => /^EXT_texture_procedurals_mx_/.test( name ) );
			if ( ! used.includes( EXTENSION_NAME ) || libraries.length !== 1 || libraries[ 0 ] !== LIBRARY_NAME ) throw proceduralError( 'Declare KHR_texture_procedurals and exactly one supported library, EXT_texture_procedurals_mx_1_39, in extensionsUsed.' );
			if ( json.extensions[ LIBRARY_NAME ] !== undefined ) throw proceduralError( 'The MaterialX library declaration has no extension payload.' );
			if ( ! Array.isArray( this.source.procedurals ) ) throw proceduralError( 'Root procedurals must be an array.' );
			// Register the asset-wide declaration with the parser as a known identifier.
			this.parser.extensions[ LIBRARY_NAME ] = true;
			for ( const animation of json.animations || [] ) {

				for ( const channel of animation.channels || [] ) {

					const pointer = channel.target && channel.target.extensions && channel.target.extensions.KHR_animation_pointer && channel.target.extensions.KHR_animation_pointer.pointer;
					if ( typeof pointer === 'string' && /^\/extensions\/(?:KHR_texture_procedurals|procedurals)\//.test( pointer ) ) {

						const error = proceduralError( 'Graph-interface animation is not supported by this experimental profile.' );
						if ( ( json.extensionsRequired || [] ).includes( 'KHR_animation_pointer' ) ) throw error;
						this.diagnose( error, { pointer } );

					}

				}

			}

			if ( this.required ) {

				for ( const graph of this.source.procedurals ) validateGLTFProceduralGraph( graph, { definitions: this.source.procedural_definitions || [], textureCount: this.textures.length } );

			}

		} catch ( error ) {

			// A required animation pointer must fail even when the procedural is optional.
			if ( ( json.extensionsRequired || [] ).includes( 'KHR_animation_pointer' ) && /animation/.test( error.message ) ) throw error;
			this.disabled = true;
			this.diagnose( error );

		}

	}

	loadMaterial( materialIndex ) {

		const parser = this.parser;
		const materialDef = parser.json.materials[ materialIndex ];
		const bindings = {};
		for ( const channel of Object.keys( PROCEDURAL_CHANNELS ) ) {

			const info = getTextureInfo( materialDef, channel );
			if ( info && info.extensions && info.extensions[ this.name ] ) bindings[ channel ] = info.extensions[ this.name ];

		}

		const allBindings = [];
		function visit( object, path = '' ) {

			if ( ! object || typeof object !== 'object' ) return;
			if ( object.extensions && object.extensions[ EXTENSION_NAME ] ) allBindings.push( path );
			for ( const key of Object.keys( object ) ) {

				visit( object[ key ], path ? `${ path }.${ key }` : key );

			}

		}

		visit( materialDef );
		if ( allBindings.length === 0 ) return null;
		return ( async () => {

			const fallback = await parser.loadMaterial( materialIndex );
			if ( this.disabled || ! this.source ) {

				if ( ! this.source ) this.diagnose( proceduralError( 'Texture-info references missing root procedural data.' ), { materialIndex } );
				return fallback;

			}

			const recognized = Object.keys( bindings ).map( channel => PROCEDURAL_CHANNELS[ channel ].path.join( '.' ) );
			for ( const path of allBindings ) {

				if ( ! recognized.includes( path ) ) this.diagnose( proceduralError( `Unsupported procedural texture-info binding ${ path }.` ), { materialIndex } );

			}

			const options = {
				textures: this.textures,
				required: this.required,
				compileGraph: ( index, graph, graphOptions ) => {

					const key = JSON.stringify( { graph, definitions: graphOptions.definitions } );
					if ( ! this.graphs.has( key ) ) this.graphs.set( key, compileGLTFProceduralGraph( graph, {
						...graphOptions,
						manager: parser.options.manager,
						resolveTexture: async textureIndex => {

							const texture = await parser.getDependency( 'texture', textureIndex );
							if ( ! texture ) throw proceduralError( `Texture ${ textureIndex } could not be loaded.` );
							this.textures[ textureIndex ] = texture;
							return texture;

						}
					} ) );
					return this.graphs.get( key );

				}
			};
			const supported = {};
			for ( const channel of Object.keys( bindings ) ) {

				try {

					const info = getTextureInfo( materialDef, channel );
					if ( ! Number.isInteger( info.index ) || ! parser.json.textures || ! parser.json.textures[ info.index ] ) throw proceduralError( `A valid fallback texture index is required for ${ channel }.` );
					await createTextureProceduralMaterial( fallback, this.source, { [ channel ]: bindings[ channel ] }, options );
					supported[ channel ] = bindings[ channel ];

				} catch ( error ) {

					this.diagnose( error, { materialIndex, channel } );

				}

			}

			if ( Object.keys( supported ).length === 0 ) return fallback;
			const material = await createTextureProceduralMaterial( fallback, this.source, supported, options );
			parser.associations.set( material, { materials: materialIndex } );
			this.materials.set( materialIndex, { state: getTextureProcedurals( material ), fallback } );
			return material;

		} )();

	}

	afterRoot( result ) {

		for ( const scene of result.scenes ) {

			scene.traverse( mesh => {

				if ( ! mesh.material ) return;
				const materials = Array.isArray( mesh.material ) ? mesh.material : [ mesh.material ];
				for ( let i = 0; i < materials.length; i ++ ) {

					const material = materials[ i ];
					const association = this.parser.associations.get( material );
					const entry = association && this.materials.get( association.materials );
					if ( ! entry ) continue;
					try {

						if ( ! mesh.isMesh ) throw proceduralError( 'Procedural materials currently support triangle meshes only.' );
						if ( ( entry.state.bindings.normalTexture || entry.state.bindings.clearcoatNormalTexture ) && ! mesh.geometry.hasAttribute( 'tangent' ) ) {

							const uv = mesh.geometry.getAttribute( 'uv' );
							if ( ! uv || uv.itemSize !== 2 ) throw proceduralError( 'Procedural tangent normals require TEXCOORD_0 when geometry has no tangents.' );

						}

						for ( const compiled of entry.state.compiled.values() ) {

							for ( const stream of compiled.streams || [] ) {

								const attribute = mesh.geometry.getAttribute( stream.attribute );
								if ( ! attribute || ( stream.semantic.startsWith( 'TEXCOORD_' ) && attribute.itemSize !== 2 ) || ( stream.semantic.startsWith( 'COLOR_' ) && attribute.itemSize !== 3 && attribute.itemSize !== 4 ) ) throw proceduralError( `Missing or invalid ${ stream.semantic } on mesh ${ mesh.name }.` );

							}

						}

						attachTextureProcedurals( material, entry.state );

					} catch ( error ) {

						this.diagnose( error, { mesh: mesh.name } );
						const primitive = { geometry: mesh.geometry, material: entry.fallback, isMesh: mesh.isMesh, isLine: mesh.isLine, isPoints: mesh.isPoints };
						this.parser.assignFinalMaterial( primitive );
						materials[ i ] = primitive.material;

					}

				}

				mesh.material = Array.isArray( mesh.material ) ? materials : materials[ 0 ];

			} );

		}

		result.userData.textureProceduralsDiagnostics = this.diagnostics;

	}

}

export { GLTFTextureProceduralsLoaderExtension };
