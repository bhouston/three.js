import { RepeatWrapping, ClampToEdgeWrapping, MirroredRepeatWrapping, Color } from 'three';

const STANDARD_SURFACE_VERSION = '1.39';
const DEFAULT_COLOR_SPACE = 'lin_rec709';
const CHANNEL_INDEX = { r: 0, g: 1, b: 2, a: 3 };
const ADDRESS_MODE = new Map( [
	[ RepeatWrapping, 'periodic' ],
	[ ClampToEdgeWrapping, 'clamp' ],
	[ MirroredRepeatWrapping, 'mirror' ],
] );

// Declarative map of standard_surface shader inputs <- MeshStandardMaterial/MeshPhysicalMaterial
// properties. `value` is skipped automatically when the material doesn't have the property
// (e.g. Physical-only properties on a plain MeshStandardMaterial), so this table also covers
// MeshStandardMaterial without a separate list - MeshPhysicalMaterial is a superset.
const STANDARD_SURFACE_PROPERTIES = [
	{ input: 'base_color', type: 'color3', value: ( m ) => m.color, map: ( m ) => m.map },
	{ input: 'specular_roughness', type: 'float', value: ( m ) => m.roughness, map: ( m ) => m.roughnessMap, channel: 'g' },
	{ input: 'metalness', type: 'float', value: ( m ) => m.metalness, map: ( m ) => m.metalnessMap, channel: 'b' },
	{ input: 'specular', type: 'float', value: ( m ) => m.specularIntensity, map: ( m ) => m.specularIntensityMap, channel: 'a' },
	{ input: 'specular_color', type: 'color3', value: ( m ) => m.specularColor, map: ( m ) => m.specularColorMap },
	{ input: 'specular_anisotropy', type: 'float', value: ( m ) => m.anisotropy },
	{ input: 'specular_rotation', type: 'float', value: ( m ) => m.anisotropyRotation !== undefined ? m.anisotropyRotation / ( Math.PI * 2 ) : undefined },
	{ input: 'transmission', type: 'float', value: ( m ) => m.transmission, map: ( m ) => m.transmissionMap, channel: 'r' },
	{ input: 'transmission_color', type: 'color3', value: ( m ) => m.attenuationColor },
	{ input: 'transmission_depth', type: 'float', value: ( m ) => m.attenuationDistance !== undefined ? ( Number.isFinite( m.attenuationDistance ) ? m.attenuationDistance : 0 ) : undefined },
	{ input: 'thin_film_thickness', type: 'float', value: ( m ) => m.iridescence !== undefined ? ( m.iridescence ? m.iridescenceThicknessRange[ 1 ] : 0 ) : undefined, map: ( m ) => m.iridescenceThicknessMap, channel: 'g' },
	{ input: 'thin_film_IOR', type: 'float', value: ( m ) => m.iridescenceIOR },
	{ input: 'sheen', type: 'float', value: ( m ) => m.sheen },
	{ input: 'sheen_color', type: 'color3', value: ( m ) => m.sheenColor, map: ( m ) => m.sheenColorMap },
	{ input: 'sheen_roughness', type: 'float', value: ( m ) => m.sheenRoughness, map: ( m ) => m.sheenRoughnessMap, channel: 'a' },
	{ input: 'coat', type: 'float', value: ( m ) => m.clearcoat, map: ( m ) => m.clearcoatMap, channel: 'r' },
	{ input: 'coat_roughness', type: 'float', value: ( m ) => m.clearcoatRoughness, map: ( m ) => m.clearcoatRoughnessMap, channel: 'r' },
	{ input: 'coat_normal', type: 'vector3', map: ( m ) => m.clearcoatNormalMap, isNormalMap: true },
	{ input: 'normal', type: 'vector3', map: ( m ) => m.normalMap, isNormalMap: true },
	{ input: 'opacity', type: 'float', value: ( m ) => m.opacity, map: ( m ) => m.alphaMap, channel: 'r', skip: ( m ) => ! m.transparent && ! m.alphaMap },
	{ input: 'specular_IOR', type: 'float', value: ( m ) => m.ior },
	{ input: 'emission', type: 'float', value: ( m ) => m.emissiveIntensity, skip: ( m ) => ! m.emissive.getHex() && ! m.emissiveMap },
	{ input: 'emission_color', type: 'color3', value: ( m ) => m.emissive, map: ( m ) => m.emissiveMap, skip: ( m ) => ! m.emissive.getHex() && ! m.emissiveMap },
];

function toValueString( components ) {

	return components.map( ( c ) => Number( c.toFixed( 6 ) ) ).join( ', ' );

}

function formatValue( value ) {

	return value instanceof Color ? toValueString( value.toArray() ) : Number( Number( value ).toFixed( 6 ) );

}

function createElement( xmlDoc, tagName, attributes = {}, children = [] ) {

	const element = xmlDoc.createElement( tagName );
	for ( const name in attributes ) {

		const value = attributes[ name ];
		if ( value !== null && value !== undefined ) element.setAttribute( name, value );

	}

	for ( const child of children ) element.appendChild( child );

	return element;

}

// MaterialX `filename` inputs are plain filesystem paths (see FilePath in MaterialXFormat) with
// no URI-scheme handling anywhere in the spec or reference implementation - unlike glTF, `.mtlx`
// has no embedded/data-URI convention, so this reuses the texture's existing URL as-is rather than
// inventing one. That's the normal case: a texture loaded via TextureLoader already carries a real,
// portable path, matching every sample file in examples/materialx/ (plain relative paths).
// ponytail: a texture with no resolvable URL (canvas-only, no `src`) can't be referenced this way -
// add a `{ text, textures }` return value with rasterized Blobs for the caller to save as sidecar
// files when that workflow is actually needed.
function getTextureURI( texture ) {

	const image = texture.image;
	const uri = image && ( image.src || image.currentSrc );

	if ( ! uri ) {

		console.warn( 'THREE.MaterialXExporter: texture has no resolvable URL (image.src) and was skipped. Load it via a URL-based loader (e.g. TextureLoader) to export it.' );
		return null;

	}

	return uri;

}

// Builds (or reuses, per unique texture) a <texcoord>/<place2d>/<image> chain inside the shared
// material nodegraph, and returns the name of the node holding the full sampled color4.
function getOrCreateImageNode( xmlDoc, nodegraph, textureImageNodes, texture ) {

	let imageName = textureImageNodes.get( texture );
	if ( imageName ) return imageName;

	const uri = getTextureURI( texture );
	if ( ! uri ) return null;

	const index = textureImageNodes.size;
	const uvIndex = texture.channel || 0;
	const hasTransform = texture.offset.x !== 0 || texture.offset.y !== 0 ||
		texture.repeat.x !== 1 || texture.repeat.y !== 1 || texture.rotation !== 0;

	let texcoordName = `texcoord_${index}`;
	nodegraph.appendChild( createElement( xmlDoc, 'texcoord', { name: texcoordName, type: 'vector2' }, [
		createElement( xmlDoc, 'input', { name: 'index', type: 'integer', value: uvIndex } ),
	] ) );

	if ( hasTransform ) {

		const place2dName = `place2d_${index}`;
		nodegraph.appendChild( createElement( xmlDoc, 'place2d', { name: place2dName, type: 'vector2' }, [
			createElement( xmlDoc, 'input', { name: 'texcoord', type: 'vector2', nodename: texcoordName } ),
			createElement( xmlDoc, 'input', { name: 'offset', type: 'vector2', value: toValueString( [ texture.offset.x, texture.offset.y ] ) } ),
			createElement( xmlDoc, 'input', { name: 'rotate', type: 'float', value: Number( ( texture.rotation * 180 / Math.PI ).toFixed( 6 ) ) } ),
			createElement( xmlDoc, 'input', { name: 'scale', type: 'vector2', value: toValueString( [ texture.repeat.x, texture.repeat.y ] ) } ),
		] ) );
		texcoordName = place2dName;

	}

	imageName = `image_${index}`;
	nodegraph.appendChild( createElement( xmlDoc, 'image', { name: imageName, type: 'color4' }, [
		createElement( xmlDoc, 'input', { name: 'file', type: 'filename', value: uri } ),
		createElement( xmlDoc, 'input', { name: 'texcoord', type: 'vector2', nodename: texcoordName } ),
		createElement( xmlDoc, 'input', { name: 'uaddressmode', type: 'string', value: ADDRESS_MODE.get( texture.wrapS ) || 'periodic' } ),
		createElement( xmlDoc, 'input', { name: 'vaddressmode', type: 'string', value: ADDRESS_MODE.get( texture.wrapT ) || 'periodic' } ),
	] ) );

	textureImageNodes.set( texture, imageName );
	return imageName;

}

// Wires a shader input up to a texture (directly, through a channel-extracting <extract> node,
// or through a <normalmap> node) and returns the <output> name exposing the result.
function wireTextureOutput( xmlDoc, nodegraph, textureImageNodes, outputNodes, spec, texture ) {

	const imageName = getOrCreateImageNode( xmlDoc, nodegraph, textureImageNodes, texture );
	if ( imageName === null ) return null;

	const key = `${imageName}|${spec.channel || ''}|${spec.isNormalMap ? 'normal' : ''}`;
	let outputName = outputNodes.get( key );
	if ( outputName ) return outputName;

	let sourceName = imageName;
	let sourceType = 'color4';

	if ( spec.isNormalMap ) {

		sourceName = `normalmap_${outputNodes.size}`;
		sourceType = 'vector3';
		nodegraph.appendChild( createElement( xmlDoc, 'normalmap', { name: sourceName, type: 'vector3' }, [
			createElement( xmlDoc, 'input', { name: 'in', type: 'vector3', nodename: imageName } ),
		] ) );

	} else if ( spec.channel ) {

		sourceName = `extract_${outputNodes.size}`;
		sourceType = 'float';
		nodegraph.appendChild( createElement( xmlDoc, 'extract', { name: sourceName, type: 'float' }, [
			createElement( xmlDoc, 'input', { name: 'in', type: 'color4', nodename: imageName } ),
			createElement( xmlDoc, 'input', { name: 'index', type: 'integer', value: CHANNEL_INDEX[ spec.channel ] } ),
		] ) );

	}

	outputName = `out_${outputNodes.size}`;
	nodegraph.appendChild( createElement( xmlDoc, 'output', { name: outputName, type: sourceType === 'color4' ? spec.type : sourceType, nodename: sourceName } ) );
	outputNodes.set( key, outputName );
	return outputName;

}

// Builds a `<materialx>` document containing a `standard_surface` shader for a
// MeshStandardMaterial/MeshPhysicalMaterial (MeshPhysicalMaterial's extra properties - clearcoat,
// sheen, transmission, iridescence, specular, anisotropy - are exported too, when present).
// ponytail: anisotropyMap direction+strength encoding and diffuseRoughness/dispersion/
// retroreflectivity have no standard_surface equivalent used here - add if a material needs them.
function buildStandardSurfaceDocument( material ) {

	const xmlDoc = globalThis.document.implementation.createDocument( null, 'materialx', null );
	const root = xmlDoc.documentElement;
	root.setAttribute( 'version', STANDARD_SURFACE_VERSION );
	root.setAttribute( 'colorspace', DEFAULT_COLOR_SPACE );

	const materialName = material.name || 'Material';
	const shaderName = `SR_${materialName}`;
	const nodegraphName = `NG_${materialName}`;

	const shader = createElement( xmlDoc, 'standard_surface', { name: shaderName, type: 'surfaceshader' } );
	const nodegraph = createElement( xmlDoc, 'nodegraph', { name: nodegraphName } );
	const textureImageNodes = new Map();
	const outputNodes = new Map();

	for ( const spec of STANDARD_SURFACE_PROPERTIES ) {

		const texture = spec.map ? spec.map( material ) : null;
		const value = spec.value ? spec.value( material ) : undefined;

		if ( spec.skip && spec.skip( material ) ) continue;
		if ( value === undefined && ! texture ) continue;

		if ( texture ) {

			const outputName = wireTextureOutput( xmlDoc, nodegraph, textureImageNodes, outputNodes, spec, texture );

			if ( outputName !== null ) {

				shader.appendChild( createElement( xmlDoc, 'input', { name: spec.input, type: spec.type, nodegraph: nodegraphName, output: outputName } ) );
				continue;

			}

		}

		if ( value === undefined ) continue;
		shader.appendChild( createElement( xmlDoc, 'input', { name: spec.input, type: spec.type, value: formatValue( value ) } ) );

	}

	if ( nodegraph.childNodes.length > 0 ) root.appendChild( nodegraph );
	root.appendChild( shader );

	const surfacematerial = createElement( xmlDoc, 'surfacematerial', { name: materialName, type: 'material' } );
	surfacematerial.appendChild( createElement( xmlDoc, 'input', { name: 'surfaceshader', type: 'surfaceshader', nodename: shaderName } ) );
	root.appendChild( surfacematerial );

	return root;

}

/**
 * An exporter for the MaterialX format.
 *
 * Accepts either a {@link MaterialXDocument} (as returned by `MaterialXLoader.parse()`) and
 * serializes it back into MaterialX XML text, or a `MeshStandardMaterial`/`MeshPhysicalMaterial`,
 * which is first translated into a `standard_surface` MaterialX document. Textures are referenced
 * by their existing URL (`.mtlx` has no embedded/data-URI convention, unlike glTF) - load them via
 * a URL-based loader such as `TextureLoader` so they have a resolvable path to export.
 *
 * ```js
 * const { document } = new MaterialXLoader().parse( mtlxText );
 * const text = new MaterialXExporter().parse( document );
 *
 * const materialText = new MaterialXExporter().parse( mesh.material );
 * ```
 *
 * @three_import import { MaterialXExporter } from 'three/addons/exporters/MaterialXExporter.js';
 */
class MaterialXExporter {

	/**
	 * Serializes the given MaterialX document, or Standard/Physical material, into a
	 * MaterialX XML string.
	 *
	 * @param {MaterialXDocument|MeshStandardMaterial} input - The document or material to export.
	 * @return {string} The MaterialX document as XML text.
	 */
	parse( input ) {

		let rootElement;

		if ( input.isMeshStandardMaterial ) {

			rootElement = buildStandardSurfaceDocument( input );

		} else if ( input.rootNode ) {

			rootElement = input.rootNode.nodeXML;

		} else {

			throw new Error( 'THREE.MaterialXExporter: Unsupported input. Expected a MaterialXDocument or a MeshStandardMaterial/MeshPhysicalMaterial.' );

		}

		return '<?xml version="1.0"?>\n' + new XMLSerializer().serializeToString( rootElement );

	}

}

export { MaterialXExporter };
