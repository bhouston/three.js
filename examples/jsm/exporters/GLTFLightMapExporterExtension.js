/**
 * A glTF exporter plugin for the vendor extension `MOZ_lightmap`.
 * Light maps must contain linear irradiance in an exportable image format.
 * Texture encoding follows GLTFExporter. Compressed textures require
 * GLTFExporter.setTextureUtils().
 *
 * ```js
 * exporter.register( writer => new GLTFLightMapExporterExtension( writer ) );
 * ```
 *
 * @three_import import { GLTFLightMapExporterExtension } from 'three/addons/exporters/GLTFLightMapExporterExtension.js';
 */
class GLTFLightMapExporterExtension {

	/**
	 * Constructs a light map exporter plugin.
	 *
	 * @param {GLTFWriter} writer - The glTF writer.
	 */
	constructor( writer ) {

		this.writer = writer;
		this.name = 'MOZ_lightmap';

	}

	/**
	 * Writes a material's light map.
	 *
	 * @param {Material} material - The material.
	 * @param {Object} materialDef - The glTF material definition.
	 * @return {Promise<void>} Resolves after the light map has been written.
	 */
	async writeMaterialAsync( material, materialDef ) {

		const map = material.lightMap;
		if ( ! map ) return;

		const extension = {
			index: await this.writer.processTextureAsync( map ),
			texCoord: map.channel,
			// Inverse of the PI scale applied to unlit materials by the loader plugin.
			intensity: material.isMeshBasicMaterial ? material.lightMapIntensity / Math.PI : material.lightMapIntensity
		};

		this.writer.applyTextureTransform( extension, map );
		materialDef.extensions = materialDef.extensions || {};
		materialDef.extensions[ this.name ] = extension;
		this.writer.extensionsUsed[ this.name ] = true;

	}

}

export { GLTFLightMapExporterExtension };
