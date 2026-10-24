/**
 * Shared work-budget starting points, to calibrate against target hardware and
 * scenes. These are candidate presets rather than universal quality optima.
 */
export const SSGI_QUALITY_PRESETS = Object.freeze( {
	Low: Object.freeze( { resolutionScale: 0.5, sliceCount: 1, stepCount: 8, maxFrames: 16, denoiseRadius: 1.5 } ),
	Balanced: Object.freeze( { resolutionScale: 0.5, sliceCount: 2, stepCount: 8, maxFrames: 16, denoiseRadius: 1.5 } ),
	High: Object.freeze( { resolutionScale: 1, sliceCount: 3, stepCount: 12, maxFrames: 16, denoiseRadius: 1 } )
} );

/**
 * Resolve radius and thickness without inferring physical surface thickness
 * from scene bounds. Optional authored metadata (for example glTF extras.ssgi)
 * can specify a stable local reference scale, individual local-unit distances,
 * or explicit world-unit distances. A local distance scales with the model;
 * an explicit world distance deliberately stays fixed when the model scales.
 *
 * @param {Object} settings - Relative radius/thickness and quality parameters.
 * @param {number} boundsScale - World-space scene diagonal, used only as a fallback.
 * @param {Object} [metadata={}] - Authored localScale/localRadius/localThickness/worldRadius/worldThickness overrides.
 * @param {number} [modelScale=1] - Uniform model-local to world-space scale.
 * @return {Object} Parameters with radius and thickness resolved to world units.
 */
export function resolveSSGISettings( settings, boundsScale, metadata = {}, modelScale = 1 ) {

	const referenceScale = metadata.localScale === undefined ? boundsScale : metadata.localScale * modelScale;
	return {
		...settings,
		radius: metadata.worldRadius ?? ( metadata.localRadius === undefined ? settings.radius * referenceScale : metadata.localRadius * modelScale ),
		thickness: metadata.worldThickness ?? ( metadata.localThickness === undefined ? settings.thickness * referenceScale : metadata.localThickness * modelScale )
	};

}
