import { Fn, PI, float, interleavedGradientNoise, screenCoordinate, vec2, vec4 } from 'three/tsl';
import { createConeTracer, gather } from './VXGIConeTracer.js';

/**
 * Creates a world-space diffuse provider for a prepared {@link VXGIVolume}.
 * Call `volume.update( renderer, scene )` before baking. The sampler traces cosine-weighted
 * cones through the volume's radiance, including cached bounces.
 * Compose it with indirectDiffuse() for {@link LightMapper} or live shading.
 * Escaping cones sample the volume's environment before hemisphere integration.
 * Coverage is spatial confidence, independently of cone opacity and surface brightness.
 * The cones are rotated per pixel with interleaved gradient noise like `VXGINode`. When baking, the
 * pixel is the light map texel. The optional sample index shifts the noise so that accumulated
 * samples, or a temporal filter like TRAA, converge.
 *
 * @param {VXGIVolume} volume - The prepared voxel volume.
 * @param {Object} [options] - Options.
 * @param {number} [options.coneCount=8] - The number of cones per sample. Fewer cones are faster but coarser.
 * @param {number} [options.coneAngle=40] - Full cone aperture in degrees.
 * @return {function(Object): Node<vec4>} A provider returning unweighted irradiance and coverage.
 */
function vxgiDiffuse( volume, { coneCount = 8, coneAngle = 40 } = {} ) {

	const tanHalfAngle = float( Math.tan( coneAngle * Math.PI / 360 ) );
	const trace = createConeTracer( volume, {
		radianceNode: volume.radianceNode,
		directionalNode: volume.directionalRadiance ? volume.directionalNode : null
	} );

	return ( { position, normal, sampleIndex = float( 0 ) } ) => Fn( () => {

		// interleaved gradient noise offsets of the stratified cone set, see VXGINode

		const shift = screenCoordinate.add( sampleIndex.mul( 5.588238 ) ).toConst();
		const elevation = interleavedGradientNoise( shift.add( vec2( 5.588238, 3.14159 ) ) ).toConst();
		const rotation = interleavedGradientNoise( shift ).toConst();

		const origin = position.add( normal.mul( volume.voxelSizeNode.mul( 1.5 ) ) ).toConst();

		const { color } = gather( trace, origin, normal, coneCount, tanHalfAngle, volume.traceDistanceNode, rotation, elevation, null, direction => volume.getEnvironmentRadiance( direction, tanHalfAngle ) );

		return vec4( color.mul( PI ), volume.getCoverage( position ) );

	} )();

}

/**
 * Compatibility irradiance sampler. New consumers can compose vxgiDiffuse()
 * with explicit fallback providers through indirectDiffuse().
 *
 * @param {VXGIVolume} volume - The prepared volume.
 * @param {Object} [options] - Cone tracing options.
 * @return {Function} The coverage-weighted irradiance sampler.
 */
function vxgiIrradiance( volume, options ) {

	const provider = vxgiDiffuse( volume, options );
	return ( position, normal, sampleIndex = float( 0 ) ) => Fn( () => {

		const sample = provider( { position, normal, sampleIndex } ).toConst();
		return sample.rgb.mul( sample.a );

	} )();

}

export { vxgiIrradiance, vxgiDiffuse };
