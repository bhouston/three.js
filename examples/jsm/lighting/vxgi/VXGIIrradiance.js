import { Fn, PI, float, interleavedGradientNoise, luminance, screenCoordinate, vec2, vec3, vec4 } from 'three/tsl';
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
 * samples, or a temporal filter like TRAA, converge. An optional `directionNode` in the query
 * receives the luminance-weighted cone directions, for the directional light maps of {@link LightMapper}.
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

	return ( { position, normal, sampleIndex = float( 0 ), directionNode = null } ) => Fn( () => {

		// interleaved gradient noise offsets of the stratified cone set, see VXGINode

		const shift = screenCoordinate.add( sampleIndex.mul( 5.588238 ) ).toConst();
		const elevation = interleavedGradientNoise( shift.add( vec2( 5.588238, 3.14159 ) ) ).toConst();
		const rotation = interleavedGradientNoise( shift ).toConst();

		const origin = position.add( normal.mul( volume.voxelSizeNode.mul( 1.5 ) ) ).toConst();

		const directionSum = directionNode !== null ? vec3( 0 ).toVar() : null;
		const onCone = directionSum !== null ? ( direction, radiance ) => directionSum.addAssign( direction.mul( luminance( radiance ) ) ) : null;

		const { color } = gather( trace, origin, normal, coneCount, tanHalfAngle, volume.traceDistanceNode, rotation, elevation, null, direction => volume.getEnvironmentRadiance( direction, tanHalfAngle ), onCone );

		if ( directionNode !== null ) directionNode.addAssign( directionSum.mul( PI.div( coneCount ) ) );

		return vec4( color.mul( PI ), volume.getCoverage( position ) );

	} )();

}

/**
 * Compatibility irradiance sampler. New consumers can compose vxgiDiffuse()
 * with explicit fallback providers through indirectDiffuse().
 *
 * @param {VXGIVolume} volume - The prepared volume.
 * @param {Object} [options] - Cone tracing options.
 * @return {function(Node<vec3>, Node<vec3>, ?Node<float>, ?Node<vec3>): Node<vec3>} The coverage-weighted irradiance sampler. The optional fourth argument receives the light directions.
 */
function vxgiIrradiance( volume, options ) {

	const provider = vxgiDiffuse( volume, options );
	return ( position, normal, sampleIndex = float( 0 ), directionNode = null ) => Fn( () => {

		const sample = provider( { position, normal, sampleIndex, directionNode } ).toConst();
		return sample.rgb.mul( sample.a );

	} )();

}

export { vxgiIrradiance, vxgiDiffuse };
