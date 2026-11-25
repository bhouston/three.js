import { AnalyticLightNode, NodeUpdateType, Vector3 } from 'three/webgpu';
import { INFINITY, If, array, float, getShIrradianceAt, normalWorld, positionWorld, renderGroup, step, texture3D, uniform, vec3, vec4 } from 'three/tsl';

// Padding texels at each boundary of every atlas sub-volume.
export const ATLAS_PADDING = 1;

/**
 * Samples the packed SH atlas and evaluates L2 irradiance for the given normal.
 *
 * The atlas stores the seven RGBA sub-volumes stacked along Z, each occupying
 * `( nz + 2 )` slices: one padding slice (a copy of the nearest edge slice) at
 * each end to prevent color bleeding when the hardware trilinear filter reads
 * across a sub-volume boundary.
 *
 * @private
 * @param {Texture3DNode} atlas - The atlas texture node.
 * @param {Node<vec3>} uvw - The probe-grid sample coordinate (texel centers).
 * @param {Node<vec3>} res - The probe resolution.
 * @param {Node<vec3>} normal - The world-space normal.
 * @return {Node<vec3>} The non-negative irradiance.
 */
function evaluateGridIrradiance( atlas, uvw, res, normal ) {

	const nz = res.z;
	const paddedSlices = nz.add( 2.0 * ATLAS_PADDING );
	const atlasDepth = paddedSlices.mul( 7.0 );
	const uvZBase = uvw.z.mul( nz ).add( ATLAS_PADDING );

	const slice = ( t ) => atlas.sample( vec3( uvw.xy, uvZBase.add( paddedSlices.mul( t ) ).div( atlasDepth ) ) ).level( 0 );

	const s0 = slice( 0 ), s1 = slice( 1 ), s2 = slice( 2 ), s3 = slice( 3 );
	const s4 = slice( 4 ), s5 = slice( 5 ), s6 = slice( 6 );

	// Unpack 9 vec3 L2 SH coefficients and evaluate irradiance.

	const sh = array( [
		s0.xyz,
		vec3( s0.w, s1.xy ),
		vec3( s1.zw, s2.x ),
		s2.yzw,
		s3.xyz,
		vec3( s3.w, s4.xy ),
		vec3( s4.zw, s5.x ),
		s5.yzw,
		s6.xyz
	] );

	return getShIrradianceAt( normal, sh ).max( vec3( 0.0 ) );

}

/**
 * Evaluates the irradiance of a light probe grid at the given world-space position and normal.
 *
 * @private
 * @param {Texture3DNode} atlas - The atlas texture node of the grid.
 * @param {Node<vec3>} min - The minimum of the grid bounds.
 * @param {Node<vec3>} max - The maximum of the grid bounds.
 * @param {Node<vec3>} res - The probe resolution.
 * @param {Node<float>} intensity - The intensity.
 * @param {Node<vec3>} position - The world-space position.
 * @param {Node<vec3>} normal - The world-space normal.
 * @return {Node<vec3>} The irradiance.
 */
function getGridIrradiance( atlas, min, max, res, intensity, position, normal ) {

	const range = max.sub( min );
	const spacing = range.div( res );

	// Offset along the normal by half a probe spacing. The bounds hold one cell per
	// probe, so they map directly to the probe grid; clamp to the outermost texel centers.

	const samplePos = position.add( normal.mul( spacing ).mul( 0.5 ) );
	const halfTexel = vec3( 0.5 ).div( res );
	const uvw = samplePos.sub( min ).div( range ).clamp( halfTexel, halfTexel.oneMinus() );

	return evaluateGridIrradiance( atlas, uvw, res, normal ).mul( intensity );

}

/**
 * Returns whether a fragment is lit by a light probe grid, which is inside of its bounds.
 *
 * @private
 * @param {Node<vec3>} min - The minimum of the grid bounds.
 * @param {Node<vec3>} max - The maximum of the grid bounds.
 * @param {Node<vec3>} res - The probe resolution.
 * @param {Node<vec3>} normal - The world-space normal.
 * @return {Node<bool>} Whether the fragment is inside of the grid.
 */
function isInsideGrid( min, max, res, normal ) {

	const spacing = max.sub( min ).div( res );

	// An axis with a single probe isn't offset along the normal, since that offset would span the whole grid.

	const samplePos = positionWorld.add( normal.mul( spacing ).mul( 0.5 ).mul( step( 1.5, res ) ) );

	// Grow the bounds by 1% of a cell, so fragments on the bounds aren't left out by
	// precision errors or samples outside of the triangle.

	const margin = spacing.mul( 0.01 );

	return samplePos.greaterThanEqual( min.sub( margin ) ).all().and( samplePos.lessThanEqual( max.add( margin ) ).all() );

}

/**
 * Returns a light probe grid as an indirect diffuse source for `builtinIndirectContext()`,
 * see {@link LightProbeGrid#getDiffuseSource}.
 *
 * @private
 * @param {LightProbeGrid} light - The light probe grid.
 * @return {function(Object): Node<vec4>} The source.
 */
export function getGridDiffuseSource( light ) {

	// the source follows the grid, whose bounds, resolution and atlas can be replaced

	const min = uniform( new Vector3() ).onRenderUpdate( () => light.boundingBox.min );
	const max = uniform( new Vector3() ).onRenderUpdate( () => light.boundingBox.max );
	const res = uniform( new Vector3() ).onRenderUpdate( () => light.resolution );
	const intensity = uniform( 1 ).onRenderUpdate( () => light.intensity );
	const atlas = texture3D( light.texture ).onRenderUpdate( () => light.texture );

	return ( { normal } ) => {

		const irradiance = getGridIrradiance( atlas, min, max, res, intensity, positionWorld, normal );

		return vec4( irradiance, 1 ).mul( float( isInsideGrid( min, max, res, normal ) ) );

	};

}

/**
 * The light node that applies a {@link LightProbeGrid} to the scene. It samples
 * the baked L2 spherical-harmonic atlas at the surface position and adds the
 * resulting irradiance to the lighting context, so every standard node material
 * picks up the grid automatically (same role as the WebGL `lights_fragment_begin`
 * integration).
 *
 * @private
 * @augments AnalyticLightNode
 */
class LightProbeGridNode extends AnalyticLightNode {

	static get type() {

		return 'LightProbeGridNode';

	}

	constructor( light = null ) {

		super( light );

		// Bounce grid intensity changes between bake passes within a frame.

		this._min = uniform( new Vector3() );
		this._max = uniform( new Vector3() );
		this._resolution = uniform( new Vector3() );
		this._intensity = uniform( 1 ).setGroup( renderGroup );
		this._priority = uniform( 0 ).setGroup( renderGroup );

		this.updateType = NodeUpdateType.RENDER;

	}

	update( /* frame */ ) {

		const light = this.light;

		this._min.value.copy( light.boundingBox.min );
		this._max.value.copy( light.boundingBox.max );
		this._resolution.value.copy( light.resolution );
		this._intensity.value = light.intensity;
		this._priority.value = light.priority;

	}

	setup( builder ) {

		const light = this.light;

		const min = this._min;
		const max = this._max;
		const res = this._resolution;

		const inside = isInsideGrid( min, max, res, normalWorld );

		builder.context.lightProbeIrradiance ||= vec3().toVar( 'lightProbeIrradiance' );
		builder.context.lightProbePriority ||= INFINITY.negate().toVar( 'lightProbePriority' );

		const prioritized = this._priority.greaterThanEqual( builder.context.lightProbePriority );

		If( inside.and( prioritized ), () => {

			const irradiance = getGridIrradiance( texture3D( light.texture ), min, max, res, this._intensity, positionWorld, normalWorld ).toConst();

			// fragments can be located in multiple grids but only the one with the highest priority defines the irradiance,
			// on ties the most recently created grid wins, see #34798

			builder.context.lightProbeIrradiance.assign( irradiance );
			builder.context.lightProbePriority.assign( this._priority );

		} );

	}

}

export { LightProbeGridNode };
