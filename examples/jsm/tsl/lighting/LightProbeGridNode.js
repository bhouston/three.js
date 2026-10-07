import { AnalyticLightNode, Vector3 } from 'three/webgpu';
import { array, cameraPosition, float, getShIrradianceAt, If, int, Loop, normalWorld, NodeUpdateType, positionWorld, renderGroup, select, texture, texture3D, uniform, vec2, vec3, vec4 } from 'three/tsl';

// Padding texels at each boundary of every atlas sub-volume.
export const ATLAS_PADDING = 1;

// Per-probe distance moments are stored as octahedral tiles in a 2D atlas: a
// DISTANCE_RESOLUTION² interior plus a 1-texel mirrored border for filtering,
// DISTANCE_COLUMNS tiles per row, in bake order.
export const DISTANCE_RESOLUTION = 16;
export const DISTANCE_TILE = DISTANCE_RESOLUTION + 2;
export const DISTANCE_COLUMNS = 32;

// Per-probe irradiance, evaluated from the SH for every normal direction, in the
// same tile layout with a smaller octahedral resolution. With visibility, shading
// takes one bilinear sample per probe instead of seven SH texels.
export const IRRADIANCE_RESOLUTION = 8;
export const IRRADIANCE_TILE = IRRADIANCE_RESOLUTION + 2;

// Slack in normalized distance before a receiver counts as occluded. Without it,
// low-variance moments flip the weights between probes on slanted surfaces.
const VISIBILITY_SLACK = 0.03;

const signNotZero = ( value ) => select( value.greaterThanEqual( 0.0 ), float( 1.0 ), float( - 1.0 ) );

/**
 * Maps a direction to octahedral coordinates in [0, 1]².
 *
 * @private
 * @param {Node<vec3>} direction - The direction.
 * @return {Node<vec2>} The octahedral coordinates.
 */
export function octEncode( direction ) {

	const p = direction.div( direction.abs().dot( vec3( 1.0 ) ).max( 1e-6 ) );
	const folded = vec2( p.y.abs().oneMinus().mul( signNotZero( p.x ) ), p.x.abs().oneMinus().mul( signNotZero( p.y ) ) );

	return select( p.z.lessThan( 0.0 ), folded, p.xy ).mul( 0.5 ).add( 0.5 );

}

/**
 * Returns the probe index and the octahedral texel of an atlas texel, for atlases
 * of square octahedral tiles with a 1-texel border, `DISTANCE_COLUMNS` tiles per
 * row in bake order. Border texels mirror the opposite side of their tile edge
 * (corners take the diagonal corner), so bilinear filtering is continuous across
 * the octahedral seams.
 *
 * @param {Node<ivec2>} texel - The atlas texel.
 * @param {number} resolution - The octahedral resolution of a tile.
 * @return {Object} The `probeIndex` and the octahedral texel `x`, `y`.
 */
export function octTileTexel( texel, resolution ) {

	const tileSize = resolution + 2;
	const probeIndex = texel.y.div( tileSize ).mul( DISTANCE_COLUMNS ).add( texel.x.div( tileSize ) );

	const x0 = texel.x.mod( tileSize ).sub( 1 );
	const y0 = texel.y.mod( tileSize ).sub( 1 );
	const x1 = x0.clamp( 0, resolution - 1 );
	const y1 = select( x0.lessThan( 0 ).or( x0.greaterThanEqual( resolution ) ), int( resolution - 1 ).sub( y0 ), y0 );
	const x = select( y1.lessThan( 0 ).or( y1.greaterThanEqual( resolution ) ), int( resolution - 1 ).sub( x1 ), x1 );
	const y = y1.clamp( 0, resolution - 1 );

	return { probeIndex, x, y };

}

/**
 * Returns the UV of a direction in a probe's tile of an octahedral atlas.
 *
 * @param {Node<int>} probeIndex - The probe index in bake order.
 * @param {Node<vec3>} direction - The unit direction.
 * @param {number} resolution - The octahedral resolution of a tile.
 * @param {Node<vec2>} atlasSize - The atlas size in texels.
 * @return {Node<vec2>} The atlas UV.
 */
export function octTileUV( probeIndex, direction, resolution, atlasSize ) {

	const tile = vec2( float( probeIndex.mod( DISTANCE_COLUMNS ) ), float( probeIndex.div( DISTANCE_COLUMNS ) ) );

	return tile.mul( resolution + 2 ).add( 1.0 ).add( octEncode( direction ).mul( resolution ) ).div( atlasSize );

}

/**
 * Maps octahedral coordinates in [-1, 1]² to a unit direction.
 *
 * @private
 * @param {Node<vec2>} e - The octahedral coordinates.
 * @return {Node<vec3>} The direction.
 */
export function octDecode( e ) {

	const z = float( 1.0 ).sub( e.x.abs() ).sub( e.y.abs() );
	const folded = vec2( e.y.abs().oneMinus().mul( signNotZero( e.x ) ), e.x.abs().oneMinus().mul( signNotZero( e.y ) ) );

	return vec3( select( z.lessThan( 0.0 ), folded, e ), z ).normalize();

}

/**
 * Returns texel `textureIndex` (0–6) of a probe's packed SH atlas data. The
 * atlas packs the 27 floats of the nine RGB L2 coefficients into seven RGBA
 * texels; {@link unpackGridSH} reverses it.
 *
 * @param {function(number): Node<vec3>} c - Returns coefficient i. Only the coefficients this texel stores are requested.
 * @param {number} textureIndex - The atlas texel (0–6).
 * @return {Node<vec4>} The packed texel.
 */
export function packGridSH( c, textureIndex ) {

	switch ( textureIndex ) {

		case 0: return vec4( c( 0 ).xyz, c( 1 ).x );
		case 1: return vec4( c( 1 ).yz, c( 2 ).xy );
		case 2: return vec4( c( 2 ).z, c( 3 ).xyz );
		case 3: return vec4( c( 4 ).xyz, c( 5 ).x );
		case 4: return vec4( c( 5 ).yz, c( 6 ).xy );
		case 5: return vec4( c( 6 ).z, c( 7 ).xyz );
		default: return vec4( c( 8 ).xyz, 0.0 );

	}

}

/**
 * Unpacks the nine L2 SH coefficients from the seven atlas texels of a probe,
 * as packed by {@link packGridSH}.
 *
 * @param {Array<Node<vec4>>} s - The seven packed texels.
 * @return {Node} The coefficient array, as expected by `getShIrradianceAt()`.
 */
export function unpackGridSH( s ) {

	return array( [
		s[ 0 ].xyz,
		vec3( s[ 0 ].w, s[ 1 ].xy ),
		vec3( s[ 1 ].zw, s[ 2 ].x ),
		s[ 2 ].yzw,
		s[ 3 ].xyz,
		vec3( s[ 3 ].w, s[ 4 ].xy ),
		vec3( s[ 4 ].zw, s[ 5 ].x ),
		s[ 5 ].yzw,
		s[ 6 ].xyz
	] );

}

/**
 * Samples the packed SH atlas with hardware trilinear filtering and returns the
 * interpolated SH coefficients.
 *
 * The atlas stores the seven RGBA sub-volumes stacked along Z, each occupying
 * `( nz + 2 )` slices: one padding slice (a copy of the nearest edge slice) at
 * each end to prevent color bleeding when the hardware trilinear filter reads
 * across a sub-volume boundary.
 *
 * @param {Texture3DNode} atlas - The atlas texture node.
 * @param {Node<vec3>} uvw - The probe-grid sample coordinate (texel centers).
 * @param {Node<float>} nz - The number of probes along Z.
 * @return {Node} The coefficient array, as expected by `getShIrradianceAt()`.
 */
export function sampleGridSH( atlas, uvw, nz ) {

	const paddedSlices = nz.add( 2.0 * ATLAS_PADDING );
	const atlasDepth = paddedSlices.mul( 7.0 );
	const uvZBase = uvw.z.mul( nz ).add( ATLAS_PADDING );

	const s = [];
	for ( let t = 0; t < 7; t ++ ) s.push( atlas.sample( vec3( uvw.xy, uvZBase.add( paddedSlices.mul( t ) ).div( atlasDepth ) ) ) );

	return unpackGridSH( s );

}

/**
 * One-sided Chebyshev upper bound on the probability that a point at the given
 * distance is visible from a probe, given the mean and mean squared distance to
 * the surfaces the probe sees in that direction. Cubed to sharpen the falloff,
 * as in "Dynamic Diffuse Global Illumination with Ray-Traced Irradiance Fields"
 * (Majercik et al. 2019). Points nearer than the mean are fully visible.
 *
 * @param {Node<vec2>} moments - The mean and mean squared distance.
 * @param {Node<float>} distance - The distance to test, in the same units.
 * @param {number} [slack=VISIBILITY_SLACK] - Distance past the mean that still counts as visible.
 * @return {Node<float>} The visibility in [0, 1].
 */
export function chebyshevVisibility( moments, distance, slack = VISIBILITY_SLACK ) {

	const variance = moments.y.sub( moments.x.mul( moments.x ) ).max( 1e-6 );
	const delta = distance.sub( moments.x ).sub( slack ).max( 0.0 );

	return variance.div( variance.add( delta.mul( delta ) ) ).pow( 3.0 );

}

/**
 * Blends the 8 probes around a surface point with visibility weights, following
 * "Dynamic Diffuse Global Illumination with Ray-Traced Irradiance Fields"
 * (Majercik et al. 2019): trilinear weight, a soft backface weight, and a
 * Chebyshev test of the receiver distance against each probe's distance moments.
 * Probes that cannot see the receiver contribute nothing, so light does not leak
 * through walls between probes. If no probe sees the receiver, it fades to black.
 *
 * @private
 * @param {Object} grid - The grid parameters.
 * @param {function(Node<vec2>): Node<vec4>} sampleIrradiance - Samples the irradiance atlas.
 * @return {Node<vec3>} The non-negative irradiance.
 */
function evaluateGridIrradianceVisibility( grid, sampleIrradiance ) {

	const { min, res, spacing, moments, distanceScale, normalBias, viewBias, momentsSize, irradianceSize } = grid;

	const nx = int( res.x ), nz = int( res.z );

	const clearance = spacing.x.min( spacing.y ).min( spacing.z );
	const viewDirection = cameraPosition.sub( positionWorld ).normalize();
	const biased = positionWorld.add( normalWorld.mul( clearance.mul( normalBias ) ) ).add( viewDirection.mul( clearance.mul( viewBias ) ) ).toVar();

	const gridPosition = biased.sub( min ).div( spacing ).clamp( vec3( 0.0 ), res.sub( 1.0 ) ).toVar();
	const base = gridPosition.floor().min( res.sub( 2.0 ) ).max( 0.0 ).toVar();
	const alpha = gridPosition.sub( base ).clamp( 0.0, 1.0 ).toVar();

	const total = vec3( 0.0 ).toVar();
	const weights = float( 0.0 ).toVar();

	// Every probe is sampled in the direction of the normal.
	const normalOct = octEncode( normalWorld ).mul( IRRADIANCE_RESOLUTION ).add( 1.0 ).toVar();

	// A shader loop over the corners keeps the node graph, and so the material
	// build time, small.

	Loop( 8, ( { i: corner } ) => {

		const offset = vec3( corner.bitAnd( 1 ), corner.shiftRight( 1 ).bitAnd( 1 ), corner.shiftRight( 2 ).bitAnd( 1 ) ).toVar();
		const cell = base.add( offset ).toVar();
		const probePosition = min.add( cell.mul( spacing ) ).toVar();

		// Chebyshev visibility from the probe's distance moments toward the receiver.

		const toReceiver = biased.sub( probePosition );
		const distance = toReceiver.length().max( 1e-6 );
		const direction = toReceiver.div( distance );

		const ix = int( cell.x ).toVar(), iy = int( cell.y ).toVar(), iz = int( cell.z ).toVar();
		const probeIndex = ix.add( iz.mul( nx ) ).add( iy.mul( nx ).mul( nz ) ).toVar();
		const momentsUV = octTileUV( probeIndex, direction, DISTANCE_RESOLUTION, momentsSize );
		const visibility = chebyshevVisibility( moments.sample( momentsUV ).xy, distance.div( distanceScale ) );

		// Soft backface weight: probes behind the surface count less.

		const toProbe = probePosition.sub( positionWorld );
		const wrap = normalWorld.dot( toProbe.div( toProbe.length().max( 1e-6 ) ) ).mul( 0.5 ).add( 0.5 ).pow( 2.0 ).add( 0.2 );

		const blend = offset.mul( alpha ).add( offset.oneMinus().mul( alpha.oneMinus() ) );
		const weight = blend.x.mul( blend.y ).mul( blend.z ).mul( wrap ).mul( visibility ).toVar();

		const tile = vec2( float( probeIndex.mod( DISTANCE_COLUMNS ) ), float( probeIndex.div( DISTANCE_COLUMNS ) ) );
		const irradianceUV = tile.mul( IRRADIANCE_TILE ).add( normalOct ).div( irradianceSize );

		total.addAssign( sampleIrradiance( irradianceUV ).rgb.mul( weight ) );
		weights.addAssign( weight );

	} );

	// No unoccluded fallback: tiny total weights fade to black instead of leaking.

	return total.div( weights.max( 1e-6 ) );

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

		// Render-group uniforms are refreshed for every render, including for
		// objects whose materials are unchanged, so grid changes always apply.

		this._min = uniform( new Vector3() ).setGroup( renderGroup );
		this._max = uniform( new Vector3() ).setGroup( renderGroup );
		this._resolution = uniform( new Vector3() ).setGroup( renderGroup );
		this._intensity = uniform( 1 ).setGroup( renderGroup );
		this._falloff = uniform( 0 ).setGroup( renderGroup );
		this._useSnapshot = uniform( 0, 'int' ).setGroup( renderGroup );
		this._distanceScale = uniform( 1 ).setGroup( renderGroup );
		this._normalBias = uniform( 0 ).setGroup( renderGroup );
		this._viewBias = uniform( 0 ).setGroup( renderGroup );
		this._visibility = uniform( 0, 'int' ).setGroup( renderGroup );

		// A bake renders the grid's captures in the same frame as the main view.
		this.updateType = NodeUpdateType.RENDER;

	}

	update( /* frame */ ) {

		const light = this.light;

		this._min.value.copy( light.boundingBox.min );
		this._max.value.copy( light.boundingBox.max );
		this._resolution.value.copy( light.resolution );
		this._intensity.value = light._captureIntensity !== null ? light._captureIntensity : light.intensity;
		this._falloff.value = light.falloff;
		this._useSnapshot.value = light._captureSnapshot ? 1 : 0;
		this._distanceScale.value = light._distanceScale;
		this._normalBias.value = light.normalBias;
		this._viewBias.value = light.viewBias;
		this._visibility.value = light.visibility ? 1 : 0;

	}

	setup( builder ) {

		const light = this.light;

		// No baked data yet: contribute nothing.

		if ( light.texture === null ) return;

		const min = this._min;
		const max = this._max;
		const res = this._resolution;

		const range = max.sub( min );
		const resMinusOne = res.sub( 1.0 );
		const spacing = range.div( resMinusOne );

		// Indirect bake passes sample the snapshot of the previous pass. Both atlases
		// stay bound so switching between them never rebuilds the materials. The
		// atlases must exist as render targets before they are bound.

		builder.renderer.initRenderTarget( light._renderTarget );
		builder.renderer.initRenderTarget( light._snapshotTarget );

		// Both sampling modes are built and a uniform selects one, so toggling
		// visibility never rebuilds the materials.

		builder.renderer.initRenderTarget( light._distanceTarget );

		const distanceTarget = light._distanceTarget;
		const grid = {
			min,
			res,
			spacing,
			moments: texture( distanceTarget.texture ),
			distanceScale: this._distanceScale,
			normalBias: this._normalBias,
			viewBias: this._viewBias,
			momentsSize: vec2( distanceTarget.width, distanceTarget.height ),
			irradianceSize: vec2( light._irradianceTarget.width, light._irradianceTarget.height )
		};

		// Without visibility: offset along the normal by half a probe spacing, then
		// remap to texel centers for hardware trilinear filtering.

		const samplePos = positionWorld.add( normalWorld.mul( spacing ).mul( 0.5 ) );
		const uvw = samplePos.sub( min ).div( range ).clamp( 0.0, 1.0 ).mul( resMinusOne ).div( res ).add( vec3( 0.5 ).div( res ) );

		// Indirect bake passes read the snapshot of the previous pass. Only the atlas
		// reads branch on it, which keeps the shaders small.

		const atlas = texture3D( light.texture );
		const snapshot = texture3D( light._snapshotTarget.texture );

		builder.renderer.initRenderTarget( light._irradianceTarget );
		builder.renderer.initRenderTarget( light._irradianceSnapshotTarget );

		const irradianceAtlas = texture( light._irradianceTarget.texture );
		const irradianceSnapshot = texture( light._irradianceSnapshotTarget.texture );

		const sampleIrradiance = ( uv ) => {

			const texel = vec4().toVar();

			If( this._useSnapshot.equal( 1 ), () => {

				texel.assign( irradianceSnapshot.sample( uv ) );

			} ).Else( () => {

				texel.assign( irradianceAtlas.sample( uv ) );

			} );

			return texel;

		};

		// Optional smooth boundary for blending grids; falloff 0 applies everywhere.

		let weight = this._intensity;

		if ( light.falloff > 0 ) {

			const outside = min.sub( positionWorld ).max( 0.0 ).add( positionWorld.sub( max ).max( 0.0 ) );
			weight = weight.mul( outside.length().smoothstep( 0.0, this._falloff ).oneMinus() );

		}

		weight = weight.toVar();

		// Skip the lookup where the grid contributes nothing: past its falloff, or
		// during the direct bake pass.

		const result = vec3( 0.0 ).toVar();

		If( weight.greaterThan( 0.0 ), () => {

			If( this._visibility.equal( 1 ), () => {

				result.assign( evaluateGridIrradianceVisibility( grid, sampleIrradiance ) );

			} ).ElseIf( this._useSnapshot.equal( 1 ), () => {

				result.assign( getShIrradianceAt( normalWorld, sampleGridSH( snapshot, uvw, res.z ) ).max( vec3( 0.0 ) ) );

			} ).Else( () => {

				result.assign( getShIrradianceAt( normalWorld, sampleGridSH( atlas, uvw, res.z ) ).max( vec3( 0.0 ) ) );

			} );

		} );

		builder.context.irradiance.addAssign( result.mul( weight ) );

	}

}

export { LightProbeGridNode };
