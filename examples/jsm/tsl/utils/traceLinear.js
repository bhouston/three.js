import { Break, If, Loop, bool, float, fract, int, interleavedGradientNoise, rand, screenCoordinate, uniform, uv, vec2 } from 'three/tsl';

const _temporalOffsets = [ 0, 0.5, 0.25, 0.75 ];

/**
 * Marches a screen-space ray through the depth one sample at a time, inside the calling TSL function.
 * The ray and the result follow `traceHiZ()`, the cost grows with the length of the ray in pixels.
 *
 * @param {TextureNode} depthNode - The depth of the scene being traced.
 * @param {Node<vec3>} origin - Start UV and projected depth.
 * @param {Node<vec3>} end - End UV and projected depth.
 * @param {Object} options - Traversal configuration.
 * @param {Node<vec2>} options.resolution - The resolution the steps are counted in.
 * @param {boolean} [options.reversedDepth=false] - Projected depth convention.
 * @param {Node<float>|number} [options.quality=1] - The steps per pixel of the ray.
 * @param {Node<float>|number} [options.offset=0] - The offset of the samples, in steps.
 * @param {Node<int>} [options.steps] - The number of samples, instead of the one given by `quality`.
 * @param {Function} [options.parameter] - Returns the ray parameter of a sample index, instead of even steps.
 * @param {Function} [options.projectDepth] - Converts raw depth to projected depth (log-depth inputs).
 * @param {Function} [options.acceptHit] - TSL predicate accepting (UV, parameter, raw depth, previous parameter).
 * @return {Object} Hit flag, UV, raw depth and parameter, and the parameter of the sample before the hit and
 * whether the ray was in front of the depth there, to refine the hit.
 */
export function traceLinear( depthNode, origin, end, { resolution, reversedDepth = false, quality = 1, offset = 0, steps = null, parameter = null, projectDepth = value => value, acceptHit = () => bool( true ) } ) {

	const delta = end.sub( origin ).toConst();

	if ( steps === null ) {

		const pixels = delta.xy.mul( resolution ).abs();

		steps = int( pixels.x.max( pixels.y ).mul( quality ) ).toConst();

	}

	if ( parameter === null ) parameter = ( index ) => index.add( offset ).div( float( steps ) );

	const hit = bool( false ).toVar();
	const hitUV = vec2( 0 ).toVar();
	const hitDepth = float( 0 ).toVar();
	const hitS = float( 0 ).toVar();
	const hitPreviousS = float( 0 ).toVar();
	const hitPreviousInFront = bool( false ).toVar();

	const previousS = float( 0 ).toVar();
	const previousInFront = bool( true ).toVar();

	Loop( steps, ( { i } ) => {

		const s = parameter( float( i ) ).toConst();
		const sampleUV = origin.xy.add( delta.xy.mul( s ) ).toConst();

		If( s.greaterThan( 1 ).or( sampleUV.lessThan( vec2( 0 ) ).any() ).or( sampleUV.greaterThan( vec2( 1 ) ).any() ), () => {

			Break();

		} );

		const raw = depthNode.sample( sampleUV ).r.toConst();
		const depth = projectDepth( raw );
		const z = origin.z.add( delta.z.mul( s ) );
		const inFront = ( reversedDepth ? z.greaterThan( depth ) : z.lessThan( depth ) ).toConst();

		const lastS = previousS.toConst();
		const lastInFront = previousInFront.toConst();

		previousS.assign( s );
		previousInFront.assign( inFront );

		If( inFront.not().and( acceptHit( sampleUV, s, raw, lastS ) ), () => {

			hit.assign( true );
			hitUV.assign( sampleUV );
			hitDepth.assign( raw );
			hitS.assign( s );
			hitPreviousS.assign( lastS );
			hitPreviousInFront.assign( lastInFront );
			Break();

		} );

	} );

	return { hit, uv: hitUV, depth: hitDepth, parameter: hitS, previousParameter: hitPreviousS, previousInFront: hitPreviousInFront };

}

/**
 * Traces screen-space rays by marching the depth linearly, see `traceLinear()`.
 *
 * @three_import import { LinearTracer } from 'three/addons/tsl/utils/traceLinear.js';
 */
export class LinearTracer {

	constructor() {

		/**
		 * The steps per pixel of a ray, in the range `[0,1]`.
		 *
		 * @type {UniformNode<float>}
		 * @default 0.5
		 */
		this.quality = uniform( 0.5 );

		/**
		 * Whether the samples are offset each frame, which requires a temporal filter like `TRAANode`.
		 *
		 * @type {boolean}
		 * @default false
		 */
		this.useTemporalFiltering = false;

		this._temporalOffset = uniform( 0 );

	}

	/**
	 * Updates the tracer, called once per frame before tracing.
	 *
	 * @param {NodeFrame} frame - The current node frame.
	 */
	update( frame ) {

		this._temporalOffset.value = this.useTemporalFiltering === true ? _temporalOffsets[ frame.frameId % 4 ] : 0;

	}

	/**
	 * Traces a ray inside the calling TSL function.
	 *
	 * @param {TextureNode} depthNode - The depth of the scene.
	 * @param {Node<vec3>} origin - Start UV and projected depth.
	 * @param {Node<vec3>} end - End UV and projected depth.
	 * @param {Object} options - The resolution, the depth convention and the predicates, see `traceLinear()`. `steps` and
	 * `parameter` replace the samples given by `quality`.
	 * @return {Object} Hit flag, UV, raw depth and parameter.
	 */
	trace( depthNode, origin, end, options ) {

		const offset = fract( interleavedGradientNoise( screenCoordinate ).add( this._temporalOffset ) ).add( rand( uv() ) ).toConst();

		return traceLinear( depthNode, origin, end, { quality: this.quality.clamp(), offset, ...options } );

	}

	dispose() {}

}
