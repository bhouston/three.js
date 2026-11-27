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
 * @param {Function} [options.projectDepth] - Converts raw depth to projected depth (log-depth inputs).
 * @param {Function} [options.acceptHit] - TSL predicate accepting (UV, parameter, raw depth).
 * @return {Object} Hit flag, UV, raw depth and parameter.
 */
export function traceLinear( depthNode, origin, end, { resolution, reversedDepth = false, quality = 1, offset = 0, projectDepth = value => value, acceptHit = () => bool( true ) } ) {

	const delta = end.sub( origin ).toConst();
	const pixels = delta.xy.mul( resolution ).abs();
	const steps = int( pixels.x.max( pixels.y ).mul( quality ) ).toConst();

	const hit = bool( false ).toVar();
	const hitUV = vec2( 0 ).toVar();
	const hitDepth = float( 0 ).toVar();
	const hitS = float( 0 ).toVar();

	Loop( steps, ( { i } ) => {

		const s = float( i ).add( offset ).div( float( steps ) ).toConst();
		const sampleUV = origin.xy.add( delta.xy.mul( s ) ).toConst();

		If( sampleUV.lessThan( vec2( 0 ) ).any().or( sampleUV.greaterThan( vec2( 1 ) ).any() ), () => {

			Break();

		} );

		const raw = depthNode.sample( sampleUV ).r.toConst();
		const depth = projectDepth( raw );
		const z = origin.z.add( delta.z.mul( s ) );

		// the ray is behind the surface

		If( ( reversedDepth ? z.lessThan( depth ) : z.greaterThan( depth ) ).and( acceptHit( sampleUV, s, raw ) ), () => {

			hit.assign( true );
			hitUV.assign( sampleUV );
			hitDepth.assign( raw );
			hitS.assign( s );
			Break();

		} );

	} );

	return { hit, uv: hitUV, depth: hitDepth, parameter: hitS };

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
	 * @param {Object} options - The resolution, the depth convention and the predicates, see `traceLinear()`.
	 * @return {Object} Hit flag, UV, raw depth and parameter.
	 */
	trace( depthNode, origin, end, options ) {

		const offset = fract( interleavedGradientNoise( screenCoordinate ).add( this._temporalOffset ) ).add( rand( uv() ) ).toConst();

		return traceLinear( depthNode, origin, end, { ...options, quality: this.quality.clamp(), offset } );

	}

	dispose() {}

}
