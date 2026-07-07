import NodeMaterial from './nodes/NodeMaterial.js';

import {
	Discard,
	Fn,
	If,
	atan,
	cameraProjectionMatrix,
	cos,
	dot,
	exp,
	float,
	highpModelViewMatrix,
	instanceIndex,
	max,
	min,
	positionGeometry,
	screenSize,
	sin,
	sqrt,
	varyingProperty,
	vec2,
	vec3,
	vec4
} from '../nodes/TSL.js';

const KERNEL_2D_SIZE = 0.3;
const MAX_SCREEN_SPACE_SPLAT_SIZE = 1024;
const CLIP_XY = 1.4;

/**
 * A material for rendering Gaussian splat objects.
 *
 * @augments NodeMaterial
 */
class GaussianSplatMaterial extends NodeMaterial {

	static get type() {

		return 'GaussianSplatMaterial';

	}

	/**
	 * Constructs a new Gaussian splat material.
	 *
	 * @param {Object} [parameters] - The configuration parameters.
	 * @param {Object} [parameters.buffers] - Renderer data buffers used by the splat shader.
	 */
	constructor( parameters = {} ) {

		super();

		/**
		 * This flag can be used for type testing.
		 *
		 * @type {boolean}
		 * @readonly
		 * @default true
		 */
		this.isGaussianSplatMaterial = true;

		/**
		 * Whether splats should be sorted automatically.
		 *
		 * @type {boolean}
		 * @default true
		 */
		this.autoSort = true;

		/**
		 * Maximum screen-space splat radius, in pixels.
		 *
		 * @type {number}
		 * @default 1024
		 */
		this.maxScreenSpaceSplatSize = MAX_SCREEN_SPACE_SPLAT_SIZE;

		/**
		 * Size of the 2D kernel added during projection.
		 *
		 * @type {number}
		 * @default 0.3
		 */
		this.kernel2DSize = KERNEL_2D_SIZE;

		this.transparent = true;
		this.depthWrite = false;
		this.depthTest = true;
		this.forceSinglePass = true;
		this.fog = false;

		if ( parameters.buffers !== undefined ) {

			this.setBuffers( parameters.buffers );

		}

		const values = { ...parameters };
		delete values.buffers;
		this.setValues( values );

	}

	/**
	 * Assigns the renderer buffers used by the generated node graph.
	 *
	 * @param {Object} buffers - The splat storage buffers.
	 * @return {GaussianSplatMaterial} A reference to this material.
	 */
	setBuffers( buffers ) {

		this._buffers = buffers;

		const splatUv = varyingProperty( 'vec2', 'vSplatUv' );
		const splatColor = varyingProperty( 'vec4', 'vSplatColor' );

		this.vertexNode = Fn( () => {

			const splatIndex = buffers.orderRead.element( instanceIndex ).toVar( 'splatIndex' );
			const center = buffers.centerRead.element( splatIndex ).xyz.toVar( 'center' );
			const covA = buffers.covarianceARead.element( splatIndex ).toVar( 'covA' );
			const covB = buffers.covarianceBRead.element( splatIndex ).toVar( 'covB' );
			const color = buffers.colorRead.element( splatIndex ).toVar( 'splatColor' );

			splatUv.assign( positionGeometry.xy );

			const viewCenter4 = highpModelViewMatrix.mul( vec4( center, 1 ) ).toVar( 'viewCenter4' );
			const viewCenter = viewCenter4.xyz.toVar( 'viewCenter' );
			const centerClip = cameraProjectionMatrix.mul( viewCenter4 ).toVar( 'centerClip' );

			const m = highpModelViewMatrix;
			const r0 = vec3( m[ 0 ].x, m[ 1 ].x, m[ 2 ].x ).toVar( 'r0' );
			const r1 = vec3( m[ 0 ].y, m[ 1 ].y, m[ 2 ].y ).toVar( 'r1' );
			const r2 = vec3( m[ 0 ].z, m[ 1 ].z, m[ 2 ].z ).toVar( 'r2' );

			const cov0 = vec3( covA.x, covA.y, covA.z ).toVar( 'cov0' );
			const cov1 = vec3( covA.y, covA.w, covB.x ).toVar( 'cov1' );
			const cov2 = vec3( covA.z, covB.x, covB.y ).toVar( 'cov2' );

			const vc0 = vec3( dot( r0, cov0 ), dot( r0, cov1 ), dot( r0, cov2 ) ).toVar( 'vc0' );
			const vc1 = vec3( dot( r1, cov0 ), dot( r1, cov1 ), dot( r1, cov2 ) ).toVar( 'vc1' );
			const vc2 = vec3( dot( r2, cov0 ), dot( r2, cov1 ), dot( r2, cov2 ) ).toVar( 'vc2' );

			const c00 = dot( vc0, r0 ).toVar( 'c00' );
			const c01 = dot( vc0, r1 ).toVar( 'c01' );
			const c02 = dot( vc0, r2 ).toVar( 'c02' );
			const c11 = dot( vc1, r1 ).toVar( 'c11' );
			const c12 = dot( vc1, r2 ).toVar( 'c12' );
			const c22 = dot( vc2, r2 ).toVar( 'c22' );

			const z = min( viewCenter.z, - 0.01 ).toVar( 'z' );
			const invZ = float( 1 ).div( z ).toVar( 'invZ' );
			const invZ2 = invZ.mul( invZ ).toVar( 'invZ2' );
			const focal = screenSize.mul( 0.5 ).mul( vec2( cameraProjectionMatrix[ 0 ].x, cameraProjectionMatrix[ 1 ].y ) ).toVar( 'focal' );

			const j00 = focal.x.negate().mul( invZ ).toVar( 'j00' );
			const j11 = focal.y.negate().mul( invZ ).toVar( 'j11' );
			const j02 = focal.x.mul( viewCenter.x ).mul( invZ2 ).toVar( 'j02' );
			const j12 = focal.y.mul( viewCenter.y ).mul( invZ2 ).toVar( 'j12' );

			const aBase = j00.mul( j00 ).mul( c00 )
				.add( j00.mul( j02 ).mul( c02 ).mul( 2 ) )
				.add( j02.mul( j02 ).mul( c22 ) )
				.toVar( 'cov2dABase' );
			const b = j00.mul( j11 ).mul( c01 )
				.add( j00.mul( j12 ).mul( c02 ) )
				.add( j02.mul( j11 ).mul( c12 ) )
				.add( j02.mul( j12 ).mul( c22 ) )
				.toVar( 'cov2dB' );
			const cBase = j11.mul( j11 ).mul( c11 )
				.add( j11.mul( j12 ).mul( c12 ).mul( 2 ) )
				.add( j12.mul( j12 ).mul( c22 ) )
				.toVar( 'cov2dCBase' );
			const a = aBase.add( this.kernel2DSize ).toVar( 'cov2dA' );
			const c = cBase.add( this.kernel2DSize ).toVar( 'cov2dC' );
			const detBase = aBase.mul( cBase ).sub( b.mul( b ) ).toVar( 'detBase' );
			const det = a.mul( c ).sub( b.mul( b ) ).toVar( 'det' );
			const alphaScale = sqrt( max( detBase.div( max( det, 0.000001 ) ), 0 ) ).toVar( 'alphaScale' );

			splatColor.assign( vec4( color.rgb, color.a.mul( alphaScale ) ) );

			const halfTrace = a.add( c ).mul( 0.5 ).toVar( 'halfTrace' );
			const radius = sqrt( max( a.sub( c ).mul( 0.5 ).pow2().add( b.mul( b ) ), 0.0000001 ) ).toVar( 'radius' );
			const lambda1 = max( halfTrace.add( radius ), 0.0000001 ).toVar( 'lambda1' );
			const lambda2 = max( halfTrace.sub( radius ), 0.0000001 ).toVar( 'lambda2' );
			const axis1 = vec2( 1, 0 ).toVar( 'axis1' );

			If( radius.greaterThan( 0.00001 ), () => {

				const angle = atan( b.mul( 2 ), a.sub( c ) ).mul( 0.5 ).toVar( 'angle' );
				axis1.assign( vec2( cos( angle ), sin( angle ) ) );

			} );

			const axis2 = vec2( axis1.y.negate(), axis1.x ).toVar( 'axis2' );

			const scale1 = min( sqrt( lambda1 ), this.maxScreenSpaceSplatSize ).toVar( 'scale1' );
			const scale2 = min( sqrt( lambda2 ), this.maxScreenSpaceSplatSize ).toVar( 'scale2' );
			const offsetPixels = axis1.mul( positionGeometry.x ).mul( scale1 ).add( axis2.mul( positionGeometry.y ).mul( scale2 ) ).toVar( 'offsetPixels' );
			const offsetNdc = offsetPixels.mul( 2 ).div( screenSize ).toVar( 'offsetNdc' );
			const clip = centerClip.add( vec4( offsetNdc.mul( centerClip.w ), 0, 0 ) ).toVar( 'clip' );

			const clipLimit = centerClip.w.mul( CLIP_XY ).toVar( 'clipLimit' );

			If( viewCenter.z.greaterThanEqual( - 0.01 )
				.or( centerClip.z.lessThan( centerClip.w.negate() ) )
				.or( centerClip.z.greaterThan( centerClip.w ) )
				.or( centerClip.x.lessThan( clipLimit.negate() ) )
				.or( centerClip.x.greaterThan( clipLimit ) )
				.or( centerClip.y.lessThan( clipLimit.negate() ) )
				.or( centerClip.y.greaterThan( clipLimit ) ), () => {

				clip.assign( vec4( 2, 2, 2, 1 ) );

			} );

			return clip;

		} )();

		this.colorNode = Fn( () => {

			const r2 = dot( splatUv, splatUv ).toVar( 'r2' );

			If( r2.greaterThan( 4 ), () => {

				Discard();

			} );

			return vec4( splatColor.rgb, exp( r2.mul( - 0.5 ) ).mul( splatColor.a ) );

		} )();

		return this;

	}

	copy( source ) {

		super.copy( source );

		this.autoSort = source.autoSort;
		this.maxScreenSpaceSplatSize = source.maxScreenSpaceSplatSize;
		this.kernel2DSize = source.kernel2DSize;

		if ( source._buffers !== undefined ) this.setBuffers( source._buffers );

		return this;

	}

}

export { GaussianSplatMaterial };
