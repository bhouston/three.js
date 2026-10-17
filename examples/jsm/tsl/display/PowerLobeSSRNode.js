import { CubeUVReflectionMapping, HalfFloatType, Matrix4, Node, NodeMaterial, NodeUpdateType, QuadMesh, RenderTarget, RendererUtils, Vector2 } from 'three/webgpu';
import { Break, Continue, Fn, If, Loop, PI, bool, cos, cross, dot, equirectUV, float, getScreenPosition, getViewPosition, int, isBackgroundDepth, luminance, max, min, mix, nodeObject, outputStruct, passTexture, pmremTexture, sin, texture, uniform, uv, vec2, vec3, vec4 } from 'three/tsl';
import { F_Schlick, ENV_RAY_LENGTH } from '../utils/SpecularHelpers.js';

/**
 * Experimental, deliberately biased power-lobe SSR with projected stepping.
 * RGB is reflected radiance (approximate Fresnel applied once); alpha is a ray
 * distance adapter for the existing specular temporal/recurrent filters, NOT coverage.
 * See docs/SSR-alternatives.md for normalization, history and composition contracts.
 *
 * @augments Node
 * @three_import import { powerLobeSSR } from 'three/addons/tsl/display/PowerLobeSSRNode.js';
 */
class PowerLobeSSRNode extends Node {

	static get type() {

		return 'PowerLobeSSRNode';

	}

	constructor( colorNode, depthNode, normalNode, options = {} ) {

		super( 'vec4' );
		this.colorNode = colorNode;
		this.depthNode = depthNode;
		this.normalNode = normalNode;
		this.camera = options.camera ?? colorNode.passNode?.camera;
		if ( ! this.camera ) throw new Error( 'PowerLobeSSRNode requires a camera.' );
		this.metalnessNode = options.metalnessNode ?? float( 0 );
		this.roughnessNode = options.roughnessNode ?? float( 0 );
		this.diffuseNode = options.diffuseNode ?? vec3( 1 );
		this.environmentNode = options.environmentNode ?? null;
		this.velocityNode = options.velocityNode ?? null;
		this.resolutionScale = 1;
		this.steps = uniform( 16 );
		this.rayCount = uniform( 1 );
		this.seed = uniform( 0 );
		this.maxDistance = uniform( 1 );
		this.thickness = uniform( 0.1 );
		this.intensity = uniform( 1 );
		this.environmentIntensity = uniform( 1 );
		// Zero disables the optional radiance clamp.
		this.maxLuminance = uniform( 0 );
		this.updateBeforeType = NodeUpdateType.FRAME;
		this._frame = uniform( 0 );
		this._ready = uniform( false );
		this._projection = uniform( this.camera.projectionMatrix );
		this._inverseProjection = uniform( this.camera.projectionMatrixInverse );
		this._world = uniform( this.camera.matrixWorld );
		this._previousView = uniform( new Matrix4() );
		this._previousProjection = uniform( new Matrix4() );
		this._lastWorld = new Matrix4();
		this._lastProjection = new Matrix4();
		this._size = new Vector2();
		this._resolution = uniform( new Vector2() );
		this._target = new RenderTarget( 1, 1, { depthBuffer: false, type: HalfFloatType, count: 3 } );
		this._history = new RenderTarget( 1, 1, { depthBuffer: false, type: HalfFloatType, count: 2 } );
		this._target.textures.forEach( ( t, i ) => t.name = 'PowerLobeSSR.' + [ 'weightedDistance', 'incidentCoverage', 'metadata' ][ i ] );
		this._textureNode = passTexture( this, this._target.textures[ 0 ] );
		this._incidentNode = passTexture( this, this._target.textures[ 1 ] );
		this._metadataNode = passTexture( this, this._target.textures[ 2 ] );
		this._material = new NodeMaterial();
		this._copyMaterial = new NodeMaterial();
		this._quad = new QuadMesh();
		this._state = undefined;

	}

	getTextureNode() {

		return this._textureNode;

	}

	getIncidentNode() {

		return this._incidentNode;

	}

	getMetadataNode() {

		return this._metadataNode;

	}

	getRenderTarget() {

		return this._target;

	}

	resetHistory() {

		this._ready.value = false;
		this._frame.value = 0;

	}

	updateBefore( frame ) {

		const { renderer } = frame;
		for ( const input of [ this.colorNode, this.depthNode, this.velocityNode ] ) {

			if ( input?.passNode ) frame.updateBeforeNode( input.passNode );

		}

		const size = renderer.getDrawingBufferSize( this._size );
		const width = Math.max( 1, Math.round( size.x * Math.min( 1, Math.max( 0.1, this.resolutionScale ) ) ) );
		const height = Math.max( 1, Math.round( size.y * Math.min( 1, Math.max( 0.1, this.resolutionScale ) ) ) );
		if ( this._target.width !== width || this._target.height !== height || this._history.width !== size.x || this._history.height !== size.y ) {

			this._target.setSize( width, height );
			this._history.setSize( size.x, size.y );
			this.resetHistory();

		}

		this._resolution.value.set( width, height );
		// Conservative camera-cut guard; applications should also reset on scene cuts.
		const delta = this.camera.matrixWorld.elements.reduce( ( sum, v, i ) => sum + Math.abs( v - this._lastWorld.elements[ i ] ), 0 );
		if ( delta > 1 || ! this.camera.projectionMatrix.equals( this._lastProjection ) ) this.resetHistory();
		this._state = RendererUtils.resetRendererState( renderer, this._state );
		renderer.setMRT( null );
		renderer.setClearColor( 0, 0 );
		try {

			this._quad.material = this._material;
			renderer.setRenderTarget( this._target );
			this._quad.render( renderer );
			// Copy ONLY linear HDR opaque beauty + geometry after tracing has read the
			// previous frame. Never copy this SSR result, transparencies, AA or grading.
			this._quad.material = this._copyMaterial;
			renderer.setRenderTarget( this._history );
			this._quad.render( renderer );

		} finally {

			RendererUtils.restoreRendererState( renderer, this._state );

		}

		this._previousView.value.copy( this.camera.matrixWorldInverse );
		this._previousProjection.value.copy( this.camera.projectionMatrix );
		this._lastWorld.copy( this.camera.matrixWorld );
		this._lastProjection.copy( this.camera.projectionMatrix );
		this._ready.value = true;
		this._frame.value = ( this._frame.value + 1 ) % 4096;

	}

	setup( builder ) {

		if ( builder.renderer.logarithmicDepthBuffer ) throw new Error( 'PowerLobeSSRNode does not support logarithmic depth.' );
		const uvNode = uv();
		const inside = ( p ) => p.x.greaterThanEqual( 0 ).and( p.y.greaterThanEqual( 0 ) ).and( p.x.lessThanEqual( 1 ) ).and( p.y.lessThanEqual( 1 ) );
		const positionAt = ( p ) => getViewPosition( p, this.depthNode.sample( p ).r, this._inverseProjection );
		const previousColor = texture( this._history.textures[ 0 ] );
		const previousGeometry = texture( this._history.textures[ 1 ] );
		const env = ( L ) => {

			if ( this.environmentNode === null ) return vec3( 0 );
			const worldL = this._world.mul( vec4( L, 0 ) ).xyz.normalize();
			return ( this.environmentNode.mapping === CubeUVReflectionMapping ? pmremTexture( this.environmentNode, worldL, float( 0 ) ).rgb : texture( this.environmentNode, equirectUV( worldL ) ).level( 0 ).rgb ).mul( this.environmentIntensity );

		};

		this._material.fragmentNode = Fn( () => {

			const depth = this.depthNode.sample( uvNode ).r.toVar();
			isBackgroundDepth( depth ).discard();
			const P = positionAt( uvNode ).toVar();
			const N = this.normalNode.sample( uvNode ).rgb.normalize().toVar();
			const V = ( this.camera.isPerspectiveCamera ? P.negate().normalize() : vec3( 0, 0, 1 ) ).toVar();
			const r = float( this.roughnessNode ).clamp( 0, 1 ).toVar();
			const r4 = r.mul( r ).mul( r ).mul( r ).toVar();
			const helper = N.z.abs().lessThan( 0.999 ).select( vec3( 0, 0, 1 ), vec3( 0, 1, 0 ) );
			const T = cross( helper, N ).normalize().toVar();
			const B = cross( N, T ).toVar();
			const epsilon = max( P.z.abs().mul( 1e-4 ), float( 1e-5 ) ).toVar();
			const origin = P.add( N.mul( epsilon ) ).toVar();
			const incidentSum = vec3( 0 ).toVar();
			const weightedSum = vec3( 0 ).toVar();
			const validCount = float( 0 ).toVar();
			const hitCount = float( 0 ).toVar();
			const distanceSum = float( 0 ).toVar();
			const adaptedDistance = float( 0 ).toVar();
			const f0 = mix( vec3( 0.04 ), vec3( this.diffuseNode.rgb ), float( this.metalnessNode ).clamp() ).toVar();
			const pixel = uvNode.mul( this._resolution ).floor().toVar();
			Loop( { start: int( 0 ), end: int( this.rayCount.clamp( 1, 4 ) ), type: 'int', condition: '<' }, ( { i } ) => {

				// Reproducible analytic noise; independent irrational per-ray/frame offsets.
				const randomX = pixel.dot( vec2( 0.06711056, 0.00583715 ) ).add( this.seed.mul( 0.131 ) ).fract().mul( 52.9829189 ).fract();
				const ex = randomX.add( this._frame.mul( 0.61803398875 ) ).add( float( i ).mul( 0.754877666 ) ).fract().toVar();
				const ey = mix( ex.mul( 38.65435 ).fract(), float( 1 ), float( 0.7 ) );
				const cosTheta = ey.max( 1e-6 ).pow( r4.div( float( 2 ).sub( r4 ).max( 1e-6 ) ) ).toVar();
				const sinTheta = float( 1 ).sub( cosTheta.mul( cosTheta ) ).max( 0 ).sqrt();
				const phi = ex.mul( PI * 2 );
				const H = T.mul( cos( phi ).mul( sinTheta ) ).add( B.mul( sin( phi ).mul( sinTheta ) ) ).add( N.mul( cosTheta ) ).normalize().toVar();
				const L = H.mul( dot( V, H ).mul( 2 ) ).sub( V ).toVar();
				If( dot( V, H ).lessThanEqual( 0 ).or( dot( N, L ).lessThanEqual( 1e-6 ) ), () => Continue() );
				validCount.addAssign( 1 );
				const radiance = env( L ).toVar();
				const rayDistance = float( ENV_RAY_LENGTH ).toVar();
				const reach = this.maxDistance.max( 0 ).toVar();
				// Clip against near plane before projection, for both camera types.
				If( L.z.greaterThan( 1e-6 ), () => reach.assign( min( reach, float( - this.camera.near ).sub( epsilon ).sub( origin.z ).div( L.z ).max( 0 ) ) ) );
				const end = origin.add( L.mul( reach ) ).toVar();
				const c0 = this._projection.mul( vec4( origin, 1 ) ).toVar();
				const c1 = this._projection.mul( vec4( end, 1 ) ).toVar();
				const k0 = float( 1 ).div( c0.w ).toVar();
				const k1 = float( 1 ).div( c1.w ).toVar();
				const q0 = origin.mul( k0 ).toVar();
				const q1 = end.mul( k1 ).toVar();
				const u0 = getScreenPosition( origin, this._projection ).toVar();
				const u1 = getScreenPosition( end, this._projection ).toVar();
				const du = u1.sub( u0 ).toVar();
				const limit = float( 1 ).toVar();
				// Clip the projected segment to all four viewport boundaries.
				for ( const axis of [ 'x', 'y' ] ) {

					If( du[ axis ].greaterThan( 1e-6 ), () => limit.assign( min( limit, u0[ axis ].oneMinus().div( du[ axis ] ) ) ) );
					If( du[ axis ].lessThan( - 1e-6 ), () => limit.assign( min( limit, u0[ axis ].negate().div( du[ axis ] ) ) ) );

				}

				const rayAt = ( s ) => mix( q0, q1, s ).div( mix( k0, k1, s ) );
				const previousS = float( 0 ).toVar();
				// D = sceneZ - rayZ: negative in front, positive behind the surface.
				const previousD = P.z.sub( origin.z ).toVar();
				const found = bool( false ).toVar();
				const hitUV = vec2( 0 ).toVar();
				const hitP = vec3( 0 ).toVar();
				Loop( { start: int( 1 ), end: int( this.steps.clamp( 1, 64 ) ), type: 'int', condition: '<=' }, ( { i: j } ) => {

					const s = float( j ).div( this.steps.clamp( 1, 64 ).floor() ).mul( limit.clamp() ).toVar();
					const p = u0.add( du.mul( s ) ).toVar();
					If( inside( p ).not(), () => Break() );
					const d = this.depthNode.sample( p ).r.toVar();
					If( isBackgroundDepth( d ), () => {

						previousD.assign( - 1e6 ); previousS.assign( s ); Continue();

					} );
					const sceneP = getViewPosition( p, d, this._inverseProjection ).toVar();
					const D = sceneP.z.sub( rayAt( s ).z ).toVar();
					If( previousD.lessThan( 0 ).and( D.greaterThanEqual( 0 ) ).and( D.lessThanEqual( this.thickness.max( epsilon ) ) ), () => {

						const denominator = previousD.sub( D ).toVar();
						const t = float( 1 ).toVar();
						If( denominator.abs().greaterThan( 1e-6 ), () => t.assign( previousD.div( denominator ).clamp() ) );
						const refined = u0.add( du.mul( mix( previousS, s, t ) ) ).toVar();
						const refinedP = positionAt( refined ).toVar();
						const refinedD = refinedP.z.sub( rayAt( mix( previousS, s, t ) ).z );
						// Depth discontinuities can invalidate secant refinement; keep the slab hit.
						const useRefined = refinedD.abs().lessThanEqual( this.thickness.max( epsilon ) ).and( refinedP.sub( sceneP ).length().lessThanEqual( this.thickness.max( epsilon ) ) );
						hitUV.assign( useRefined.select( refined, p ) );
						hitP.assign( useRefined.select( refinedP, sceneP ) );
						found.assign( true );
						Break();

					} );
					previousD.assign( D );
					previousS.assign( s );

				} );
				If( found.and( this._ready ).and( reach.greaterThan( epsilon ) ), () => {

					const worldHit = this._world.mul( vec4( hitP, 1 ) ).toVar();
					const previousViewHit = this._previousView.mul( worldHit ).xyz.toVar();
					const historyUV = getScreenPosition( previousViewHit, this._previousProjection ).toVar();
					if ( this.velocityNode !== null ) historyUV.assign( hitUV.sub( this.velocityNode.sample( hitUV ).xy.mul( vec2( 0.5, - 0.5 ) ) ) );
					If( inside( historyUV ).and( previousViewHit.z.lessThan( - this.camera.near ) ), () => {

						const oldGeometry = previousGeometry.sample( historyUV ).toVar();
						// A moving surface with changed view depth conservatively falls back to env.
						const tolerance = max( float( 1e-3 ), previousViewHit.z.abs().mul( 0.002 ) );
						If( oldGeometry.a.greaterThan( 0.5 ).and( oldGeometry.z.sub( previousViewHit.z ).abs().lessThan( tolerance ) ), () => {

							radiance.assign( previousColor.sample( historyUV ).rgb );
							rayDistance.assign( hitP.sub( P ).length() );
							hitCount.addAssign( 1 );
							distanceSum.addAssign( rayDistance );

						} );

					} );

				} );
				If( this.maxLuminance.greaterThan( 0 ), () => radiance.mulAssign( this.maxLuminance.div( luminance( radiance ).max( 1e-6 ) ).min( 1 ) ) );
				incidentSum.addAssign( radiance );
				// Deliberately approximate response: Schlick only, no BRDF/PDF correction.
				weightedSum.addAssign( radiance.mul( F_Schlick( f0, dot( V, H ).clamp() ) ) );
				adaptedDistance.addAssign( rayDistance );

			} );
			const divisor = validCount.max( 1 );
			return outputStruct(
				vec4( weightedSum.div( divisor ).mul( this.intensity ), adaptedDistance.div( divisor ) ),
				vec4( incidentSum.div( divisor ), hitCount.div( divisor ) ),
				vec4( hitCount.div( divisor ), distanceSum.div( hitCount.max( 1 ) ), r, validCount )
			);

		} )().context( builder.getSharedContext() );
		this._copyMaterial.fragmentNode = Fn( () => {

			const d = this.depthNode.sample( uvNode ).r;
			return outputStruct( vec4( this.colorNode.sample( uvNode ).rgb, 1 ), vec4( positionAt( uvNode ), isBackgroundDepth( d ).not() ) );

		} )().context( builder.getSharedContext() );
		return this._textureNode;

	}

	dispose() {

		super.dispose();
		this._target.dispose();
		this._history.dispose();
		this._material.dispose();
		this._copyMaterial.dispose();

	}

}

export default PowerLobeSSRNode;
export const powerLobeSSR = ( colorNode, depthNode, normalNode, options = {} ) => new PowerLobeSSRNode( nodeObject( colorNode ), nodeObject( depthNode ), nodeObject( normalNode ), options );
