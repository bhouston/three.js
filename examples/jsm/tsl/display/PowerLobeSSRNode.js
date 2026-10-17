import { HalfFloatType, Matrix4, Node, NodeMaterial, NodeUpdateType, QuadMesh, RenderTarget, RendererUtils, Vector2 } from 'three/webgpu';
import { Break, Continue, Fn, If, Loop, PI, bool, context, cos, cross, dot, equirectUV, float, getScreenPosition, getViewPosition, int, isBackgroundDepth, luminance, max, min, mix, nodeObject, outputStruct, passTexture, pmremTexture, property, reference, sin, texture, uniform, uv, vec2, vec3, vec4 } from 'three/tsl';
import { F_Schlick, ENV_RAY_LENGTH } from '../utils/SpecularHelpers.js';

/**
 * @typedef {Object} PowerLobeSSRNodeOptions
 * @property {Camera} [camera] - Camera; inferred from the opaque color pass if omitted.
 * @property {Node<float>} [metalnessNode] - Metalness, default zero.
 * @property {Node<float>} [roughnessNode] - Perceptual roughness, clamped to [0,1]. Default zero is a stable mirror limit.
 * @property {Node<vec3>} [diffuseNode] - Unattenuated base color for metal Fresnel tint; default white.
 * @property {Texture} [environmentNode] - Linear HDR equirectangular, cube or PMREM environment; missing environment means black fallback.
 * @property {Node<vec2>} [velocityNode] - Hit-surface unjittered NDC motion (current minus previous). Omit for static-object camera reprojection.
 */

/**
 * Experimental, deliberately biased power-lobe SSR with projected stepping.
 * RGB is reflected radiance (approximate Fresnel applied once); alpha is a ray
 * distance adapter for the existing specular temporal/recurrent filters, NOT coverage.
 *
 * The sampler restricts Ey to [0.7,1] and uses a roughness-dependent power lobe.
 * It is NOT GGX VNDF or an unbiased Monte Carlo estimator. There is no BRDF/PDF
 * throughput correction. Schlick Fresnel, with dielectric f0=0.04 and metal base
 * color tint, is the sole approximate receiver response and is applied once to
 * each incident sample, including environment misses. No extra BRDF should be
 * applied to this node's default RGB. This omits masking/shadowing, diffuse energy
 * compensation and multilayer materials; lobe shape and energy are biased.
 *
 * One to four directions are launched per pixel; below-surface directions are
 * rejected, without replacement. Sums divide by the launched VALID direction
 * count, including misses with environment radiance. No valid directions gives
 * zero. Coverage is accepted previous-color hits / valid directions, not opacity.
 * It must not multiply the resolved RGB again. A constant incident environment
 * stays constant across ray counts, before Fresnel and optional luminance clamp.
 * The clamp defaults OFF (maxLuminance=0); enabling it changes energy.
 *
 * getIncidentNode(): RGB = mean incident radiance after optional clamp, A = coverage.
 * getMetadataNode(): R = coverage, G = conditional mean accepted-hit distance in
 * view/world units, B = roughness, A = valid launched count. Missing hits have G=0.
 * getHitNode(): first geometric hit UV.xy, receiver-to-hit distance.z, validity.w.
 * A geometric hit can fail history validation; it then contributes environment,
 * zero accepted coverage, but remains visible in getHitNode(). No hit is (-1,-1,10000,0).
 * Default output: weighted RGB and conditional mean accepted-hit distance alpha;
 * all misses use ENV_RAY_LENGTH=10000. For multiple rays this single distance is
 * only a representative adapter to the existing temporal/recurrent specular path,
 * not an exact multilobe history model. That path outputs inverse history age in
 * alpha; neither distance alpha nor inverse age is coverage.
 *
 * colorNode MUST be opaque linear HDR scene radiance, before transparency/glass,
 * AA, sharpening, exposure, tone mapping or grading. The node copies it and view
 * geometry AFTER tracing, at full drawing-buffer resolution. It never feeds back
 * its own output or denoised reflections. First/reset frames use environment.
 * Hits use HIT-surface velocity (NDC to UV = (0.5,-0.5)), or camera reprojection
 * when velocity is omitted. Full-viewport TRAA jitter is accounted for separately.
 * Offscreen, background and inconsistent previous view-depth hits fall back to
 * environment. Motion along view depth is conservatively rejected because NDC
 * velocity has no previous-depth component. Camera-only fallback assumes static
 * objects. The history depth tolerance is max(0.001, abs(previousViewZ)*0.002).
 * Lighting changes appear one frame later by construction. Exposure remains a
 * downstream display operation, so exposure changes do not rescale this history.
 *
 * Reach is a finite distance from a normal-offset origin, in view/world units.
 * Epsilon = max(abs(viewZ)*1e-4,1e-5). Endpoints clip to the near plane and viewport;
 * steps interpolate Q=viewPosition/clipW and k=1/clipW, then reconstruct Q/k.
 * Signed D=sceneViewZ-rayViewZ is negative in front and nonnegative behind. The
 * first negative-to-nonnegative crossing in the slab [0,max(thickness,epsilon)]
 * is accepted, with safeguarded signed-depth interpolation. Discontinuities can
 * reject refinement and retain the coarse slab hit. This is not Hi-Z traversal.
 * Single-depth screen-space geometry misses thin/hidden/offscreen surfaces, and
 * oblique slabs can produce false hits. Perspective/orthographic and reversed
 * depth use the backend projection helpers; logarithmic depth throws explicitly.
 * The examples' existing recurrent specular filter is restricted to perspective.
 *
 * resetHistory() restarts the deterministic frame sequence with the same seed.
 * Resize, large camera transforms and lens changes invalidate radiance history.
 * Applications must also reset their temporal/recurrent filters on scene cuts.
 * Small camera cuts and topology/material changes require an explicit reset.
 *
 * The examples add this weighted reflection to opaque/transparent beauty after
 * disabling the beauty's indirect environment specular in explicit comparison
 * modes. Transparent beauty remains visible but is excluded from reflection
 * history; reflections are still composed after transparency, an approximation
 * that does not model glass attenuation or layered transmission. Existing mirror
 * and GGX tracers retain their own composition, reach and fallback conventions.
 * Explicit comparison URLs opt into SSRNode.maxRayLength for a shared absolute
 * reach cap, while retaining the stock samplers, slab test and BRDF weights.
 *
 * @augments Node
 * @three_import import { powerLobeSSR } from 'three/addons/tsl/display/PowerLobeSSRNode.js';
 */
class PowerLobeSSRNode extends Node {

	static get type() {

		return 'PowerLobeSSRNode';

	}

	/**
	 * @param {Node<vec4>} colorNode - Opaque linear HDR beauty; never a filtered final frame.
	 * @param {Node<float>} depthNode - Matching opaque depth.
	 * @param {Node<vec3>} normalNode - Matching normalized view normals.
	 * @param {PowerLobeSSRNodeOptions} [options] - Material, motion and environment inputs.
	 */
	constructor( colorNode, depthNode, normalNode, options = {} ) {

		super( 'vec4' );
		this.colorNode = colorNode;
		this.depthNode = depthNode;
		this.normalNode = normalNode;
		this.camera = options.camera ?? colorNode.passNode?.camera;
		if ( ! this.camera ) throw new Error( 'PowerLobeSSRNode requires a camera.' );
		if ( colorNode.passNode?.transparent === true ) throw new Error( 'PowerLobeSSRNode requires an opaque color pass (transparent=false).' );
		this.metalnessNode = options.metalnessNode ?? float( 0 );
		this.roughnessNode = options.roughnessNode ?? float( 0 );
		this.diffuseNode = options.diffuseNode ?? vec3( 1 );
		this.environmentNode = options.environmentNode ?? null;
		this.velocityNode = options.velocityNode ?? null;
		this.resolutionScale = 1;
		/** Projected depth samples per ray, clamped to integer [1,64]. */
		this.steps = uniform( 16 );
		/** Launched directions per pixel, clamped to integer [1,4]. */
		this.rayCount = uniform( 1 );
		/** Reproducible comparison seed. Call resetHistory() after changing it. */
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
		this._near = reference( 'near', 'float', this.camera );
		this._previousView = uniform( new Matrix4() );
		this._previousProjection = uniform( new Matrix4() );
		this._lastWorld = new Matrix4();
		this._lastProjection = new Matrix4();
		this._lastJitter = new Vector2();
		this._jitterDelta = uniform( new Vector2() );
		this._size = new Vector2();
		this._resolution = uniform( new Vector2() );
		this._target = new RenderTarget( 1, 1, { depthBuffer: false, type: HalfFloatType, count: 4 } );
		this._history = new RenderTarget( 1, 1, { depthBuffer: false, type: HalfFloatType, count: 2 } );
		this._target.textures.forEach( ( t, i ) => t.name = 'PowerLobeSSR.' + [ 'weightedDistance', 'incidentCoverage', 'metadata', 'hit' ][ i ] );
		this._textureNode = passTexture( this, this._target.textures[ 0 ] );
		this._incidentNode = passTexture( this, this._target.textures[ 1 ] );
		this._metadataNode = passTexture( this, this._target.textures[ 2 ] );
		this._hitNode = passTexture( this, this._target.textures[ 3 ] );
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

	/** First geometric hit: UV.xy, surface distance.z, validity.w; history can still reject it. */
	getHitNode() {

		return this._hitNode;

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
		// Ignore subpixel projection shifts from TRAA. Other lens changes reset history.
		const view = this.camera.view;
		const fullViewportJitter = view?.enabled && view.fullWidth === view.width && view.fullHeight === view.height;
		const jitterElements = this.camera.isPerspectiveCamera ? [ 8, 9 ] : [ 12, 13 ];
		const projectionDelta = this.camera.projectionMatrix.elements.reduce( ( sum, v, i ) => sum + ( fullViewportJitter && jitterElements.includes( i ) ? 0 : Math.abs( v - this._lastProjection.elements[ i ] ) ), 0 );
		if ( delta > 1 || projectionDelta > 0.02 ) this.resetHistory();
		const jitter = new Vector2();
		if ( fullViewportJitter ) jitter.set( - view.offsetX / view.fullWidth, - view.offsetY / view.fullHeight );
		this._jitterDelta.value.copy( this._lastJitter ).sub( jitter );
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
		this._lastJitter.copy( jitter );
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
			return ( this.environmentNode.isPMREMTexture || this.environmentNode.isCubeTexture ? pmremTexture( this.environmentNode, worldL, float( 0 ) ).rgb : texture( this.environmentNode, equirectUV( worldL ) ).level( 0 ).rgb ).mul( this.environmentIntensity );

		};

		const weightedOutput = property( 'vec4' );
		const incidentOutput = property( 'vec4' );
		const metadataOutput = property( 'vec4' );
		const hitOutput = property( 'vec4' );
		this._material.contextNode = context( builder.getSharedContext() );
		this._material.outputNode = outputStruct( weightedOutput, incidentOutput, metadataOutput, hitOutput );
		this._material.colorNode = Fn( () => {

			const depth = this.depthNode.sample( uvNode ).r.toVar();
			// Auxiliary MRT attachments clear to alpha=1 by default. Write all
			// background contracts explicitly rather than leaving a false valid sample.
			weightedOutput.assign( vec4( 0 ) );
			incidentOutput.assign( vec4( 0 ) );
			metadataOutput.assign( vec4( 0 ) );
			hitOutput.assign( vec4( - 1, - 1, ENV_RAY_LENGTH, 0 ) );
			If( isBackgroundDepth( depth ).not(), () => {

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
				const representativeHit = vec4( - 1, - 1, ENV_RAY_LENGTH, 0 ).toVar();
				const f0 = mix( vec3( 0.04 ), vec3( this.diffuseNode.rgb ), float( this.metalnessNode ).clamp() ).toVar();
				const pixel = uvNode.mul( this._resolution ).floor().toVar();
				Loop( { start: int( 0 ), end: int( this.rayCount.clamp( 1, 4 ) ), type: 'int', condition: '<' }, ( { i } ) => {

					// Reproducible analytic noise; independent irrational per-ray/frame offsets.
					const randomX = pixel.dot( vec2( 0.06711056, 0.00583715 ) ).add( this.seed.mul( 0.131 ) ).fract().mul( 52.9829189 ).fract();
					const ex = randomX.add( this._frame.mul( 0.61803398875 ) ).add( float( i ).mul( 0.754877666 ) ).fract().toVar();
					const ey = mix( ex.mul( 38.65435 ).fract(), float( 1 ), float( 0.7 ) );
					const cosTheta = ey.max( 1e-6 ).pow( r4.div( float( 2 ).sub( r4 ).max( 1e-6 ) ) ).toVar();
					const sinTheta = float( 1 ).sub( cosTheta.mul( cosTheta ) ).max( 0 ).sqrt();
					const phi = ex.mul( PI ).mul( 2 );
					const H = T.mul( cos( phi ).mul( sinTheta ) ).add( B.mul( sin( phi ).mul( sinTheta ) ) ).add( N.mul( cosTheta ) ).normalize().toVar();
					const L = H.mul( dot( V, H ).mul( 2 ) ).sub( V ).toVar();
					If( dot( V, H ).lessThanEqual( 0 ).or( dot( N, L ).lessThanEqual( 1e-6 ) ), () => {

						Continue();

					} );
					validCount.addAssign( 1 );
					const radiance = env( L ).toVar();
					const rayDistance = float( ENV_RAY_LENGTH ).toVar();
					const reach = this.maxDistance.max( 0 ).toVar();
					// Clip against near plane before projection, for both camera types.
					If( L.z.greaterThan( 1e-6 ), () => {

						reach.assign( min( reach, this._near.negate().sub( epsilon ).sub( origin.z ).div( L.z ).max( 0 ) ) );

					} );
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

						If( du[ axis ].greaterThan( 1e-6 ), () => {

							limit.assign( min( limit, u0[ axis ].oneMinus().div( du[ axis ] ) ) );

						} );
						If( du[ axis ].lessThan( - 1e-6 ), () => {

							limit.assign( min( limit, u0[ axis ].negate().div( du[ axis ] ) ) );

						} );

					}

					const rayAt = ( s ) => mix( q0, q1, s ).div( mix( k0, k1, s ) );
					const previousS = float( 0 ).toVar();
					// D = sceneZ - rayZ: negative in front, positive behind the surface.
					const previousD = positionAt( u0 ).z.sub( origin.z ).toVar();
					const found = bool( false ).toVar();
					const hitUV = vec2( 0 ).toVar();
					const hitP = vec3( 0 ).toVar();
					Loop( { start: int( 1 ), end: int( this.steps.clamp( 1, 64 ) ), type: 'int', condition: '<=' }, ( { i: j } ) => {

						const s = float( j ).div( this.steps.clamp( 1, 64 ).floor() ).mul( limit.clamp() ).toVar();
						const p = u0.add( du.mul( s ) ).toVar();
						If( inside( p ).not(), () => {

							Break();

						} );
						const d = this.depthNode.sample( p ).r.toVar();
						If( isBackgroundDepth( d ), () => {

							previousD.assign( - 1e6 ); previousS.assign( s ); Continue();

						} );
						const sceneP = getViewPosition( p, d, this._inverseProjection ).toVar();
						const D = sceneP.z.sub( rayAt( s ).z ).toVar();
						If( previousD.lessThan( 0 ).and( D.greaterThanEqual( 0 ) ).and( D.lessThanEqual( this.thickness.max( epsilon ) ) ).and( sceneP.sub( origin ).length().lessThanEqual( reach ) ).and( sceneP.sub( P ).length().greaterThan( epsilon.mul( 2 ) ) ), () => {

							const denominator = previousD.sub( D ).toVar();
							const t = float( 1 ).toVar();
							If( denominator.abs().greaterThan( 1e-6 ), () => {

								t.assign( previousD.div( denominator ).clamp() );

							} );
							const refined = u0.add( du.mul( mix( previousS, s, t ) ) ).toVar();
							const refinedP = positionAt( refined ).toVar();
							const refinedD = refinedP.z.sub( rayAt( mix( previousS, s, t ) ).z );
							// Depth discontinuities can invalidate secant refinement; keep the slab hit.
							const useRefined = refinedD.abs().lessThanEqual( this.thickness.max( epsilon ) ).and( refinedP.sub( sceneP ).length().lessThanEqual( this.thickness.max( epsilon ) ) ).and( refinedP.sub( origin ).length().lessThanEqual( reach ) );
							hitUV.assign( useRefined.select( refined, p ) );
							hitP.assign( useRefined.select( refinedP, sceneP ) );
							found.assign( true );
							Break();

						} );
						previousD.assign( D );
						previousS.assign( s );

					} );
					If( found.and( representativeHit.w.equal( 0 ) ).and( reach.greaterThan( epsilon ) ), () => {

						representativeHit.assign( vec4( hitUV, hitP.sub( P ).length(), 1 ) );

					} );
					If( found.and( this._ready ).and( reach.greaterThan( epsilon ) ), () => {

						const worldHit = this._world.mul( vec4( hitP, 1 ) ).toVar();
						const previousViewHit = this._previousView.mul( worldHit ).xyz.toVar();
						const historyUV = getScreenPosition( previousViewHit, this._previousProjection ).toVar();
						if ( this.velocityNode !== null ) historyUV.assign( hitUV.sub( this.velocityNode.sample( hitUV ).xy.mul( vec2( 0.5, - 0.5 ) ) ).add( this._jitterDelta ) );
						If( inside( historyUV ).and( previousViewHit.z.lessThan( this._near.negate() ) ), () => {

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
					If( this.maxLuminance.greaterThan( 0 ), () => {

						radiance.mulAssign( this.maxLuminance.div( luminance( radiance ).max( 1e-6 ) ).min( 1 ) );

					} );
					incidentSum.addAssign( radiance );
					// Deliberately approximate response: Schlick only, no BRDF/PDF correction.
					weightedSum.addAssign( radiance.mul( F_Schlick( f0, dot( V, H ).clamp() ) ) );

				} );
				const divisor = validCount.max( 1 );
				weightedOutput.assign( vec4( weightedSum.div( divisor ).mul( this.intensity ), hitCount.greaterThan( 0 ).select( distanceSum.div( hitCount.max( 1 ) ), float( ENV_RAY_LENGTH ) ) ) );
				incidentOutput.assign( vec4( incidentSum.div( divisor ), hitCount.div( divisor ) ) );
				metadataOutput.assign( vec4( hitCount.div( divisor ), distanceSum.div( hitCount.max( 1 ) ), r, validCount ) );
				hitOutput.assign( representativeHit );

			} );
			return vec4( 0 );

		} )();
		const copyColor = property( 'vec4' );
		const copyGeometry = property( 'vec4' );
		this._copyMaterial.contextNode = context( builder.getSharedContext() );
		this._copyMaterial.outputNode = outputStruct( copyColor, copyGeometry );
		this._copyMaterial.colorNode = Fn( () => {

			const d = this.depthNode.sample( uvNode ).r;
			copyColor.assign( vec4( this.colorNode.sample( uvNode ).rgb, 1 ) );
			copyGeometry.assign( vec4( positionAt( uvNode ), isBackgroundDepth( d ).not() ) );
			return vec4( 0 );

		} )();
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
