import { HalfFloatType, Matrix4, NearestFilter, Node, NodeMaterial, NodeUpdateType, QuadMesh, RenderTarget, RendererUtils, Vector2 } from 'three/webgpu';
import { Break, Continue, Fn, If, Loop, abs, bool, context, cross, dot, float, floor, fract, getScreenPosition, getViewPosition, int, interleavedGradientNoise, isBackgroundDepth, max, min, mix, mrt, nodeObject, normalize, passTexture, reflect, texture, uniform, uv, vec2, vec3, vec4 } from 'three/tsl';

const _quad = /*@__PURE__*/ new QuadMesh();
const _size = /*@__PURE__*/ new Vector2();
const _phases = [[ 0, 0 ], [ 1, 1 ], [ 1, 0 ], [ 0, 1 ]];

/**
 * Experimental reflection tracer with rotating half-resolution sampling and
 * geometry-aware reconstruction. Outputs incident radiance premultiplied by
 * screen-space coverage, not BRDF-weighted reflected lighting. Misses are zero;
 * the compositor must preserve its existing environment lighting on misses.
 *
 * @augments Node
 */
class SSRTraceNode extends Node {

	static get type() {

		return 'SSRTraceNode';

	}

	constructor( colorNode, depthNode, normalNode, { roughnessNode, camera } ) {

		super( 'vec4' );
		this.colorNode = colorNode;
		this.depthNode = depthNode;
		this.normalNode = normalNode;
		this.roughnessNode = roughnessNode;
		this.camera = camera;
		this.resolutionScale = 0.5;
		this.maxDistance = uniform( 8 );
		this.thickness = uniform( 0.15 );
		this.rayCount = uniform( 2 );
		this.quality = uniform( 1 );
		this.roughnessFactor = uniform( 1 );
		this.maxRoughness = uniform( 0.9 );
		this.mirrorRoughness = uniform( 0.1 );
		this.maxRadiance = uniform( 16 );
		this.updateBeforeType = NodeUpdateType.FRAME;
		this._projection = uniform( new Matrix4() );
		this._projectionInverse = uniform( new Matrix4() );
		this._size = uniform( new Vector2() );
		this._traceSize = uniform( new Vector2() );
		this._phase = uniform( new Vector2() );
		this._frame = uniform( 0 );
		this._mirrorOnly = uniform( false );
		this._halfResolution = uniform( true );
		this._traceTarget = this._createTarget( 'SSRTrace.Trace' );
		this._mirrorTarget = this._createTarget( 'SSRTrace.Mirror' );
		this._outputTarget = this._createTarget( 'SSRTrace.Output' );
		this._traceMaterial = new NodeMaterial();
		this._resolveMaterial = new NodeMaterial();
		this._traceMaterial.name = 'SSRTrace';
		this._resolveMaterial.name = 'SSRTrace.Resolve';
		this._textureNode = passTexture( this, this._outputTarget.textures[ 0 ] );
		this._distanceNode = passTexture( this, this._outputTarget.textures[ 1 ] ).r;

	}

	_createTarget( name ) {

		const target = new RenderTarget( 1, 1, { count: 2, type: HalfFloatType, depthBuffer: false, minFilter: NearestFilter, magFilter: NearestFilter } );
		target.textures[ 0 ].name = 'output';
		target.textures[ 1 ].name = 'distance';
		target.name = name;
		return target;

	}

	getTextureNode() {

		return this._textureNode;

	}

	getHitDistanceNode() {

		return this._distanceNode;

	}

	getRenderTarget() {

		return this._outputTarget;

	}

	updateBefore( { renderer, frameId } ) {

		const state = RendererUtils.resetRendererState( renderer );
		try {

			renderer.getDrawingBufferSize( _size );
			const half = this.resolutionScale < 0.75;
			this._size.value.copy( _size );
			this._traceSize.value.set( half ? Math.ceil( _size.x / 2 ) : _size.x, half ? Math.ceil( _size.y / 2 ) : _size.y );
			this._traceTarget.setSize( this._traceSize.value.x, this._traceSize.value.y );
			this._mirrorTarget.setSize( _size.x, _size.y );
			this._outputTarget.setSize( _size.x, _size.y );
			this._halfResolution.value = half;
			this._frame.value = frameId % 4096;
			this._phase.value.fromArray( half ? _phases[ frameId % 4 ] : _phases[ 0 ] );
			this._projection.value.copy( this.camera.projectionMatrix );
			this._projectionInverse.value.copy( this.camera.projectionMatrixInverse );
			this._mirrorOnly.value = false;
			_quad.material = this._traceMaterial;
			renderer.setRenderTarget( this._traceTarget );
			_quad.render( renderer );
			if ( half ) {

				// Mirror pixels get a full-resolution trace, rather than inheriting
				// the sparse trace's rotating phase or a glossy reconstruction filter.
				this._mirrorOnly.value = true;
				renderer.setRenderTarget( this._mirrorTarget );
				_quad.render( renderer );

			}

			_quad.material = this._resolveMaterial;
			renderer.setRenderTarget( this._outputTarget );
			_quad.render( renderer );

		} finally {

			RendererUtils.restoreRendererState( renderer, state );

		}

	}

	setup( builder ) {

		const shared = context( builder.getSharedContext() );
		const trace = Fn( () => {

			const pixel = floor( uv().mul( this._traceSize ) ).toVar();
			const sparseUV = pixel.mul( 2 ).add( this._phase ).add( 0.5 ).div( this._size );
			const coord = this._halfResolution.and( this._mirrorOnly.not() ).select( sparseUV, uv() ).toVar();
			const depth = this.depthNode.sample( coord ).r.toVar();
			const roughness = this.roughnessNode.sample( coord ).r.clamp( 0, 1 ).toVar();
			const result = vec4( 0 ).toVar();
			const distanceResult = float( 0 ).toVar();
			const eligible = isBackgroundDepth( depth ).not().and( roughness.lessThan( this.maxRoughness ) );
			const mirror = roughness.lessThan( this.mirrorRoughness );
			If( eligible.and( this._mirrorOnly.not().or( mirror ) ), () => {

				const position = getViewPosition( coord, depth, this._projectionInverse ).toVar();
				const normal = normalize( this.normalNode.sample( coord ).xyz ).toVar();
				const incident = this.camera.isPerspectiveCamera ? normalize( position ) : vec3( 0, 0, - 1 );
				const origin = position.add( normal.mul( max( abs( position.z ).mul( 0.0001 ), 0.001 ) ) ).toVar();
				const tangent = normalize( cross( abs( normal.z ).lessThan( 0.999 ).select( vec3( 0, 0, 1 ), vec3( 1, 0, 0 ) ), normal ) ).toVar();
				const bitangent = cross( normal, tangent ).toVar();
				const phase = this._frame.mod( 64 );
				const block = floor( this._frame.div( 64 ) );
				const steps = int( this.quality.clamp( 0, 1 ).mul( 32 ).max( 1 ) ).toVar();
				const rays = int( this.rayCount.clamp( 1, 8 ) ).toVar();
				const hits = float( 0 ).toVar();
				const radiance = vec3( 0 ).toVar();
				const distanceSum = float( 0 ).toVar();
				Loop( { start: int( 0 ), end: int( 8 ), type: 'int', condition: '<' }, ( { i } ) => {

					If( i.greaterThanEqual( rays ), () => {

						Break();

					} );
					const seed = float( i );
					const noisePosition = coord.mul( this._size );
					const e1 = fract( interleavedGradientNoise( noisePosition.add( phase.mul( 34 ).add( seed ) ) ).add( block.mul( 0.6180339887 ) ) ).toVar();
					const e2 = mix( fract( e1.mul( 38.65435 ) ), 1, 0.7 );
					const r = roughness.mul( this.roughnessFactor ).clamp( 0, 1 );
					const r4 = r.pow( 4 );
					const cosTheta = e2.max( 0.000001 ).pow( r4.div( float( 2 ).sub( r4 ) ) );
					const sinTheta = cosTheta.mul( cosTheta ).oneMinus().max( 0 ).sqrt();
					const phi = e1.mul( Math.PI * 2 );
					const halfVector = tangent.mul( sinTheta.mul( phi.cos() ) ).add( bitangent.mul( sinTheta.mul( phi.sin() ) ) ).add( normal.mul( cosTheta ) );
					const direction = normalize( reflect( incident, mirror.select( normal, halfVector ) ) ).toVar();
					If( dot( direction, normal ).lessThanEqual( 0 ), () => {

						Continue();

					} );
					const scale = i.mod( 3 ).equal( 0 ).select( 1, i.mod( 3 ).equal( 1 ).select( 0.5, 1.5 ) );
					const rayLength = this.maxDistance.max( 0.001 ).mul( scale ).toVar();
					// Clip at the near plane before projecting; screen-linear marching
					// interpolates reciprocal view depth for perspective cameras.
					If( direction.z.greaterThan( 0 ), () => {

						rayLength.assign( min( rayLength, max( float( - this.camera.near ).sub( origin.z ).div( direction.z ), 0.001 ) ) );

					} );
					const end = origin.add( direction.mul( rayLength ) ).toVar();
					const startUV = getScreenPosition( origin, this._projection ).toVar();
					const endUV = getScreenPosition( end, this._projection ).toVar();
					const deltaUV = endUV.sub( startUV ).toVar();
					const jitter = fract( interleavedGradientNoise( noisePosition.add( phase ).add( seed ) ).add( block.mul( 0.4142135624 ) ) );
					const previousDelta = float( 0 ).toVar();
					const found = bool( false ).toVar();
					const hitUV = vec2( 0 ).toVar();
					const lastS = float( 0 ).toVar();
					Loop( { start: int( 0 ), end: int( 32 ), type: 'int', condition: '<' }, ( { i: k } ) => {

						If( k.greaterThanEqual( steps ), () => {

							Break();

						} );
						const s = float( k ).add( jitter ).add( 0.5 ).div( float( steps ) ).min( 1 ).toVar();
						const sampleUV = startUV.add( deltaUV.mul( s ) ).toVar();
						If( sampleUV.x.lessThan( 0 ).or( sampleUV.y.lessThan( 0 ) ).or( sampleUV.x.greaterThan( 1 ) ).or( sampleUV.y.greaterThan( 1 ) ), () => {

							Break();

						} );
						const sceneDepth = this.depthNode.sample( sampleUV ).r.toVar();
						const sceneZ = getViewPosition( sampleUV, sceneDepth, this._projectionInverse ).z;
						const rayZ = this.camera.isPerspectiveCamera ? float( 1 ).div( mix( origin.z.reciprocal(), end.z.reciprocal(), s ) ) : mix( origin.z, end.z, s );
						const delta = rayZ.sub( sceneZ ).toVar();
						If( isBackgroundDepth( sceneDepth ).not().and( delta.lessThan( 0 ) ).and( delta.greaterThan( this.thickness.negate() ) ), () => {

							const fraction = previousDelta.div( previousDelta.sub( delta ).max( 0.000001 ) ).clamp( 0, 1 );
							const refinedS = mix( lastS, s, fraction );
							hitUV.assign( startUV.add( deltaUV.mul( refinedS ) ) );
							found.assign( true );
							Break();

						} );
						previousDelta.assign( delta );
						lastS.assign( s );

					} );
					If( found, () => {

						const hitDepth = this.depthNode.sample( hitUV ).r;
						const hitPosition = getViewPosition( hitUV, hitDepth, this._projectionInverse );
						radiance.addAssign( this.colorNode.sample( hitUV ).rgb.max( 0 ).min( this.maxRadiance ) );
						distanceSum.addAssign( hitPosition.sub( origin ).length() );
						hits.addAssign( 1 );

					} );

				} );
				If( hits.greaterThan( 0 ), () => {

					// Average valid hits, track coverage independently. This is a
					// biased glossy reflection estimator, not GGX/PDF integration.
					result.assign( vec4( radiance.div( hits ), 1 ) );
					distanceResult.assign( distanceSum.div( hits ) );

				} );

			} );
			return mrt( { output: result, distance: vec4( distanceResult, 0, 0, 0 ) } );

		} );
		this._traceMaterial.contextNode = shared;
		this._traceMaterial.fragmentNode = trace();
		const resolve = Fn( () => {

			const coord = uv();
			const roughness = this.roughnessNode.sample( coord ).r;
			const color = vec4( 0 ).toVar();
			const hitDistance = float( 0 ).toVar();
			If( this._halfResolution.not(), () => {

				color.assign( texture( this._traceTarget.textures[ 0 ], coord ) );
				hitDistance.assign( texture( this._traceTarget.textures[ 1 ], coord ).r );

			} ).ElseIf( roughness.lessThan( this.mirrorRoughness ), () => {

				color.assign( texture( this._mirrorTarget.textures[ 0 ], coord ) );
				hitDistance.assign( texture( this._mirrorTarget.textures[ 1 ], coord ).r );

			} ).Else( () => {

				const depth = this.depthNode.sample( coord ).r;
				If( isBackgroundDepth( depth ).not(), () => {

					const position = getViewPosition( coord, depth, this._projectionInverse ).toVar();
					const normal = normalize( this.normalNode.sample( coord ).xyz ).toVar();
					const pixel = floor( coord.mul( this._size ) ).toVar();
					const base = floor( pixel.sub( this._phase ).mul( 0.5 ) ).toVar();
					const total = float( 0 ).toVar();
					const distanceWeight = float( 0 ).toVar();
					Loop( { start: int( 0 ), end: int( 4 ), type: 'int', condition: '<' }, ( { i } ) => {

						const halfPixel = base.add( vec2( float( i.mod( 2 ) ), floor( float( i ).div( 2 ) ) ) ).clamp( 0, this._traceSize.sub( 1 ) );
						const tracedPixel = halfPixel.mul( 2 ).add( this._phase );
						const tracedUV = tracedPixel.add( 0.5 ).div( this._size );
						const tracedDepth = this.depthNode.sample( tracedUV ).r;
						If( isBackgroundDepth( tracedDepth ), () => {

							Continue();

						} );
						const tracedPosition = getViewPosition( tracedUV, tracedDepth, this._projectionInverse );
						const tracedNormal = normalize( this.normalNode.sample( tracedUV ).xyz );
						const plane = dot( normal, tracedPosition.sub( position ) ).abs();
						const planeWeight = plane.negate().div( abs( position.z ).mul( 0.02 ).max( 0.00001 ) ).exp();
						const normalWeight = dot( normal, tracedNormal ).max( 0 ).pow( 32 );
						const delta = tracedPixel.sub( pixel );
						const weight = planeWeight.mul( normalWeight ).div( dot( delta, delta ).add( 1 ) );
						const halfUV = halfPixel.add( 0.5 ).div( this._traceSize );
						color.addAssign( texture( this._traceTarget.textures[ 0 ], halfUV ).mul( weight ) );
						total.addAssign( weight );
						const distance = texture( this._traceTarget.textures[ 1 ], halfUV ).r;
						If( distance.greaterThan( 0 ), () => {

							hitDistance.addAssign( distance.mul( weight ) );
							distanceWeight.addAssign( weight );

						} );

					} );
					color.divAssign( total.max( 0.00001 ) );
					hitDistance.divAssign( distanceWeight.max( 0.00001 ) );

				} );

			} );
			return mrt( { output: color, distance: vec4( hitDistance, 0, 0, 0 ) } );

		} );
		this._resolveMaterial.contextNode = shared;
		this._resolveMaterial.fragmentNode = resolve();
		return this._textureNode;

	}

	dispose() {

		super.dispose();
		this._traceTarget.dispose();
		this._mirrorTarget.dispose();
		this._outputTarget.dispose();
		this._traceMaterial.dispose();
		this._resolveMaterial.dispose();

	}

}

export default SSRTraceNode;
export const ssrTrace = ( colorNode, depthNode, normalNode, options ) => new SSRTraceNode( nodeObject( colorNode ), nodeObject( depthNode ), nodeObject( normalNode ), options );
