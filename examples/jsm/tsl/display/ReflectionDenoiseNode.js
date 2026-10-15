import { Fn, If, abs, clamp, context, convertToTexture, cos, dot, exp, float, floor, getViewPosition, isBackgroundDepth, luminance, max, min, mix, mrt, normalize, passTexture, property, sin, smoothstep, sqrt, texture, uniform, uv, vec2, vec4 } from 'three/tsl';
import { FloatType, HalfFloatType, Matrix4, NearestFilter, Node, NodeMaterial, NodeUpdateType, QuadMesh, RenderTarget, RendererUtils, Vector2 } from 'three/webgpu';

const _quad = /*@__PURE__*/ new QuadMesh();
const _size = /*@__PURE__*/ new Vector2();
const signal = c => luminance( c.rgb ).add( c.a );

function target( names ) {

	const result = new RenderTarget( 1, 1, { count: names.length, type: HalfFloatType, depthBuffer: false } );
	for ( let i = 0; i < names.length; i ++ ) result.textures[ i ].name = names[ i ];
	// Raw depth requires full precision for reliable distant-surface rejection.
	if ( names.includes( 'geometry' ) ) {

		const geometry = result.textures[ names.indexOf( 'geometry' ) ];
		geometry.type = FloatType;
		geometry.minFilter = NearestFilter;
		geometry.magFilter = NearestFilter;

	}

	return result;

}

/**
 * Spatiotemporal reflection reconstruction. RGB is premultiplied incident radiance;
 * alpha is hit coverage, never a history length or ray distance. Normals are decoded
 * view-space vectors and velocities are NDC displacement, as produced by the scene MRT.
 * Hit distance is supplied separately in scene units (zero denotes a miss).
 *
 * Temporal feedback deliberately precedes spatial filtering to avoid blur feedback.
 * @augments Node
 */
class ReflectionDenoiseNode extends Node {

	static get type() {

		return 'ReflectionDenoiseNode';

	}

	constructor( colorNode, depthNode, normalNode, roughnessNode, velocityNode, camera, { hitDistanceNode = float( 0 ) } = {} ) {

		super( 'vec4' );
		this.updateBeforeType = NodeUpdateType.FRAME;
		this.colorNode = colorNode;
		this.depthNode = depthNode;
		this.normalNode = normalNode;
		this.roughnessNode = roughnessNode;
		this.velocityNode = velocityNode;
		this.hitDistanceNode = hitDistanceNode;
		this.camera = camera;
		this.maxHistory = uniform( 16 );
		this.maxHistoryStill = uniform( 256 );
		this.spatialIterations = 4;
		this.prefilterRadius = uniform( 0.5 );
		this.historyClamp = uniform( 1.5 );
		this.parallaxHistory = uniform( 2 );
		this.historyOnStop = uniform( 4 );
		this.phiLuminance = uniform( 4 );
		this.phiNormal = uniform( 64 );
		this.phiDepth = uniform( 1 );
		this._size = uniform( new Vector2( 1, 1 ) );
		this._projection = uniform( new Matrix4() );
		this._projectionInverse = uniform( new Matrix4() );
		this._world = uniform( new Matrix4() );
		this._previousWorld = uniform( new Matrix4() );
		this._previousViewProjection = uniform( new Matrix4() );
		this._previousProjectionInverse = uniform( new Matrix4() );
		this._previousProjection = new Matrix4();
		this._previousView = new Matrix4();
		this._valid = uniform( false, 'bool' );
		this._still = uniform( false, 'bool' );
		this._cap = uniform( 1e9 );
		this._frame = uniform( 0 );
		this._step = uniform( 1 );
		this._history = [ target( [ 'output', 'moments', 'geometry' ] ), target( [ 'output', 'moments', 'geometry' ] ) ];
		this._prefilter = target( [ 'output', 'distance' ] );
		this._spatial = [ target( [ 'output' ] ), target( [ 'output' ] ) ];
		this._historyColor = texture( this._history[ 0 ].textures[ 0 ] );
		this._historyMoments = texture( this._history[ 0 ].textures[ 1 ] );
		this._historyGeometry = texture( this._history[ 0 ].textures[ 2 ] );
		this._filtered = texture( this._prefilter.textures[ 0 ] );
		this._distance = texture( this._prefilter.textures[ 1 ] );
		this._spatialInput = texture( this._history[ 1 ].textures[ 0 ] );
		this._moments = texture( this._history[ 1 ].textures[ 1 ] );
		this._textureNode = passTexture( this, this._history[ 1 ].textures[ 0 ] );
		this._momentsNode = passTexture( this, this._history[ 1 ].textures[ 1 ] );
		this._prefilterMaterial = new NodeMaterial();
		this._temporalMaterial = new NodeMaterial();
		this._spatialMaterial = new NodeMaterial();
		this._index = 0;
		this._hasHistory = false;
		this._wasStill = false;
		this._cameraUUID = '';
		this._lastFrameId = - 1;

	}

	getTextureNode() {

		return this._textureNode;

	}

	getMomentsTextureNode() {

		return this._momentsNode;

	}

	reset() {

		this._hasHistory = false;

	}

	setSize( width, height ) {

		if ( this._size.value.x === width && this._size.value.y === height ) return;
		this._size.value.set( width, height );
		for ( const rt of [ ...this._history, this._prefilter, ...this._spatial ] ) rt.setSize( width, height );
		this.reset();

	}

	_geometry( coord ) {

		const depth = this.depthNode.sample( coord ).r;
		const position = getViewPosition( coord, depth, this._projectionInverse );
		const normal = normalize( this.normalNode.sample( coord ).rgb );
		return { depth, position, normal };

	}

	_footprint( position, roughness, distance ) {

		const pixelSize = abs( position.z ).mul( 2 ).div( abs( this._projection.element( 1 ).y ).mul( this._size.y ) ).max( 1e-6 );
		return { pixelSize, footprint: roughness.pow2().mul( distance.greaterThan( 0 ).select( distance, abs( position.z ).mul( 0.25 ) ) ).div( pixelSize ).max( 1 ) };

	}

	_buildPrefilter() {

		const output = property( 'vec4' ), distanceOutput = property( 'vec4' );
		this._prefilterMaterial.outputNode = mrt( { output, distance: distanceOutput } );
		return Fn( () => {

			const coord = uv();
			const g = this._geometry( coord );
			const still = this._still.and( this.velocityNode.sample( coord ).xy.length().lessThan( 1e-6 ) );
			const center = this.colorNode.sample( coord ).toVar();
			const roughness = this.roughnessNode.sample( coord ).r;
			const distance = this.hitDistanceNode.sample( coord ).r.toVar();
			If( distance.lessThanEqual( 0 ), () => {

				const total = float( 0 ).toVar(), count = float( 0 ).toVar();
				for ( let y = - 1; y <= 1; y ++ ) for ( let x = - 1; x <= 1; x ++ ) {

					const d = this.hitDistanceNode.sample( coord.add( vec2( x * 4, y * 4 ).div( this._size ) ).clamp( 0, 1 ) ).r;
					total.addAssign( d );
					count.addAssign( d.greaterThan( 0 ).select( 1, 0 ) );

				}

				distance.assign( total.div( count.max( 1 ) ) );

			} );
			const { footprint } = this._footprint( g.position, roughness, distance );
			const radius = footprint.mul( this.prefilterRadius ).clamp( 1, this._size.y.mul( 0.03 ).max( 1 ) );
			const angle = dot( floor( coord.mul( this._size ) ), vec2( 0.06711056, 0.00583715 ) ).fract().mul( 52.9829189 ).fract().add( this._frame.mul( 0.6180339887 ) ).fract().mul( Math.PI * 2 );
			const sum = center.toVar(), weights = float( 1 ).toVar();
			const distanceSum = this.hitDistanceNode.sample( coord ).r.toVar();
			const distanceWeight = distanceSum.greaterThan( 0 ).select( float( 1 ), float( 0 ) ).toVar();
			If( isBackgroundDepth( g.depth ).not().and( roughness.lessThan( 0.9 ) ).and( this.prefilterRadius.greaterThan( 0 ) ).and( still.not() ), () => {

				for ( let i = 0; i < 16; i ++ ) {

					const t = Math.sqrt( ( i + 0.5 ) / 16 );
					const a = angle.add( ( i + 0.5 ) * 2.39996323 );
					const q = floor( coord.mul( this._size ) ).add( floor( vec2( cos( a ), sin( a ) ).mul( radius ).mul( t ).add( 0.5 ) ) ).add( 0.5 ).div( this._size );
					const n = this._geometry( q );
					const r = this.roughnessNode.sample( q ).r;
					const w = exp( abs( dot( g.normal, n.position.sub( g.position ) ) ).div( abs( g.position.z ).mul( 0.02 ).max( 1e-6 ) ).negate() )
						.mul( dot( g.normal, n.normal ).max( 0 ).pow( 32 ) ).mul( exp( abs( r.sub( roughness ) ).mul( - 20 ) ) ).mul( Math.exp( - 2 * t * t ) );
					If( q.greaterThanEqual( 0 ).all().and( q.lessThanEqual( 1 ).all() ).and( isBackgroundDepth( n.depth ).not() ), () => {

						sum.addAssign( this.colorNode.sample( q ).mul( w ) );
						weights.addAssign( w );
						const d = this.hitDistanceNode.sample( q ).r;
						If( d.greaterThan( 0 ), () => {

							distanceSum.addAssign( d.mul( w ) ); distanceWeight.addAssign( w );

						} );

					} );

				}

			} );
			output.assign( sum.div( weights ) );
			distanceOutput.assign( vec4( distanceSum.div( distanceWeight.max( 1e-6 ) ), 0, 0, 0 ) );
			return vec4( 0 );

		} )();

	}

	_buildTemporal() {

		const output = property( 'vec4' ), momentsOutput = property( 'vec4' ), geometryOutput = property( 'vec4' );
		this._temporalMaterial.outputNode = mrt( { output, moments: momentsOutput, geometry: geometryOutput } );
		return Fn( () => {

			const coord = uv(), g = this._geometry( coord );
			const raw = this._filtered.sample( coord ).toVar();
			const worldPosition = this._world.mul( vec4( g.position, 1 ) ).xyz;
			const worldNormal = normalize( this._world.mul( vec4( g.normal, 0 ) ).xyz );
			const mean = vec4( 0 ).toVar(), second = vec4( 0 ).toVar();
			const lumMean = float( 0 ).toVar(), lumSecond = float( 0 ).toVar();
			for ( let y = - 1; y <= 1; y ++ ) for ( let x = - 1; x <= 1; x ++ ) {

				const c = this._filtered.sample( coord.add( vec2( x, y ).div( this._size ) ).clamp( 0, 1 ) );
				mean.addAssign( c.div( 9 ) ); second.addAssign( c.mul( c ).div( 9 ) );
				lumMean.addAssign( signal( c ).div( 9 ) ); lumSecond.addAssign( signal( c ).pow2().div( 9 ) );

			}

			const spatialVariance = lumSecond.sub( lumMean.pow2() ).max( 0 );
			const motion = this.velocityNode.sample( coord ).xy;
			const still = this._still.and( motion.length().lessThan( 1e-6 ) );
			const surfaceUV = coord.sub( motion.mul( vec2( 0.5, - 0.5 ) ) );
			const previousUV = surfaceUV.toVar();
			const maxAge = still.select( this.maxHistoryStill.max( this.maxHistory ), this.maxHistory ).max( 1 ).toVar();
			const distance = this._distance.sample( coord ).r;
			const roughness = this.roughnessNode.sample( coord ).r;
			If( still, () => {

				previousUV.assign( coord );

			} ).Else( () => {

				If( distance.greaterThan( 0 ), () => {

					const virtualPosition = this._world.mul( vec4( g.position.add( normalize( g.position ).mul( distance ) ), 1 ) );
					const clip = this._previousViewProjection.mul( virtualPosition );
					previousUV.assign( clip.xy.div( clip.w ).mul( vec2( 0.5, - 0.5 ) ).add( 0.5 ) );
					const { footprint } = this._footprint( g.position, roughness, distance );
					const parallax = previousUV.sub( surfaceUV ).mul( this._size ).length();
					maxAge.assign( maxAge.div( parallax.mul( this.parallaxHistory ).div( footprint ).add( 1 ) ).max( 1 ) );

				} );

			} );
			const history = vec4( 0 ).toVar(), moments = vec4( 0 ).toVar(), weight = float( 0 ).toVar();
			const pixel = previousUV.mul( this._size ).sub( 0.5 ), base = floor( pixel ), f = pixel.fract();
			If( this._valid.and( isBackgroundDepth( g.depth ).not() ), () => {

				for ( let y = 0; y <= 1; y ++ ) for ( let x = 0; x <= 1; x ++ ) {

					const q = base.add( vec2( x, y ) ).add( 0.5 ).div( this._size );
					const geom = this._historyGeometry.sample( q );
					const previousPosition = this._previousWorld.mul( vec4( getViewPosition( q, geom.a, this._previousProjectionInverse ), 1 ) ).xyz;
					const plane = abs( dot( worldNormal, previousPosition.sub( worldPosition ) ) );
					const w = ( x ? f.x : f.x.oneMinus() ).mul( y ? f.y : f.y.oneMinus() );
					If( q.greaterThanEqual( 0 ).all().and( q.lessThanEqual( 1 ).all() ).and( isBackgroundDepth( geom.a ).not() )
						.and( plane.lessThan( abs( g.position.z ).mul( 0.03 ) ) ).and( dot( worldNormal, geom.rgb ).greaterThan( 0.8 ) ), () => {

						history.addAssign( this._historyColor.sample( q ).mul( w ) );
						moments.addAssign( this._historyMoments.sample( q ).mul( w ) ); weight.addAssign( w );

					} );

				}

			} );
			const age = float( 1 ).toVar();
			If( weight.greaterThan( 1e-3 ), () => {

				history.divAssign( weight ); moments.divAssign( weight );
				If( still.not(), () => {

					const sigma = sqrt( second.sub( mean.mul( mean ) ).max( 0 ) );
					const clipped = clamp( history, mean.sub( sigma.mul( this.historyClamp ) ), mean.add( sigma.mul( this.historyClamp ) ) );
					history.assign( mix( history, clipped, smoothstep( 0.2, 0.6, mean.a ) ) );
					const coverage = mean.a.add( sigma.a.mul( this.historyClamp ) ).add( 0.1 );
					history.mulAssign( min( 1, coverage.div( history.a.max( 1e-6 ) ) ) );

				} );
				const disagreement = abs( lumMean.sub( signal( history ) ) ).div( sqrt( spatialVariance ).div( 3 ).add( max( signal( history ), lumMean ).mul( 0.05 ) ).add( 0.01 ) );
				const keep = disagreement.sub( 2 ).mul( 0.25 ).oneMinus().clamp();
				age.assign( min( moments.b, this._cap ).min( maxAge.mul( keep ).max( 1 ) ).add( 1 ).min( maxAge ) );

			} );
			const alpha = age.reciprocal();
			const color = raw.toVar();
			const lum = signal( raw );
			const m = vec2( lum, lum.pow2() ).toVar();
			// Do not read stale temporal values when seeding or rejecting history.
			If( age.greaterThan( 1 ), () => {

				color.assign( mix( history, raw, alpha ) );
				m.assign( mix( moments.rg, m, alpha.max( 0.2 ) ) );

			} );
			const variance = m.y.sub( m.x.pow2() ).max( 0 ).max( spatialVariance.mul( float( 4 ).sub( age ).max( 0 ) ).div( 3 ) ).div( age );
			If( isBackgroundDepth( g.depth ), () => {

				color.assign( vec4( 0 ) );

			} );
			output.assign( color );
			momentsOutput.assign( vec4( m, age, variance ) );
			geometryOutput.assign( vec4( worldNormal, g.depth ) );
			return vec4( 0 );

		} )();

	}

	_buildSpatial() {

		return Fn( () => {

			const coord = uv(), g = this._geometry( coord );
			const center = this._spatialInput.sample( coord );
			const roughness = this.roughnessNode.sample( coord ).r;
			const { footprint, pixelSize } = this._footprint( g.position, roughness, this._distance.sample( coord ).r );
			const variance = float( 0 ).toVar();
			for ( let y = - 1; y <= 1; y ++ ) for ( let x = - 1; x <= 1; x ++ ) {

				variance.addAssign( this._moments.sample( coord.add( vec2( x, y ).div( this._size ) ).clamp( 0, 1 ) ).a.mul( ( x === 0 ? 0.5 : 0.25 ) * ( y === 0 ? 0.5 : 0.25 ) ) );

			}

			const sum = center.toVar(), weights = float( 1 ).toVar();
			If( this._step.lessThanEqual( footprint ).and( isBackgroundDepth( g.depth ).not() ), () => {

				for ( let y = - 1; y <= 1; y ++ ) for ( let x = - 1; x <= 1; x ++ ) {

					if ( x === 0 && y === 0 ) continue;
					const offset = vec2( x, y ).mul( this._step ), q = coord.add( offset.div( this._size ) );
					const n = this._geometry( q ), c = this._spatialInput.sample( q );
					const w = dot( g.normal, n.normal ).max( 0 ).pow( this.phiNormal )
						.mul( exp( abs( dot( g.normal, n.position.sub( g.position ) ) ).div( this.phiDepth.mul( pixelSize ).mul( offset.length() ).add( 1e-6 ) ).negate() ) )
						.mul( exp( abs( signal( c ).sub( signal( center ) ) ).div( this.phiLuminance.mul( sqrt( variance.max( 0 ) ) ).add( 1e-4 ) ).negate() ) )
						.mul( exp( abs( this.roughnessNode.sample( q ).r.sub( roughness ) ).mul( - 20 ) ) ).mul( ( x === 0 ? 1 : 0.5 ) * ( y === 0 ? 1 : 0.5 ) );
					If( q.greaterThanEqual( 0 ).all().and( q.lessThanEqual( 1 ).all() ).and( isBackgroundDepth( n.depth ).not() ), () => {

						sum.addAssign( c.mul( w ) ); weights.addAssign( w );

					} );

				}

			} );
			return sum.div( weights );

		} )();

	}

	setup( builder ) {

		// Build all pass dependencies before registering this frame update.
		for ( const input of [ this.colorNode, this.depthNode, this.normalNode, this.roughnessNode, this.velocityNode, this.hitDistanceNode ] ) input.build( builder );
		const shared = context( builder.getSharedContext() );
		for ( const material of [ this._prefilterMaterial, this._temporalMaterial, this._spatialMaterial ] ) material.contextNode = shared;
		this._prefilterMaterial.colorNode = this._buildPrefilter();
		this._temporalMaterial.colorNode = this._buildTemporal();
		this._spatialMaterial.fragmentNode = this._buildSpatial();
		return this._textureNode;

	}

	updateBefore( { renderer, frameId } ) {

		const camera = this.camera;
		if ( this._lastFrameId >= 0 && frameId - this._lastFrameId > 1 ) this.reset();
		this._lastFrameId = frameId;
		renderer.getDrawingBufferSize( _size );
		this.setSize( _size.x, _size.y );
		// Ignore subpixel jitter when classifying camera movement and storing geometry.
		const projection = camera.projectionMatrix.clone();
		if ( camera.view && camera.view.enabled ) {

			const view = camera.view;
			camera.view = null;
			camera.updateProjectionMatrix();
			projection.copy( camera.projectionMatrix );
			camera.view = view;
			camera.updateProjectionMatrix();

		}

		const still = this._previousView.equals( camera.matrixWorldInverse ) && this._previousProjection.equals( projection );
		if ( this._cameraUUID !== camera.uuid || ! this._previousProjection.equals( projection ) ) this.reset();
		if ( this._hasHistory ) {

			const a = this._previousWorld.value.elements, b = camera.matrixWorld.elements;
			const translation = Math.hypot( a[ 12 ] - b[ 12 ], a[ 13 ] - b[ 13 ], a[ 14 ] - b[ 14 ] );
			const rotation = a[ 8 ] * b[ 8 ] + a[ 9 ] * b[ 9 ] + a[ 10 ] * b[ 10 ];
			if ( translation > Math.max( 1, camera.far * 0.1 ) || rotation < 0.5 ) this.reset();

		}

		this._valid.value = this._hasHistory;
		this._still.value = still;
		this._cap.value = still && ! this._wasStill && this.historyOnStop.value > 0 ? this.historyOnStop.value : 1e9;
		this._projection.value.copy( projection );
		this._projectionInverse.value.copy( camera.projectionMatrixInverse );
		this._world.value.copy( camera.matrixWorld );
		const read = this._history[ this._index ], write = this._history[ this._index ^ 1 ];
		this._historyColor.value = read.textures[ 0 ]; this._historyMoments.value = read.textures[ 1 ]; this._historyGeometry.value = read.textures[ 2 ];
		const state = RendererUtils.resetRendererState( renderer );
		try {

			const render = ( material, rt ) => {

				renderer.setRenderTarget( rt ); _quad.material = material; _quad.render( renderer );

			};

			render( this._prefilterMaterial, this._prefilter );
			render( this._temporalMaterial, write );
			this._moments.value = write.textures[ 1 ];
			let source = write.textures[ 0 ];
			for ( let i = 0; i < Math.min( 8, Math.max( 0, Math.round( this.spatialIterations ) ) ); i ++ ) {

				this._spatialInput.value = source;
				this._step.value = 2 ** i;
				const destination = this._spatial[ i % 2 ];
				render( this._spatialMaterial, destination ); source = destination.texture;

			}

			this._textureNode.value = source;
			this._momentsNode.value = write.textures[ 1 ];

		} finally {

			RendererUtils.restoreRendererState( renderer, state );

		}

		this._previousWorld.value.copy( camera.matrixWorld );
		this._previousView.copy( camera.matrixWorldInverse );
		this._previousProjection.copy( projection );
		this._previousProjectionInverse.value.copy( camera.projectionMatrixInverse );
		this._previousViewProjection.value.multiplyMatrices( projection, camera.matrixWorldInverse );
		this._cameraUUID = camera.uuid;
		this._hasHistory = true;
		this._wasStill = still;
		this._index ^= 1;
		this._frame.value = ( this._frame.value + 1 ) % 4096;

	}

	dispose() {

		super.dispose();
		for ( const rt of [ ...this._history, this._prefilter, ...this._spatial ] ) rt.dispose();
		for ( const material of [ this._prefilterMaterial, this._temporalMaterial, this._spatialMaterial ] ) material.dispose();

	}

}

export default ReflectionDenoiseNode;

export const reflectionDenoise = ( colorNode, depthNode, normalNode, roughnessNode, velocityNode, camera, options = {} ) => new ReflectionDenoiseNode(
	convertToTexture( colorNode ), convertToTexture( depthNode ), convertToTexture( normalNode ), convertToTexture( roughnessNode ), convertToTexture( velocityNode ), camera,
	{ ...options, hitDistanceNode: convertToTexture( options.hitDistanceNode || float( 0 ) ) }
);
