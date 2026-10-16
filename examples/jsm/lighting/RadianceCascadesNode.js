import { Node, NodeUpdateType, NodeMaterial, QuadMesh, RenderTarget, StorageTexture, HalfFloatType, Vector2, RendererUtils } from 'three/webgpu';
import { Fn, If, Loop, Break, Return, uniform, texture, textureStore, instanceIndex, float, ivec2, ivec3, vec2, vec3, vec4, uv, floor, fract, min, max, abs, dot, normalize, select, PI, getViewPosition, isBackgroundDepth, passTexture, context, log2, exp2 } from 'three/tsl';

import { VXGIVolume } from './vxgi/VXGIVolume.js';
import { intersectVolume, sampleDirectional } from './vxgi/VXGIConeTracer.js';

const _size = new Vector2();
const TILE_SIZE = 6;

/**
 * Experimental diffuse radiance cascades. Screen-space probes trace finite
 * world-space intervals through a VXGI voxel volume. Optional filtered tracing
 * and surface-aware reconstruction reduce grid artifacts.
 * Directions use nested equal-area azimuth/height bins, not Fibonacci indices.
 * No temporal accumulation, software BVH, or additional bounce feedback is used.
 *
 * Requires a WebGPU backend and an ordinary perspective depth buffer.
 * Screen-space probe interpolation remains approximate at depth discontinuities.
 *
 * @augments Node
 * @three_import import { radianceCascades } from 'three/addons/lighting/RadianceCascadesNode.js';
 */
class RadianceCascadesNode extends Node {

	static get type() {

		return 'RadianceCascadesNode';

	}

	/**
	 * @param {TextureNode} depthNode - Scene depth, before GI.
	 * @param {Node} normalNode - View-space normal texture node.
	 * @param {Scene} scene - Scene to voxelize.
	 * @param {Camera} camera - Perspective camera.
	 * @param {Object} [options={}] - Configuration.
	 * @param {?VXGIVolume} [options.volume=null] - Borrowed volume; not disposed by this node.
	 * @param {number} [options.resolution=64] - Voxel resolution when creating a volume.
	 * @param {number} [options.probeSpacing=16] - Base probe spacing in drawing-buffer pixels.
	 * @param {number} [options.cascadeCount=4] - Number of distance intervals, between two and five.
	 */
	constructor( depthNode, normalNode, scene, camera, options = {} ) {

		super( 'vec4' );

		const { volume = null, resolution = 64, probeSpacing = 16, cascadeCount = 4 } = options;

		if ( ! Number.isInteger( probeSpacing ) || probeSpacing < 4 || ! Number.isInteger( cascadeCount ) || cascadeCount < 2 || cascadeCount > 5 ) {

			throw new RangeError( 'RadianceCascadesNode: Invalid probe spacing or cascade count.' );

		}

		this.depthNode = depthNode;
		this.normalNode = normalNode;
		this.scene = scene;
		this.camera = camera;
		this.volume = volume || new VXGIVolume( resolution );
		this._ownsVolume = volume === null;
		if ( this._ownsVolume ) this.volume.bounces = 0;

		this.probeSpacing = probeSpacing;
		/** Angular resolution multiplier: 1 = 32 base directions, 2 = 128. */
		this.directionResolution = 1;
		this._directionResolution = null;
		this.cascadeCount = cascadeCount;
		this.intervalLength = uniform( 0.3 );
		this.normalOffset = uniform( 1.5 );
		this.giIntensity = uniform( 1 );
		/** Integer uniform: 0 = binary DDA, 1 = trilinear mip zero, 2 = angular-footprint mips. */
		this.traceMode = uniform( 0, 'int' );
		/** Diameter multiplier for angular-footprint filtering. */
		this.filterScale = uniform( 1 );
		/** Boolean uniform: use a positive cubic B-spline over 4×4 surface-aware probes. */
		this.smoothReconstruction = uniform( false );
		/** Boolean uniform: retrace to displaced upper interval starts before composition. */
		this.connectIntervals = uniform( false );
		/** Boolean uniform: trace a receiver hemisphere when no compatible probe exists. */
		this.fillMissingProbes = uniform( false );
		this.updateBeforeType = NodeUpdateType.FRAME;

		this._projectionInverse = uniform( camera.projectionMatrixInverse );
		this._cameraWorld = uniform( camera.matrixWorld );
		this._grid = uniform( new Vector2( 1, 1 ) );
		this._quad = new QuadMesh();
		this._material = new NodeMaterial();
		this._material.name = 'RadianceCascades.Gather';
		this._target = new RenderTarget( 1, 1, { depthBuffer: false, type: HalfFloatType } );
		this._target.texture.name = 'RadianceCascades.Irradiance';
		this._giNode = passTexture( this, this._target.texture );
		this._irradiance = this._createTexture( 'RadianceCascades.Prefilter' );
		this.cascades = Array.from( { length: cascadeCount }, ( _, level ) => ( {
			level,
			width: 1,
			height: 1,
			azimuths: 8 * 2 ** level,
			elevations: 4 * 2 ** level,
			intervals: this._createTexture( `RadianceCascades.Interval${ level }` ),
			merged: this._createTexture( `RadianceCascades.Merged${ level }` )
		} ) );
		this._kernels = null;
		this._sharedContext = null;

	}

	_createTexture( name ) {

		const result = new StorageTexture();
		result.type = HalfFloatType;
		result.name = name;
		result.generateMipmaps = false;
		return result;

	}

	getGINode() {

		return this._giNode;

	}

	/** Number of bytes in interval, merged and irradiance textures, excluding the shared volume. */
	get memoryBytes() {

		return this.cascades.reduce( ( total, c ) => total + c.width * c.height * c.azimuths * c.elevations * 16, 0 ) + this._grid.value.x * this._grid.value.y * TILE_SIZE ** 2 * 8 + this._target.width * this._target.height * 8;

	}

	setSize( width, height ) {

		if ( this.directionResolution !== 1 && this.directionResolution !== 2 ) {

			throw new RangeError( 'RadianceCascadesNode: Direction resolution must be 1 or 2.' );

		}

		const gridWidth = Math.max( 1, Math.ceil( width / this.probeSpacing ) );
		const gridHeight = Math.max( 1, Math.ceil( height / this.probeSpacing ) );
		this._target.setSize( width, height );

		if ( this._grid.value.x === gridWidth && this._grid.value.y === gridHeight && this._kernels !== null && this._directionResolution === this.directionResolution ) return;

		this._disposeKernels();
		this._directionResolution = this.directionResolution;
		this._grid.value.set( gridWidth, gridHeight );
		this._irradiance.setSize( gridWidth * TILE_SIZE, gridHeight * TILE_SIZE );

		for ( const c of this.cascades ) {

			c.azimuths = 8 * this.directionResolution * 2 ** c.level;
			c.elevations = 4 * this.directionResolution * 2 ** c.level;
			c.width = Math.max( 1, Math.ceil( gridWidth / 2 ** c.level ) );
			c.height = Math.max( 1, Math.ceil( gridHeight / 2 ** c.level ) );
			c.intervals.setSize( c.width * c.azimuths, c.height * c.elevations );
			c.merged.setSize( c.width * c.azimuths, c.height * c.elevations );

		}

	}

	// Equal-area spherical cells: doubling both axes creates exactly four children.
	_direction( coords, azimuths, elevations ) {

		const y = float( 1 ).sub( float( coords.y ).add( 0.5 ).mul( 2 / elevations ) ).toConst();
		const radius = y.mul( y ).oneMinus().max( 0 ).sqrt().toConst();
		const phi = float( coords.x ).add( 0.5 ).mul( 2 * Math.PI / azimuths ).toConst();
		return vec3( radius.mul( phi.cos() ), y, radius.mul( phi.sin() ) );

	}

	_surface( coords, width, height ) {

		const sampleUV = vec2( coords ).add( 0.5 ).div( vec2( width, height ) ).toConst();
		const depth = this.depthNode.sample( sampleUV ).r.toConst();
		const viewPosition = getViewPosition( sampleUV, depth, this._projectionInverse ).toConst();
		const position = this._cameraWorld.mul( vec4( viewPosition, 1 ) ).xyz.toConst();
		const normal = normalize( this._cameraWorld.mul( vec4( this.normalNode.sample( sampleUV ).rgb, 0 ) ).xyz ).toConst();
		const origin = position.add( normal.mul( this.volume.voxelSizeNode.mul( this.normalOffset ) ) ).toConst();
		return { position, normal, origin, valid: isBackgroundDepth( depth ).not() };

	}

	// Filter the scene representation, not the final shaded image. Integrate fractional
	// directional opacity front-to-back, correcting it for the actual step length.
	_traceFiltered( origin, direction, start, end, rayCount ) {

		const volume = this.volume;
		const { tEnter, tExit } = intersectVolume( volume, origin, direction );
		const t = max( start, tEnter ).toVar();
		const limit = min( end, tExit ).toConst();
		const result = vec4( 0, 0, 0, 1 ).toVar();
		const directionWeights = direction.mul( direction ).toConst();

		Loop( { start: 0, end: 768, type: 'int', name: 'filteredStep' }, () => {

			If( t.greaterThanEqual( limit ).or( result.a.lessThan( 0.01 ) ), () => Break() );
			// Equal-area angular cell represented as a disk at distance t.
			const diameter = max( volume.voxelSizeNode, t.mul( 2 * Math.sqrt( Math.PI / rayCount ) ).mul( this.filterScale ) ).toConst();
			const lod = select( this.traceMode.equal( 2 ), log2( diameter.div( volume.voxelSizeNode ) ).clamp( 0, volume.maxLevelNode ), float( 0 ) ).toConst();
			const texelSize = volume.voxelSizeNode.mul( exp2( lod ) ).toConst();
			const step = min( texelSize.mul( 0.5 ), limit.sub( t ) ).toConst();
			const uvw = origin.add( direction.mul( t.add( step.mul( 0.5 ) ) ) ).sub( volume.boundsMinNode ).div( volume.volumeSizeNode ).toConst();
			const opacity = dot( volume.opacityNode.sample( uvw ).level( lod ).xyz, directionWeights ).clamp( 0, 1 ).toConst();
			const alpha = opacity.oneMinus().pow( step.div( texelSize ) ).oneMinus().toConst();
			const radiance = volume.radianceNode.sample( uvw ).level( lod ).toVar();

			if ( volume.directionalRadiance ) {

				If( lod.greaterThanEqual( 1 ), () => {

					radiance.assign( sampleDirectional( volume, volume.directionalNode, uvw, lod.sub( 1 ), direction ) );

				} );

			}

			result.rgb.addAssign( radiance.rgb.div( radiance.a.max( 1e-4 ) ).mul( alpha.mul( result.a ) ) );
			result.a.mulAssign( alpha.oneMinus() );
			t.addAssign( step );

		} );
		return result;

	}

	_trace( origin, direction, start, end, rayCount ) {

		const result = vec4( 0, 0, 0, 1 ).toVar();
		If( this.traceMode.equal( 0 ), () => {

			result.assign( this._traceInterval( origin, direction, start, end ) );

		} ).Else( () => {

			result.assign( this._traceFiltered( origin, direction, start, end, rayCount ) );

		} );
		return result;

	}

	// Exact cell stepping at mip zero. Traversal never substitutes cone mip samples.
	_traceInterval( origin, direction, start, end ) {

		const volume = this.volume;
		const { tEnter, tExit } = intersectVolume( volume, origin, direction );
		const t = max( start, tEnter ).add( volume.voxelSizeNode.mul( 1e-4 ) ).toVar();
		const limit = min( end, tExit ).toConst();
		const size = volume.volumeSizeNode.div( volume.voxelSizeNode ).round().toConst();
		const result = vec4( 0, 0, 0, 1 ).toVar();
		const safeDirection = vec3(
			select( abs( direction.x ).lessThan( 1e-6 ), 1e-6, direction.x ),
			select( abs( direction.y ).lessThan( 1e-6 ), 1e-6, direction.y ),
			select( abs( direction.z ).lessThan( 1e-6 ), 1e-6, direction.z )
		).toConst();

		Loop( { start: 0, end: 768, type: 'int', name: 'voxelStep' }, () => {

			If( t.greaterThanEqual( limit ), () => Break() );
			const p = origin.add( direction.mul( t ) ).sub( volume.boundsMinNode ).div( volume.voxelSizeNode ).toConst();
			const cell = floor( p ).toConst();
			If( cell.lessThan( vec3( 0 ) ).any().or( cell.greaterThanEqual( size ).any() ), () => Break() );
			const opacity = volume.opacityNode.load( ivec3( cell ) ).level( 0 ).a.toConst();

			If( opacity.greaterThan( 0.1 ), () => {

				const radiance = volume.radianceNode.load( ivec3( cell ) ).level( 0 ).toConst();
				result.assign( vec4( radiance.rgb.div( max( radiance.a, 1e-4 ) ), 0 ) );
				Break();

			} );

			const boundary = cell.add( select( direction.greaterThan( vec3( 0 ) ), vec3( 1 ), vec3( 0 ) ) );
			const distance = boundary.sub( p ).mul( volume.voxelSizeNode ).div( safeDirection ).toConst();
			const next = min( min( distance.x, distance.y ), distance.z ).toConst();
			t.addAssign( max( next, 0 ).add( volume.voxelSizeNode.mul( 1e-4 ) ) );

		} );

		return result;

	}

	_createKernels() {

		const kernels = [];

		for ( const c of this.cascades ) {

			const width = c.width * c.azimuths;
			const count = width * c.height * c.elevations;
			const kernel = Fn( () => {

				If( instanceIndex.greaterThanEqual( count ), () => {

					Return();

				} );
				const pixel = ivec2( instanceIndex.mod( width ), instanceIndex.div( width ) ).toConst();
				const probe = pixel.div( ivec2( c.azimuths, c.elevations ) ).toConst();
				const angular = pixel.mod( ivec2( c.azimuths, c.elevations ) ).toConst();
				const surface = this._surface( probe, c.width, c.height );
				const result = vec4( 0, 0, 0, 1 ).toVar();
				const start = this.intervalLength.mul( ( 4 ** c.level - 1 ) / 3 ).toConst();
				const end = this.intervalLength.mul( ( 4 ** ( c.level + 1 ) - 1 ) / 3 ).toConst();

				If( surface.valid, () => {

					result.assign( this._trace( surface.origin, this._direction( angular, c.azimuths, c.elevations ), start, end, c.azimuths * c.elevations ) );

				} );

				textureStore( c.intervals, pixel, result ).toWriteOnly();

			} )().compute( count );
			kernel.setName( `RadianceCascades.Trace${ c.level }` );
			kernels.push( kernel );

		}

		for ( let level = this.cascadeCount - 1; level >= 0; level -- ) {

			const c = this.cascades[ level ];
			const parent = this.cascades[ level + 1 ];
			const width = c.width * c.azimuths;
			const count = width * c.height * c.elevations;
			const kernel = Fn( () => {

				const pixel = ivec2( instanceIndex.mod( width ), instanceIndex.div( width ) ).toConst();
				If( instanceIndex.greaterThanEqual( count ), () => Return() );
				const local = texture( c.intervals ).load( pixel ).toConst();
				const result = local.toVar();

				if ( parent ) {

					const probe = pixel.div( ivec2( c.azimuths, c.elevations ) ).toConst();
					const angular = pixel.mod( ivec2( c.azimuths, c.elevations ) ).toConst();
					const surface = this._surface( probe, c.width, c.height );
					const coord = vec2( probe ).add( 0.5 ).div( vec2( c.width, c.height ) ).mul( vec2( parent.width, parent.height ) ).sub( 0.5 ).toConst();
					const blend = fract( coord ).toConst();
					const far = vec4( 0 ).toVar();
					const connected = vec4( 0 ).toVar();
					const weights = float( 0 ).toVar();

					If( local.a.greaterThan( 0 ).or( this.connectIntervals ).and( surface.valid ), () => {

						for ( let y = 0; y < 2; y ++ ) {

							for ( let x = 0; x < 2; x ++ ) {

								const neighbor = ivec2( floor( coord ) ).add( ivec2( x, y ) ).clamp( ivec2( 0 ), ivec2( parent.width - 1, parent.height - 1 ) ).toConst();
								const nextSurface = this._surface( neighbor, parent.width, parent.height );
								const weight = ( x ? blend.x : blend.x.oneMinus() ).mul( y ? blend.y : blend.y.oneMinus() ).toConst();
								const radius = this.intervalLength.mul( ( 4 ** ( level + 1 ) - 1 ) / 3 ).max( this.volume.voxelSizeNode );
								const separation = nextSurface.position.sub( surface.position ).length();
								const compatible = nextSurface.valid.and( dot( nextSurface.normal, surface.normal ).greaterThan( 0.5 ) ).and( separation.lessThan( radius.mul( 2 ) ) );

								If( compatible, () => {

									const angularSum = vec4( 0 ).toVar();
									for ( let v = 0; v < 2; v ++ ) {

										for ( let u = 0; u < 2; u ++ ) {

											const child = angular.mul( 2 ).add( ivec2( u, v ) );
											angularSum.addAssign( texture( parent.merged ).load( neighbor.mul( ivec2( parent.azimuths, parent.elevations ) ).add( child ) ) );

										}

									}

									const upper = angularSum.mul( 0.25 ).toConst();
									If( this.connectIntervals, () => {

										const direction = this._direction( angular, c.azimuths, c.elevations ).toConst();
										const start = this.intervalLength.mul( ( 4 ** level - 1 ) / 3 );
										const a = surface.origin.add( direction.mul( start ) ).toConst();
										const b = nextSurface.origin.add( direction.mul( this.intervalLength.mul( ( 4 ** ( level + 1 ) - 1 ) / 3 ) ) ).toConst();
										const segment = b.sub( a ).toConst();
										const near = this._trace( a, segment.div( segment.length().max( 1e-5 ) ), float( 0 ), segment.length(), c.azimuths * c.elevations );
										connected.addAssign( vec4( near.rgb.add( upper.rgb.mul( near.a ) ), near.a.mul( upper.a ) ).mul( weight ) );

									} );
									far.addAssign( upper.mul( weight ) );
									weights.addAssign( weight );

								} );

							}

						}

						If( weights.greaterThan( 1e-5 ), () => {

							far.divAssign( weights );
							result.assign( select( this.connectIntervals, connected.div( weights ), vec4( local.rgb.add( far.rgb.mul( local.a ) ), local.a.mul( far.a ) ) ) );

						} );

					} );

				}

				textureStore( c.merged, pixel, result ).toWriteOnly();

			} )().compute( count );
			kernel.setName( `RadianceCascades.Merge${ level }` );
			kernels.push( kernel );

		}

		const c0 = this.cascades[ 0 ];
		const width = c0.width * TILE_SIZE;
		const prefilter = Fn( () => {

			If( instanceIndex.greaterThanEqual( width * c0.height * TILE_SIZE ), () => Return() );
			const pixel = ivec2( instanceIndex.mod( width ), instanceIndex.div( width ) ).toConst();
			const probe = pixel.div( TILE_SIZE ).toConst();
			const oct = vec2( pixel.mod( TILE_SIZE ) ).div( TILE_SIZE - 1 ).mul( 2 ).sub( 1 ).toConst();
			const normal = vec3( oct, float( 1 ).sub( abs( oct.x ) ).sub( abs( oct.y ) ) ).toVar();
			If( normal.z.lessThan( 0 ), () => {

				normal.xy.assign( normal.yx.abs().oneMinus().mul( select( normal.xy.greaterThanEqual( vec2( 0 ) ), vec2( 1 ), vec2( - 1 ) ) ) );

			} );
			normal.assign( normalize( normal ) );
			const irradiance = vec3( 0 ).toVar();

			Loop( c0.azimuths * c0.elevations, ( { i } ) => {

				const angular = ivec2( i.mod( c0.azimuths ), i.div( c0.azimuths ) );
				const radiance = texture( c0.merged ).load( probe.mul( ivec2( c0.azimuths, c0.elevations ) ).add( angular ) ).rgb;
				irradiance.addAssign( radiance.mul( dot( normal, this._direction( angular, c0.azimuths, c0.elevations ) ).max( 0 ) ) );

			} );
			textureStore( this._irradiance, pixel, vec4( irradiance.mul( PI.mul( 4 / ( c0.azimuths * c0.elevations ) ) ), 1 ) ).toWriteOnly();

		} )().compute( width * c0.height * TILE_SIZE );
		prefilter.setName( 'RadianceCascades.Prefilter' );
		kernels.push( prefilter );
		this._kernels = kernels;

	}

	setup( builder ) {

		if ( builder.renderer.backend.isWebGPUBackend !== true || builder.renderer.logarithmicDepthBuffer === true || this.camera.isPerspectiveCamera !== true ) {

			throw new Error( 'RadianceCascadesNode: Requires WebGPU and ordinary perspective depth.' );

		}

		this._sharedContext = builder.getSharedContext();
		this._material.contextNode = context( this._sharedContext );
		this._material.fragmentNode = Fn( () => {

			const sampleUV = uv();
			const depth = this.depthNode.sample( sampleUV ).r.toConst();
			isBackgroundDepth( depth ).discard();
			const p = this._cameraWorld.mul( vec4( getViewPosition( sampleUV, depth, this._projectionInverse ), 1 ) ).xyz.toConst();
			const normal = normalize( this._cameraWorld.mul( vec4( this.normalNode.sample( sampleUV ).rgb, 0 ) ).xyz ).toConst();
			const oct = normal.div( abs( normal.x ).add( abs( normal.y ) ).add( abs( normal.z ) ) ).xy.toVar();
			If( normal.z.lessThan( 0 ), () => {

				oct.assign( oct.yx.abs().oneMinus().mul( select( oct.greaterThanEqual( vec2( 0 ) ), vec2( 1 ), vec2( - 1 ) ) ) );

			} );
			const octUV = oct.mul( 0.5 ).add( 0.5 ).toConst();
			const coord = sampleUV.mul( this._grid ).sub( 0.5 ).toConst();
			const blend = fract( coord ).toConst();
			const color = vec3( 0 ).toVar();
			const weights = float( 0 ).toVar();

			const spline = ( distance ) => {

				const d = abs( distance );
				return select( d.lessThan( 1 ), float( 2 / 3 ).sub( d.mul( d ) ).add( d.pow( 3 ).mul( 0.5 ) ), float( 2 ).sub( d ).max( 0 ).pow( 3 ).div( 6 ) );

			};

			for ( let y = - 1; y <= 2; y ++ ) {

				for ( let x = - 1; x <= 2; x ++ ) {

					const bilinear = ( x === 0 ? blend.x.oneMinus() : x === 1 ? blend.x : float( 0 ) ).mul( y === 0 ? blend.y.oneMinus() : y === 1 ? blend.y : float( 0 ) );
					const spatialWeight = select( this.smoothReconstruction, spline( blend.x.sub( x ) ).mul( spline( blend.y.sub( y ) ) ), bilinear ).toConst();
					If( spatialWeight.greaterThan( 0 ), () => {

						const probe = floor( coord ).add( vec2( x, y ) ).clamp( vec2( 0 ), this._grid.sub( 1 ) ).toConst();
						const surface = this._surface( probe, this._grid.x, this._grid.y );
						const delta = surface.position.sub( p ).toConst();
						// Plane distance accepts distant samples on the same wall, but rejects
						// parallel surfaces at another depth. Smooth normal weights avoid a cutoff.
						const normalWeight = dot( normal, surface.normal ).sub( 0.5 ).max( 0 ).mul( 2 ).pow( 2 );
						const planeWeight = float( 1 ).div( float( 1 ).add( dot( delta, normal ).div( this.volume.voxelSizeNode.max( 1e-4 ) ).pow( 2 ).mul( 16 ) ) );
						const originalWeight = select( dot( normal, surface.normal ).greaterThan( 0.8 ), float( 1 ).div( float( 1 ).add( delta.length().div( this.intervalLength.max( 1e-4 ) ).pow( 2 ) ) ), float( 0 ) );
						const weight = spatialWeight.mul( select( this.smoothReconstruction, normalWeight.mul( planeWeight ), originalWeight ) ).toConst();

						// Reject a different parallel surface before normalizing weights;
						// tiny plane weights alone would still borrow its full irradiance.
						const samePlane = abs( dot( delta, normal ) ).lessThan( this.volume.voxelSizeNode.mul( 0.75 ) );
						If( surface.valid.and( this.smoothReconstruction.not().or( samePlane ) ), () => {

							const atlasUV = probe.mul( TILE_SIZE ).add( 0.5 ).add( octUV.mul( TILE_SIZE - 1 ) ).div( this._grid.mul( TILE_SIZE ) );
							color.addAssign( texture( this._irradiance ).sample( atlasUV ).rgb.mul( weight ) );
							weights.addAssign( weight );

						} );

					} );

				}

			}

			const irradiance = color.div( weights.max( 1e-5 ) ).toVar();
			// Sparse screen probes can completely miss narrow upward-facing surfaces.
			// Only unsupported receivers pay for this full-range reference gather.
			If( this.fillMissingProbes.and( weights.lessThan( 1e-5 ) ), () => {

				const origin = p.add( normal.mul( this.volume.voxelSizeNode.mul( this.normalOffset ) ) ).toConst();
				const end = this.intervalLength.mul( ( 4 ** this.cascadeCount - 1 ) / 3 ).toConst();
				const count = 32;
				Loop( count, ( { i } ) => {

					const direction = this._direction( ivec2( i.mod( 8 ), i.div( 8 ) ), 8, 4 ).toConst();
					const cosine = dot( normal, direction ).max( 0 ).toConst();
					If( cosine.greaterThan( 0 ), () => {

						const radiance = this._trace( origin, direction, float( 0 ), end, count );
						irradiance.addAssign( radiance.rgb.mul( cosine.mul( PI.mul( 4 / count ) ) ) );

					} );

				} );

			} );
			return vec4( irradiance.mul( this.giIntensity ), 1 );

		} )();
		this._material.needsUpdate = true;
		return this._giNode;

	}

	updateBefore( frame ) {

		const { renderer } = frame;
		const state = RendererUtils.resetRendererState( renderer );
		const previousContext = renderer.contextNode;
		try {

			renderer.contextNode = context();
			// Populate current depth, normals and shadow maps before light injection.
			if ( this.depthNode.isPassTextureNode ) frame.updateBeforeNode( this.depthNode.passNode );
			renderer.getDrawingBufferSize( _size );
			this.setSize( _size.x, _size.y );
			this.volume.update( renderer, this.scene );
			if ( this._kernels === null ) this._createKernels();
			for ( const kernel of this._kernels ) renderer.compute( kernel );
			renderer.setRenderTarget( this._target );
			renderer.setClearColor( 0, 1 );
			this._quad.material = this._material;
			this._quad.render( renderer );

		} finally {

			renderer.contextNode = previousContext;
			RendererUtils.restoreRendererState( renderer, state );

		}

	}

	_disposeKernels() {

		if ( this._kernels ) for ( const kernel of this._kernels ) kernel.dispose();
		this._kernels = null;

	}

	dispose() {

		super.dispose();
		this._disposeKernels();
		for ( const c of this.cascades ) {

			c.intervals.dispose();
			c.merged.dispose();

		}

		this._irradiance.dispose();
		this._target.dispose();
		this._material.dispose();
		if ( this._ownsVolume ) this.volume.dispose();

	}

}

export default RadianceCascadesNode;

export const radianceCascades = ( depthNode, normalNode, scene, camera, options ) => new RadianceCascadesNode( depthNode, normalNode, scene, camera, options );
