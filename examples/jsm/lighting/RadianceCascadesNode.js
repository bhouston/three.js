import { Node, NodeUpdateType, NodeMaterial, QuadMesh, RenderTarget, StorageTexture, HalfFloatType, Vector2, RendererUtils } from 'three/webgpu';
import { Fn, If, Loop, Break, Return, uniform, texture, textureStore, instanceIndex, float, ivec2, ivec3, vec2, vec3, vec4, uv, floor, fract, min, max, abs, dot, normalize, select, PI, getViewPosition, isBackgroundDepth, passTexture, context } from 'three/tsl';

import { VXGIVolume } from './vxgi/VXGIVolume.js';
import { intersectVolume } from './vxgi/VXGIConeTracer.js';

const _size = new Vector2();
const TILE_SIZE = 6;

/**
 * Experimental diffuse radiance cascades. Screen-space probes trace finite
 * world-space intervals through the finest level of a VXGI voxel volume.
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
		this.cascadeCount = cascadeCount;
		this.intervalLength = uniform( 0.3 );
		this.normalOffset = uniform( 1.5 );
		this.giIntensity = uniform( 1 );
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

		const gridWidth = Math.max( 1, Math.ceil( width / this.probeSpacing ) );
		const gridHeight = Math.max( 1, Math.ceil( height / this.probeSpacing ) );
		this._target.setSize( width, height );

		if ( this._grid.value.x === gridWidth && this._grid.value.y === gridHeight && this._kernels !== null ) return;

		this._disposeKernels();
		this._grid.value.set( gridWidth, gridHeight );
		this._irradiance.setSize( gridWidth * TILE_SIZE, gridHeight * TILE_SIZE );

		for ( const c of this.cascades ) {

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

					result.assign( this._traceInterval( surface.origin, this._direction( angular, c.azimuths, c.elevations ), start, end ) );

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
					const weights = float( 0 ).toVar();

					If( local.a.greaterThan( 0 ).and( surface.valid ), () => {

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

									far.addAssign( angularSum.mul( weight.mul( 0.25 ) ) );
									weights.addAssign( weight );

								} );

							}

						}

						If( weights.greaterThan( 1e-5 ), () => {

							far.divAssign( weights );
							result.assign( vec4( local.rgb.add( far.rgb.mul( local.a ) ), local.a.mul( far.a ) ) );

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

			Loop( 32, ( { i } ) => {

				const angular = ivec2( i.mod( 8 ), i.div( 8 ) );
				const radiance = texture( c0.merged ).load( probe.mul( ivec2( 8, 4 ) ).add( angular ) ).rgb;
				irradiance.addAssign( radiance.mul( dot( normal, this._direction( angular, 8, 4 ) ).max( 0 ) ) );

			} );
			textureStore( this._irradiance, pixel, vec4( irradiance.mul( PI.mul( 4 / 32 ) ), 1 ) ).toWriteOnly();

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

			for ( let y = 0; y < 2; y ++ ) {

				for ( let x = 0; x < 2; x ++ ) {

					const probe = floor( coord ).add( vec2( x, y ) ).clamp( vec2( 0 ), this._grid.sub( 1 ) ).toConst();
					const surface = this._surface( probe, this._grid.x, this._grid.y );
					const compatible = surface.valid.and( dot( normal, surface.normal ).greaterThan( 0.8 ) );
					const weight = ( x ? blend.x : blend.x.oneMinus() ).mul( y ? blend.y : blend.y.oneMinus() ).div( float( 1 ).add( surface.position.sub( p ).length().div( this.intervalLength.max( 1e-4 ) ).pow( 2 ) ) ).toConst();

					If( compatible, () => {

						const atlasUV = probe.mul( TILE_SIZE ).add( 0.5 ).add( octUV.mul( TILE_SIZE - 1 ) ).div( this._grid.mul( TILE_SIZE ) );
						color.addAssign( texture( this._irradiance ).sample( atlasUV ).rgb.mul( weight ) );
						weights.addAssign( weight );

					} );

				}

			}

			return vec4( color.div( weights.max( 1e-5 ) ).mul( this.giIntensity ), 1 );

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
