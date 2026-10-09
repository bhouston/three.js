import { BackSide, Color, CustomBlending, DoubleSide, HalfFloatType, Mesh, NodeMaterial, OneFactor, OrthographicCamera, QuadMesh, RenderTarget, Scene, Vector2 } from 'three/webgpu';
import { Fn, If, Loop, ivec2, normalWorldGeometry, positionWorld, screenCoordinate, textureLoad, uniform, uv, vec4 } from 'three/tsl';

const _clearColor = /*@__PURE__*/ new Color();

/**
 * Bakes indirect diffuse irradiance into a light map. The irradiance is defined by a
 * function that returns a TSL node for a world-space position and normal, so any
 * global illumination technique can be baked, e.g. a {@link LightProbeGrid} via
 * `lightProbeGridIrradiance()`.
 *
 * Meshes need light map UVs in the `uv1` attribute, e.g. generated with {@link UVUnwrapper}.
 * Each sample rasterizes the meshes in UV space with a sub-texel jitter and accumulates
 * the result. Call `update()` once per frame to bake progressively, or pass a sample
 * count to bake at once.
 *
 * ```js
 * const lightMapper = new LightMapper( renderer, size, ( position, normal ) => lightProbeGridIrradiance( probes, position, normal ) );
 * lightMapper.addObjects( meshes );
 * lightMapper.update( 16 );
 *
 * material.lightMap = lightMapper.texture;
 * ```
 *
 * This class can only be used with {@link WebGPURenderer}.
 *
 * @three_import import { LightMapper } from 'three/addons/lighting/LightMapper.js';
 */
class LightMapper {

	/**
	 * Constructs a new light mapper.
	 *
	 * @param {WebGPURenderer} renderer - The renderer.
	 * @param {number} size - The light map size in texels.
	 * @param {function(Node<vec3>, Node<vec3>): Node<vec3>} irradianceNode - Returns the irradiance for a world-space position and normal.
	 */
	constructor( renderer, size, irradianceNode ) {

		/**
		 * The renderer.
		 *
		 * @type {WebGPURenderer}
		 */
		this.renderer = renderer;

		/**
		 * The light map size in texels.
		 *
		 * @type {number}
		 */
		this.size = size;

		/**
		 * The number of accumulated samples.
		 *
		 * @type {number}
		 * @readonly
		 * @default 0
		 */
		this.samples = 0;

		this._accumulation = new RenderTarget( size, size, { type: HalfFloatType, depthBuffer: false } );
		this._renderTarget = new RenderTarget( size, size, { type: HalfFloatType, depthBuffer: false } );
		this._renderTarget.texture.channel = 1;

		/**
		 * The baked light map. It stores irradiance and uses the `uv1` attribute.
		 *
		 * @type {Texture}
		 */
		this.texture = this._renderTarget.texture;

		this._scene = new Scene();
		this._camera = new OrthographicCamera();
		this._objects = [];
		this._jitter = uniform( new Vector2() );

		this._frontMaterial = this._createBakeMaterial( irradianceNode, false );
		this._backMaterial = this._createBakeMaterial( irradianceNode, true );

		// normalize the accumulated samples and fill the chart padding

		const accumulation = this._accumulation.texture;
		const maxCoord = ivec2( size - 1 );

		const resolve = Fn( () => {

			const coord = ivec2( screenCoordinate.xy );
			const center = textureLoad( accumulation, coord ).toVar();
			const sum = vec4( 0 ).toVar();

			If( center.a.equal( 0 ), () => {

				Loop( { start: - 2, end: 2, condition: '<=', name: 'x' }, { start: - 2, end: 2, condition: '<=', name: 'y' }, ( { x, y } ) => {

					const value = textureLoad( accumulation, coord.add( ivec2( x, y ) ).clamp( ivec2( 0 ), maxCoord ) );

					If( value.a.greaterThan( 0 ), () => {

						sum.addAssign( vec4( value.rgb.div( value.a ), 1 ) );

					} );

				} );

				center.assign( sum );

			} );

			return vec4( center.rgb.div( center.a.max( 1e-4 ) ), 1 );

		} );

		const resolveMaterial = new NodeMaterial();
		resolveMaterial.fragmentNode = resolve();

		this._quad = new QuadMesh( resolveMaterial );

	}

	/**
	 * Adds meshes to the light map. Their geometries must provide `uv1`.
	 *
	 * @param {Array<Mesh>} objects - The meshes to add.
	 */
	addObjects( objects ) {

		for ( const object of objects ) {

			const material = Array.isArray( object.material ) ? object.material[ 0 ] : object.material;

			const proxy = new Mesh( object.geometry, material.side === BackSide ? this._backMaterial : this._frontMaterial );
			proxy.matrixAutoUpdate = false;
			proxy.frustumCulled = false;

			this._scene.add( proxy );
			this._objects.push( object );

		}

	}

	/**
	 * Accumulates samples into the light map.
	 *
	 * @param {number} [samples=1] - The number of samples to add.
	 */
	update( samples = 1 ) {

		const renderer = this.renderer;

		const renderTarget = renderer.getRenderTarget();
		const autoClear = renderer.autoClear;
		const clearAlpha = renderer.getClearAlpha();
		renderer.getClearColor( _clearColor );

		renderer.autoClear = false;
		renderer.setClearColor( 0x000000, 0 );
		renderer.setRenderTarget( this._accumulation );

		if ( this.samples === 0 ) renderer.clear();

		for ( let i = 0; i < this._objects.length; i ++ ) {

			this._scene.children[ i ].matrix.copy( this._objects[ i ].matrixWorld );

		}

		for ( let i = 0; i < samples; i ++ ) {

			this.samples ++;

			this._jitter.value.set( halton( this.samples, 2 ) - 0.5, halton( this.samples, 3 ) - 0.5 ).multiplyScalar( 2 / this.size );

			renderer.render( this._scene, this._camera );

		}

		renderer.setRenderTarget( this._renderTarget );
		this._quad.render( renderer );

		renderer.setRenderTarget( renderTarget );
		renderer.setClearColor( _clearColor, clearAlpha );
		renderer.autoClear = autoClear;

	}

	/**
	 * Discards the accumulated samples.
	 */
	reset() {

		this.samples = 0;

	}

	/**
	 * Frees all internal resources.
	 */
	dispose() {

		this._accumulation.dispose();
		this._renderTarget.dispose();
		this._frontMaterial.dispose();
		this._backMaterial.dispose();
		this._quad.material.dispose();

	}

	/**
	 * Creates a material that rasterizes meshes in UV space and outputs irradiance.
	 *
	 * @private
	 * @param {function(Node<vec3>, Node<vec3>): Node<vec3>} irradianceNode - The irradiance function.
	 * @param {boolean} flipNormal - Whether to flip the normal, for back-sided meshes.
	 * @return {NodeMaterial} The material.
	 */
	_createBakeMaterial( irradianceNode, flipNormal ) {

		const material = new NodeMaterial();
		material.side = DoubleSide;
		material.depthTest = false;
		material.depthWrite = false;

		// additive blending, the sample count accumulates in alpha

		material.blending = CustomBlending;
		material.blendSrc = OneFactor;
		material.blendDst = OneFactor;

		material.vertexNode = vec4( uv( 1 ).flipY().mul( 2 ).sub( 1 ).add( this._jitter ), 0, 1 );

		const normal = flipNormal ? normalWorldGeometry.negate() : normalWorldGeometry;

		material.fragmentNode = vec4( irradianceNode( positionWorld, normal ), 1 );

		return material;

	}

}

function halton( index, base ) {

	let result = 0;
	let f = 1;

	while ( index > 0 ) {

		f /= base;
		result += f * ( index % base );
		index = Math.floor( index / base );

	}

	return result;

}

export { LightMapper };
