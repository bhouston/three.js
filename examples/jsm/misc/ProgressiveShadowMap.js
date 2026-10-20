import { BackSide, Color, DoubleSide, HalfFloatType, Mesh, NodeMaterial, QuadMesh, RenderTarget, Scene, Vector2, Vector3 } from 'three/webgpu';
import { Fn, If, Discard, mix, normalViewGeometry, screenCoordinate, shadow, textureLoad, uniform, uv, vec2, vec4 } from 'three/tsl';

/**
 * Accumulates directional shadow visibility in a pre-unwrapped uv1 atlas.
 * Static, opaque meshes only. The first mesh is planar ground in the left
 * atlas half; the other meshes occupy the right half. Call reset() after
 * geometry, transforms or light
 * settings change. Camera movement does not invalidate texture-space history.
 * The light must be a separate sampling light, not the scene's shading light.
 *
 * @three_import import { ProgressiveShadowMap } from 'three/addons/misc/ProgressiveShadowMap.js';
 */
class ProgressiveShadowMap {

	/**
	 * @param {WebGPURenderer} renderer - Renderer with shadows enabled.
	 * @param {Array<Mesh>} objects - Casters and receivers with unique uv1 coordinates.
	 * @param {DirectionalLight} light - Shadow sampling light.
	 * @param {number} [resolution=2048] - Height of the 2:1 atlas.
	 */
	constructor( renderer, objects, light, resolution = 2048 ) {

		this.renderer = renderer;
		this.light = light;
		this.samples = 0;
		this._scene = new Scene();
		this._scene.add( light, light.target );
		this._objects = objects.map( object => {

			const proxy = new Mesh( object.geometry );
			proxy.matrixAutoUpdate = false;
			proxy.frustumCulled = false;
			proxy.castShadow = proxy.receiveShadow = true;
			this._scene.add( proxy );
			return { object, proxy };

		} );
		const options = { type: HalfFloatType, depthBuffer: false };
		this._sample = new RenderTarget( resolution * 2, resolution, options );
		this._history = new RenderTarget( resolution * 2, resolution, options );
		this.texture = this._history.texture;
		this._weight = uniform( 1 );
		// Read the exact history texel. Interpolated planar UVs on the 10 km
		// ground lose enough precision to blur an edge over repeated averages.
		this._previous = textureLoad( this.texture, screenCoordinate.xy );
		this._shadow = shadow( light );
		this._material = new NodeMaterial();
		this._material.side = DoubleSide;
		this._material.shadowSide = BackSide;
		this._material.depthTest = false;
		this._material.depthWrite = false;
		// UV winding is unrelated to surface orientation. Do not flip the
		// receiver's geometric normal when rasterizing the back of a UV chart.
		this._material.normalNode = normalViewGeometry;
		this._material.vertexNode = vec4( uv( 1 ).flipY().sub( 0.5 ).mul( 2 ), 0, 1 );
		this._material.fragmentNode = vec4( mix( this._previous.r, this._shadow, this._weight ), 0, 0, 1 );
		// The ground occupies the left half. Its world-planar UVs extend outside
		// that half, so clip them before they can overwrite the house atlas.
		this._groundMaterial = this._material.clone();
		this._groundMaterial.fragmentNode = Fn( () => {

			If( uv( 1 ).x.lessThan( 0 ).or( uv( 1 ).x.greaterThan( 0.5 ) ), () => Discard() );
			return this._material.fragmentNode;

		} )();
		for ( const { object, proxy } of this._objects ) proxy.material = object === objects[ 0 ] ? this._groundMaterial : this._material;

		// A one-texel gutter makes bilinear sampling safe at chart edges. Only
		// empty texels are filled: valid visibility is never spatially blurred.
		const source = textureLoad( this._sample.texture, screenCoordinate.xy );
		const dilation = new NodeMaterial();
		dilation.fragmentNode = Fn( () => {

			const center = source.toVar();
			const sum = vec2( 0 ).toVar();
			for ( let y = - 1; y <= 1; y ++ ) for ( let x = - 1; x <= 1; x ++ ) {

				const coord = screenCoordinate.xy.add( vec2( x, y ) ).clamp( vec2( 0 ), vec2( resolution * 2 - 1, resolution - 1 ) );
				const neighbor = textureLoad( this._sample.texture, coord );
				sum.addAssign( vec2( neighbor.r.mul( neighbor.a ), neighbor.a ) );

			}

			return center.a.greaterThan( 0 ).select( center, vec4( sum.x.div( sum.y.max( 1 ) ), 0, 0, sum.y.greaterThan( 0 ).select( 1, 0 ) ) );

		} )();
		this._quad = new QuadMesh( dilation );
		this._direction = new Vector3();
		this._u = new Vector3();
		this._v = new Vector3();
		this._disk = new Vector2();

	}

	/** Starts a new arithmetic average on the next update. */
	reset() {

		this.samples = 0;

	}

	/**
	 * Adds one sample, uniformly distributed over the sun's apparent disk.
	 *
	 * @param {Camera} camera - Camera used to submit the atlas render.
	 * @param {Vector3} direction - Unjittered direction towards the sun.
	 * @param {number} radius - Angular radius in degrees.
	 * @param {number} count - Total sample budget (must stay fixed until reset).
	 */
	update( camera, direction, radius, count ) {

		if ( this.samples >= count ) return;
		const renderer = this.renderer;
		const target = renderer.getRenderTarget();
		const autoClear = renderer.autoClear;
		const clearColor = renderer.getClearColor( new Color() );
		const clearAlpha = renderer.getClearAlpha();
		this._direction.copy( direction ).normalize();
		this._u.set( 0, 1, 0 );
		if ( Math.abs( this._direction.y ) > 0.99 ) this._u.set( 1, 0, 0 );
		this._u.cross( this._direction ).normalize();
		this._v.crossVectors( this._direction, this._u );
		// Progressive disk samples: radical inverse for equal-area radius and
		// golden angle for azimuth. Short prefixes cover the full sun disk.
		let index = this.samples + 1, radial = 0, digit = 0.5;
		while ( index > 0 ) {

			radial += ( index % 2 ) * digit;
			index = Math.floor( index / 2 );
			digit *= 0.5;

		}

		const r = Math.sqrt( radial ) * Math.tan( radius * Math.PI / 180 );
		const phi = this.samples * Math.PI * ( 3 - Math.sqrt( 5 ) );
		this._disk.set( Math.cos( phi ) * r, Math.sin( phi ) * r );
		this.light.position.copy( this._direction ).addScaledVector( this._u, this._disk.x ).addScaledVector( this._v, this._disk.y ).normalize().multiplyScalar( 200 );
		this.light.updateMatrixWorld();
		this.light.target.updateMatrixWorld();
		this.light.shadow.updateMatrices( this.light );
		this.light.shadow.needsUpdate = true;
		this._weight.value = 1 / ( this.samples + 1 );
		for ( const { object, proxy } of this._objects ) {

			object.updateWorldMatrix( true, false );
			proxy.geometry = object.geometry;
			proxy.matrix.copy( object.matrixWorld );
			proxy.matrixWorldNeedsUpdate = true;
			proxy.visible = true;
			for ( let ancestor = object; ancestor !== null; ancestor = ancestor.parent ) proxy.visible &&= ancestor.visible;

		}

		try {

			renderer.autoClear = true;
			renderer.setClearColor( 0, 0 );
			// ShadowNode normally updates once per animation frame/camera. Explicit
			// updates also permit multiple samples between animation frames (bakes,
			// resets and offline references) without reusing stale depth maps.
			if ( this._shadow.shadowMap !== null ) {

				this._scene.updateMatrixWorld( true );
				this._shadow.updateShadow( { renderer, scene: this._scene, camera } );
				this.light.shadow.needsUpdate = false;

			}

			if ( this.samples === 0 ) {

				renderer.setRenderTarget( this._history );
				renderer.clear();

			}

			renderer.setRenderTarget( this._sample );
			renderer.render( this._scene, camera );
			renderer.setRenderTarget( this._history );
			this._quad.render( renderer );
			this.samples ++;

		} finally {

			renderer.setRenderTarget( target );
			renderer.autoClear = autoClear;
			renderer.setClearColor( clearColor, clearAlpha );

		}

	}

	/** Frees the accumulator's targets, materials and shadow map. */
	dispose() {

		this._sample.dispose();
		this._history.dispose();
		this._material.dispose();
		this._groundMaterial.dispose();
		this._quad.material.dispose();
		this._shadow.dispose();

	}

}

export { ProgressiveShadowMap };
