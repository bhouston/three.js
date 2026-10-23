import { Color, CubeCamera, CubeRenderTarget, HalfFloatType, LinearMipmapLinearFilter, Scene } from 'three/webgpu';
import { backgroundRotation, cubeTexture, equirectUV, normalWorldGeometry, texture, vec3 } from 'three/tsl';

/**
 * Raw distant radiance shared by GI capture and cone tracing. The mip chain
 * filters small HDR emitters before low-resolution sampling. This is not a
 * diffuse convolution: visibility and cosine integration belong to the solver.
 * Scene environment rotation and intensity are captured into the cube, so its
 * node can also be sampled in compute shaders without material dependencies.
 *
 * @private
 */
class EnvironmentRadiance {

	constructor( size = 512 ) {

		this.size = size;
		this.target = new CubeRenderTarget( size, { type: HalfFloatType, generateMipmaps: true, minFilter: LinearMipmapLinearFilter } );
		this.node = cubeTexture( this.target.texture, vec3( 0, 1, 0 ), 0 );
		this._key = null;
		this._scene = new Scene();
		this._camera = new CubeCamera( 1, 10, this.target );

	}

	update( renderer, scene ) {

		const environment = scene.environment;
		const key = `${ environment ? environment.uuid + ':' + environment.version : '-' }:${ scene.environmentIntensity }:${ scene.environmentRotation.toArray() }`;
		if ( key === this._key ) return false;

		const sky = this._scene;
		sky.backgroundNode = null;
		sky.background = environment || new Color( 0 );
		if ( environment && ! environment.isCubeTexture ) {

			// Sample the original equirectangular map directly, avoiding another
			// resampling stage and a cached conversion that can miss texture updates.
			sky.background = null;
			sky.backgroundNode = texture( environment, equirectUV( backgroundRotation.mul( normalWorldGeometry ) ), 0 );

		}

		sky.backgroundRotation.copy( scene.environmentRotation );
		sky.backgroundIntensity = scene.environmentIntensity;
		this._camera.update( renderer, sky );
		this._key = key;
		return true;

	}

	sample( direction, level = 0 ) {

		return this.node.sample( direction ).level( level ).rgb;

	}

	dispose() {

		this.target.dispose();
		this._key = null;

	}

}

export { EnvironmentRadiance };
