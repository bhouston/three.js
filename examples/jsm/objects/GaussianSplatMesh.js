import { GaussianSplat } from 'three/webgpu';

/**
 * Backwards-compatible alias for {@link GaussianSplat}.
 *
 * ```js
 * const splats = new GaussianSplatMesh( geometry );
 * scene.add( splats );
 * ```
 *
 * @augments GaussianSplat
 * @three_import import { GaussianSplatMesh } from 'three/addons/objects/GaussianSplatMesh.js';
 */
class GaussianSplatMesh extends GaussianSplat {

	/**
	 * Constructs a new Gaussian splat mesh.
	 *
	 * @param {BufferGeometry} splatGeometry - The splat geometry to render.
	 * @param {Object} [options] - Options.
	 * @param {boolean} [options.autoSort=true] - Whether to sort automatically in `onBeforeRender`.
	 */
	constructor( splatGeometry, { autoSort = true } = {} ) {

		super( splatGeometry, { autoSort } );

		/**
		 * This flag can be used for type testing.
		 *
		 * @type {boolean}
		 * @readonly
		 * @default true
		 */
		this.isGaussianSplatMesh = true;

		this.type = 'GaussianSplatMesh';

	}

}

export { GaussianSplatMesh };
