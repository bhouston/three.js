import { DataTexture, FloatType, LinearFilter, NodeMaterial, NoBlending, NoColorSpace, QuadMesh, RendererUtils, RenderTarget, RGBAFormat } from 'three/webgpu';
import { texture, uv } from 'three/tsl';

const WEIGHTS_URL = 'https://media.githubusercontent.com/media/RenderKit/oidn-weights/28883d1769d5930e13cf7f1676dd852bd81ed9e7/rtlightmap_hdr.tza';

/**
 * Experimental browser OIDN RTLightmap filter. Requires the optional `oidn-webgpu`
 * package (tested with 0.6.0) or a matching browser import map entry.
 *
 * This model has no chart or geometry guides and may blur unrelated UV islands.
 * Only HDR color is filtered; directional light maps are left to the caller.
 * Whole-image inference is restricted to 512 texels per edge to bound GPU memory
 * and avoid the library's independent HDR exposure normalization across tiles.
 * The raw light mapper accumulation is never modified.
 *
 * @three_import import { LightMapOIDNDenoiser } from 'three/addons/lighting/LightMapOIDNDenoiser.js';
 */
class LightMapOIDNDenoiser {

	/**
	 * @param {WebGPURenderer} renderer - A hardware WebGPU renderer.
	 * @param {LightMapper} lightMapper - The source light mapper.
	 * @param {Object} [options] - Options.
	 * @param {number} [options.page=0] - Source atlas page.
	 */
	constructor( renderer, lightMapper, { page = 0 } = {} ) {

		const size = lightMapper.size;
		if ( size > 512 ) throw new Error( 'LightMapOIDNDenoiser: Experimental whole-image inference supports atlas sizes up to 512.' );
		if ( ! lightMapper.pages[ page ] ) throw new Error( 'LightMapOIDNDenoiser: Invalid atlas page.' );

		this.renderer = renderer;
		this.lightMapper = lightMapper;
		this.page = page;
		this.texture = new DataTexture( new Float32Array( size * size * 4 ), size, size, RGBAFormat, FloatType );
		this.texture.channel = 1;
		this.texture.colorSpace = NoColorSpace;
		this.texture.minFilter = LinearFilter;
		this.texture.magFilter = LinearFilter;
		this.texture.needsUpdate = true;

		this._target = new RenderTarget( size, size, { type: FloatType, depthBuffer: false } );
		this._material = new NodeMaterial();
		this._material.fragmentNode = texture( lightMapper.pages[ page ].texture, uv() );
		this._material.blending = NoBlending;
		this._material.depthTest = false;
		this._material.depthWrite = false;
		this._quad = new QuadMesh( this._material );
		this._model = null;
		this._modelPromise = null;
		this._pending = null;
		this._disposed = false;

	}

	async _loadModel() {

		if ( this._modelPromise === null ) {

			this._modelPromise = ( async () => {

				await this.renderer.init();
				const device = this.renderer.backend.device;
				if ( ! device ) throw new Error( 'LightMapOIDNDenoiser: A WebGPU device is required.' );
				const { initUNetFromURL } = await import( 'oidn-webgpu' );
				const model = await initUNetFromURL( WEIGHTS_URL, { device }, { hdr: true, hdrTransfer: 'log', precision: 'fp32', dynamicTile: false } );
				if ( this._disposed ) {

					model.dispose();
					throw new Error( 'LightMapOIDNDenoiser: Disposed during initialization.' );

				}

				this._model = model;
				return model;

			} )().catch( ( error ) => {

				this._modelPromise = null;
				throw error;

			} );

		}

		return this._modelPromise;

	}

	/**
	 * Filters a snapshot of the normalized light map and updates {@link LightMapOIDNDenoiser#texture}.
	 * The returned promise resolves only after GPU inference and CPU readback finish.
	 * Concurrent updates are rejected. Dispose rejects a pending update.
	 *
	 * @param {Object} [options] - Processing options.
	 * @param {AbortSignal} [options.signal] - Cancels the update.
	 * @return {Promise<Texture>} The output texture.
	 */
	async update( { signal } = {} ) {

		signal?.throwIfAborted();
		if ( this._disposed ) throw new Error( 'LightMapOIDNDenoiser: Already disposed.' );
		if ( this._pending !== null ) throw new Error( 'LightMapOIDNDenoiser: An update is already running.' );

		let rejectCancelled;
		const cancelled = new Promise( ( resolve, reject ) => {

			rejectCancelled = reject;

		} );
		const pending = { reject: rejectCancelled, abort: null, rejectInference: null };
		this._pending = pending;
		const onAbort = () => {

			pending.abort?.();
			pending.rejectInference?.( signal.reason );
			pending.reject( signal.reason );

		};

		signal?.addEventListener( 'abort', onAbort, { once: true } );

		try {

			const operation = ( async () => {

				const model = await this._loadModel();
				signal?.throwIfAborted();
				if ( this._disposed ) throw new Error( 'LightMapOIDNDenoiser: Already disposed.' );
				const renderer = this.renderer;
				const state = RendererUtils.resetRendererState( renderer );
				try {

					renderer.setRenderTarget( this._target );
					this._quad.render( renderer );

				} finally {

					RendererUtils.restoreRendererState( renderer, state );

				}

				const size = this.lightMapper.size;
				const pixels = await renderer.readRenderTargetPixelsAsync( this._target, 0, 0, size, size );
				signal?.throwIfAborted();
				if ( this._disposed ) throw new Error( 'LightMapOIDNDenoiser: Already disposed.' );
				const result = await new Promise( ( resolve, reject ) => {

					pending.rejectInference = reject;
					pending.abort = model.tileExecute( {
						color: { data: pixels, width: size, height: size },
						wholeImage: true,
						done: resolve,
						error: reject
					} );

				} );
				signal?.throwIfAborted();
				if ( this._disposed ) throw new Error( 'LightMapOIDNDenoiser: Already disposed.' );
				this.texture.image.data.set( result.data );
				// OIDN replaces alpha with one; preserve the atlas coverage and padding.
				for ( let i = 3; i < pixels.length; i += 4 ) this.texture.image.data[ i ] = pixels[ i ];
				this.texture.needsUpdate = true;
				return this.texture;

			} )();
			return await Promise.race( [ operation, cancelled ] );

		} finally {

			signal?.removeEventListener( 'abort', onAbort );
			if ( this._pending === pending ) this._pending = null;

		}

	}

	/** Releases resources and cancels a pending update. The shared GPU device is retained. */
	dispose() {

		if ( this._disposed ) return;
		this._disposed = true;
		if ( this._pending !== null ) {

			if ( this._pending.abort ) this._pending.abort();
			const error = new Error( 'LightMapOIDNDenoiser: Disposed during update.' );
			if ( this._pending.rejectInference ) this._pending.rejectInference( error );
			this._pending.reject( error );

		}

		if ( this._model !== null ) this._model.dispose();
		this._target.dispose();
		this._material.dispose();
		this.texture.dispose();

	}

}

export { LightMapOIDNDenoiser };
