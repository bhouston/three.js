import * as THREE from 'three';
import * as TSL from 'three/tsl';
import {
	computeFootprintArea,
	createGaussianSampleKernel,
	getGaussianSampleGridSize
} from './NeuralAppearanceFilterUtils.js';

const DEFAULT_BATCH_SIZE = 1024;
const DEFAULT_TILE_SIZE = 8;
const DEFAULT_UV_GRADIENT_SCALE = 1 / 1024;
const PhysicalLightingModel = THREE.PhysicalLightingModel || class {};

class NeuralAppearanceTeacherEvaluator {

	constructor( material, renderer, options = {} ) {

		this.material = material;
		this.renderer = renderer;
		this.batchSize = options.teacherBatchSize || options.batchSize || DEFAULT_BATCH_SIZE;
		this.tileSize = options.teacherTileSize || DEFAULT_TILE_SIZE;
		this.uvGradientScale = options.uvGradientScale || DEFAULT_UV_GRADIENT_SCALE;
		this.filterMode = options.teacherFilterMode || 'gaussian';
		this.filterMinSamples = options.teacherFilterMinSamples || 1;
		this.filterMaxSamples = Math.min( options.teacherFilterMaxSamples || 64, this.tileSize * this.tileSize );
		this.filterSigma = options.teacherFilterSigma || 0.25;
		this.sourceResolution = options.sourceResolution || options.resolution || 1;

		this._scene = null;
		this._camera = null;
		this._geometry = null;
		this._material = null;
		this._mesh = null;
		this._light = null;
		this._target = null;
		this._atlasWidth = 0;
		this._atlasHeight = 0;
		this._atlasColumns = 0;
		this._sampleTextures = null;
		this._filterKernels = new Map();
		this._initialized = false;

	}

	async init() {

		if ( this._initialized ) return;

		if ( ! this.renderer || this.renderer.isWebGPURenderer !== true ) {

			throw new Error( 'THREE.NeuralAppearanceTeacherEvaluator: WebGPU renderer is required for MaterialX node-material teacher sampling.' );

		}

		if ( this.renderer.init ) await this.renderer.init();

		this._createResources();
		this._initialized = true;

	}

	encodeInputs( uv = [ 0.5, 0.5 ] ) {

		return [
			uv[ 0 ], uv[ 1 ],
			0, 0, 1,
			1, 0, 0,
			0, 1, 0,
			0, 0, 1
		];

	}

	async evaluate( sample ) {

		const results = await this.evaluateBatch( [ sample ] );
		return results[ 0 ];

	}

	async evaluateBatch( samples ) {

		if ( this._initialized === false ) await this.init();
		if ( samples.length === 0 ) return [];

		const targets = new Array( samples.length );

		for ( let offset = 0; offset < samples.length; offset += this.batchSize ) {

			const batch = samples.slice( offset, offset + this.batchSize );
			this._uploadSamples( batch );

			const pixels = await this._renderAndRead();

			for ( let i = 0; i < batch.length; i ++ ) {

				targets[ offset + i ] = this.filterMode === 'point' ?
					this._readSamplePixel( pixels, i ) :
					this._readFilteredSample( pixels, i, batch[ i ] );

			}

		}

		return targets;

	}

	dispose() {

		if ( this._target ) this._target.dispose();
		if ( this._geometry ) this._geometry.dispose();
		if ( this._material && this._material !== this.material && this._material.dispose ) this._material.dispose();

		if ( this._sampleTextures ) {

			for ( const texture of Object.values( this._sampleTextures ) ) {

				texture.dispose();

			}

		}

		this._scene = null;
		this._camera = null;
		this._geometry = null;
		this._material = null;
		this._mesh = null;
		this._light = null;
		this._target = null;
		this._sampleTextures = null;
		this._filterKernels.clear();
		this._initialized = false;

	}

	_createResources() {

		const capacity = Math.max( 1, this.batchSize );
		const tileSize = this.tileSize;
		this._atlasColumns = alignTo( Math.ceil( Math.sqrt( capacity ) ), 16 );
		const atlasRows = Math.ceil( capacity / this._atlasColumns );
		this._atlasWidth = this._atlasColumns * tileSize;
		this._atlasHeight = atlasRows * tileSize;

		this._sampleTextures = {
			uv: createSampleTexture( this._atlasColumns, atlasRows ),
			normal: createSampleTexture( this._atlasColumns, atlasRows ),
			tangent: createSampleTexture( this._atlasColumns, atlasRows ),
			bitangent: createSampleTexture( this._atlasColumns, atlasRows ),
			wi: createSampleTexture( this._atlasColumns, atlasRows ),
			wo: createSampleTexture( this._atlasColumns, atlasRows )
		};

		const sampleCoord = createSampleCoordNode( this._atlasColumns, atlasRows, tileSize );
		const sampleUv = sampleTextureNode( this._sampleTextures.uv, sampleCoord );
		const sampleNormal = sampleTextureNode( this._sampleTextures.normal, sampleCoord );
		const sampleTangent = sampleTextureNode( this._sampleTextures.tangent, sampleCoord );
		const sampleBitangent = sampleTextureNode( this._sampleTextures.bitangent, sampleCoord );
		const sampleWi = sampleTextureNode( this._sampleTextures.wi, sampleCoord );
		const sampleWo = sampleTextureNode( this._sampleTextures.wo, sampleCoord );
		const materialUv = createMaterialUvNode( sampleUv, sampleTangent, sampleBitangent, tileSize );
		const normal = sampleNormal.xyz.normalize();
		const tangent = sampleTangent.xyz.normalize();
		const bitangent = sampleBitangent.xyz.normalize();
		const wi = sampleWi.xyz.normalize();
		const wo = sampleWo.xyz.normalize();

		const sampleMaterial = this.material.clone ? this.material.clone() : this.material;
		sampleMaterial.contextNode = TSL.replaceUV( materialUv, TSL.overrideNodes( [
			[ TSL.normalView, normal ],
			[ TSL.tangentView, tangent ],
			[ TSL.bitangentView, bitangent ],
			[ TSL.positionViewDirection, wo ]
		] ) );
		sampleMaterial.lightsNode = TSL.lights( [ this._createLight() ] ).context( {
			lightingModel: new NeuralTeacherLightingModel( sampleMaterial, wi )
		} );
		sampleMaterial.toneMapped = false;
		sampleMaterial.needsUpdate = true;

		this._scene = new THREE.Scene();
		this._camera = new THREE.OrthographicCamera( - 1, 1, 1, - 1, 0, 4 );
		this._camera.position.set( 0, 0, 2 );
		this._geometry = new THREE.PlaneGeometry( 2, 2 );
		this._material = sampleMaterial;
		this._mesh = new THREE.Mesh( this._geometry, this._material );
		this._scene.add( this._mesh );
		this._scene.add( this._light );
		this._target = createTeacherRenderTarget( this._atlasWidth, this._atlasHeight );

	}

	_createLight() {

		this._light = new THREE.DirectionalLight( 0xffffff, 1 );
		this._light.position.set( 0, 0, 1 );
		return this._light;

	}

	_uploadSamples( samples ) {

		const capacity = this._atlasColumns * ( this._atlasHeight / this.tileSize );
		const textureData = {};

		for ( const [ name, texture ] of Object.entries( this._sampleTextures ) ) {

			texture.image.data.fill( 0 );
			textureData[ name ] = texture.image.data;

		}

		for ( let i = 0; i < Math.min( samples.length, capacity ); i ++ ) {

			const sample = samples[ i ];
			const offset = i * 4;
			const uv = sample.uv || [ 0.5, 0.5 ];
			const normal = normalize( sample.normal || [ 0, 0, 1 ] );
			const tangent = normalize( sample.tangent || [ 1, 0, 0 ] );
			const bitangent = normalize( sample.bitangent || [ 0, 1, 0 ] );
			const wi = normalize( sample.wi || [ 0, 0, 1 ] );
			const wo = normalize( sample.wo || [ 0, 0, 1 ] );
			const duvDx = sample.duvDx || [ this.uvGradientScale, 0 ];
			const duvDy = sample.duvDy || [ 0, this.uvGradientScale ];

			write4( textureData.uv, offset, uv[ 0 ], uv[ 1 ], duvDx[ 0 ], duvDx[ 1 ] );
			write4( textureData.normal, offset, normal[ 0 ], normal[ 1 ], normal[ 2 ], 1 );
			write4( textureData.tangent, offset, tangent[ 0 ], tangent[ 1 ], tangent[ 2 ], duvDy[ 0 ] );
			write4( textureData.bitangent, offset, bitangent[ 0 ], bitangent[ 1 ], bitangent[ 2 ], duvDy[ 1 ] );
			write4( textureData.wi, offset, wi[ 0 ], wi[ 1 ], wi[ 2 ], 0 );
			write4( textureData.wo, offset, wo[ 0 ], wo[ 1 ], wo[ 2 ], 0 );

		}

		for ( const texture of Object.values( this._sampleTextures ) ) {

			texture.needsUpdate = true;

		}

	}

	async _renderAndRead() {

		const previousTarget = this.renderer.getRenderTarget();
		const previousToneMapping = this.renderer.toneMapping;
		const previousClearColor = new THREE.Color();
		const previousClearAlpha = this.renderer.getClearAlpha();

		this.renderer.getClearColor( previousClearColor );

		try {

			this.renderer.toneMapping = THREE.NoToneMapping;
			this.renderer.setClearColor( 0x000000, 1 );
			this.renderer.setRenderTarget( this._target );
			this.renderer.render( this._scene, this._camera );

			const pixels = await this.renderer.readRenderTargetPixelsAsync( this._target, 0, 0, this._atlasWidth, this._atlasHeight );

			if ( pixels instanceof Uint16Array === false ) {

				const type = pixels && pixels.constructor ? pixels.constructor.name : typeof pixels;
				throw new Error( `THREE.NeuralAppearanceTeacherEvaluator: Half-float teacher readback is required; received ${ type }.` );

			}

			return pixels;

		} finally {

			this.renderer.setRenderTarget( previousTarget );
			this.renderer.setClearColor( previousClearColor, previousClearAlpha );
			this.renderer.toneMapping = previousToneMapping;

		}

	}

	_readSamplePixel( pixels, sampleIndex ) {

		const tileX = sampleIndex % this._atlasColumns;
		const tileY = Math.floor( sampleIndex / this._atlasColumns );
		const x = tileX * this.tileSize + Math.floor( this.tileSize / 2 );
		const y = tileY * this.tileSize + Math.floor( this.tileSize / 2 );
		const index = ( y * this._atlasWidth + x ) * 4;

		return [
			readPixelValue( pixels, index ),
			readPixelValue( pixels, index + 1 ),
			readPixelValue( pixels, index + 2 )
		];

	}

	_readFilteredSample( pixels, sampleIndex, sample ) {

		const footprintArea = computeFootprintArea(
			sample.duvDx || [ this.uvGradientScale, 0 ],
			sample.duvDy || [ 0, this.uvGradientScale ],
			this.sourceResolution,
			this.sourceResolution
		);
		const gridSize = getGaussianSampleGridSize( footprintArea, this.filterMinSamples, this.filterMaxSamples );
		let kernel = this._filterKernels.get( gridSize );

		if ( kernel === undefined ) {

			kernel = createGaussianSampleKernel( gridSize, this.tileSize, this.filterSigma );
			this._filterKernels.set( gridSize, kernel );

		}

		const tileX = sampleIndex % this._atlasColumns;
		const tileY = Math.floor( sampleIndex / this._atlasColumns );
		const target = [ 0, 0, 0 ];

		for ( const tap of kernel ) {

			const x = tileX * this.tileSize + tap.x;
			const y = tileY * this.tileSize + tap.y;
			const index = ( y * this._atlasWidth + x ) * 4;

			target[ 0 ] += readPixelValue( pixels, index ) * tap.weight;
			target[ 1 ] += readPixelValue( pixels, index + 1 ) * tap.weight;
			target[ 2 ] += readPixelValue( pixels, index + 2 ) * tap.weight;

		}

		return target;

	}

}

class NeuralTeacherLightingModel extends PhysicalLightingModel {

	constructor( material, lightDirectionNode ) {

		if ( THREE.PhysicalLightingModel === undefined ) {

			throw new Error( 'THREE.NeuralAppearanceTeacherEvaluator: PhysicalLightingModel is required for directional GPU teacher sampling. Import three/webgpu for training.' );

		}

		super( material.useClearcoat, material.useSheen, material.useIridescence, material.useAnisotropy, material.useTransmission, material.useDispersion, material.useRetroreflection );
		this.lightDirectionNode = lightDirectionNode;

	}

	direct( { reflectedLight }, builder ) {

		super.direct( {
			lightDirection: this.lightDirectionNode,
			lightColor: TSL.vec3( 1 ),
			reflectedLight
		}, builder );

	}

	indirect( /*builder*/ ) {}

}

function createTeacherRenderTarget( width, height ) {

	return new THREE.RenderTarget( width, height, {
		type: THREE.HalfFloatType,
		format: THREE.RGBAFormat,
		colorSpace: THREE.NoColorSpace,
		minFilter: THREE.NearestFilter,
		magFilter: THREE.NearestFilter,
		depthBuffer: false,
		stencilBuffer: false
	} );

}

function createSampleTexture( width, height ) {

	const texture = new THREE.DataTexture( new Float32Array( width * height * 4 ), width, height, THREE.RGBAFormat, THREE.FloatType );
	texture.colorSpace = THREE.NoColorSpace;
	texture.minFilter = THREE.NearestFilter;
	texture.magFilter = THREE.NearestFilter;
	texture.wrapS = THREE.ClampToEdgeWrapping;
	texture.wrapT = THREE.ClampToEdgeWrapping;
	texture.generateMipmaps = false;
	texture.needsUpdate = true;
	return texture;

}

function createSampleCoordNode( atlasColumns, atlasRows, tileSize ) {

	const tileCoord = TSL.floor( TSL.viewportCoordinate.div( tileSize ) );
	return tileCoord.add( 0.5 ).div( TSL.vec2( atlasColumns, atlasRows ) );

}

function createMaterialUvNode( sampleUv, sampleTangent, sampleBitangent, tileSize ) {

	const tilePixel = TSL.fract( TSL.viewportCoordinate.div( tileSize ) ).sub( 0.5 );
	const duvDx = sampleUv.zw;
	const duvDy = TSL.vec2( sampleTangent.w, sampleBitangent.w );

	return sampleUv.xy.add( duvDx.mul( tilePixel.x ) ).add( duvDy.mul( tilePixel.y ) );

}

function sampleTextureNode( texture, sampleCoord ) {

	return TSL.texture( texture, sampleCoord );

}

function write4( data, offset, x, y, z, w ) {

	data[ offset ] = x;
	data[ offset + 1 ] = y;
	data[ offset + 2 ] = z;
	data[ offset + 3 ] = w;

}

function readPixelValue( pixels, index ) {

	if ( pixels instanceof Uint16Array ) return THREE.DataUtils.fromHalfFloat( pixels[ index ] );
	return pixels[ index ];

}

function normalize( value ) {

	const length = Math.hypot( value[ 0 ], value[ 1 ], value[ 2 ] ) || 1;
	return [ value[ 0 ] / length, value[ 1 ] / length, value[ 2 ] / length ];

}

function alignTo( value, alignment ) {

	return Math.ceil( value / alignment ) * alignment;

}

function createGpuMaterialTeacher( material, renderer, options ) {

	return new NeuralAppearanceTeacherEvaluator( material, renderer, options );

}

export {
	NeuralAppearanceTeacherEvaluator,
	createGpuMaterialTeacher
};
