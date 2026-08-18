import * as THREE from 'three';
import * as TSL from 'three/tsl';
import { EnvironmentNode, PhysicalLightingModel } from 'three/webgpu';
import {
	computeAtlasDimensions,
	createAtlasTextures,
	createAtlasShaderNodes,
	createTeacherRenderTarget,
	uploadAtlasSamples
} from './NeuralAppearanceTeacherAtlas.js';
import {
	readSamplePixel,
	readFilteredSample,
	renderAndReadTeacher
} from './NeuralAppearanceTeacherReadback.js';

const DEFAULT_BATCH_SIZE = 1024;
const DEFAULT_TILE_SIZE = 8;
const DEFAULT_UV_GRADIENT_SCALE = 1 / 1024;

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
		this.supportsEmission = material.emissiveNode !== null && material.emissiveNode !== undefined;
		this.supportsOpacity = material.opacityNode !== null && material.opacityNode !== undefined;
		this.alphaCutoff = Number.isFinite( material.alphaTest ) ? material.alphaTest : 0.5;

		const materialOpacityMode = ( material && material.userData && material.userData.opacityMode ) ? material.userData.opacityMode : ( material ? material._opacityMode : undefined );
		if ( options.opacityMode === 'mask' || options.opacityMode === 'blend' ) {

			this.opacityMode = options.opacityMode;

		} else if ( materialOpacityMode === 'mask' || materialOpacityMode === 'blend' ) {

			this.opacityMode = materialOpacityMode;

		} else if ( this.supportsOpacity && material && material.transparent === true && ( ! Number.isFinite( material.alphaTest ) || material.alphaTest <= 0 ) ) {

			this.opacityMode = 'blend';

		} else {

			this.opacityMode = 'mask';

		}

		this._targetMode = 'brdf';
		this.environment = options.environment || null;
		this.supportsIBL = this.environment !== null && this.environment !== undefined;

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

	async evaluateBatch( samples, targetMode = 'brdf' ) {

		if ( this._initialized === false ) {

			this._targetMode = targetMode;
			await this.init();

		} else if ( targetMode !== this._targetMode ) {

			this.dispose();
			this._targetMode = targetMode;
			await this.init();

		}

		if ( samples.length === 0 ) return [];

		const targets = new Array( samples.length );

		for ( let offset = 0; offset < samples.length; offset += this.batchSize ) {

			const batch = samples.slice( offset, offset + this.batchSize );
			this._uploadSamples( batch );

			const pixels = await this._renderAndRead();

			const usePointFilter = this.filterMode === 'point' || targetMode === 'iblQuery';

			for ( let i = 0; i < batch.length; i ++ ) {

				const pixel = usePointFilter ?
					this._readSamplePixel( pixels, i ) :
					this._readFilteredSample( pixels, i, batch[ i ] );
				targets[ offset + i ] = targetMode === 'iblQuery' ? pixel.slice( 0, 4 ) : pixel.slice( 0, 3 );

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
		const { atlasColumns, atlasRows, atlasWidth, atlasHeight } = computeAtlasDimensions( capacity, this.tileSize );
		this._atlasColumns = atlasColumns;
		this._atlasWidth = atlasWidth;
		this._atlasHeight = atlasHeight;

		this._sampleTextures = createAtlasTextures( atlasColumns, atlasRows );
		const nodes = createAtlasShaderNodes( this._sampleTextures, atlasColumns, atlasRows, this.tileSize );

		const sampleMaterial = this.material.clone ? this.material.clone() : this.material;
		sampleMaterial.contextNode = TSL.replaceUV( nodes.materialUv, TSL.overrideNodes( [
			[ TSL.normalView, nodes.normal ],
			[ TSL.tangentView, nodes.tangent ],
			[ TSL.bitangentView, nodes.bitangent ],
			[ TSL.positionViewDirection, nodes.wo ]
		] ) );

		if ( this._targetMode === 'brdf' ) {

			sampleMaterial.lightsNode = TSL.lights( [ this._createLight() ] ).context( {
				lightingModel: new NeuralTeacherLightingModel( sampleMaterial, nodes.wi )
			} );

		} else if ( this._targetMode === 'emission' ) {

			const emission = sampleMaterial.emissiveNode || TSL.vec3( 0 );
			sampleMaterial.lights = false;
			sampleMaterial.lightsNode = TSL.lights( [] );
			sampleMaterial.outputNode = TSL.vec4( emission, 1 );

		} else if ( this._targetMode === 'opacity' ) {

			const opacity = sampleMaterial.opacityNode || TSL.float( 1 );
			sampleMaterial.lights = false;
			sampleMaterial.lightsNode = TSL.lights( [] );
			sampleMaterial.alphaTest = 0;
			sampleMaterial.alphaTestNode = null;
			sampleMaterial.outputNode = TSL.vec4( opacity, opacity, opacity, 1 );

		} else if ( this._targetMode === 'iblQuery' || this._targetMode === 'iblIncoming' ) {

			const query = createTeacherIBLQueryNodes( sampleMaterial );
			sampleMaterial.lights = false;
			sampleMaterial.lightsNode = TSL.lights( [] );

			if ( this._targetMode === 'iblQuery' ) {

				sampleMaterial.outputNode = TSL.vec4( query.radianceDir, query.roughness );

			} else {

				if ( this.environment === null ) {

					throw new Error( 'THREE.NeuralAppearanceTeacherEvaluator: An environment texture is required for IBL incoming sampling.' );

				}

				const envNode = TSL.pmremTexture( this.environment );
				const incoming = envNode.context( {
					getUV: () => query.radianceDir.transformDirection( TSL.cameraWorldMatrix ),
					getTextureLevel: () => query.roughness
				} );
				sampleMaterial.outputNode = TSL.vec4( incoming, 1 );

			}

		} else if ( this._targetMode === 'iblIrradiance' ) {

			if ( this.environment === null ) {

				throw new Error( 'THREE.NeuralAppearanceTeacherEvaluator: An environment texture is required for IBL irradiance sampling.' );

			}

			// Incoming PMREM irradiance (N, mip 1). Training input only; outgoing
			// IBL is sampled with iblIndirect / PhysicalLightingModel.
			sampleMaterial.lights = false;
			sampleMaterial.lightsNode = TSL.lights( [] );
			const envNode = TSL.pmremTexture( this.environment );
			const irradiance = envNode.context( {
				getUV: () => TSL.normalWorld,
				getTextureLevel: () => TSL.float( 1 )
			} );
			sampleMaterial.outputNode = TSL.vec4( irradiance, 1 );

		} else if ( this._targetMode === 'iblIndirect' || this._targetMode === 'iblIndirectRadiance' || this._targetMode === 'iblIndirectIrradiance' ) {

			if ( this.environment === null ) {

				throw new Error( 'THREE.NeuralAppearanceTeacherEvaluator: An environment texture is required for IBL teacher sampling.' );

			}

			const isolate = this._targetMode === 'iblIndirectRadiance' ? 'radiance' :
				this._targetMode === 'iblIndirectIrradiance' ? 'irradiance' :
					'full';

			sampleMaterial.lightsNode = TSL.lights( [] ).context( {
				lightingModel: new NeuralTeacherIBLLightingModel( sampleMaterial, isolate )
			} );
			sampleMaterial.setupEnvironment = function ( builder ) {

				let envNode = THREE.NodeMaterial.prototype.setupEnvironment.call( this, builder );
				if ( envNode === null && builder.environmentNode ) envNode = builder.environmentNode;
				return envNode ? new NeuralTeacherIBLEnvironmentNode( envNode, isolate ) : null;

			};
			sampleMaterial.customProgramCacheKey = function () {

				return THREE.NodeMaterial.prototype.customProgramCacheKey.call( this ) + '|teacherIblIsolate:' + isolate;

			};

		} else {

			throw new Error( `THREE.NeuralAppearanceTeacherEvaluator: Unsupported target mode "${ this._targetMode }".` );

		}

		sampleMaterial.toneMapped = false;
		sampleMaterial.needsUpdate = true;

		this._scene = new THREE.Scene();
		this._camera = new THREE.OrthographicCamera( - 1, 1, 1, - 1, 0, 4 );
		this._camera.position.set( 0, 0, 2 );
		this._geometry = new THREE.PlaneGeometry( 2, 2 );
		this._material = sampleMaterial;
		this._mesh = new THREE.Mesh( this._geometry, this._material );
		this._scene.add( this._mesh );
		if ( this._light ) this._scene.add( this._light );
		if ( this._targetMode === 'iblIndirect' || this._targetMode === 'iblIndirectRadiance' || this._targetMode === 'iblIndirectIrradiance' || this._targetMode === 'iblIncoming' || this._targetMode === 'iblIrradiance' ) {

			this._scene.environment = this.environment;

		}

		this._target = createTeacherRenderTarget( this._atlasWidth, this._atlasHeight );

	}

	_createLight() {

		this._light = new THREE.DirectionalLight( 0xffffff, 1 );
		this._light.position.set( 0, 0, 1 );
		return this._light;

	}

	_uploadSamples( samples ) {

		uploadAtlasSamples( this._sampleTextures, samples, this._atlasColumns, this._atlasHeight, this.tileSize, this.uvGradientScale );

	}

	async _renderAndRead() {

		return renderAndReadTeacher( this.renderer, this._scene, this._camera, this._target, this._atlasWidth, this._atlasHeight );

	}

	_readSamplePixel( pixels, sampleIndex ) {

		return readSamplePixel( pixels, sampleIndex, this._atlasColumns, this._atlasWidth, this.tileSize );

	}

	_readFilteredSample( pixels, sampleIndex, sample ) {

		return readFilteredSample( pixels, sampleIndex, sample, {
			atlasColumns: this._atlasColumns,
			atlasWidth: this._atlasWidth,
			tileSize: this.tileSize,
			sourceResolution: this.sourceResolution,
			uvGradientScale: this.uvGradientScale,
			filterMinSamples: this.filterMinSamples,
			filterMaxSamples: this.filterMaxSamples,
			filterSigma: this.filterSigma,
			filterKernels: this._filterKernels
		} );

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

class NeuralTeacherIBLLightingModel extends PhysicalLightingModel {

	constructor( material, isolate = 'full' ) {

		super( material.useClearcoat, material.useSheen, material.useIridescence, material.useAnisotropy, material.useTransmission, material.useDispersion, material.useRetroreflection );

		this.isolate = isolate;

	}

	direct( /*input, builder*/ ) {}

}

class NeuralTeacherIBLEnvironmentNode extends EnvironmentNode {

	static get type() {

		return 'NeuralTeacherIBLEnvironmentNode';

	}

	constructor( envNode = null, isolate = 'full' ) {

		super( envNode );
		this.isolate = isolate;

	}

	customCacheKey() {

		const isolateKey = this.isolate === 'radiance' ? 1 : this.isolate === 'irradiance' ? 2 : 0;
		return super.customCacheKey() + isolateKey;

	}

	setup( builder ) {

		const { material } = builder;
		let envNode = this.envNode;

		if ( envNode.isTextureNode || envNode.isMaterialReferenceNode ) {

			const value = ( envNode.isTextureNode ) ? envNode.value : material[ envNode.property ];
			const cache = this._getPMREMNodeCache( builder.renderer );
			let cacheEnvNode = cache.get( value );

			if ( cacheEnvNode === undefined ) {

				cacheEnvNode = TSL.pmremTexture( value );
				cache.set( value, cacheEnvNode );

			}

			envNode = cacheEnvNode;

		}

		const useAnisotropy = material.useAnisotropy === true || material.anisotropy > 0;
		const radianceNormalView = useAnisotropy ? TSL.bentNormalView : TSL.normalView;
		const radiance = envNode.context( createRadianceContext( TSL.roughness, radianceNormalView ) ).mul( TSL.materialEnvIntensity );
		const irradiance = envNode.context( createIrradianceContext( TSL.normalWorld ) ).mul( Math.PI ).mul( TSL.materialEnvIntensity );

		if ( this.isolate !== 'irradiance' ) {

			builder.context.radiance.addAssign( radiance.isolate() );

		}

		if ( this.isolate !== 'radiance' ) {

			builder.context.iblIrradiance.addAssign( irradiance.isolate() );

		}

		const clearcoatRadiance = builder.context.lightingModel.clearcoatRadiance;

		if ( clearcoatRadiance && this.isolate !== 'irradiance' ) {

			clearcoatRadiance.addAssign(
				envNode.context( createRadianceContext( TSL.clearcoatRoughness, TSL.clearcoatNormalView ) ).mul( TSL.materialEnvIntensity ).isolate()
			);

		}

	}

}

function createRadianceContext( roughnessNode, normalViewNode ) {

	let reflectVec = null;

	return {
		getUV: () => {

			if ( reflectVec === null ) {

				reflectVec = TSL.positionViewDirection.negate().reflect( normalViewNode );
				reflectVec = TSL.pow4( roughnessNode ).mix( reflectVec, normalViewNode ).normalize();
				reflectVec = reflectVec.transformDirection( TSL.cameraWorldMatrix );

			}

			return reflectVec;

		},
		getTextureLevel: () => roughnessNode
	};

}

function createIrradianceContext( normalWorldNode ) {

	return {
		getUV: () => normalWorldNode,
		getTextureLevel: () => TSL.float( 1 )
	};

}

function createTeacherIBLQueryNodes( material ) {

	const shadingNormal = TSL.normalView.normalize();
	const viewDir = TSL.positionViewDirection.normalize();
	const roughnessSource = material.roughnessNode !== undefined && material.roughnessNode !== null ?
		TSL.float( material.roughnessNode ) :
		TSL.materialRoughness;
	const roughness = TSL.getRoughness( { roughness: roughnessSource } );
	const reflectDir = viewDir.negate().reflect( shadingNormal );
	const roughness4 = roughness.mul( roughness ).mul( roughness ).mul( roughness );
	const radianceDir = roughness4.mix( reflectDir, shadingNormal ).normalize();

	return { roughness, radianceDir };

}

function createGpuMaterialTeacher( material, renderer, options ) {

	if ( material === null || material === undefined || material.isMeshPhysicalNodeMaterial !== true ) {

		const type = material && material.type ? material.type : typeof material;
		throw new Error( `THREE.NeuralAppearanceTeacherEvaluator: A supported MeshPhysicalNodeMaterial teacher is required; received ${ type }.` );

	}

	return new NeuralAppearanceTeacherEvaluator( material, renderer, options );

}

export {
	NeuralAppearanceTeacherEvaluator,
	NeuralTeacherIBLLightingModel,
	NeuralTeacherIBLEnvironmentNode,
	createGpuMaterialTeacher
};
