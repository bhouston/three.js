import * as THREE from 'three/webgpu';
import * as TSL from 'three/tsl';
import {
	evaluateNeuralBRDF,
	evaluateNeuralEmission,
	evaluateNeuralIBL,
	evaluateNeuralOpacity,
	evaluateNeuralDebugShading,
	packDebugDirection,
	packDebugScalar,
	createEvaluateNeuralBRDFFn,
	createNeuralFragmentContext,
	createOutputUniforms,
	isCompatibleNeuralAppearanceData,
	updateOutputUniforms,
	copyLatentTextureData
} from './NeuralAppearanceTSL.js';

const DEFAULT_PARAMETERS = {
	lodMode: 'deterministic',
	fixedMipLevel: - 1,
	intensity: 1,
	emissiveIntensity: 1,
	debugView: 'shaded'
};

/**
 * Evaluates a compact neural appearance model in a WebGPU node material.
 *
 * @augments NodeMaterial
 * @three_import import { NeuralAppearanceNodeMaterial } from 'three/addons/neural/NeuralAppearanceNodeMaterial.js';
 */
class NeuralAppearanceNodeMaterial extends THREE.NodeMaterial {

	static get type() {

		return 'NeuralAppearanceNodeMaterial';

	}

	/**
	 * Constructs a new neural appearance node material.
	 *
	 * @param {Object} neuralAppearanceData - Data returned by `NeuralAppearanceLoader`.
	 * @param {Object} [parameters] - Material parameters.
	 */
	constructor( neuralAppearanceData, parameters = {} ) {

		super();

		if ( ! neuralAppearanceData || neuralAppearanceData.isNeuralAppearanceData !== true ) {

			throw new Error( 'THREE.NeuralAppearanceNodeMaterial: Expected data from NeuralAppearanceLoader.' );

		}

		this.isNeuralAppearanceNodeMaterial = true;
		this.lights = true;
		this.transparent = false;

		this.neuralAppearanceData = neuralAppearanceData;
		this.lodMode = parameters.lodMode || DEFAULT_PARAMETERS.lodMode;
		this.fixedMipLevel = parameters.fixedMipLevel !== undefined ? parameters.fixedMipLevel : DEFAULT_PARAMETERS.fixedMipLevel;
		this.intensity = parameters.intensity !== undefined ? parameters.intensity : DEFAULT_PARAMETERS.intensity;
		this.emissiveIntensity = parameters.emissiveIntensity !== undefined ? parameters.emissiveIntensity : DEFAULT_PARAMETERS.emissiveIntensity;
		this.debugView = parameters.debugView || DEFAULT_PARAMETERS.debugView;
		this._outputUniforms = createOutputUniforms( neuralAppearanceData.outputs );

		this._fixedMipLevelNode = TSL.uniform( this.fixedMipLevel ).onObjectUpdate( ( { material } ) => material.fixedMipLevel );
		this._intensityNode = TSL.uniform( this.intensity ).onObjectUpdate( ( { material } ) => material.intensity );
		this._emissiveIntensityNode = TSL.uniform( this.emissiveIntensity ).onObjectUpdate( ( { material } ) => material.emissiveIntensity );

		if ( neuralAppearanceData.outputs.emission ) {

			this.emissiveNode = evaluateNeuralEmission( this ).mul( this._emissiveIntensityNode );

		}

		if ( neuralAppearanceData.outputs.opacity ) {

			const opacityHead = neuralAppearanceData.outputs.opacity;
			const opacity = evaluateNeuralOpacity( this );
			this.opacityNode = opacity;

			if ( opacityHead.mode === 'blend' ) {

				this.transparent = true;
				this.depthWrite = false;

			} else {

				const alphaCutoff = TSL.uniform( opacityHead.alphaCutoff );
				this.alphaTestNode = alphaCutoff;
				this.alphaTest = opacityHead.alphaCutoff;
				this.maskNode = opacity.greaterThanEqual( alphaCutoff );
				this.maskShadowNode = this.maskNode;

			}

		}

		this.setValues( parameters );

	}

	/**
	 * Uploads matching latent textures and decoder weights without rebuilding the shader.
	 *
	 * Returns `false` when the new data has a different layout. On success the incoming
	 * latent textures are disposed.
	 *
	 * @param {Object} neuralAppearanceData - Data returned by `NeuralAppearanceLoader`.
	 * @return {boolean} True if the existing shader was reused.
	 */
	updateFromData( neuralAppearanceData ) {

		if ( ! neuralAppearanceData || neuralAppearanceData.isNeuralAppearanceData !== true ) {

			throw new Error( 'THREE.NeuralAppearanceNodeMaterial: Expected data from NeuralAppearanceLoader.' );

		}

		if ( isCompatibleNeuralAppearanceData( this.neuralAppearanceData, neuralAppearanceData ) !== true ) {

			return false;

		}

		copyLatentTextureData( this.neuralAppearanceData.latentTextures, neuralAppearanceData.latentTextures );
		updateOutputUniforms( this._outputUniforms, neuralAppearanceData.outputs );

		this.neuralAppearanceData.name = neuralAppearanceData.name;
		this.neuralAppearanceData.outputs = neuralAppearanceData.outputs;
		this.neuralAppearanceData.referenceEvaluations = neuralAppearanceData.referenceEvaluations;

		for ( const texture of neuralAppearanceData.latentTextures ) {

			texture.dispose();

		}

		return true;

	}

	/**
	 * Setups the lighting model.
	 *
	 * @return {NeuralAppearanceLightingModel} The lighting model.
	 */
	setupLightingModel() {

		return new NeuralAppearanceLightingModel( this );

	}

	/**
	 * Setups neural environment lighting from material or scene environments.
	 *
	 * @param {NodeBuilder} builder - The current node builder.
	 * @return {?NeuralAppearanceEnvironmentNode} The neural environment node.
	 */
	setupEnvironment( builder ) {

		let envNode = super.setupEnvironment( builder );

		if ( envNode === null && builder.environmentNode ) {

			envNode = builder.environmentNode;

		}

		return envNode ? new NeuralAppearanceEnvironmentNode( this, envNode ) : null;

	}

	/**
	 * Neural lighting is accumulated by the custom lighting model.
	 *
	 * @return {Node<vec3>} Black outgoing light before direct lights are applied.
	 */
	setupOutgoingLight() {

		return TSL.vec3( 0 );

	}

	/**
	 * Replaces shaded lighting with packed decoder-frame debug values.
	 *
	 * @param {NodeBuilder} builder - The current node builder.
	 * @return {Node<vec3>} Shaded lighting or a packed debug direction.
	 */
	setupLighting( builder ) {

		if ( this.debugView === 'normal' || this.debugView === 'reflection' || this.debugView === 'roughness' ) {

			const debug = evaluateNeuralDebugShading( this );

			if ( this.debugView === 'roughness' ) {

				return packDebugScalar( debug.roughness );

			}

			const direction = this.debugView === 'normal' ? debug.viewNormal : debug.viewReflect;

			return packDebugDirection( direction );

		}

		return super.setupLighting( builder );

	}

}

class NeuralAppearanceEnvironmentNode extends THREE.LightingNode {

	constructor( material, envNode = null ) {

		super();
		this.material = material;
		this.envNode = envNode;

	}

	setup( builder ) {

		let envNode = this.envNode;

		if ( envNode.isTextureNode || envNode.isMaterialReferenceNode ) {

			const value = envNode.isTextureNode ? envNode.value : builder.material[ envNode.property ];
			envNode = TSL.pmremTexture( value );

		}

		builder.context.neuralEnvironmentNode = envNode;

		return TSL.vec3( 0 );

	}

}

class NeuralAppearanceLightingModel extends THREE.LightingModel {

	constructor( material ) {

		super();
		this.material = material;
		this._evaluateBRDF = null;
		this._fragmentContext = null;

	}

	start( builder ) {

		this._evaluateBRDF = createEvaluateNeuralBRDFFn( this.material );
		this._fragmentContext = createNeuralFragmentContext( this.material );
		super.start( builder );

	}

	direct( { lightDirection, lightColor, reflectedLight } ) {

		const material = this.material;
		const brdf = evaluateNeuralBRDF( material, lightDirection, this._fragmentContext, this._evaluateBRDF );

		reflectedLight.directDiffuse.addAssign( brdf.mul( lightColor ).mul( material._intensityNode ) );

	}

	indirect( builder ) {

		const envNode = builder.context.neuralEnvironmentNode;
		if ( envNode === undefined || envNode === null ) return;

		builder.context.reflectedLight.indirectSpecular.addAssign(
			evaluateNeuralIBL( this.material, envNode, this._fragmentContext ).mul( this.material._intensityNode )
		);

	}

}

export { NeuralAppearanceNodeMaterial };
