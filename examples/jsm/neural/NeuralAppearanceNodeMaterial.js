import * as THREE from 'three';
import * as TSL from 'three/tsl';
import {
	evaluateNeuralBRDF,
	evaluateNeuralEmission,
	evaluateNeuralOpacity,
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
	emissiveIntensity: 1
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
		this._outputUniforms = createOutputUniforms( neuralAppearanceData.outputs );

		this._fixedMipLevelNode = TSL.uniform( this.fixedMipLevel ).onObjectUpdate( ( { material } ) => material.fixedMipLevel );
		this._intensityNode = TSL.uniform( this.intensity ).onObjectUpdate( ( { material } ) => material.intensity );
		this._emissiveIntensityNode = TSL.uniform( this.emissiveIntensity ).onObjectUpdate( ( { material } ) => material.emissiveIntensity );

		if ( neuralAppearanceData.outputs.emission ) {

			this.emissiveNode = evaluateNeuralEmission( this ).mul( this._emissiveIntensityNode );

		}

		if ( neuralAppearanceData.outputs.opacity ) {

			const opacity = evaluateNeuralOpacity( this );
			const alphaCutoff = TSL.uniform( neuralAppearanceData.outputs.opacity.alphaCutoff );
			this.opacityNode = opacity;
			this.alphaTestNode = alphaCutoff;
			this.alphaTest = neuralAppearanceData.outputs.opacity.alphaCutoff;
			this.maskNode = opacity.greaterThanEqual( alphaCutoff );
			this.maskShadowNode = this.maskNode;

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
	 * Neural lighting is accumulated by the custom lighting model.
	 *
	 * @return {Node<vec3>} Black outgoing light before direct lights are applied.
	 */
	setupOutgoingLight() {

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

}

export { NeuralAppearanceNodeMaterial };
