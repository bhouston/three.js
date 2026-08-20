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
	intensity: 1,
	emissiveIntensity: 1,
	debugView: 'shaded'
};

/**
 * Evaluates a compact neural appearance model in a WebGPU node material.
 *
 * @augments NodeMaterial
 * @three_import import { NeuralAppearanceNodeMaterial } from 'three/addons/neural-appearance/NeuralAppearanceNodeMaterial.js';
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

		// The shader graph below (see NeuralAppearanceTSL.js's fetchLatentTexels/
		// createEvaluateNeuralBRDFFn) generates its named latent-texel inputs
		// (latent0..latentN-1) from this model's own `levels` count at
		// construction time, so any level count `createModel`/
		// `NeuralAppearanceTrainer` can produce renders correctly here too -
		// this is just a basic shape check, not a fixed-count limitation.
		if ( ! Array.isArray( neuralAppearanceData.latentTextures ) || neuralAppearanceData.latentTextures.length === 0 ) {

			const actual = neuralAppearanceData.latentTextures ? neuralAppearanceData.latentTextures.length : 0;
			throw new Error( `THREE.NeuralAppearanceNodeMaterial: neuralAppearanceData.latentTextures must be a non-empty array (got ${ actual }).` );

		}

		this.isNeuralAppearanceNodeMaterial = true;
		this.lights = true;
		this.transparent = false;

		this.neuralAppearanceData = neuralAppearanceData;
		this.intensity = parameters.intensity !== undefined ? parameters.intensity : DEFAULT_PARAMETERS.intensity;
		this.emissiveIntensity = parameters.emissiveIntensity !== undefined ? parameters.emissiveIntensity : DEFAULT_PARAMETERS.emissiveIntensity;
		this.debugView = parameters.debugView || DEFAULT_PARAMETERS.debugView;
		this._outputUniforms = createOutputUniforms( neuralAppearanceData.outputs );

		// Built once and shared with the BRDF/IBL lighting model (see
		// NeuralAppearanceLightingModel.start(), which reuses this same
		// instance instead of building its own) -- every decoder head reads
		// its already-sampled grid latents/rotation frames from here rather
		// than each independently re-sampling the 4 grid-level textures and
		// re-running the rotation decoder per head.
		this._fragmentContext = createNeuralFragmentContext( this );

		this._intensityNode = TSL.uniform( this.intensity ).onObjectUpdate( ( { material } ) => material.intensity );
		this._emissiveIntensityNode = TSL.uniform( this.emissiveIntensity ).onObjectUpdate( ( { material } ) => material.emissiveIntensity );

		if ( neuralAppearanceData.outputs.emission ) {

			this.emissiveNode = evaluateNeuralEmission( this, this._fragmentContext ).mul( this._emissiveIntensityNode );

		}

		if ( neuralAppearanceData.outputs.opacity ) {

			const opacityHead = neuralAppearanceData.outputs.opacity;
			const opacity = evaluateNeuralOpacity( this, this._fragmentContext );
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
	 * Releases the latent grid textures this material owns. The decoder
	 * weight/bias uniforms (`_outputUniforms`, built via `TSL.uniformArray`)
	 * don't need an explicit release here - unlike `latentTextures`, they're
	 * plain node-graph state torn down by the renderer's standard
	 * node-disposal path when this material's own `dispose()` event fires
	 * (via the inherited `Material.dispose()` below), not a GPU resource
	 * this class owns directly.
	 */
	dispose() {

		for ( const texture of this.neuralAppearanceData.latentTextures ) texture.dispose();

		super.dispose();

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

		if ( this.debugView === 'normal' || this.debugView === 'reflection' || this.debugView === 'roughness' || this.debugView === 'irradiance' ) {

			const debug = evaluateNeuralDebugShading( this );

			if ( this.debugView === 'roughness' ) {

				return packDebugScalar( debug.roughness );

			}

			const direction = this.debugView === 'normal' ? debug.viewNormal :
				this.debugView === 'irradiance' ? debug.viewIrradiance :
					debug.viewReflect;

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
		// Reuses the material's own shared context (built once in its
		// constructor) instead of building a second, independent one --
		// see NeuralAppearanceNodeMaterial's constructor comment.
		this._fragmentContext = this.material._fragmentContext;
		super.start( builder );

	}

	direct( { lightDirection, lightColor, reflectedLight } ) {

		if ( isIsolatedIblDebugView( this.material.debugView ) ) return;

		const material = this.material;
		const brdf = evaluateNeuralBRDF( material, lightDirection, this._fragmentContext, this._evaluateBRDF );

		reflectedLight.directDiffuse.addAssign( brdf.mul( lightColor ).mul( material._intensityNode ) );

	}

	indirect( builder ) {

		const envNode = builder.context.neuralEnvironmentNode;
		if ( envNode === undefined || envNode === null ) return;

		const isolate = this.material.debugView === 'iblRadiance' ? 'radiance' :
			this.material.debugView === 'iblIrradiance' ? 'irradiance' :
				'full';

		builder.context.reflectedLight.indirectSpecular.addAssign(
			evaluateNeuralIBL( this.material, envNode, this._fragmentContext, isolate ).mul( this.material._intensityNode )
		);

	}

}

function isIsolatedIblDebugView( debugView ) {

	return debugView === 'ibl' || debugView === 'iblRadiance' || debugView === 'iblIrradiance';

}

export { NeuralAppearanceNodeMaterial };
