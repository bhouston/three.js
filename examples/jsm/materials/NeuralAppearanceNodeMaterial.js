import * as THREE from 'three';
import * as TSL from 'three/tsl';

const DEFAULT_PARAMETERS = {
	lodMode: 'deterministic',
	fixedMipLevel: - 1,
	intensity: 1
};

/**
 * Evaluates a compact neural appearance model in a WebGPU node material.
 *
 * The material implements the rasterization path described by NVIDIA's
 * Real-Time Neural Appearance Models paper: an 8D latent texture is fetched at
 * the surface UV, transformed into learned shading frames, and decoded by a
 * small MLP for each direct light. The initial implementation is intentionally
 * WebGPU-only and direct-light-only; the paper's importance sampler requires a
 * path tracer with `sample`/`pdf` material hooks, which three.js does not
 * provide in core.
 *
 * @augments NodeMaterial
 * @three_import import { NeuralAppearanceNodeMaterial } from 'three/addons/materials/NeuralAppearanceNodeMaterial.js';
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
		this._decoderUniforms = createDecoderUniforms( neuralAppearanceData.decoder );

		this._fixedMipLevelNode = TSL.uniform( this.fixedMipLevel ).onObjectUpdate( ( { material } ) => material.fixedMipLevel );
		this._intensityNode = TSL.uniform( this.intensity ).onObjectUpdate( ( { material } ) => material.intensity );

		this.setValues( parameters );

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

	}

	direct( { lightDirection, lightColor, reflectedLight } ) {

		const material = this.material;
		const brdf = evaluateNeuralBRDF( material, lightDirection );

		reflectedLight.directDiffuse.addAssign( brdf.mul( lightColor ).mul( material._intensityNode ) );

	}

}

function evaluateNeuralBRDF( material, lightDirection ) {

	const data = material.neuralAppearanceData;
	const latentCode = fetchLatentCode( material );
	const viewDirection = TSL.positionViewDirection.mul( TSL.TBNViewMatrix ).normalize();
	const incomingDirection = lightDirection.mul( TSL.TBNViewMatrix ).normalize();
	const decoderInput = buildDecoderInput( data.decoder, material._decoderUniforms, latentCode, incomingDirection, viewDirection );
	const decoded = evaluateMLP( data.decoder.layers, material._decoderUniforms.layers, decoderInput );

	return applyOutputActivation( decoded, data.decoder.outputActivation );

}

function fetchLatentCode( material ) {

	const data = material.neuralAppearanceData;
	const uvNode = TSL.uv();
	const lod = computeLOD( material, uvNode );
	const texel0 = TSL.texture( data.latentTextures[ 0 ], uvNode ).level( lod );
	const texel1 = TSL.texture( data.latentTextures[ 1 ], uvNode ).level( lod );

	return [
		texel0.x, texel0.y, texel0.z, texel0.w,
		texel1.x, texel1.y, texel1.z, texel1.w
	];

}

function computeLOD( material, uvNode ) {

	const data = material.neuralAppearanceData;
	const fixedMip = material._fixedMipLevelNode;
	const duvdx = TSL.dFdx( uvNode ).mul( TSL.vec2( data.latentWidth, data.latentHeight ) );
	const duvdy = TSL.dFdy( uvNode ).mul( TSL.vec2( data.latentWidth, data.latentHeight ) );
	const footprint = TSL.max( TSL.length( duvdx ), TSL.length( duvdy ) ).max( 1.0 );
	const computed = TSL.log2( footprint ).clamp( 0, data.mipLevels - 1 );
	const nearest = TSL.floor( computed.add( 0.5 ) );

	if ( material.lodMode === 'stochastic' ) {

		const base = TSL.floor( computed );
		const probability = TSL.fract( computed );
		const random = TSL.fract( TSL.sin( TSL.dot( uvNode, TSL.vec2( 12.9898, 78.233 ) ) ).mul( 43758.5453 ) );
		const stochastic = TSL.select( random.lessThan( probability ), base.add( 1 ), base ).clamp( 0, data.mipLevels - 1 );

		return TSL.select( fixedMip.greaterThanEqual( 0 ), fixedMip, stochastic );

	}

	return TSL.select( fixedMip.greaterThanEqual( 0 ), fixedMip, nearest );

}

function createDecoderUniforms( decoder ) {

	return {
		rotationWeights: decoder.rotation ? TSL.uniformArray( packLayerWeights( decoder.rotation.weights, decoder.rotation.inputSize, decoder.rotation.outputSize ), 'vec4' ) : null,
		layers: decoder.layers.map( ( layer ) => ( {
			weights: TSL.uniformArray( packLayerWeights( layer.weights, layer.inputSize, layer.outputSize ), 'vec4' ),
			biases: TSL.uniformArray( layer.biases, 'float' )
		} ) )
	};

}

function packLayerWeights( weights, inputSize, outputSize ) {

	const inputVectorCount = Math.ceil( inputSize / 4 );
	const packed = [];

	for ( let outputIndex = 0; outputIndex < outputSize; outputIndex ++ ) {

		for ( let vectorIndex = 0; vectorIndex < inputVectorCount; vectorIndex ++ ) {

			const offset = outputIndex * inputSize + vectorIndex * 4;

			packed.push( new THREE.Vector4(
				weights[ offset ] || 0,
				weights[ offset + 1 ] || 0,
				weights[ offset + 2 ] || 0,
				weights[ offset + 3 ] || 0
			) );

		}

	}

	return packed;

}

function buildDecoderInput( decoder, decoderUniforms, latents, wi, wo ) {

	if ( decoder.rotation === null ) {

		throw new Error( 'THREE.NeuralAppearanceNodeMaterial: A two-frame rotation decoder is required.' );

	}

	const frames = linearLayer( latents, decoderUniforms.rotationWeights, null, decoder.rotation.outputSize, 'linear' );
	const input = [];

	for ( let i = 0; i < 8; i ++ ) {

		input.push( latents[ i ] );

	}

	for ( let frame = 0; frame < 2; frame ++ ) {

		const offset = frame * 6;
		const n = TSL.vec3( frames[ offset ], frames[ offset + 1 ], frames[ offset + 2 ].add( 1 ) ).normalize();
		const t = TSL.vec3( frames[ offset + 3 ].add( 1 ), frames[ offset + 4 ], frames[ offset + 5 ] ).normalize();
		const b = TSL.cross( n, t );

		input.push( wi.dot( t ), wi.dot( b ), wi.dot( n ) );
		input.push( wo.dot( t ), wo.dot( b ), wo.dot( n ) );

	}

	if ( input.length !== decoder.inputSize ) {

		throw new Error( `THREE.NeuralAppearanceNodeMaterial: Decoder input has ${ input.length } values, expected ${ decoder.inputSize }.` );

	}

	return input;

}

function evaluateMLP( layers, layerUniforms, inputs ) {

	let activations = inputs;

	for ( let i = 0; i < layers.length; i ++ ) {

		const layer = layers[ i ];
		const layerUniform = layerUniforms[ i ];

		activations = linearLayer( activations, layerUniform.weights, layerUniform.biases, layer.outputSize, layer.activation );

	}

	if ( activations.length !== 3 ) {

		throw new Error( 'THREE.NeuralAppearanceNodeMaterial: Decoder output must be RGB.' );

	}

	return TSL.vec3( activations[ 0 ], activations[ 1 ], activations[ 2 ] );

}

function linearLayer( inputs, weights, biases, outputSize, activation ) {

	const outputs = [];
	const inputVectors = [];
	const inputVectorCount = Math.ceil( inputs.length / 4 );

	for ( let i = 0; i < inputVectorCount; i ++ ) {

		const offset = i * 4;

		inputVectors.push( TSL.vec4(
			inputs[ offset ] || 0,
			inputs[ offset + 1 ] || 0,
			inputs[ offset + 2 ] || 0,
			inputs[ offset + 3 ] || 0
		) );

	}

	for ( let outputIndex = 0; outputIndex < outputSize; outputIndex ++ ) {

		let value = biases ? biases.element( outputIndex ) : TSL.float( 0 );

		for ( let vectorIndex = 0; vectorIndex < inputVectorCount; vectorIndex ++ ) {

			value = value.add( TSL.dot( inputVectors[ vectorIndex ], weights.element( outputIndex * inputVectorCount + vectorIndex ) ) );

		}

		if ( activation === 'relu' ) {

			value = value.max( 0 );

		}

		outputs.push( value );

	}

	return outputs;

}

function applyOutputActivation( value, activation ) {

	if ( activation.type === 'exp' ) {

		return TSL.exp( value.add( activation.offset || 0 ) );

	}

	if ( activation.type === 'scaledSigmoid' ) {

		const scale = activation.scale !== undefined ? activation.scale : 1;
		return TSL.float( scale ).div( TSL.float( 1 ).add( TSL.exp( value.negate() ) ) );

	}

	return value.max( 0 );

}

export { NeuralAppearanceNodeMaterial };
