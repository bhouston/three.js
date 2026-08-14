import * as THREE from 'three';
import * as TSL from 'three/tsl';

function evaluateNeuralBRDF( material, lightDirection ) {

	const data = material.neuralAppearanceData;
	const latentCode = fetchLatentCode( material );
	const viewDirection = transformToCanonicalFrame( TSL.positionViewDirection );
	const incomingDirection = transformToCanonicalFrame( lightDirection );
	const brdf = data.outputs.brdf;
	const uniforms = material._outputUniforms.brdf;
	const decoderInput = buildDecoderInput( brdf, uniforms, latentCode, incomingDirection, viewDirection );
	const decoded = toVec3( evaluateMLP( brdf.layers, uniforms.layers, decoderInput ) );
	const nDotL = incomingDirection.z.max( 0 );

	return applyOutputActivation( decoded, brdf.outputActivation ).mul( nDotL );

}

function evaluateNeuralEmission( material ) {

	const output = material.neuralAppearanceData.outputs.emission;
	const decoded = toVec3( evaluateMLP( output.layers, material._outputUniforms.emission.layers, fetchLatentCode( material ) ) );
	return applyOutputActivation( decoded, output.outputActivation );

}

function evaluateNeuralOpacity( material ) {

	const output = material.neuralAppearanceData.outputs.opacity;
	const decoded = evaluateMLP( output.layers, material._outputUniforms.opacity.layers, fetchLatentCode( material ) )[ 0 ];
	return applyScalarOutputActivation( decoded, output.outputActivation );

}

function transformToCanonicalFrame( direction ) {

	const frame = TSL.TBNViewMatrix;
	const normal = frame[ 2 ].normalize();

	// The derivative TBN fallback uses a shared scale for both axes. Rebuild an
	// orthonormal basis so runtime directions match the canonical training frame.
	const projectedTangent = frame[ 0 ].sub( normal.mul( frame[ 0 ].dot( normal ) ) );
	const tangentLengthSquared = projectedTangent.dot( projectedTangent );
	const normalizedTangent = projectedTangent.mul( tangentLengthSquared.max( 1e-10 ).inverseSqrt() );
	const fallbackAxis = normal.y.abs().lessThan( 0.999 ).select( TSL.vec3( 0, 1, 0 ), TSL.vec3( 1, 0, 0 ) );
	const fallbackTangent = fallbackAxis.cross( normal ).normalize();
	const tangent = tangentLengthSquared.greaterThan( 1e-10 ).select( normalizedTangent, fallbackTangent );
	const unhandedBitangent = normal.cross( tangent );
	const handedness = unhandedBitangent.dot( frame[ 1 ] ).lessThan( 0 ).select( - 1, 1 );
	const bitangent = unhandedBitangent.mul( handedness );

	return TSL.vec3(
		direction.dot( tangent ),
		direction.dot( bitangent ),
		direction.dot( normal )
	).normalize();

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

function createOutputUniforms( outputs ) {

	const uniforms = {
		brdf: createHeadUniforms( outputs.brdf )
	};

	if ( outputs.emission ) uniforms.emission = createHeadUniforms( outputs.emission );
	if ( outputs.opacity ) uniforms.opacity = createHeadUniforms( outputs.opacity );

	return uniforms;

}

function createHeadUniforms( decoder ) {

	return {
		rotationWeights: decoder.rotation ? TSL.uniformArray( packLayerWeights( decoder.rotation.weights, decoder.rotation.inputSize, decoder.rotation.outputSize ), 'vec4' ) : null,
		layers: decoder.layers.map( ( layer ) => ( {
			weights: TSL.uniformArray( packLayerWeights( layer.weights, layer.inputSize, layer.outputSize ), 'vec4' ),
			biases: TSL.uniformArray( packLayerBiases( layer.biases ), 'vec4' )
		} ) )
	};

}

function packLayerWeights( weights, inputSize, outputSize ) {

	const inputVectorCount = Math.ceil( inputSize / 4 );
	const packed = [];

	for ( let outputIndex = 0; outputIndex < outputSize; outputIndex ++ ) {

		for ( let vectorIndex = 0; vectorIndex < inputVectorCount; vectorIndex ++ ) {

			const inputBase = vectorIndex * 4;
			const rowOffset = outputIndex * inputSize;

			packed.push( new THREE.Vector4(
				inputBase < inputSize ? ( weights[ rowOffset + inputBase ] || 0 ) : 0,
				inputBase + 1 < inputSize ? ( weights[ rowOffset + inputBase + 1 ] || 0 ) : 0,
				inputBase + 2 < inputSize ? ( weights[ rowOffset + inputBase + 2 ] || 0 ) : 0,
				inputBase + 3 < inputSize ? ( weights[ rowOffset + inputBase + 3 ] || 0 ) : 0
			) );

		}

	}

	return packed;

}

function packLayerBiases( biases ) {

	const packed = [];
	const vectorCount = Math.ceil( biases.length / 4 );

	for ( let vectorIndex = 0; vectorIndex < vectorCount; vectorIndex ++ ) {

		const offset = vectorIndex * 4;

		packed.push( new THREE.Vector4(
			biases[ offset ] || 0,
			biases[ offset + 1 ] || 0,
			biases[ offset + 2 ] || 0,
			biases[ offset + 3 ] || 0
		) );

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
		const b = TSL.cross( n, t ).normalize();

		input.push( wi.dot( t ), wi.dot( b ), wi.dot( n ) );
		input.push( wo.dot( t ), wo.dot( b ), wo.dot( n ) );

	}

	if ( input.length !== decoder.inputSize ) {

		throw new Error( `THREE.NeuralAppearanceNodeMaterial: Decoder input has ${ input.length } values, expected ${ decoder.inputSize }.` );

	}

	return input;

}

function evaluateMLP( layers, layerUniforms, inputs ) {

	let activations = packNodeInputs( inputs );

	for ( let i = 0; i < layers.length; i ++ ) {

		const layer = layers[ i ];
		const layerUniform = layerUniforms[ i ];

		activations = linearLayerPacked( activations, layerUniform.weights, layerUniform.biases, layer.inputSize, layer.outputSize, layer.activation );

	}

	return unpackNodeInputs( activations, layers[ layers.length - 1 ].outputSize );

}

function packNodeInputs( inputs ) {

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

	return inputVectors;

}

function unpackNodeInputs( inputs, outputSize ) {

	const outputs = [];

	for ( let outputIndex = 0; outputIndex < outputSize; outputIndex ++ ) {

		const vector = inputs[ Math.floor( outputIndex / 4 ) ];
		outputs.push( vector.element( outputIndex % 4 ) );

	}

	return outputs;

}

function linearLayerPacked( inputs, weights, biases, inputSize, outputSize, activation ) {

	const outputs = [];
	const inputVectorCount = Math.ceil( inputSize / 4 );
	const outputVectorCount = Math.ceil( outputSize / 4 );

	for ( let outputVector = 0; outputVector < outputVectorCount; outputVector ++ ) {

		const outputBase = outputVector * 4;
		const sums = [ TSL.float( 0 ), TSL.float( 0 ), TSL.float( 0 ), TSL.float( 0 ) ];

		for ( let vectorIndex = 0; vectorIndex < inputVectorCount; vectorIndex ++ ) {

			const inputVector = inputs[ vectorIndex ];

			for ( let component = 0; component < 4; component ++ ) {

				const outputIndex = outputBase + component;

				if ( outputIndex < outputSize ) {

					sums[ component ] = sums[ component ].add( TSL.dot( inputVector, weights.element( outputIndex * inputVectorCount + vectorIndex ) ) );

				}

			}

		}

		let value = biases.element( outputVector ).add( TSL.vec4( sums[ 0 ], sums[ 1 ], sums[ 2 ], sums[ 3 ] ) );

		if ( activation === 'relu' ) {

			value = value.max( 0 );

		}

		outputs.push( value );

	}

	return outputs;

}

function toVec3( values ) {

	if ( values.length !== 3 ) {

		throw new Error( 'THREE.NeuralAppearanceNodeMaterial: Decoder output must be RGB.' );

	}

	return TSL.vec3( values[ 0 ], values[ 1 ], values[ 2 ] );

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

function applyScalarOutputActivation( value, activation ) {

	if ( activation.type === 'sigmoid' ) {

		return TSL.float( 1 ).div( TSL.float( 1 ).add( TSL.exp( value.negate() ) ) );

	}

	return applyOutputActivation( value, activation );

}

export {
	evaluateNeuralBRDF,
	evaluateNeuralEmission,
	evaluateNeuralOpacity,
	transformToCanonicalFrame,
	fetchLatentCode,
	computeLOD,
	createOutputUniforms,
	createHeadUniforms,
	packLayerWeights,
	packLayerBiases,
	buildDecoderInput,
	evaluateMLP,
	packNodeInputs,
	unpackNodeInputs,
	linearLayerPacked,
	toVec3,
	linearLayer,
	applyOutputActivation,
	applyScalarOutputActivation
};
