import * as THREE from 'three';
import * as TSL from 'three/tsl';

function createEvaluateNeuralBRDFFn( material ) {

	const brdf = material.neuralAppearanceData.outputs.brdf;
	const uniforms = material._outputUniforms.brdf;

	return TSL.Fn( ( { wi, wo, latent0, latent1 } ) => {

		const latents = [
			latent0.x, latent0.y, latent0.z, latent0.w,
			latent1.x, latent1.y, latent1.z, latent1.w
		];
		const frames = buildDecoderFrames( brdf, uniforms, latents );
		const input = projectDecoderInput( latents, frames, wi, wo, brdf.inputSize );
		const decoded = toVec3( evaluateMLP( brdf.layers, uniforms.layers, input ) );

		return applyOutputActivation( decoded, brdf.outputActivation ).mul( wi.z.max( 0 ) );

	} ).setLayout( {
		name: `evaluateNeuralBRDF_${ material.id }`,
		type: 'vec3',
		inputs: [
			{ name: 'wi', type: 'vec3' },
			{ name: 'wo', type: 'vec3' },
			{ name: 'latent0', type: 'vec4' },
			{ name: 'latent1', type: 'vec4' }
		]
	} );

}

function evaluateNeuralBRDF( material, lightDirection, context, evaluateFn ) {

	const fragment = context || createNeuralFragmentContext( material );
	const fn = evaluateFn || createEvaluateNeuralBRDFFn( material );
	const wi = transformToCanonicalFrame( lightDirection );

	if ( fragment.trilinear ) {

		const rgb0 = fn( wi, fragment.viewDirection, fragment.texel00, fragment.texel01 );
		const rgb1 = fn( wi, fragment.viewDirection, fragment.texel10, fragment.texel11 );

		return TSL.mix( rgb0, rgb1, fragment.fracMip );

	}

	return fn( wi, fragment.viewDirection, fragment.texel0, fragment.texel1 );

}

function evaluateNeuralEmission( material ) {

	return TSL.Fn( () => {

		const output = material.neuralAppearanceData.outputs.emission;
		const uniforms = material._outputUniforms.emission;
		const uvNode = TSL.uv();
		const data = material.neuralAppearanceData;

		if ( material.lodMode === 'trilinear' ) {

			const continuousLod = computeContinuousLOD( material, uvNode ).toVar();
			const baseMip = TSL.floor( continuousLod ).toVar();
			const fracMip = TSL.fract( continuousLod ).toVar();
			const nextMip = TSL.min( baseMip.add( 1 ), data.mipLevels - 1 );
			const decoded = TSL.array( 'vec3', 2 ).toVar();

			TSL.Loop( { start: 0, end: 2, type: 'int', name: 'm', condition: '<' }, ( { m } ) => {

				const level = TSL.select( m.equal( TSL.int( 0 ) ), baseMip, nextMip );
				const latents = fetchLatentCodeAtLevel( material, uvNode, level );
				const rgb = applyOutputActivation( toVec3( evaluateMLP( output.layers, uniforms.layers, latents ) ), output.outputActivation );

				decoded.element( m ).assign( rgb );

			} );

			return TSL.mix( decoded.element( 0 ), decoded.element( 1 ), fracMip );

		}

		const lod = computeLOD( material, uvNode );
		const latents = fetchLatentCodeAtLevel( material, uvNode, lod );
		const decoded = toVec3( evaluateMLP( output.layers, uniforms.layers, latents ) );

		return applyOutputActivation( decoded, output.outputActivation );

	}, 'vec3' )();

}

function evaluateNeuralOpacity( material ) {

	return TSL.Fn( () => {

		const output = material.neuralAppearanceData.outputs.opacity;
		const uniforms = material._outputUniforms.opacity;
		const uvNode = TSL.uv();
		const data = material.neuralAppearanceData;

		if ( material.lodMode === 'trilinear' ) {

			const continuousLod = computeContinuousLOD( material, uvNode ).toVar();
			const baseMip = TSL.floor( continuousLod ).toVar();
			const fracMip = TSL.fract( continuousLod ).toVar();
			const nextMip = TSL.min( baseMip.add( 1 ), data.mipLevels - 1 );
			const decoded = TSL.array( 'float', 2 ).toVar();

			TSL.Loop( { start: 0, end: 2, type: 'int', name: 'm', condition: '<' }, ( { m } ) => {

				const level = TSL.select( m.equal( TSL.int( 0 ) ), baseMip, nextMip );
				const latents = fetchLatentCodeAtLevel( material, uvNode, level );
				const opacity = applyScalarOutputActivation( evaluateMLP( output.layers, uniforms.layers, latents )[ 0 ], output.outputActivation );

				decoded.element( m ).assign( opacity );

			} );

			return TSL.mix( decoded.element( 0 ), decoded.element( 1 ), fracMip );

		}

		const lod = computeLOD( material, uvNode );
		const latents = fetchLatentCodeAtLevel( material, uvNode, lod );
		const decoded = evaluateMLP( output.layers, uniforms.layers, latents )[ 0 ];

		return applyScalarOutputActivation( decoded, output.outputActivation );

	}, 'float' )();

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

	const uvNode = TSL.uv();
	const lod = computeLOD( material, uvNode );
	return fetchLatentCodeAtLevel( material, uvNode, lod );

}

function fetchLatentTexels( material, uvNode, levelNode ) {

	const data = material.neuralAppearanceData;

	return {
		texel0: TSL.texture( data.latentTextures[ 0 ], uvNode ).level( levelNode ).toVar(),
		texel1: TSL.texture( data.latentTextures[ 1 ], uvNode ).level( levelNode ).toVar()
	};

}

function fetchLatentCodeAtLevel( material, uvNode, levelNode ) {

	const texels = fetchLatentTexels( material, uvNode, levelNode );

	return [
		texels.texel0.x, texels.texel0.y, texels.texel0.z, texels.texel0.w,
		texels.texel1.x, texels.texel1.y, texels.texel1.z, texels.texel1.w
	];

}

function createNeuralFragmentContext( material ) {

	const uvNode = TSL.uv();
	const viewDirection = transformToCanonicalFrame( TSL.positionViewDirection ).toVar();

	if ( material.lodMode === 'trilinear' ) {

		const data = material.neuralAppearanceData;
		const continuousLod = computeContinuousLOD( material, uvNode ).toVar();
		const baseMip = TSL.floor( continuousLod ).toVar();
		const fracMip = TSL.fract( continuousLod ).toVar();
		const nextMip = TSL.min( baseMip.add( 1 ), data.mipLevels - 1 );
		const mip0 = fetchLatentTexels( material, uvNode, baseMip );
		const mip1 = fetchLatentTexels( material, uvNode, nextMip );

		return {
			trilinear: true,
			fracMip,
			viewDirection,
			texel00: mip0.texel0,
			texel01: mip0.texel1,
			texel10: mip1.texel0,
			texel11: mip1.texel1
		};

	}

	const lod = computeLOD( material, uvNode ).toVar();
	const texels = fetchLatentTexels( material, uvNode, lod );

	return {
		trilinear: false,
		viewDirection,
		texel0: texels.texel0,
		texel1: texels.texel1
	};

}

function computeContinuousLOD( material, uvNode ) {

	const data = material.neuralAppearanceData;
	const fixedMip = material._fixedMipLevelNode;
	const duvdx = TSL.dFdx( uvNode ).mul( TSL.vec2( data.latentWidth, data.latentHeight ) );
	const duvdy = TSL.dFdy( uvNode ).mul( TSL.vec2( data.latentWidth, data.latentHeight ) );
	const footprint = TSL.max( TSL.length( duvdx ), TSL.length( duvdy ) ).max( 1.0 );
	const computed = TSL.log2( footprint ).clamp( 0, data.mipLevels - 1 );

	return TSL.select( fixedMip.greaterThanEqual( 0 ), fixedMip, computed );

}

function computeLOD( material, uvNode ) {

	const data = material.neuralAppearanceData;
	const fixedMip = material._fixedMipLevelNode;
	const computed = computeContinuousLOD( material, uvNode );
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

	const frames = buildDecoderFrames( decoder, decoderUniforms, latents );
	return projectDecoderInput( latents, frames, wi, wo, decoder.inputSize );

}

function buildDecoderFrames( decoder, decoderUniforms, latents ) {

	if ( decoder.rotation === null ) {

		throw new Error( 'THREE.NeuralAppearanceNodeMaterial: A two-frame rotation decoder is required.' );

	}

	const rotation = linearLayer( latents, decoderUniforms.rotationWeights, null, decoder.rotation.outputSize, 'linear' );
	const frames = [];

	for ( let frame = 0; frame < 2; frame ++ ) {

		const offset = frame * 6;
		const n = TSL.vec3( rotation.element( offset ), rotation.element( offset + 1 ), rotation.element( offset + 2 ).add( 1 ) ).normalize().toVar();
		const t = TSL.vec3( rotation.element( offset + 3 ).add( 1 ), rotation.element( offset + 4 ), rotation.element( offset + 5 ) ).normalize().toVar();
		const b = TSL.cross( n, t ).normalize().toVar();

		frames.push( { n, t, b } );

	}

	return frames;

}

function projectDecoderInput( latents, frames, wi, wo, inputSize ) {

	const input = [];

	for ( let i = 0; i < 8; i ++ ) {

		input.push( latents[ i ] );

	}

	for ( let frame = 0; frame < frames.length; frame ++ ) {

		const basis = frames[ frame ];

		input.push( wi.dot( basis.t ), wi.dot( basis.b ), wi.dot( basis.n ) );
		input.push( wo.dot( basis.t ), wo.dot( basis.b ), wo.dot( basis.n ) );

	}

	if ( input.length !== inputSize ) {

		throw new Error( `THREE.NeuralAppearanceNodeMaterial: Decoder input has ${ input.length } values, expected ${ inputSize }.` );

	}

	return input;

}

function evaluateMLP( layers, layerUniforms, inputs ) {

	const maxVecCount = Math.max(
		Math.ceil( inputs.length / 4 ),
		...layers.map( ( layer ) => Math.max(
			Math.ceil( layer.inputSize / 4 ),
			Math.ceil( layer.outputSize / 4 )
		) )
	);
	const bufferA = TSL.array( 'vec4', maxVecCount ).toVar();
	const bufferB = TSL.array( 'vec4', maxVecCount ).toVar();

	copyInputsToBuffer( inputs, bufferA );

	let readBuffer = bufferA;
	let writeBuffer = bufferB;

	for ( let i = 0; i < layers.length; i ++ ) {

		const layer = layers[ i ];
		const layerUniform = layerUniforms[ i ];

		linearLayerPacked( readBuffer, writeBuffer, layerUniform.weights, layerUniform.biases, layer.inputSize, layer.outputSize, layer.activation );

		const swap = readBuffer;
		readBuffer = writeBuffer;
		writeBuffer = swap;

	}

	return unpackNodeInputs( readBuffer, layers[ layers.length - 1 ].outputSize );

}

function copyInputsToBuffer( inputs, buffer ) {

	const inputVectorCount = Math.ceil( inputs.length / 4 );

	for ( let i = 0; i < inputVectorCount; i ++ ) {

		const offset = i * 4;

		buffer.element( i ).assign( TSL.vec4(
			inputs[ offset ] || 0,
			inputs[ offset + 1 ] || 0,
			inputs[ offset + 2 ] || 0,
			inputs[ offset + 3 ] || 0
		) );

	}

}

function packNodeInputs( inputs ) {

	const buffer = TSL.array( 'vec4', Math.ceil( inputs.length / 4 ) ).toVar();
	copyInputsToBuffer( inputs, buffer );
	return buffer;

}

function unpackNodeInputs( inputs, outputSize ) {

	const outputs = [];

	for ( let outputIndex = 0; outputIndex < outputSize; outputIndex ++ ) {

		const vector = inputs.element( Math.floor( outputIndex / 4 ) );
		const component = outputIndex % 4;

		outputs.push( component === 0 ? vector.x : ( component === 1 ? vector.y : ( component === 2 ? vector.z : vector.w ) ) );

	}

	return outputs;

}

function packedWeightDot( inputVector, weights, outputIndex, inputVectorCount, outputSize, vectorIndex ) {

	const clampedOutput = TSL.min( outputIndex, TSL.int( outputSize - 1 ) );
	const contrib = TSL.dot( inputVector, weights.element( clampedOutput.mul( inputVectorCount ).add( vectorIndex ) ) );

	return TSL.select( outputIndex.lessThan( TSL.int( outputSize ) ), contrib, TSL.float( 0 ) );

}

function linearLayerPacked( inputs, outputs, weights, biases, inputSize, outputSize, activation ) {

	const inputVectorCount = Math.ceil( inputSize / 4 );
	const outputVectorCount = Math.ceil( outputSize / 4 );
	const inputVectorCountNode = TSL.int( inputVectorCount );

	TSL.Loop( { start: 0, end: outputVectorCount, type: 'int', name: 'o', condition: '<' }, ( { o } ) => {

		const acc = biases.element( o ).toVar();
		const row = o.mul( TSL.int( 4 ) );

		TSL.Loop( { start: 0, end: inputVectorCount, type: 'int', name: 'v', condition: '<' }, ( { v } ) => {

			const inputVector = inputs.element( v );
			const contrib = TSL.vec4( 0 ).toVar();

			TSL.Loop( { start: 0, end: 4, type: 'int', name: 'c', condition: '<' }, ( { c } ) => {

				const d = packedWeightDot( inputVector, weights, row.add( c ), inputVectorCountNode, outputSize, v );

				contrib.addAssign( TSL.vec4(
					TSL.select( c.equal( TSL.int( 0 ) ), d, TSL.float( 0 ) ),
					TSL.select( c.equal( TSL.int( 1 ) ), d, TSL.float( 0 ) ),
					TSL.select( c.equal( TSL.int( 2 ) ), d, TSL.float( 0 ) ),
					TSL.select( c.equal( TSL.int( 3 ) ), d, TSL.float( 0 ) )
				) );

			} );

			acc.addAssign( contrib );

		} );

		if ( activation === 'relu' ) {

			acc.assign( acc.max( 0 ) );

		}

		outputs.element( o ).assign( acc );

	} );

}

function linearLayer( inputs, weights, biases, outputSize, activation ) {

	const inputVectors = packNodeInputs( inputs );
	const inputVectorCount = Math.ceil( inputs.length / 4 );
	const outputs = TSL.array( 'float', outputSize ).toVar();
	const inputVectorCountNode = TSL.int( inputVectorCount );

	TSL.Loop( { start: 0, end: outputSize, type: 'int', name: 'o', condition: '<' }, ( { o } ) => {

		const acc = TSL.float( 0 ).toVar();

		if ( biases ) {

			acc.assign( biases.element( o ) );

		}

		TSL.Loop( { start: 0, end: inputVectorCount, type: 'int', name: 'v', condition: '<' }, ( { v } ) => {

			acc.addAssign( TSL.dot( inputVectors.element( v ), weights.element( o.mul( inputVectorCountNode ).add( v ) ) ) );

		} );

		if ( activation === 'relu' ) {

			acc.assign( acc.max( 0 ) );

		}

		outputs.element( o ).assign( acc );

	} );

	return outputs;

}

function toVec3( values ) {

	if ( values.length !== 3 ) {

		throw new Error( 'THREE.NeuralAppearanceNodeMaterial: Decoder output must be RGB.' );

	}

	return TSL.vec3( values[ 0 ], values[ 1 ], values[ 2 ] );

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
	createEvaluateNeuralBRDFFn,
	createNeuralFragmentContext,
	transformToCanonicalFrame,
	fetchLatentCode,
	fetchLatentCodeAtLevel,
	computeContinuousLOD,
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
