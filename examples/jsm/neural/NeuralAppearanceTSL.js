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
		const decoded = toVec3( evaluateMLP( brdf.layers, uniforms, input ) );

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

	const output = material.neuralAppearanceData.outputs.emission;
	const uniforms = material._outputUniforms.emission;
	const uvNode = TSL.uv();

	if ( material.lodMode === 'trilinear' ) {

		const data = material.neuralAppearanceData;
		const continuousLod = computeContinuousLOD( material, uvNode );
		const baseMip = TSL.floor( continuousLod );
		const fracMip = TSL.fract( continuousLod );
		const nextMip = TSL.min( baseMip.add( 1 ), data.mipLevels - 1 );
		const rgb0 = applyOutputActivation( toVec3( evaluateMLP( output.layers, uniforms, fetchLatentCodeAtLevel( material, uvNode, baseMip ) ) ), output.outputActivation );
		const rgb1 = applyOutputActivation( toVec3( evaluateMLP( output.layers, uniforms, fetchLatentCodeAtLevel( material, uvNode, nextMip ) ) ), output.outputActivation );

		return TSL.mix( rgb0, rgb1, fracMip );

	}

	const decoded = toVec3( evaluateMLP( output.layers, uniforms, fetchLatentCode( material ) ) );

	return applyOutputActivation( decoded, output.outputActivation );

}

function evaluateNeuralOpacity( material ) {

	const output = material.neuralAppearanceData.outputs.opacity;
	const uniforms = material._outputUniforms.opacity;
	const uvNode = TSL.uv();

	if ( material.lodMode === 'trilinear' ) {

		const data = material.neuralAppearanceData;
		const continuousLod = computeContinuousLOD( material, uvNode );
		const baseMip = TSL.floor( continuousLod );
		const fracMip = TSL.fract( continuousLod );
		const nextMip = TSL.min( baseMip.add( 1 ), data.mipLevels - 1 );
		const opacity0 = applyScalarOutputActivation( evaluateMLP( output.layers, uniforms, fetchLatentCodeAtLevel( material, uvNode, baseMip ) )[ 0 ], output.outputActivation );
		const opacity1 = applyScalarOutputActivation( evaluateMLP( output.layers, uniforms, fetchLatentCodeAtLevel( material, uvNode, nextMip ) )[ 0 ], output.outputActivation );

		return TSL.mix( opacity0, opacity1, fracMip );

	}

	const decoded = evaluateMLP( output.layers, uniforms, fetchLatentCode( material ) )[ 0 ];

	return applyScalarOutputActivation( decoded, output.outputActivation );

}

function evaluateNeuralIBL( material, envNode, context = null ) {

	const fragment = context || createNeuralFragmentContext( material );

	if ( fragment.trilinear ) {

		const ibl0 = evaluateNeuralIBLForTexels( material, envNode, fragment.viewDirection, fragment.texel00, fragment.texel01 );
		const ibl1 = evaluateNeuralIBLForTexels( material, envNode, fragment.viewDirection, fragment.texel10, fragment.texel11 );

		return TSL.mix( ibl0, ibl1, fragment.fracMip );

	}

	return evaluateNeuralIBLForTexels( material, envNode, fragment.viewDirection, fragment.texel0, fragment.texel1 );

}

function evaluateNeuralIBLForTexels( material, envNode, wo, latent0, latent1 ) {

	const ibl = material.neuralAppearanceData.outputs.ibl;
	const uniforms = material._outputUniforms.ibl;
	const latents = [
		latent0.x, latent0.y, latent0.z, latent0.w,
		latent1.x, latent1.y, latent1.z, latent1.w
	];
	const frames = buildDecoderFrames( material.neuralAppearanceData.outputs.brdf, material._outputUniforms.brdf, latents );
	const queryInput = projectIBLInput( latents, frames, wo, ibl.inputSize );
	const query = evaluateMLP( ibl.layers, uniforms, queryInput );
	const queryDirection = TSL.vec3( query[ 0 ], query[ 1 ], query[ 2 ] ).normalize();
	const queryRoughness = TSL.float( 1 ).div( TSL.float( 1 ).add( TSL.exp( query[ 3 ].negate() ) ) );
	const incoming = envNode.context( {
		getUV: () => canonicalToWorldDirection( queryDirection ),
		getTextureLevel: () => queryRoughness
	} ).isolate().mul( TSL.materialEnvIntensity );

	if ( material.neuralAppearanceData.outputs.indirect ) {

		const indirect = material.neuralAppearanceData.outputs.indirect;
		const indirectInput = projectIndirectInput( latents, wo, incoming, indirect.inputSize );
		const decoded = toVec3( evaluateMLP( indirect.layers, material._outputUniforms.indirect, indirectInput ) );

		return applyOutputActivation( decoded, indirect.outputActivation );

	}

	return incoming;

}

function getCanonicalViewBasis() {

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

	return { tangent, bitangent, normal };

}

function canonicalToViewDirection( direction ) {

	const { tangent, bitangent, normal } = getCanonicalViewBasis();

	return tangent.mul( direction.x ).add( bitangent.mul( direction.y ) ).add( normal.mul( direction.z ) ).normalize();

}

function canonicalToWorldDirection( direction ) {

	return canonicalToViewDirection( direction ).transformDirection( TSL.cameraWorldMatrix );

}

function transformToCanonicalFrame( direction ) {

	const { tangent, bitangent, normal } = getCanonicalViewBasis();

	return TSL.vec3(
		direction.dot( tangent ),
		direction.dot( bitangent ),
		direction.dot( normal )
	).normalize();

}

function latentsFromTexels( texel0, texel1 ) {

	return [
		texel0.x, texel0.y, texel0.z, texel0.w,
		texel1.x, texel1.y, texel1.z, texel1.w
	];

}

function evaluateLearnedCanonicalNormal( material, fragment ) {

	const brdf = material.neuralAppearanceData.outputs.brdf;
	const uniforms = material._outputUniforms.brdf;

	if ( fragment.trilinear ) {

		const normal0 = buildDecoderFrames( brdf, uniforms, latentsFromTexels( fragment.texel00, fragment.texel01 ) )[ 0 ].n;
		const normal1 = buildDecoderFrames( brdf, uniforms, latentsFromTexels( fragment.texel10, fragment.texel11 ) )[ 0 ].n;

		return TSL.mix( normal0, normal1, fragment.fracMip ).normalize();

	}

	return buildDecoderFrames( brdf, uniforms, latentsFromTexels( fragment.texel0, fragment.texel1 ) )[ 0 ].n;

}

function sigmoidNode( value ) {

	return TSL.float( 1 ).div( TSL.float( 1 ).add( TSL.exp( value.negate() ) ) );

}

function evaluateLearnedIBLQueryForTexels( material, texel0, texel1, wo ) {

	const latents = latentsFromTexels( texel0, texel1 );
	const ibl = material.neuralAppearanceData.outputs.ibl;
	const frames = buildDecoderFrames( material.neuralAppearanceData.outputs.brdf, material._outputUniforms.brdf, latents );
	const input = projectIBLInput( latents, frames, wo, ibl.inputSize );
	const output = evaluateMLP( ibl.layers, material._outputUniforms.ibl, input );

	return {
		direction: TSL.vec3( output[ 0 ], output[ 1 ], output[ 2 ] ).normalize(),
		roughness: sigmoidNode( output[ 3 ] )
	};

}

function evaluateLearnedIBLQuery( material, fragment ) {

	if ( fragment.trilinear ) {

		const query0 = evaluateLearnedIBLQueryForTexels( material, fragment.texel00, fragment.texel01, fragment.viewDirection );
		const query1 = evaluateLearnedIBLQueryForTexels( material, fragment.texel10, fragment.texel11, fragment.viewDirection );

		return {
			direction: TSL.mix( query0.direction, query1.direction, fragment.fracMip ).normalize(),
			roughness: TSL.mix( query0.roughness, query1.roughness, fragment.fracMip )
		};

	}

	return evaluateLearnedIBLQueryForTexels( material, fragment.texel0, fragment.texel1, fragment.viewDirection );

}

/**
 * Decodes learned shading-frame intermediates used by the debug visualizer.
 *
 * @param {NeuralAppearanceNodeMaterial} material - The neural appearance material.
 * @return {{ viewNormal: Node<vec3>, viewReflect: Node<vec3>, roughness: Node<float> }} Debug values.
 */
function evaluateNeuralDebugShading( material ) {

	const fragment = createNeuralFragmentContext( material );
	const canonicalNormal = evaluateLearnedCanonicalNormal( material, fragment );
	const query = material.neuralAppearanceData.outputs.ibl ?
		evaluateLearnedIBLQuery( material, fragment ) :
		{ direction: fragment.viewDirection.negate().reflect( canonicalNormal ).normalize(), roughness: TSL.float( 1 ) };

	return {
		viewNormal: canonicalToViewDirection( canonicalNormal ),
		viewReflect: canonicalToViewDirection( query.direction ),
		roughness: query.roughness
	};

}

function packDebugDirection( direction ) {

	return TSL.colorSpaceToWorking( TSL.vec4( TSL.packNormalToRGB( direction ), 1 ), THREE.SRGBColorSpace ).xyz;

}

function packDebugScalar( value ) {

	return TSL.colorSpaceToWorking( TSL.vec4( TSL.vec3( value ), 1 ), THREE.SRGBColorSpace ).xyz;

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
		brdf: createHeadUniforms( outputs.brdf ),
		ibl: createHeadUniforms( outputs.ibl )
	};

	if ( outputs.indirect ) uniforms.indirect = createHeadUniforms( outputs.indirect );
	if ( outputs.emission ) uniforms.emission = createHeadUniforms( outputs.emission );
	if ( outputs.opacity ) uniforms.opacity = createHeadUniforms( outputs.opacity );

	return uniforms;

}

function createHeadUniforms( decoder ) {

	const packed = packHeadParameters( decoder );

	return {
		parameters: TSL.uniformArray( packed.values, 'vec4' ),
		rotationWeightsOffset: packed.rotationWeightsOffset,
		layers: packed.layers
	};

}

function isCompatibleNeuralAppearanceData( current, next ) {

	if ( ! current || ! next ) return false;
	if ( current.latentWidth !== next.latentWidth ) return false;
	if ( current.latentHeight !== next.latentHeight ) return false;
	if ( current.mipLevels !== next.mipLevels ) return false;
	if ( current.wrap !== next.wrap ) return false;
	if ( ! sameLatentTextureLayout( current.latentTextures, next.latentTextures ) ) return false;
	if ( ! sameHeadArchitecture( current.outputs.brdf, next.outputs.brdf ) ) return false;
	if ( ! sameHeadArchitecture( current.outputs.ibl, next.outputs.ibl ) ) return false;
	if ( ! sameHeadArchitecture( current.outputs.indirect, next.outputs.indirect ) ) return false;
	if ( ! sameHeadArchitecture( current.outputs.emission, next.outputs.emission ) ) return false;
	if ( ! sameHeadArchitecture( current.outputs.opacity, next.outputs.opacity ) ) return false;

	return true;

}

function sameLatentTextureLayout( currentTextures, nextTextures ) {

	if ( ! currentTextures || ! nextTextures || currentTextures.length !== nextTextures.length ) return false;

	for ( let i = 0; i < currentTextures.length; i ++ ) {

		const currentMips = currentTextures[ i ].mipmaps;
		const nextMips = nextTextures[ i ].mipmaps;

		if ( currentMips.length !== nextMips.length ) return false;

		for ( let mip = 0; mip < currentMips.length; mip ++ ) {

			if ( currentMips[ mip ].width !== nextMips[ mip ].width ) return false;
			if ( currentMips[ mip ].height !== nextMips[ mip ].height ) return false;
			if ( currentMips[ mip ].data.length !== nextMips[ mip ].data.length ) return false;

		}

	}

	return true;

}

function sameHeadArchitecture( current, next ) {

	if ( Boolean( current ) !== Boolean( next ) ) return false;
	if ( ! current ) return true;
	if ( current.inputSize !== next.inputSize ) return false;
	if ( Boolean( current.rotation ) !== Boolean( next.rotation ) ) return false;

	if ( current.rotation ) {

		if ( current.rotation.inputSize !== next.rotation.inputSize ) return false;
		if ( current.rotation.outputSize !== next.rotation.outputSize ) return false;

	}

	if ( current.layers.length !== next.layers.length ) return false;

	for ( let i = 0; i < current.layers.length; i ++ ) {

		if ( current.layers[ i ].inputSize !== next.layers[ i ].inputSize ) return false;
		if ( current.layers[ i ].outputSize !== next.layers[ i ].outputSize ) return false;
		if ( current.layers[ i ].activation !== next.layers[ i ].activation ) return false;

	}

	const currentActivation = current.outputActivation || { type: 'linear' };
	const nextActivation = next.outputActivation || { type: 'linear' };

	if ( currentActivation.type !== nextActivation.type ) return false;
	if ( ( currentActivation.offset || 0 ) !== ( nextActivation.offset || 0 ) ) return false;
	if ( ( currentActivation.scale !== undefined ? currentActivation.scale : 1 ) !== ( nextActivation.scale !== undefined ? nextActivation.scale : 1 ) ) return false;
	if ( ( current.mode || 'mask' ) !== ( next.mode || 'mask' ) ) return false;
	if ( ( current.alphaCutoff !== undefined ? current.alphaCutoff : null ) !== ( next.alphaCutoff !== undefined ? next.alphaCutoff : null ) ) return false;

	return true;

}

function updateOutputUniforms( uniforms, outputs ) {

	updateHeadUniforms( uniforms.brdf, outputs.brdf );
	updateHeadUniforms( uniforms.ibl, outputs.ibl );
	if ( outputs.indirect ) updateHeadUniforms( uniforms.indirect, outputs.indirect );
	if ( outputs.emission ) updateHeadUniforms( uniforms.emission, outputs.emission );
	if ( outputs.opacity ) updateHeadUniforms( uniforms.opacity, outputs.opacity );

}

function updateHeadUniforms( uniforms, decoder ) {

	copyPackedVectors( uniforms.parameters.array, packHeadParameters( decoder ).values );

}

function copyPackedVectors( targetArray, packed ) {

	if ( targetArray.length !== packed.length ) {

		throw new Error( `THREE.NeuralAppearanceNodeMaterial: Packed uniform length mismatch (${ targetArray.length } !== ${ packed.length }).` );

	}

	for ( let i = 0; i < packed.length; i ++ ) {

		targetArray[ i ].copy( packed[ i ] );

	}

}

function copyLatentTextureData( destinationTextures, sourceTextures ) {

	for ( let i = 0; i < destinationTextures.length; i ++ ) {

		const destination = destinationTextures[ i ];
		const source = sourceTextures[ i ];

		for ( let mip = 0; mip < destination.mipmaps.length; mip ++ ) {

			destination.mipmaps[ mip ].data.set( source.mipmaps[ mip ].data );

		}

		if ( destination.image && destination.image.data && destination.image.data !== destination.mipmaps[ 0 ].data ) {

			destination.image.data.set( source.image.data );

		}

		destination.needsUpdate = true;

	}

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

function packHeadParameters( decoder ) {

	const values = [];
	const layers = [];
	let rotationWeightsOffset = null;

	if ( decoder.rotation ) {

		rotationWeightsOffset = values.length;
		values.push( ...packLayerWeights( decoder.rotation.weights, decoder.rotation.inputSize, decoder.rotation.outputSize ) );

	}

	for ( const layer of decoder.layers ) {

		const weightsOffset = values.length;
		values.push( ...packLayerWeights( layer.weights, layer.inputSize, layer.outputSize ) );

		const biasesOffset = values.length;
		values.push( ...packLayerBiases( layer.biases ) );

		layers.push( { weightsOffset, biasesOffset } );

	}

	return { values, rotationWeightsOffset, layers };

}

function buildDecoderInput( decoder, decoderUniforms, latents, wi, wo ) {

	const frames = buildDecoderFrames( decoder, decoderUniforms, latents );
	return projectDecoderInput( latents, frames, wi, wo, decoder.inputSize );

}

function buildDecoderFrames( decoder, decoderUniforms, latents ) {

	if ( decoder.rotation === null ) {

		throw new Error( 'THREE.NeuralAppearanceNodeMaterial: A two-frame rotation decoder is required.' );

	}

	const rotation = linearLayer( latents, decoderUniforms.parameters, decoderUniforms.rotationWeightsOffset, null, decoder.rotation.outputSize, 'linear' );
	const frames = [];

	for ( let frame = 0; frame < 2; frame ++ ) {

		const offset = frame * 6;
		const n = TSL.vec3( rotation[ offset ], rotation[ offset + 1 ], rotation[ offset + 2 ].add( 1 ) ).normalize().toVar();
		const t = TSL.vec3( rotation[ offset + 3 ].add( 1 ), rotation[ offset + 4 ], rotation[ offset + 5 ] ).normalize().toVar();
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

function projectIBLInput( latents, frames, wo, inputSize ) {

	const input = [];

	for ( let i = 0; i < 8; i ++ ) {

		input.push( latents[ i ] );

	}

	for ( let frame = 0; frame < frames.length; frame ++ ) {

		const basis = frames[ frame ];
		input.push( wo.dot( basis.t ), wo.dot( basis.b ), wo.dot( basis.n ) );

	}

	if ( input.length !== inputSize ) {

		throw new Error( `THREE.NeuralAppearanceNodeMaterial: IBL decoder input has ${ input.length } values, expected ${ inputSize }.` );

	}

	return input;

}

function projectIndirectInput( latents, wo, incoming, inputSize ) {

	const input = [];

	for ( let i = 0; i < 8; i ++ ) {

		input.push( latents[ i ] );

	}

	input.push( wo.x, wo.y, wo.z );
	input.push( incoming.x, incoming.y, incoming.z );

	if ( input.length !== inputSize ) {

		throw new Error( `THREE.NeuralAppearanceNodeMaterial: Indirect decoder input has ${ input.length } values, expected ${ inputSize }.` );

	}

	return input;

}

function evaluateMLP( layers, uniforms, inputs ) {

	let activations = packNodeInputs( inputs );

	for ( let i = 0; i < layers.length; i ++ ) {

		const layer = layers[ i ];
		const layerUniform = uniforms.layers[ i ];

		activations = linearLayerPacked( activations, uniforms.parameters, layerUniform.weightsOffset, layerUniform.biasesOffset, layer.inputSize, layer.outputSize, layer.activation );

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

function linearLayerPacked( inputs, parameters, weightsOffset, biasesOffset, inputSize, outputSize, activation ) {

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

					sums[ component ] = sums[ component ].add( TSL.dot( inputVector, parameters.element( weightsOffset + outputIndex * inputVectorCount + vectorIndex ) ) );

				}

			}

		}

		let value = parameters.element( biasesOffset + outputVector ).add( TSL.vec4( sums[ 0 ], sums[ 1 ], sums[ 2 ], sums[ 3 ] ) );

		if ( activation === 'relu' ) {

			value = value.max( 0 );

		}

		outputs.push( value );

	}

	return outputs;

}

function linearLayer( inputs, parameters, weightsOffset, biasesOffset, outputSize, activation ) {

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

		let value = biasesOffset !== null ? parameters.element( biasesOffset + outputIndex ) : TSL.float( 0 );

		for ( let vectorIndex = 0; vectorIndex < inputVectorCount; vectorIndex ++ ) {

			value = value.add( TSL.dot( inputVectors[ vectorIndex ], parameters.element( weightsOffset + outputIndex * inputVectorCount + vectorIndex ) ) );

		}

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
	evaluateNeuralIBL,
	evaluateNeuralOpacity,
	evaluateNeuralDebugShading,
	packDebugDirection,
	packDebugScalar,
	createEvaluateNeuralBRDFFn,
	createNeuralFragmentContext,
	transformToCanonicalFrame,
	fetchLatentCode,
	fetchLatentCodeAtLevel,
	computeContinuousLOD,
	computeLOD,
	createOutputUniforms,
	createHeadUniforms,
	isCompatibleNeuralAppearanceData,
	updateOutputUniforms,
	copyLatentTextureData,
	packLayerWeights,
	packLayerBiases,
	buildDecoderInput,
	projectIBLInput,
	canonicalToViewDirection,
	canonicalToWorldDirection,
	evaluateMLP,
	packNodeInputs,
	unpackNodeInputs,
	linearLayerPacked,
	toVec3,
	linearLayer,
	applyOutputActivation,
	applyScalarOutputActivation
};
