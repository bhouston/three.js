import {
	Fn,
	If,
	Loop,
	atomicAdd,
	atomicLoad,
	clamp,
	cross,
	exp,
	float,
	floor,
	instanceIndex,
	int,
	log,
	max,
	pow,
	select,
	sign,
	vec3
} from 'three/tsl';
import {
	FIXED_POINT_SCALE,
	GRADIENT_NORM_SCALE
} from '../neural/NeuralGPUTrainingConstants.js';
import {
	wrapIndexTSL,
	forwardDenseLayerTSL,
	accumulateDenseLayerGradTSL,
	backwardDenseLayerReLUTSL,
	createAdamComputeNode
} from '../neural/NeuralGPUComputeTSL.js';
import {
	IBL_OUTPUT_SIZE,
	INDIRECT_OUTPUT_SIZE,
	CHANNELS_PER_LEVEL
} from './NeuralAppearanceFormat.js';

const OUTPUT_CLAMP_GRADIENT_LEAK = 0.01;

function backwardNormalizeTSL( raw, norm, gradNorm ) {

	const rawLen = max( raw.length(), float( 1e-10 ) );
	const invLen = float( 1.0 ).div( rawLen );
	const proj = norm.dot( gradNorm );

	return gradNorm.sub( norm.mul( proj ) ).mul( invLen );

}

/**
 * Computes one learned tangent-space coordinate frame (n, t, b) from the
 * latent code at `actA0Base`, via the two 3-row rotation projections
 * (raw N, raw T) at `rotationOffset + frameOffset`, each Gram-Schmidt'd
 * into an orthonormal frame. Shared between the forward pass (step 3 below,
 * which also stores wi/wo (.) frame projections) and the backward pass
 * (step 11, which needs the same n/t/b - and the un-normalized rawN/rawT/
 * rawB - to backprop through) - the two are otherwise identical
 * computations that used to be hand-duplicated ~80 lines apart.
 */
function computeFrameTSL( { activationsStorage, weightsStorage, actA0Base, rotationOffset, latentChannels, frameOffset } ) {

	const rawN = vec3( 0.0 ).toVar();
	const rawT = vec3( 0.0 ).toVar();

	Loop( { start: 0, end: 3, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

		const rotRowOffset = int( rotationOffset ).add( frameOffset.add( j ).mul( latentChannels ) );
		const r_j = float( 0.0 ).toVar();

		Loop( { start: 0, end: latentChannels, type: 'int', name: 'c', condition: '<' }, ( { c } ) => {

			const z_c = activationsStorage.element( actA0Base.add( c ) );
			const rotW = weightsStorage.element( rotRowOffset.add( c ) );
			r_j.addAssign( rotW.mul( z_c ) );

		} );

		If( j.equal( 0 ), () => {

			rawN.x.assign( r_j );

		} ).ElseIf( j.equal( 1 ), () => {

			rawN.y.assign( r_j );

		} ).Else( () => {

			rawN.z.assign( r_j.add( 1.0 ) );

		} );

	} );

	Loop( { start: 0, end: 3, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

		const rotRowOffset = int( rotationOffset ).add( frameOffset.add( j.add( 3 ) ).mul( latentChannels ) );
		const r_j = float( 0.0 ).toVar();

		Loop( { start: 0, end: latentChannels, type: 'int', name: 'c', condition: '<' }, ( { c } ) => {

			const z_c = activationsStorage.element( actA0Base.add( c ) );
			const rotW = weightsStorage.element( rotRowOffset.add( c ) );
			r_j.addAssign( rotW.mul( z_c ) );

		} );

		If( j.equal( 0 ), () => {

			rawT.x.assign( r_j.add( 1.0 ) );

		} ).ElseIf( j.equal( 1 ), () => {

			rawT.y.assign( r_j );

		} ).Else( () => {

			rawT.z.assign( r_j );

		} );

	} );

	const n = rawN.normalize();
	const t = rawT.normalize();
	const rawB = cross( n, t );
	const b = rawB.normalize();

	return { rawN, rawT, n, t, rawB, b };

}

function trainIndirectProbeHead( {
	samplesStorage,
	sampleOffset,
	activationsStorage,
	actBase,
	weightsStorage,
	gradWeightsAtomic,
	wo,
	a0,
	iblScale,
	sampleIblLoss,
	iblHiddenSize,
	probeSampleOffset,
	targetSampleOffset,
	layer0WeightsOffset,
	layer0BiasesOffset,
	layer1WeightsOffset,
	layer1BiasesOffset,
	actIndirectA0Offset,
	actIndirectZ1Offset,
	actIndirectA1Offset,
	actIndirectZ2Offset,
	actIndirectDelta2Offset,
	actIndirectDelta1Offset,
	actGradLatentsOffset,
	latentChannels,
	indirectInputSize
} ) {

	const indirectA0 = actBase.add( int( actIndirectA0Offset ) );

	Loop( { start: 0, end: latentChannels, type: 'int', name: 'c', condition: '<' }, ( { c } ) => {

		activationsStorage.element( indirectA0.add( c ) ).assign( activationsStorage.element( a0.add( c ) ) );

	} );

	activationsStorage.element( indirectA0.add( latentChannels ) ).assign( wo.x );
	activationsStorage.element( indirectA0.add( latentChannels + 1 ) ).assign( wo.y );
	activationsStorage.element( indirectA0.add( latentChannels + 2 ) ).assign( wo.z );
	activationsStorage.element( indirectA0.add( latentChannels + 3 ) ).assign( samplesStorage.element( sampleOffset.add( probeSampleOffset ) ) );
	activationsStorage.element( indirectA0.add( latentChannels + 4 ) ).assign( samplesStorage.element( sampleOffset.add( probeSampleOffset + 1 ) ) );
	activationsStorage.element( indirectA0.add( latentChannels + 5 ) ).assign( samplesStorage.element( sampleOffset.add( probeSampleOffset + 2 ) ) );

	const indirectA1 = actBase.add( int( actIndirectA1Offset ) );

	forwardDenseLayerTSL( {
		activationsStorage, weightsStorage,
		inputBase: indirectA0, inputSize: indirectInputSize, outputSize: iblHiddenSize,
		weightsOffset: layer0WeightsOffset, biasesOffset: layer0BiasesOffset,
		zBase: actBase.add( int( actIndirectZ1Offset ) ), aBase: indirectA1
	} );

	forwardDenseLayerTSL( {
		activationsStorage, weightsStorage,
		inputBase: indirectA1, inputSize: iblHiddenSize, outputSize: INDIRECT_OUTPUT_SIZE,
		weightsOffset: layer1WeightsOffset, biasesOffset: layer1BiasesOffset,
		zBase: actBase.add( int( actIndirectZ2Offset ) ), aBase: null
	} );

	const indirectDelta2 = actBase.add( int( actIndirectDelta2Offset ) );

	Loop( { start: 0, end: INDIRECT_OUTPUT_SIZE, type: 'int', name: 'c', condition: '<' }, ( { c } ) => {

		const z3_c = activationsStorage.element( actBase.add( int( actIndirectZ2Offset ) ).add( c ) );
		const yHat_c = max( z3_c, float( 0.0 ) );
		const target_c = samplesStorage.element( sampleOffset.add( targetSampleOffset ).add( c ) );
		const predClamped = max( yHat_c, float( 1e-6 ) );
		const refClamped = max( target_c, float( 1e-6 ) );
		const predLog = pow( predClamped, float( 1.0 / 3.0 ) ).sub( 1.0 ).mul( 3.0 );
		const refLog = pow( refClamped, float( 1.0 / 3.0 ) ).sub( 1.0 ).mul( 3.0 );
		const diff = predLog.sub( refLog );
		sampleIblLoss.addAssign( diff.abs().mul( iblScale ).div( 3.0 ) );
		const leak = select( z3_c.greaterThan( 0.0 ), float( 1.0 ), float( OUTPUT_CLAMP_GRADIENT_LEAK ) );
		const grad = sign( diff ).mul( pow( predClamped, float( - 2.0 / 3.0 ) ) ).mul( iblScale ).div( 3.0 ).mul( leak );
		activationsStorage.element( indirectDelta2.add( c ) ).assign( grad );

	} );

	accumulateDenseLayerGradTSL( {
		activationsStorage, weightsStorage, gradWeightsAtomic,
		deltaBase: indirectDelta2, inputBase: indirectA1, inputSize: iblHiddenSize, outputSize: INDIRECT_OUTPUT_SIZE,
		weightsOffset: layer1WeightsOffset, biasesOffset: layer1BiasesOffset
	} );

	const indirectDelta1 = actBase.add( int( actIndirectDelta1Offset ) );

	backwardDenseLayerReLUTSL( {
		activationsStorage, weightsStorage,
		deltaBase: indirectDelta2, deltaSize: INDIRECT_OUTPUT_SIZE,
		weightsOffset: layer1WeightsOffset, prevSize: iblHiddenSize,
		prevZBase: actBase.add( int( actIndirectZ1Offset ) ), outDeltaBase: indirectDelta1
	} );

	accumulateDenseLayerGradTSL( {
		activationsStorage, weightsStorage, gradWeightsAtomic,
		deltaBase: indirectDelta1, inputBase: indirectA0, inputSize: indirectInputSize, outputSize: iblHiddenSize,
		weightsOffset: layer0WeightsOffset, biasesOffset: layer0BiasesOffset
	} );

	Loop( { start: 0, end: latentChannels, type: 'int', name: 'c', condition: '<' }, ( { c } ) => {

		const gradLatent_c = float( 0.0 ).toVar();

		Loop( { start: 0, end: iblHiddenSize, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

			const delta1_j = activationsStorage.element( indirectDelta1.add( j ) );
			const w_jc = weightsStorage.element( int( layer0WeightsOffset ).add( j.mul( indirectInputSize ) ).add( c ) );
			gradLatent_c.addAssign( delta1_j.mul( w_jc ) );

		} );

		const curGradZ = activationsStorage.element( actBase.add( int( actGradLatentsOffset ) ).add( c ) );
		activationsStorage.element( actBase.add( int( actGradLatentsOffset ) ).add( c ) ).assign( curGradZ.add( gradLatent_c ) );

	} );

}

/**
 * Creates the primary forward, loss, and backward backpropagation compute node.
 * Evaluates 1 invocation per batch sample.
 */
function createTrainBatchComputeNode( gpuModel ) {

	const {
		layout,
		batchSize,
		samplesStorage,
		activationsStorage,
		lossAtomic,
		invBatchUniform
	} = gpuModel;
	const { valuesStorage: weightsStorage, gradAtomic: gradWeightsAtomic } = gpuModel.weightsBuffers;
	const { valuesStorage: latentsStorage, gradAtomic: gradLatentsAtomic } = gpuModel.latentsBuffers;

	const {
		gridLevels,
		hiddenSize,
		iblHiddenSize,
		supportsEmission,
		supportsOpacity,
		latentChannels,
		decoderInputSize,
		iblInputSize,
		indirectInputSize,
		rotationOffset,
		layer0WeightsOffset,
		layer0BiasesOffset,
		layer1WeightsOffset,
		layer1BiasesOffset,
		layer2WeightsOffset,
		layer2BiasesOffset,
		iblLayer0WeightsOffset,
		iblLayer0BiasesOffset,
		iblLayer1WeightsOffset,
		iblLayer1BiasesOffset,
		indirectRadianceLayer0WeightsOffset,
		indirectRadianceLayer0BiasesOffset,
		indirectRadianceLayer1WeightsOffset,
		indirectRadianceLayer1BiasesOffset,
		indirectIrradianceLayer0WeightsOffset,
		indirectIrradianceLayer0BiasesOffset,
		indirectIrradianceLayer1WeightsOffset,
		indirectIrradianceLayer1BiasesOffset,
		emissionWeightsOffset,
		emissionBiasesOffset,
		opacityWeightsOffset,
		opacityBiasesOffset,
		activationStride,
		sampleStride,
		actA0Offset,
		actZ1Offset,
		actA1Offset,
		actZ2Offset,
		actA2Offset,
		actZ3Offset,
		actDelta3Offset,
		actDelta2Offset,
		actDelta1Offset,
		actGradA0Offset,
		actGradLatentsOffset,
		actIblA0Offset,
		actIblZ1Offset,
		actIblA1Offset,
		actIblZ2Offset,
		actIblDelta2Offset,
		actIblDelta1Offset,
		actIndirectA0Offset,
		actIndirectZ1Offset,
		actIndirectA1Offset,
		actIndirectZ2Offset,
		actIndirectDelta2Offset,
		actIndirectDelta1Offset
	} = layout;

	return Fn( () => {

		const sampleIdx = int( instanceIndex );
		const sampleOffset = sampleIdx.mul( int( sampleStride ) );

		// 1. Fetch sample fields
		const uvX = samplesStorage.element( sampleOffset.add( 0 ) );
		const uvY = samplesStorage.element( sampleOffset.add( 1 ) );
		const wi = vec3(
			samplesStorage.element( sampleOffset.add( 2 ) ),
			samplesStorage.element( sampleOffset.add( 3 ) ),
			samplesStorage.element( sampleOffset.add( 4 ) )
		);
		const wo = vec3(
			samplesStorage.element( sampleOffset.add( 5 ) ),
			samplesStorage.element( sampleOffset.add( 6 ) ),
			samplesStorage.element( sampleOffset.add( 7 ) )
		);
		const sampleWeight = samplesStorage.element( sampleOffset.add( 8 ) );

		// Guard against zero-weight samples
		If( sampleWeight.greaterThan( 0.0 ), () => {

			const actBase = sampleIdx.mul( int( activationStride ) );

			// 2. Bilinear-sample every latent grid level (wrap addressing) and
			// concatenate their features into a0 - this is the trainable
			// positional encoding (same multiresolution grid encoding as
			// neural-texture / neural-material, see NeuralGridModel.js). No
			// mip/LOD concept: every level always contributes.
			const levelTaps = [];

			for ( let g = 0; g < gridLevels.length; g ++ ) {

				const level = gridLevels[ g ];
				const x = uvX.mul( level.width ).sub( 0.5 );
				const y = uvY.mul( level.height ).sub( 0.5 );
				const x0 = int( floor( x ) );
				const y0 = int( floor( y ) );
				const tx = x.sub( float( x0 ) );
				const ty = y.sub( float( y0 ) );

				const w0 = float( 1.0 ).sub( tx ).mul( float( 1.0 ).sub( ty ) );
				const w1 = tx.mul( float( 1.0 ).sub( ty ) );
				const w2 = float( 1.0 ).sub( tx ).mul( ty );
				const w3 = tx.mul( ty );

				const tapX0 = wrapIndexTSL( x0, level.width );
				const tapY0 = wrapIndexTSL( y0, level.height );
				const tapX1 = wrapIndexTSL( x0.add( 1 ), level.width );
				const tapY1 = wrapIndexTSL( y0, level.height );
				const tapX2 = wrapIndexTSL( x0, level.width );
				const tapY2 = wrapIndexTSL( y0.add( 1 ), level.height );
				const tapX3 = wrapIndexTSL( x0.add( 1 ), level.width );
				const tapY3 = wrapIndexTSL( y0.add( 1 ), level.height );

				const off0 = int( level.offset ).add( tapY0.mul( level.width ).add( tapX0 ).mul( CHANNELS_PER_LEVEL ) );
				const off1 = int( level.offset ).add( tapY1.mul( level.width ).add( tapX1 ).mul( CHANNELS_PER_LEVEL ) );
				const off2 = int( level.offset ).add( tapY2.mul( level.width ).add( tapX2 ).mul( CHANNELS_PER_LEVEL ) );
				const off3 = int( level.offset ).add( tapY3.mul( level.width ).add( tapX3 ).mul( CHANNELS_PER_LEVEL ) );

				levelTaps.push( { off0, off1, off2, off3, w0, w1, w2, w3 } );

				for ( let c = 0; c < CHANNELS_PER_LEVEL; c ++ ) {

					const z_c = latentsStorage.element( off0.add( c ) ).mul( w0 )
						.add( latentsStorage.element( off1.add( c ) ).mul( w1 ) )
						.add( latentsStorage.element( off2.add( c ) ).mul( w2 ) )
						.add( latentsStorage.element( off3.add( c ) ).mul( w3 ) );

					activationsStorage.element( actBase.add( int( actA0Offset + g * CHANNELS_PER_LEVEL + c ) ) ).assign( z_c );

				}

			}

			const a0Base = actBase.add( int( actA0Offset ) );
			const a1Base = actBase.add( int( actA1Offset ) );
			const a2Base = actBase.add( int( actA2Offset ) );

			// 3. Learned coordinate frames (Frame 0 and Frame 1)
			Loop( { start: 0, end: 2, type: 'int', name: 'f', condition: '<' }, ( { f } ) => {

				const frameOffset = f.mul( 6 );
				const { n, t, b } = computeFrameTSL( { activationsStorage, weightsStorage, actA0Base: a0Base, rotationOffset, latentChannels, frameOffset } );

				const projBase = a0Base.add( latentChannels ).add( f.mul( 6 ) );
				activationsStorage.element( projBase.add( 0 ) ).assign( wi.dot( t ) );
				activationsStorage.element( projBase.add( 1 ) ).assign( wi.dot( b ) );
				activationsStorage.element( projBase.add( 2 ) ).assign( wi.dot( n ) );
				activationsStorage.element( projBase.add( 3 ) ).assign( wo.dot( t ) );
				activationsStorage.element( projBase.add( 4 ) ).assign( wo.dot( b ) );
				activationsStorage.element( projBase.add( 5 ) ).assign( wo.dot( n ) );

			} );

			// 4. Forward Layer 0 (decoderInputSize -> hiddenSize)
			forwardDenseLayerTSL( {
				activationsStorage, weightsStorage,
				inputBase: a0Base, inputSize: decoderInputSize, outputSize: hiddenSize,
				weightsOffset: layer0WeightsOffset, biasesOffset: layer0BiasesOffset,
				zBase: actBase.add( int( actZ1Offset ) ), aBase: a1Base
			} );

			// 5. Forward Layer 1 (hiddenSize -> hiddenSize)
			forwardDenseLayerTSL( {
				activationsStorage, weightsStorage,
				inputBase: a1Base, inputSize: hiddenSize, outputSize: hiddenSize,
				weightsOffset: layer1WeightsOffset, biasesOffset: layer1BiasesOffset,
				zBase: actBase.add( int( actZ2Offset ) ), aBase: a2Base
			} );

			// 6. Forward Layer 2 (hiddenSize -> 3, Linear)
			forwardDenseLayerTSL( {
				activationsStorage, weightsStorage,
				inputBase: a2Base, inputSize: hiddenSize, outputSize: 3,
				weightsOffset: layer2WeightsOffset, biasesOffset: layer2BiasesOffset,
				zBase: actBase.add( int( actZ3Offset ) ), aBase: null
			} );

			// 7. Cube-Root Power Loss & Delta 3
			const sampleDirectLoss = float( 0.0 ).toVar();
			const sampleIblLoss = float( 0.0 ).toVar();

			Loop( { start: 0, end: 3, type: 'int', name: 'c', condition: '<' }, ( { c } ) => {

				const z3_c = activationsStorage.element( actBase.add( int( actZ3Offset ) ).add( c ) );
				const yHat_c = max( z3_c, float( 0.0 ) );
				const target_c = samplesStorage.element( sampleOffset.add( 9 ).add( c ) );

				const predClamped = max( yHat_c, float( 1e-6 ) );
				const refClamped = max( target_c, float( 1e-6 ) );
				const predLog = pow( predClamped, float( 1.0 / 3.0 ) ).sub( 1.0 ).mul( 3.0 );
				const refLog = pow( refClamped, float( 1.0 / 3.0 ) ).sub( 1.0 ).mul( 3.0 );
				const diff = predLog.sub( refLog );

				const channelLoss = diff.abs().mul( sampleWeight ).mul( invBatchUniform ).div( 3.0 );
				sampleDirectLoss.addAssign( channelLoss );

				const gradPred = sign( diff ).mul( pow( predClamped, float( - 2.0 / 3.0 ) ) ).mul( sampleWeight ).mul( invBatchUniform ).div( 3.0 );
				const outputClampGradient = select( z3_c.greaterThan( 0.0 ), float( 1.0 ), float( OUTPUT_CLAMP_GRADIENT_LEAK ) );
				activationsStorage.element( actBase.add( int( actDelta3Offset ) ).add( c ) ).assign( gradPred.mul( outputClampGradient ) );

			} );

			const delta3Base = actBase.add( int( actDelta3Offset ) );
			const delta2Base = actBase.add( int( actDelta2Offset ) );
			const delta1Base = actBase.add( int( actDelta1Offset ) );

			// 8. Backward Layer 2 (hiddenSize -> 3)
			accumulateDenseLayerGradTSL( {
				activationsStorage, weightsStorage, gradWeightsAtomic,
				deltaBase: delta3Base, inputBase: a2Base, inputSize: hiddenSize, outputSize: 3,
				weightsOffset: layer2WeightsOffset, biasesOffset: layer2BiasesOffset
			} );

			backwardDenseLayerReLUTSL( {
				activationsStorage, weightsStorage,
				deltaBase: delta3Base, deltaSize: 3,
				weightsOffset: layer2WeightsOffset, prevSize: hiddenSize,
				prevZBase: actBase.add( int( actZ2Offset ) ), outDeltaBase: delta2Base
			} );

			// 9. Backward Layer 1 (hiddenSize -> hiddenSize)
			accumulateDenseLayerGradTSL( {
				activationsStorage, weightsStorage, gradWeightsAtomic,
				deltaBase: delta2Base, inputBase: a1Base, inputSize: hiddenSize, outputSize: hiddenSize,
				weightsOffset: layer1WeightsOffset, biasesOffset: layer1BiasesOffset
			} );

			backwardDenseLayerReLUTSL( {
				activationsStorage, weightsStorage,
				deltaBase: delta2Base, deltaSize: hiddenSize,
				weightsOffset: layer1WeightsOffset, prevSize: hiddenSize,
				prevZBase: actBase.add( int( actZ1Offset ) ), outDeltaBase: delta1Base
			} );

			// 10. Backward Layer 0 (decoderInputSize -> hiddenSize)
			accumulateDenseLayerGradTSL( {
				activationsStorage, weightsStorage, gradWeightsAtomic,
				deltaBase: delta1Base, inputBase: a0Base, inputSize: decoderInputSize, outputSize: hiddenSize,
				weightsOffset: layer0WeightsOffset, biasesOffset: layer0BiasesOffset
			} );

			// Compute gradA0 (decoderInputSize floats) - unlike the two
			// propagate steps above, a0 is raw (unactivated) input, so there's
			// no ReLU derivative to gate by; this stays hand-written rather
			// than using backwardDenseLayerReLUTSL.
			Loop( { start: 0, end: decoderInputSize, type: 'int', name: 'i', condition: '<' }, ( { i } ) => {

				const gradA0_i = float( 0.0 ).toVar();
				Loop( { start: 0, end: hiddenSize, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

					const delta1_j = activationsStorage.element( delta1Base.add( j ) );
					const w_ji = weightsStorage.element( int( layer0WeightsOffset ).add( j.mul( decoderInputSize ) ).add( i ) );
					gradA0_i.addAssign( delta1_j.mul( w_ji ) );

				} );

				activationsStorage.element( actBase.add( int( actGradA0Offset ) ).add( i ) ).assign( gradA0_i );

			} );

			// 11. Shading Frames Backward
			// Initialize gradLatents with gradA0[0..latentChannels)
			Loop( { start: 0, end: latentChannels, type: 'int', name: 'c', condition: '<' }, ( { c } ) => {

				const gradA0_c = activationsStorage.element( actBase.add( int( actGradA0Offset ) ).add( c ) );
				activationsStorage.element( actBase.add( int( actGradLatentsOffset ) ).add( c ) ).assign( gradA0_c );

			} );

			// Backward through both frames
			Loop( { start: 0, end: 2, type: 'int', name: 'f', condition: '<' }, ( { f } ) => {

				const inOffset = actBase.add( int( actGradA0Offset ) ).add( latentChannels ).add( f.mul( 6 ) );
				const gradT_val = activationsStorage.element( inOffset.add( 0 ) );
				const gradB_val = activationsStorage.element( inOffset.add( 1 ) );
				const gradN_val = activationsStorage.element( inOffset.add( 2 ) );
				const gradWoT_val = activationsStorage.element( inOffset.add( 3 ) );
				const gradWoB_val = activationsStorage.element( inOffset.add( 4 ) );
				const gradWoN_val = activationsStorage.element( inOffset.add( 5 ) );

				const gradT = wi.mul( gradT_val ).add( wo.mul( gradWoT_val ) );
				const gradB = wi.mul( gradB_val ).add( wo.mul( gradWoB_val ) );
				const gradN = wi.mul( gradN_val ).add( wo.mul( gradWoN_val ) );

				const frameOffset = f.mul( 6 );
				// Recompute the same (n, t, b) frame the forward pass built at
				// step 3 - see computeFrameTSL's doc comment for why this isn't
				// instead cached from the forward pass.
				const { rawN, rawT, n, t, rawB, b } = computeFrameTSL( { activationsStorage, weightsStorage, actA0Base: a0Base, rotationOffset, latentChannels, frameOffset } );

				const gradRawB = backwardNormalizeTSL( rawB, b, gradB );
				const gradNormN = gradN.add( cross( t, gradRawB ) );
				const gradNormT = gradT.add( cross( gradRawB, n ) );

				const gradRawN = backwardNormalizeTSL( rawN, n, gradNormN );
				const gradRawT = backwardNormalizeTSL( rawT, t, gradNormT );

				Loop( { start: 0, end: 6, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

					const rotRowOffset = int( rotationOffset ).add( frameOffset.add( j ).mul( latentChannels ) );
					const gradFrameJ = float( 0.0 ).toVar();

					If( j.equal( 0 ), () => {

						gradFrameJ.assign( gradRawN.x );

					} ).ElseIf( j.equal( 1 ), () => {

						gradFrameJ.assign( gradRawN.y );

					} ).ElseIf( j.equal( 2 ), () => {

						gradFrameJ.assign( gradRawN.z );

					} ).ElseIf( j.equal( 3 ), () => {

						gradFrameJ.assign( gradRawT.x );

					} ).ElseIf( j.equal( 4 ), () => {

						gradFrameJ.assign( gradRawT.y );

					} ).Else( () => {

						gradFrameJ.assign( gradRawT.z );

					} );

					Loop( { start: 0, end: latentChannels, type: 'int', name: 'c', condition: '<' }, ( { c } ) => {

						const z_c = activationsStorage.element( actBase.add( int( actA0Offset ) ).add( c ) );
						const gradRotW = gradFrameJ.mul( z_c );
						atomicAdd( gradWeightsAtomic.element( rotRowOffset.add( c ) ), int( gradRotW.mul( float( FIXED_POINT_SCALE ) ) ) );

						const rotW = weightsStorage.element( rotRowOffset.add( c ) );
						const curGradZ = activationsStorage.element( actBase.add( int( actGradLatentsOffset ) ).add( c ) );
						activationsStorage.element( actBase.add( int( actGradLatentsOffset ) ).add( c ) ).assign( curGradZ.add( gradFrameJ.mul( rotW ) ) );

					} );

				} );

			} );

			// 12. Auxiliary Emission Head Backward (if supported)
			if ( supportsEmission ) {

				If( samplesStorage.element( sampleOffset.add( 12 ) ).greaterThan( 0.5 ), () => {

					Loop( { start: 0, end: 3, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

						const val = weightsStorage.element( int( emissionBiasesOffset ).add( j ) ).toVar();
						const rowOffset = int( emissionWeightsOffset ).add( j.mul( latentChannels ) );

						Loop( { start: 0, end: latentChannels, type: 'int', name: 'c', condition: '<' }, ( { c } ) => {

							const z_c = activationsStorage.element( actBase.add( int( actA0Offset ) ).add( c ) );
							val.addAssign( weightsStorage.element( rowOffset.add( c ) ).mul( z_c ) );

						} );

						const yHat_j = max( val, float( 0.0 ) );
						const predClamped = max( yHat_j, float( 1e-6 ) );
						const refClamped = max( samplesStorage.element( sampleOffset.add( 13 ).add( j ) ), float( 1e-6 ) );
						const predLog = pow( predClamped, float( 1.0 / 3.0 ) ).sub( 1.0 ).mul( 3.0 );
						const refLog = pow( refClamped, float( 1.0 / 3.0 ) ).sub( 1.0 ).mul( 3.0 );
						const diff = predLog.sub( refLog );

						sampleDirectLoss.addAssign( diff.abs().mul( sampleWeight ).mul( invBatchUniform ).div( 3.0 ) );

						const gradPred = sign( diff ).mul( pow( predClamped, float( - 2.0 / 3.0 ) ) ).mul( sampleWeight ).mul( invBatchUniform ).div( 3.0 );
						const outputClampGradient = select( val.greaterThan( 0.0 ), float( 1.0 ), float( OUTPUT_CLAMP_GRADIENT_LEAK ) );
						const delta_j = gradPred.mul( outputClampGradient );

						atomicAdd( gradWeightsAtomic.element( int( emissionBiasesOffset ).add( j ) ), int( delta_j.mul( float( FIXED_POINT_SCALE ) ) ) );

						Loop( { start: 0, end: latentChannels, type: 'int', name: 'c', condition: '<' }, ( { c } ) => {

							const z_c = activationsStorage.element( actBase.add( int( actA0Offset ) ).add( c ) );
							atomicAdd( gradWeightsAtomic.element( rowOffset.add( c ) ), int( delta_j.mul( z_c ).mul( float( FIXED_POINT_SCALE ) ) ) );

							const curGradZ = activationsStorage.element( actBase.add( int( actGradLatentsOffset ) ).add( c ) );
							const rotW = weightsStorage.element( rowOffset.add( c ) );
							activationsStorage.element( actBase.add( int( actGradLatentsOffset ) ).add( c ) ).assign( curGradZ.add( delta_j.mul( rotW ) ) );

						} );

					} );

				} );

			}

			// 13. Auxiliary Opacity Head Backward (if supported)
			if ( supportsOpacity ) {

				If( samplesStorage.element( sampleOffset.add( 16 ) ).greaterThanEqual( 0.0 ), () => {

					const opTarget = clamp( samplesStorage.element( sampleOffset.add( 16 ) ), float( 0.0 ), float( 1.0 ) );
					const val = weightsStorage.element( int( opacityBiasesOffset ) ).toVar();

					Loop( { start: 0, end: latentChannels, type: 'int', name: 'c', condition: '<' }, ( { c } ) => {

						const z_c = activationsStorage.element( actBase.add( int( actA0Offset ) ).add( c ) );
						val.addAssign( weightsStorage.element( int( opacityWeightsOffset ).add( c ) ).mul( z_c ) );

					} );

					const opPred = float( 1.0 ).div( float( 1.0 ).add( exp( val.negate() ) ) );
					const bce = opTarget.mul( log( max( opPred, float( 1e-7 ) ) ) )
						.add( float( 1.0 ).sub( opTarget ).mul( log( max( float( 1.0 ).sub( opPred ), float( 1e-7 ) ) ) ) ).negate();
					sampleDirectLoss.addAssign( bce.mul( sampleWeight ).mul( invBatchUniform ) );

					const delta_op = opPred.sub( opTarget ).mul( sampleWeight ).mul( invBatchUniform );
					atomicAdd( gradWeightsAtomic.element( int( opacityBiasesOffset ) ), int( delta_op.mul( float( FIXED_POINT_SCALE ) ) ) );

					Loop( { start: 0, end: latentChannels, type: 'int', name: 'c', condition: '<' }, ( { c } ) => {

						const z_c = activationsStorage.element( actBase.add( int( actA0Offset ) ).add( c ) );
						atomicAdd( gradWeightsAtomic.element( int( opacityWeightsOffset ).add( c ) ), int( delta_op.mul( z_c ).mul( float( FIXED_POINT_SCALE ) ) ) );

						const curGradZ = activationsStorage.element( actBase.add( int( actGradLatentsOffset ) ).add( c ) );
						const w_c = weightsStorage.element( int( opacityWeightsOffset ).add( c ) );
						activationsStorage.element( actBase.add( int( actGradLatentsOffset ) ).add( c ) ).assign( curGradZ.add( delta_op.mul( w_c ) ) );

					} );

				} );

			}

			// 14. IBL query (iblInputSize -> H -> 4) and separate radiance/irradiance (indirectInputSize -> H -> 3) heads.
			const iblWeight = samplesStorage.element( sampleOffset.add( 17 ) );

			If( iblWeight.greaterThan( 0.0 ), () => {

				const iblScale = iblWeight.mul( invBatchUniform );
				const iblA0 = actBase.add( int( actIblA0Offset ) );
				const a0 = actBase.add( int( actA0Offset ) );

				Loop( { start: 0, end: latentChannels, type: 'int', name: 'c', condition: '<' }, ( { c } ) => {

					activationsStorage.element( iblA0.add( c ) ).assign( activationsStorage.element( a0.add( c ) ) );

				} );

				activationsStorage.element( iblA0.add( latentChannels ) ).assign( activationsStorage.element( a0.add( latentChannels + 3 ) ) );
				activationsStorage.element( iblA0.add( latentChannels + 1 ) ).assign( activationsStorage.element( a0.add( latentChannels + 4 ) ) );
				activationsStorage.element( iblA0.add( latentChannels + 2 ) ).assign( activationsStorage.element( a0.add( latentChannels + 5 ) ) );
				activationsStorage.element( iblA0.add( latentChannels + 3 ) ).assign( activationsStorage.element( a0.add( latentChannels + 9 ) ) );
				activationsStorage.element( iblA0.add( latentChannels + 4 ) ).assign( activationsStorage.element( a0.add( latentChannels + 10 ) ) );
				activationsStorage.element( iblA0.add( latentChannels + 5 ) ).assign( activationsStorage.element( a0.add( latentChannels + 11 ) ) );

				const iblA1 = actBase.add( int( actIblA1Offset ) );

				forwardDenseLayerTSL( {
					activationsStorage, weightsStorage,
					inputBase: iblA0, inputSize: iblInputSize, outputSize: iblHiddenSize,
					weightsOffset: iblLayer0WeightsOffset, biasesOffset: iblLayer0BiasesOffset,
					zBase: actBase.add( int( actIblZ1Offset ) ), aBase: iblA1
				} );

				forwardDenseLayerTSL( {
					activationsStorage, weightsStorage,
					inputBase: iblA1, inputSize: iblHiddenSize, outputSize: IBL_OUTPUT_SIZE,
					weightsOffset: iblLayer1WeightsOffset, biasesOffset: iblLayer1BiasesOffset,
					zBase: actBase.add( int( actIblZ2Offset ) ), aBase: null
				} );

				const iblZ2 = actBase.add( int( actIblZ2Offset ) );
				const iblDelta2 = actBase.add( int( actIblDelta2Offset ) );

				Loop( { start: 0, end: IBL_OUTPUT_SIZE, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

					activationsStorage.element( iblDelta2.add( j ) ).assign( 0.0 );

				} );

				const rawDirection = vec3(
					activationsStorage.element( iblZ2.add( 0 ) ),
					activationsStorage.element( iblZ2.add( 1 ) ),
					activationsStorage.element( iblZ2.add( 2 ) )
				);
				const predDirection = rawDirection.normalize();
				const targetDirection = vec3(
					samplesStorage.element( sampleOffset.add( 18 ) ),
					samplesStorage.element( sampleOffset.add( 19 ) ),
					samplesStorage.element( sampleOffset.add( 20 ) )
				).normalize();
				const dirScale = iblScale;
				sampleIblLoss.addAssign( float( 1.0 ).sub( predDirection.dot( targetDirection ) ).mul( dirScale ) );
				const gradDirection = backwardNormalizeTSL( rawDirection, predDirection, targetDirection.negate().mul( dirScale ) );
				activationsStorage.element( iblDelta2.add( 0 ) ).assign( gradDirection.x );
				activationsStorage.element( iblDelta2.add( 1 ) ).assign( gradDirection.y );
				activationsStorage.element( iblDelta2.add( 2 ) ).assign( gradDirection.z );

				const zRough = activationsStorage.element( iblZ2.add( 3 ) );
				const predRough = float( 1.0 ).div( float( 1.0 ).add( exp( zRough.negate() ) ) );
				const targetRough = samplesStorage.element( sampleOffset.add( 21 ) );
				const roughDiff = predRough.sub( targetRough );
				sampleIblLoss.addAssign( roughDiff.abs().mul( iblScale ) );
				activationsStorage.element( iblDelta2.add( 3 ) ).assign(
					sign( roughDiff ).mul( predRough.mul( float( 1.0 ).sub( predRough ) ) ).mul( iblScale )
				);

				const iblDelta1 = actBase.add( int( actIblDelta1Offset ) );

				accumulateDenseLayerGradTSL( {
					activationsStorage, weightsStorage, gradWeightsAtomic,
					deltaBase: iblDelta2, inputBase: iblA1, inputSize: iblHiddenSize, outputSize: IBL_OUTPUT_SIZE,
					weightsOffset: iblLayer1WeightsOffset, biasesOffset: iblLayer1BiasesOffset
				} );

				backwardDenseLayerReLUTSL( {
					activationsStorage, weightsStorage,
					deltaBase: iblDelta2, deltaSize: IBL_OUTPUT_SIZE,
					weightsOffset: iblLayer1WeightsOffset, prevSize: iblHiddenSize,
					prevZBase: actBase.add( int( actIblZ1Offset ) ), outDeltaBase: iblDelta1
				} );

				accumulateDenseLayerGradTSL( {
					activationsStorage, weightsStorage, gradWeightsAtomic,
					deltaBase: iblDelta1, inputBase: iblA0, inputSize: iblInputSize, outputSize: iblHiddenSize,
					weightsOffset: iblLayer0WeightsOffset, biasesOffset: iblLayer0BiasesOffset
				} );

				// gradLatent_c only propagates the *latent* subset of iblA0 (c <
				// latentChannels, not the full iblInputSize) back into the shared
				// grid gradients - iblA0's remaining components are external
				// sample inputs (wo/probe directions), not something to backprop
				// into - so this stays hand-written rather than reusing
				// backwardDenseLayerReLUTSL (which also isn't a fit: iblA0 is raw,
				// unactivated input, with no ReLU derivative to gate by).
				Loop( { start: 0, end: latentChannels, type: 'int', name: 'c', condition: '<' }, ( { c } ) => {

					const gradLatent_c = float( 0.0 ).toVar();

					Loop( { start: 0, end: iblHiddenSize, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

						const delta1_j = activationsStorage.element( iblDelta1.add( j ) );
						const w_jc = weightsStorage.element( int( iblLayer0WeightsOffset ).add( j.mul( iblInputSize ) ).add( c ) );
						gradLatent_c.addAssign( delta1_j.mul( w_jc ) );

					} );

					const curGradZ = activationsStorage.element( actBase.add( int( actGradLatentsOffset ) ).add( c ) );
					activationsStorage.element( actBase.add( int( actGradLatentsOffset ) ).add( c ) ).assign( curGradZ.add( gradLatent_c ) );

				} );

				trainIndirectProbeHead( {
					samplesStorage,
					sampleOffset,
					activationsStorage,
					actBase,
					weightsStorage,
					gradWeightsAtomic,
					wo,
					a0,
					iblScale,
					sampleIblLoss,
					iblHiddenSize,
					probeSampleOffset: 22,
					targetSampleOffset: 28,
					layer0WeightsOffset: indirectRadianceLayer0WeightsOffset,
					layer0BiasesOffset: indirectRadianceLayer0BiasesOffset,
					layer1WeightsOffset: indirectRadianceLayer1WeightsOffset,
					layer1BiasesOffset: indirectRadianceLayer1BiasesOffset,
					actIndirectA0Offset,
					actIndirectZ1Offset,
					actIndirectA1Offset,
					actIndirectZ2Offset,
					actIndirectDelta2Offset,
					actIndirectDelta1Offset,
					actGradLatentsOffset,
					latentChannels,
					indirectInputSize
				} );

				trainIndirectProbeHead( {
					samplesStorage,
					sampleOffset,
					activationsStorage,
					actBase,
					weightsStorage,
					gradWeightsAtomic,
					wo,
					a0,
					iblScale,
					sampleIblLoss,
					iblHiddenSize,
					probeSampleOffset: 25,
					targetSampleOffset: 31,
					layer0WeightsOffset: indirectIrradianceLayer0WeightsOffset,
					layer0BiasesOffset: indirectIrradianceLayer0BiasesOffset,
					layer1WeightsOffset: indirectIrradianceLayer1WeightsOffset,
					layer1BiasesOffset: indirectIrradianceLayer1BiasesOffset,
					actIndirectA0Offset,
					actIndirectZ1Offset,
					actIndirectA1Offset,
					actIndirectZ2Offset,
					actIndirectDelta2Offset,
					actIndirectDelta1Offset,
					actGradLatentsOffset,
					latentChannels,
					indirectInputSize
				} );

			} );

			// Accumulate loss after all active heads contribute.
			atomicAdd( lossAtomic.element( 0 ), int( sampleDirectLoss.add( sampleIblLoss ).mul( float( FIXED_POINT_SCALE ) ) ) );
			atomicAdd( lossAtomic.element( 1 ), int( sampleDirectLoss.mul( float( FIXED_POINT_SCALE ) ) ) );
			atomicAdd( lossAtomic.element( 2 ), int( sampleIblLoss.mul( float( FIXED_POINT_SCALE ) ) ) );

			// 15. Scatter Bilinear Latent Gradients back into each grid level,
			// using the same per-level taps/weights computed in step 2.
			for ( let g = 0; g < gridLevels.length; g ++ ) {

				const taps = levelTaps[ g ];

				for ( let c = 0; c < CHANNELS_PER_LEVEL; c ++ ) {

					const gradZ_c = activationsStorage.element( actBase.add( int( actGradLatentsOffset + g * CHANNELS_PER_LEVEL + c ) ) );
					atomicAdd( gradLatentsAtomic.element( taps.off0.add( c ) ), int( gradZ_c.mul( taps.w0 ).mul( float( FIXED_POINT_SCALE ) ) ) );
					atomicAdd( gradLatentsAtomic.element( taps.off1.add( c ) ), int( gradZ_c.mul( taps.w1 ).mul( float( FIXED_POINT_SCALE ) ) ) );
					atomicAdd( gradLatentsAtomic.element( taps.off2.add( c ) ), int( gradZ_c.mul( taps.w2 ).mul( float( FIXED_POINT_SCALE ) ) ) );
					atomicAdd( gradLatentsAtomic.element( taps.off3.add( c ) ), int( gradZ_c.mul( taps.w3 ).mul( float( FIXED_POINT_SCALE ) ) ) );

				}

			}

		} );

	} )().compute( batchSize ).setName( 'NeuralAppearanceTrainBatch' );

}

/**
 * Accumulates squared weight and latent gradients for global norm clipping.
 */
function createAccumulateGradientNormComputeNode( gpuModel, { weightOffset = 0, weightCount = null, includeLatents = true } = {} ) {

	const { layout, gradNormAtomic } = gpuModel;
	const { gradAtomic: gradWeightsAtomic } = gpuModel.weightsBuffers;
	const { gradAtomic: gradLatentsAtomic } = gpuModel.latentsBuffers;

	const resolvedWeightCount = weightCount === null ? layout.totalWeights : weightCount;
	const dispatchCount = resolvedWeightCount + ( includeLatents ? layout.totalLatents : 0 );

	return Fn( () => {

		const idx = int( instanceIndex );
		const grad = float( 0.0 ).toVar();

		If( idx.lessThan( int( resolvedWeightCount ) ), () => {

			grad.assign( float( atomicLoad( gradWeightsAtomic.element( int( weightOffset ).add( idx ) ) ) ).div( float( FIXED_POINT_SCALE ) ) );

		} ).Else( () => {

			const latentIdx = idx.sub( int( resolvedWeightCount ) );
			grad.assign( float( atomicLoad( gradLatentsAtomic.element( latentIdx ) ) ).div( float( FIXED_POINT_SCALE ) ) );

		} );

		atomicAdd( gradNormAtomic.element( 0 ), int( grad.mul( grad ).mul( float( GRADIENT_NORM_SCALE ) ) ) );

	} )().compute( dispatchCount ).setName( 'NeuralAppearanceAccumulateGradientNorm' );

}

/**
 * Creates the Adam optimizer compute node for MLP and frame weights.
 */
function createAdamWeightsComputeNode( gpuModel, { beta1 = 0.9, beta2 = 0.999, epsilon = 1e-7, weightOffset = 0, weightCount = null } = {} ) {

	const {
		layout,
		learningRateUniform,
		stepUniform,
		gradNormAtomic,
		maxGradientNormUniform
	} = gpuModel;
	const { valuesStorage, gradAtomic, mStorage, vStorage } = gpuModel.weightsBuffers;

	return createAdamComputeNode( {
		valuesStorage,
		gradAtomic,
		mStorage,
		vStorage,
		gradNormAtomic,
		maxGradientNormUniform,
		learningRateUniform,
		stepUniform,
		offset: weightOffset,
		count: weightCount === null ? layout.totalWeights : weightCount,
		beta1,
		beta2,
		epsilon,
		name: 'NeuralAppearanceAdamWeights'
	} );

}

/**
 * Creates the Adam optimizer compute node for the multiresolution latent
 * grid levels (all levels updated together, one contiguous buffer).
 */
function createAdamLatentsComputeNode( gpuModel, { beta1 = 0.9, beta2 = 0.999, epsilon = 1e-7 } = {} ) {

	const {
		layout,
		learningRateUniform,
		stepUniform,
		gradNormAtomic,
		maxGradientNormUniform
	} = gpuModel;
	const { valuesStorage, gradAtomic, mStorage, vStorage } = gpuModel.latentsBuffers;

	return createAdamComputeNode( {
		valuesStorage,
		gradAtomic,
		mStorage,
		vStorage,
		gradNormAtomic,
		maxGradientNormUniform,
		learningRateUniform,
		stepUniform,
		count: layout.totalLatents,
		beta1,
		beta2,
		epsilon,
		name: 'NeuralAppearanceAdamLatents'
	} );

}

export {
	createTrainBatchComputeNode,
	createAccumulateGradientNormComputeNode,
	createAdamWeightsComputeNode,
	createAdamLatentsComputeNode
};
