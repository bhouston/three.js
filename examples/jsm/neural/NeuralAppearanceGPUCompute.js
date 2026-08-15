import {
	Fn,
	If,
	Loop,
	atomicAdd,
	atomicLoad,
	atomicStore,
	clamp,
	cross,
	exp,
	float,
	floor,
	instanceIndex,
	int,
	log,
	max,
	min,
	pow,
	select,
	sign,
	sqrt,
	vec3
} from 'three/tsl';
import {
	FIXED_POINT_SCALE,
	GRADIENT_NORM_SCALE
} from './NeuralAppearanceGPUModel.js';
import { IBL_INPUT_SIZE, IBL_OUTPUT_SIZE, INDIRECT_INPUT_SIZE, INDIRECT_OUTPUT_SIZE } from './NeuralAppearanceFormat.js';

const OUTPUT_CLAMP_GRADIENT_LEAK = 0.01;

function backwardNormalizeTSL( raw, norm, gradNorm ) {

	const rawLen = max( raw.length(), float( 1e-10 ) );
	const invLen = float( 1.0 ).div( rawLen );
	const proj = norm.dot( gradNorm );

	return gradNorm.sub( norm.mul( proj ) ).mul( invLen );

}

function wrapIndexTSL( val, size ) {

	return val.mod( size ).add( size ).mod( size );

}

function computeGradientClipScale( gradNormAtomic, maxGradientNormUniform ) {

	const normSquared = float( atomicLoad( gradNormAtomic.element( 0 ) ) ).div( float( GRADIENT_NORM_SCALE ) );
	const unclippedScale = maxGradientNormUniform.div( sqrt( max( normSquared, float( 1e-20 ) ) ) );

	return min( float( 1.0 ), unclippedScale );

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
		weightsStorage,
		gradWeightsAtomic,
		latentsStorage,
		gradLatentsAtomic,
		lossAtomic,
		invBatchUniform,
		mipInfoArray
	} = gpuModel;

	const {
		hiddenSize,
		iblHiddenSize,
		supportsEmission,
		supportsOpacity,
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
		indirectLayer0WeightsOffset,
		indirectLayer0BiasesOffset,
		indirectLayer1WeightsOffset,
		indirectLayer1BiasesOffset,
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
			samplesStorage.element( sampleOffset.add( 4 ) ),
			samplesStorage.element( sampleOffset.add( 5 ) ),
			samplesStorage.element( sampleOffset.add( 6 ) )
		);
		const mip = int( samplesStorage.element( sampleOffset.add( 7 ) ) );
		const wo = vec3(
			samplesStorage.element( sampleOffset.add( 8 ) ),
			samplesStorage.element( sampleOffset.add( 9 ) ),
			samplesStorage.element( sampleOffset.add( 10 ) )
		);
		const sampleWeight = samplesStorage.element( sampleOffset.add( 11 ) );

		// Guard against zero-weight samples
		If( sampleWeight.greaterThan( 0.0 ), () => {

			// 2. Sample latents with bilinear interpolation
			const mipData = mipInfoArray.element( mip );
			const gridWidth = int( mipData.x );
			const gridHeight = int( mipData.y );
			const baseOffset = int( mipData.z );

			const x = uvX.mul( mipData.x ).sub( 0.5 );
			const y = uvY.mul( mipData.y ).sub( 0.5 );
			const x0 = int( floor( x ) );
			const y0 = int( floor( y ) );
			const tx = x.sub( float( x0 ) );
			const ty = y.sub( float( y0 ) );

			const w0 = float( 1.0 ).sub( tx ).mul( float( 1.0 ).sub( ty ) );
			const w1 = tx.mul( float( 1.0 ).sub( ty ) );
			const w2 = float( 1.0 ).sub( tx ).mul( ty );
			const w3 = tx.mul( ty );

			const tapX0 = wrapIndexTSL( x0, gridWidth );
			const tapY0 = wrapIndexTSL( y0, gridHeight );
			const tapX1 = wrapIndexTSL( x0.add( 1 ), gridWidth );
			const tapY1 = wrapIndexTSL( y0, gridHeight );
			const tapX2 = wrapIndexTSL( x0, gridWidth );
			const tapY2 = wrapIndexTSL( y0.add( 1 ), gridHeight );
			const tapX3 = wrapIndexTSL( x0.add( 1 ), gridWidth );
			const tapY3 = wrapIndexTSL( y0.add( 1 ), gridHeight );

			const off0 = baseOffset.add( tapY0.mul( gridWidth ).add( tapX0 ).mul( 8 ) );
			const off1 = baseOffset.add( tapY1.mul( gridWidth ).add( tapX1 ).mul( 8 ) );
			const off2 = baseOffset.add( tapY2.mul( gridWidth ).add( tapX2 ).mul( 8 ) );
			const off3 = baseOffset.add( tapY3.mul( gridWidth ).add( tapX3 ).mul( 8 ) );

			const actBase = sampleIdx.mul( int( activationStride ) );

			// Fetch 8 latents and write to actA0Offset
			Loop( { start: 0, end: 8, type: 'int', name: 'c', condition: '<' }, ( { c } ) => {

				const z_c = latentsStorage.element( off0.add( c ) ).mul( w0 )
					.add( latentsStorage.element( off1.add( c ) ).mul( w1 ) )
					.add( latentsStorage.element( off2.add( c ) ).mul( w2 ) )
					.add( latentsStorage.element( off3.add( c ) ).mul( w3 ) );

				activationsStorage.element( actBase.add( int( actA0Offset ) ).add( c ) ).assign( z_c );

			} );

			// 3. Learned coordinate frames (Frame 0 and Frame 1)
			Loop( { start: 0, end: 2, type: 'int', name: 'f', condition: '<' }, ( { f } ) => {

				const frameOffset = f.mul( 6 );
				const rawN = vec3( 0.0 ).toVar();
				const rawT = vec3( 0.0 ).toVar();

				// Raw N (j = 0, 1, 2)
				Loop( { start: 0, end: 3, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

					const rotRowOffset = int( rotationOffset ).add( frameOffset.add( j ).mul( 8 ) );
					const r_j = float( 0.0 ).toVar();

					Loop( { start: 0, end: 8, type: 'int', name: 'c', condition: '<' }, ( { c } ) => {

						const z_c = activationsStorage.element( actBase.add( int( actA0Offset ) ).add( c ) );
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

				// Raw T (j = 0, 1, 2 for rotation outputs 3, 4, 5)
				Loop( { start: 0, end: 3, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

					const rotRowOffset = int( rotationOffset ).add( frameOffset.add( j.add( 3 ) ).mul( 8 ) );
					const r_j = float( 0.0 ).toVar();

					Loop( { start: 0, end: 8, type: 'int', name: 'c', condition: '<' }, ( { c } ) => {

						const z_c = activationsStorage.element( actBase.add( int( actA0Offset ) ).add( c ) );
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

				const projBase = actBase.add( int( actA0Offset ) ).add( 8 ).add( f.mul( 6 ) );
				activationsStorage.element( projBase.add( 0 ) ).assign( wi.dot( t ) );
				activationsStorage.element( projBase.add( 1 ) ).assign( wi.dot( b ) );
				activationsStorage.element( projBase.add( 2 ) ).assign( wi.dot( n ) );
				activationsStorage.element( projBase.add( 3 ) ).assign( wo.dot( t ) );
				activationsStorage.element( projBase.add( 4 ) ).assign( wo.dot( b ) );
				activationsStorage.element( projBase.add( 5 ) ).assign( wo.dot( n ) );

			} );

			// 4. Forward Layer 0 (20 -> hiddenSize)
			Loop( { start: 0, end: hiddenSize, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

				const val = weightsStorage.element( int( layer0BiasesOffset ).add( j ) ).toVar();
				const rowOffset = int( layer0WeightsOffset ).add( j.mul( 20 ) );

				Loop( { start: 0, end: 20, type: 'int', name: 'i', condition: '<' }, ( { i } ) => {

					const a0_i = activationsStorage.element( actBase.add( int( actA0Offset ) ).add( i ) );
					const w_ji = weightsStorage.element( rowOffset.add( i ) );
					val.addAssign( w_ji.mul( a0_i ) );

				} );

				activationsStorage.element( actBase.add( int( actZ1Offset ) ).add( j ) ).assign( val );
				activationsStorage.element( actBase.add( int( actA1Offset ) ).add( j ) ).assign( max( val, float( 0.0 ) ) );

			} );

			// 5. Forward Layer 1 (hiddenSize -> hiddenSize)
			Loop( { start: 0, end: hiddenSize, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

				const val = weightsStorage.element( int( layer1BiasesOffset ).add( j ) ).toVar();
				const rowOffset = int( layer1WeightsOffset ).add( j.mul( hiddenSize ) );

				Loop( { start: 0, end: hiddenSize, type: 'int', name: 'i', condition: '<' }, ( { i } ) => {

					const a1_i = activationsStorage.element( actBase.add( int( actA1Offset ) ).add( i ) );
					const w_ji = weightsStorage.element( rowOffset.add( i ) );
					val.addAssign( w_ji.mul( a1_i ) );

				} );

				activationsStorage.element( actBase.add( int( actZ2Offset ) ).add( j ) ).assign( val );
				activationsStorage.element( actBase.add( int( actA2Offset ) ).add( j ) ).assign( max( val, float( 0.0 ) ) );

			} );

			// 6. Forward Layer 2 (hiddenSize -> 3, Linear)
			Loop( { start: 0, end: 3, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

				const val = weightsStorage.element( int( layer2BiasesOffset ).add( j ) ).toVar();
				const rowOffset = int( layer2WeightsOffset ).add( j.mul( hiddenSize ) );

				Loop( { start: 0, end: hiddenSize, type: 'int', name: 'i', condition: '<' }, ( { i } ) => {

					const a2_i = activationsStorage.element( actBase.add( int( actA2Offset ) ).add( i ) );
					const w_ji = weightsStorage.element( rowOffset.add( i ) );
					val.addAssign( w_ji.mul( a2_i ) );

				} );

				activationsStorage.element( actBase.add( int( actZ3Offset ) ).add( j ) ).assign( val );

			} );

			// 7. Cube-Root Power Loss & Delta 3
			const sampleLoss = float( 0.0 ).toVar();

			Loop( { start: 0, end: 3, type: 'int', name: 'c', condition: '<' }, ( { c } ) => {

				const z3_c = activationsStorage.element( actBase.add( int( actZ3Offset ) ).add( c ) );
				const yHat_c = max( z3_c, float( 0.0 ) );
				const target_c = samplesStorage.element( sampleOffset.add( 12 ).add( c ) );

				const predClamped = max( yHat_c, float( 1e-6 ) );
				const refClamped = max( target_c, float( 1e-6 ) );
				const predLog = pow( predClamped, float( 1.0 / 3.0 ) ).sub( 1.0 ).mul( 3.0 );
				const refLog = pow( refClamped, float( 1.0 / 3.0 ) ).sub( 1.0 ).mul( 3.0 );
				const diff = predLog.sub( refLog );

				const channelLoss = diff.abs().mul( sampleWeight ).mul( invBatchUniform ).div( 3.0 );
				sampleLoss.addAssign( channelLoss );

				const gradPred = sign( diff ).mul( pow( predClamped, float( - 2.0 / 3.0 ) ) ).mul( sampleWeight ).mul( invBatchUniform ).div( 3.0 );
				const outputClampGradient = select( z3_c.greaterThan( 0.0 ), float( 1.0 ), float( OUTPUT_CLAMP_GRADIENT_LEAK ) );
				activationsStorage.element( actBase.add( int( actDelta3Offset ) ).add( c ) ).assign( gradPred.mul( outputClampGradient ) );

			} );

			// 8. Backward Layer 2 (hiddenSize -> 3)
			Loop( { start: 0, end: 3, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

				const delta3_j = activationsStorage.element( actBase.add( int( actDelta3Offset ) ).add( j ) );
				atomicAdd( gradWeightsAtomic.element( int( layer2BiasesOffset ).add( j ) ), int( delta3_j.mul( float( FIXED_POINT_SCALE ) ) ) );

				const rowOffset = int( layer2WeightsOffset ).add( j.mul( hiddenSize ) );
				Loop( { start: 0, end: hiddenSize, type: 'int', name: 'i', condition: '<' }, ( { i } ) => {

					const a2_i = activationsStorage.element( actBase.add( int( actA2Offset ) ).add( i ) );
					const gradW = delta3_j.mul( a2_i );
					atomicAdd( gradWeightsAtomic.element( rowOffset.add( i ) ), int( gradW.mul( float( FIXED_POINT_SCALE ) ) ) );

				} );

			} );

			Loop( { start: 0, end: hiddenSize, type: 'int', name: 'i', condition: '<' }, ( { i } ) => {

				const gradInput_i = float( 0.0 ).toVar();
				Loop( { start: 0, end: 3, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

					const delta3_j = activationsStorage.element( actBase.add( int( actDelta3Offset ) ).add( j ) );
					const w_ji = weightsStorage.element( int( layer2WeightsOffset ).add( j.mul( hiddenSize ) ).add( i ) );
					gradInput_i.addAssign( delta3_j.mul( w_ji ) );

				} );

				const z2_i = activationsStorage.element( actBase.add( int( actZ2Offset ) ).add( i ) );
				const delta2_i = select( z2_i.greaterThan( 0.0 ), gradInput_i, float( 0.0 ) );
				activationsStorage.element( actBase.add( int( actDelta2Offset ) ).add( i ) ).assign( delta2_i );

			} );

			// 9. Backward Layer 1 (hiddenSize -> hiddenSize)
			Loop( { start: 0, end: hiddenSize, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

				const delta2_j = activationsStorage.element( actBase.add( int( actDelta2Offset ) ).add( j ) );
				atomicAdd( gradWeightsAtomic.element( int( layer1BiasesOffset ).add( j ) ), int( delta2_j.mul( float( FIXED_POINT_SCALE ) ) ) );

				const rowOffset = int( layer1WeightsOffset ).add( j.mul( hiddenSize ) );
				Loop( { start: 0, end: hiddenSize, type: 'int', name: 'i', condition: '<' }, ( { i } ) => {

					const a1_i = activationsStorage.element( actBase.add( int( actA1Offset ) ).add( i ) );
					const gradW = delta2_j.mul( a1_i );
					atomicAdd( gradWeightsAtomic.element( rowOffset.add( i ) ), int( gradW.mul( float( FIXED_POINT_SCALE ) ) ) );

				} );

			} );

			Loop( { start: 0, end: hiddenSize, type: 'int', name: 'i', condition: '<' }, ( { i } ) => {

				const gradInput_i = float( 0.0 ).toVar();
				Loop( { start: 0, end: hiddenSize, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

					const delta2_j = activationsStorage.element( actBase.add( int( actDelta2Offset ) ).add( j ) );
					const w_ji = weightsStorage.element( int( layer1WeightsOffset ).add( j.mul( hiddenSize ) ).add( i ) );
					gradInput_i.addAssign( delta2_j.mul( w_ji ) );

				} );

				const z1_i = activationsStorage.element( actBase.add( int( actZ1Offset ) ).add( i ) );
				const delta1_i = select( z1_i.greaterThan( 0.0 ), gradInput_i, float( 0.0 ) );
				activationsStorage.element( actBase.add( int( actDelta1Offset ) ).add( i ) ).assign( delta1_i );

			} );

			// 10. Backward Layer 0 (20 -> hiddenSize)
			Loop( { start: 0, end: hiddenSize, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

				const delta1_j = activationsStorage.element( actBase.add( int( actDelta1Offset ) ).add( j ) );
				atomicAdd( gradWeightsAtomic.element( int( layer0BiasesOffset ).add( j ) ), int( delta1_j.mul( float( FIXED_POINT_SCALE ) ) ) );

				const rowOffset = int( layer0WeightsOffset ).add( j.mul( 20 ) );
				Loop( { start: 0, end: 20, type: 'int', name: 'i', condition: '<' }, ( { i } ) => {

					const a0_i = activationsStorage.element( actBase.add( int( actA0Offset ) ).add( i ) );
					const gradW = delta1_j.mul( a0_i );
					atomicAdd( gradWeightsAtomic.element( rowOffset.add( i ) ), int( gradW.mul( float( FIXED_POINT_SCALE ) ) ) );

				} );

			} );

			// Compute gradA0 (20 floats)
			Loop( { start: 0, end: 20, type: 'int', name: 'i', condition: '<' }, ( { i } ) => {

				const gradA0_i = float( 0.0 ).toVar();
				Loop( { start: 0, end: hiddenSize, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

					const delta1_j = activationsStorage.element( actBase.add( int( actDelta1Offset ) ).add( j ) );
					const w_ji = weightsStorage.element( int( layer0WeightsOffset ).add( j.mul( 20 ) ).add( i ) );
					gradA0_i.addAssign( delta1_j.mul( w_ji ) );

				} );

				activationsStorage.element( actBase.add( int( actGradA0Offset ) ).add( i ) ).assign( gradA0_i );

			} );

			// 11. Shading Frames Backward
			// Initialize gradLatents with gradA0[0..7]
			Loop( { start: 0, end: 8, type: 'int', name: 'c', condition: '<' }, ( { c } ) => {

				const gradA0_c = activationsStorage.element( actBase.add( int( actGradA0Offset ) ).add( c ) );
				activationsStorage.element( actBase.add( int( actGradLatentsOffset ) ).add( c ) ).assign( gradA0_c );

			} );

			// Backward through both frames
			Loop( { start: 0, end: 2, type: 'int', name: 'f', condition: '<' }, ( { f } ) => {

				const inOffset = actBase.add( int( actGradA0Offset ) ).add( 8 ).add( f.mul( 6 ) );
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
				const rawN = vec3( 0.0 ).toVar();
				const rawT = vec3( 0.0 ).toVar();

				// Recompute rawN
				Loop( { start: 0, end: 3, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

					const rotRowOffset = int( rotationOffset ).add( frameOffset.add( j ).mul( 8 ) );
					const r_j = float( 0.0 ).toVar();

					Loop( { start: 0, end: 8, type: 'int', name: 'c', condition: '<' }, ( { c } ) => {

						const z_c = activationsStorage.element( actBase.add( int( actA0Offset ) ).add( c ) );
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

				// Recompute rawT
				Loop( { start: 0, end: 3, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

					const rotRowOffset = int( rotationOffset ).add( frameOffset.add( j.add( 3 ) ).mul( 8 ) );
					const r_j = float( 0.0 ).toVar();

					Loop( { start: 0, end: 8, type: 'int', name: 'c', condition: '<' }, ( { c } ) => {

						const z_c = activationsStorage.element( actBase.add( int( actA0Offset ) ).add( c ) );
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

				const gradRawB = backwardNormalizeTSL( rawB, b, gradB );
				const gradNormN = gradN.add( cross( t, gradRawB ) );
				const gradNormT = gradT.add( cross( gradRawB, n ) );

				const gradRawN = backwardNormalizeTSL( rawN, n, gradNormN );
				const gradRawT = backwardNormalizeTSL( rawT, t, gradNormT );

				Loop( { start: 0, end: 6, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

					const rotRowOffset = int( rotationOffset ).add( frameOffset.add( j ).mul( 8 ) );
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

					Loop( { start: 0, end: 8, type: 'int', name: 'c', condition: '<' }, ( { c } ) => {

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

				If( samplesStorage.element( sampleOffset.add( 15 ) ).greaterThan( 0.5 ), () => {

					Loop( { start: 0, end: 3, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

						const val = weightsStorage.element( int( emissionBiasesOffset ).add( j ) ).toVar();
						const rowOffset = int( emissionWeightsOffset ).add( j.mul( 8 ) );

						Loop( { start: 0, end: 8, type: 'int', name: 'c', condition: '<' }, ( { c } ) => {

							const z_c = activationsStorage.element( actBase.add( int( actA0Offset ) ).add( c ) );
							val.addAssign( weightsStorage.element( rowOffset.add( c ) ).mul( z_c ) );

						} );

						const yHat_j = max( val, float( 0.0 ) );
						const predClamped = max( yHat_j, float( 1e-6 ) );
						const refClamped = max( samplesStorage.element( sampleOffset.add( 16 ).add( j ) ), float( 1e-6 ) );
						const predLog = pow( predClamped, float( 1.0 / 3.0 ) ).sub( 1.0 ).mul( 3.0 );
						const refLog = pow( refClamped, float( 1.0 / 3.0 ) ).sub( 1.0 ).mul( 3.0 );
						const diff = predLog.sub( refLog );

						sampleLoss.addAssign( diff.abs().mul( sampleWeight ).mul( invBatchUniform ).div( 3.0 ) );

						const gradPred = sign( diff ).mul( pow( predClamped, float( - 2.0 / 3.0 ) ) ).mul( sampleWeight ).mul( invBatchUniform ).div( 3.0 );
						const outputClampGradient = select( val.greaterThan( 0.0 ), float( 1.0 ), float( OUTPUT_CLAMP_GRADIENT_LEAK ) );
						const delta_j = gradPred.mul( outputClampGradient );

						atomicAdd( gradWeightsAtomic.element( int( emissionBiasesOffset ).add( j ) ), int( delta_j.mul( float( FIXED_POINT_SCALE ) ) ) );

						Loop( { start: 0, end: 8, type: 'int', name: 'c', condition: '<' }, ( { c } ) => {

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

				If( samplesStorage.element( sampleOffset.add( 19 ) ).greaterThanEqual( 0.0 ), () => {

					const opTarget = clamp( samplesStorage.element( sampleOffset.add( 19 ) ), float( 0.0 ), float( 1.0 ) );
					const val = weightsStorage.element( int( opacityBiasesOffset ) ).toVar();

					Loop( { start: 0, end: 8, type: 'int', name: 'c', condition: '<' }, ( { c } ) => {

						const z_c = activationsStorage.element( actBase.add( int( actA0Offset ) ).add( c ) );
						val.addAssign( weightsStorage.element( int( opacityWeightsOffset ).add( c ) ).mul( z_c ) );

					} );

					const opPred = float( 1.0 ).div( float( 1.0 ).add( exp( val.negate() ) ) );
					const bce = opTarget.mul( log( max( opPred, float( 1e-7 ) ) ) )
						.add( float( 1.0 ).sub( opTarget ).mul( log( max( float( 1.0 ).sub( opPred ), float( 1e-7 ) ) ) ) ).negate();
					sampleLoss.addAssign( bce.mul( sampleWeight ).mul( invBatchUniform ) );

					const delta_op = opPred.sub( opTarget ).mul( sampleWeight ).mul( invBatchUniform );
					atomicAdd( gradWeightsAtomic.element( int( opacityBiasesOffset ) ), int( delta_op.mul( float( FIXED_POINT_SCALE ) ) ) );

					Loop( { start: 0, end: 8, type: 'int', name: 'c', condition: '<' }, ( { c } ) => {

						const z_c = activationsStorage.element( actBase.add( int( actA0Offset ) ).add( c ) );
						atomicAdd( gradWeightsAtomic.element( int( opacityWeightsOffset ).add( c ) ), int( delta_op.mul( z_c ).mul( float( FIXED_POINT_SCALE ) ) ) );

						const curGradZ = activationsStorage.element( actBase.add( int( actGradLatentsOffset ) ).add( c ) );
						const w_c = weightsStorage.element( int( opacityWeightsOffset ).add( c ) );
						activationsStorage.element( actBase.add( int( actGradLatentsOffset ) ).add( c ) ).assign( curGradZ.add( delta_op.mul( w_c ) ) );

					} );

				} );

			}

			// 14. IBL query (14 -> H -> 4) and indirect (14 -> H -> 3) heads.
			const iblWeight = samplesStorage.element( sampleOffset.add( 20 ) );

			If( iblWeight.greaterThan( 0.0 ), () => {

				const iblScale = iblWeight.mul( invBatchUniform );
				const iblA0 = actBase.add( int( actIblA0Offset ) );
				const a0 = actBase.add( int( actA0Offset ) );

				Loop( { start: 0, end: 8, type: 'int', name: 'c', condition: '<' }, ( { c } ) => {

					activationsStorage.element( iblA0.add( c ) ).assign( activationsStorage.element( a0.add( c ) ) );

				} );

				activationsStorage.element( iblA0.add( 8 ) ).assign( activationsStorage.element( a0.add( 11 ) ) );
				activationsStorage.element( iblA0.add( 9 ) ).assign( activationsStorage.element( a0.add( 12 ) ) );
				activationsStorage.element( iblA0.add( 10 ) ).assign( activationsStorage.element( a0.add( 13 ) ) );
				activationsStorage.element( iblA0.add( 11 ) ).assign( activationsStorage.element( a0.add( 17 ) ) );
				activationsStorage.element( iblA0.add( 12 ) ).assign( activationsStorage.element( a0.add( 18 ) ) );
				activationsStorage.element( iblA0.add( 13 ) ).assign( activationsStorage.element( a0.add( 19 ) ) );

				Loop( { start: 0, end: iblHiddenSize, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

					const val = weightsStorage.element( int( iblLayer0BiasesOffset ).add( j ) ).toVar();
					const rowOffset = int( iblLayer0WeightsOffset ).add( j.mul( IBL_INPUT_SIZE ) );

					Loop( { start: 0, end: IBL_INPUT_SIZE, type: 'int', name: 'i', condition: '<' }, ( { i } ) => {

						val.addAssign( weightsStorage.element( rowOffset.add( i ) ).mul( activationsStorage.element( iblA0.add( i ) ) ) );

					} );

					activationsStorage.element( actBase.add( int( actIblZ1Offset ) ).add( j ) ).assign( val );
					activationsStorage.element( actBase.add( int( actIblA1Offset ) ).add( j ) ).assign( max( val, float( 0.0 ) ) );

				} );

				Loop( { start: 0, end: IBL_OUTPUT_SIZE, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

					const val = weightsStorage.element( int( iblLayer1BiasesOffset ).add( j ) ).toVar();
					const rowOffset = int( iblLayer1WeightsOffset ).add( j.mul( iblHiddenSize ) );

					Loop( { start: 0, end: iblHiddenSize, type: 'int', name: 'i', condition: '<' }, ( { i } ) => {

						val.addAssign( weightsStorage.element( rowOffset.add( i ) ).mul( activationsStorage.element( actBase.add( int( actIblA1Offset ) ).add( i ) ) ) );

					} );

					activationsStorage.element( actBase.add( int( actIblZ2Offset ) ).add( j ) ).assign( val );

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
					samplesStorage.element( sampleOffset.add( 21 ) ),
					samplesStorage.element( sampleOffset.add( 22 ) ),
					samplesStorage.element( sampleOffset.add( 23 ) )
				).normalize();
				const dirScale = iblScale;
				sampleLoss.addAssign( float( 1.0 ).sub( predDirection.dot( targetDirection ) ).mul( dirScale ) );
				const gradDirection = backwardNormalizeTSL( rawDirection, predDirection, targetDirection.negate().mul( dirScale ) );
				activationsStorage.element( iblDelta2.add( 0 ) ).assign( gradDirection.x );
				activationsStorage.element( iblDelta2.add( 1 ) ).assign( gradDirection.y );
				activationsStorage.element( iblDelta2.add( 2 ) ).assign( gradDirection.z );

				const zRough = activationsStorage.element( iblZ2.add( 3 ) );
				const predRough = float( 1.0 ).div( float( 1.0 ).add( exp( zRough.negate() ) ) );
				const targetRough = samplesStorage.element( sampleOffset.add( 24 ) );
				const roughDiff = predRough.sub( targetRough );
				sampleLoss.addAssign( roughDiff.abs().mul( iblScale ) );
				activationsStorage.element( iblDelta2.add( 3 ) ).assign(
					sign( roughDiff ).mul( predRough.mul( float( 1.0 ).sub( predRough ) ) ).mul( iblScale )
				);

				Loop( { start: 0, end: IBL_OUTPUT_SIZE, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

					const delta_j = activationsStorage.element( iblDelta2.add( j ) );
					atomicAdd( gradWeightsAtomic.element( int( iblLayer1BiasesOffset ).add( j ) ), int( delta_j.mul( float( FIXED_POINT_SCALE ) ) ) );
					const rowOffset = int( iblLayer1WeightsOffset ).add( j.mul( iblHiddenSize ) );

					Loop( { start: 0, end: iblHiddenSize, type: 'int', name: 'i', condition: '<' }, ( { i } ) => {

						const a1_i = activationsStorage.element( actBase.add( int( actIblA1Offset ) ).add( i ) );
						atomicAdd( gradWeightsAtomic.element( rowOffset.add( i ) ), int( delta_j.mul( a1_i ).mul( float( FIXED_POINT_SCALE ) ) ) );

					} );

				} );

				Loop( { start: 0, end: iblHiddenSize, type: 'int', name: 'i', condition: '<' }, ( { i } ) => {

					const gradInput_i = float( 0.0 ).toVar();

					Loop( { start: 0, end: IBL_OUTPUT_SIZE, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

						const delta_j = activationsStorage.element( iblDelta2.add( j ) );
						const w_ji = weightsStorage.element( int( iblLayer1WeightsOffset ).add( j.mul( iblHiddenSize ) ).add( i ) );
						gradInput_i.addAssign( delta_j.mul( w_ji ) );

					} );

					const z1_i = activationsStorage.element( actBase.add( int( actIblZ1Offset ) ).add( i ) );
					activationsStorage.element( actBase.add( int( actIblDelta1Offset ) ).add( i ) ).assign(
						select( z1_i.greaterThan( 0.0 ), gradInput_i, float( 0.0 ) )
					);

				} );

				Loop( { start: 0, end: iblHiddenSize, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

					const delta1_j = activationsStorage.element( actBase.add( int( actIblDelta1Offset ) ).add( j ) );
					atomicAdd( gradWeightsAtomic.element( int( iblLayer0BiasesOffset ).add( j ) ), int( delta1_j.mul( float( FIXED_POINT_SCALE ) ) ) );
					const rowOffset = int( iblLayer0WeightsOffset ).add( j.mul( IBL_INPUT_SIZE ) );

					Loop( { start: 0, end: IBL_INPUT_SIZE, type: 'int', name: 'i', condition: '<' }, ( { i } ) => {

						const a0_i = activationsStorage.element( iblA0.add( i ) );
						atomicAdd( gradWeightsAtomic.element( rowOffset.add( i ) ), int( delta1_j.mul( a0_i ).mul( float( FIXED_POINT_SCALE ) ) ) );

					} );

				} );

				Loop( { start: 0, end: 8, type: 'int', name: 'c', condition: '<' }, ( { c } ) => {

					const gradLatent_c = float( 0.0 ).toVar();

					Loop( { start: 0, end: iblHiddenSize, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

						const delta1_j = activationsStorage.element( actBase.add( int( actIblDelta1Offset ) ).add( j ) );
						const w_jc = weightsStorage.element( int( iblLayer0WeightsOffset ).add( j.mul( IBL_INPUT_SIZE ) ).add( c ) );
						gradLatent_c.addAssign( delta1_j.mul( w_jc ) );

					} );

					const curGradZ = activationsStorage.element( actBase.add( int( actGradLatentsOffset ) ).add( c ) );
					activationsStorage.element( actBase.add( int( actGradLatentsOffset ) ).add( c ) ).assign( curGradZ.add( gradLatent_c ) );

				} );

				const indirectA0 = actBase.add( int( actIndirectA0Offset ) );

				Loop( { start: 0, end: 8, type: 'int', name: 'c', condition: '<' }, ( { c } ) => {

					activationsStorage.element( indirectA0.add( c ) ).assign( activationsStorage.element( a0.add( c ) ) );

				} );

				activationsStorage.element( indirectA0.add( 8 ) ).assign( wo.x );
				activationsStorage.element( indirectA0.add( 9 ) ).assign( wo.y );
				activationsStorage.element( indirectA0.add( 10 ) ).assign( wo.z );
				activationsStorage.element( indirectA0.add( 11 ) ).assign( samplesStorage.element( sampleOffset.add( 25 ) ) );
				activationsStorage.element( indirectA0.add( 12 ) ).assign( samplesStorage.element( sampleOffset.add( 26 ) ) );
				activationsStorage.element( indirectA0.add( 13 ) ).assign( samplesStorage.element( sampleOffset.add( 27 ) ) );

				Loop( { start: 0, end: iblHiddenSize, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

					const val = weightsStorage.element( int( indirectLayer0BiasesOffset ).add( j ) ).toVar();
					const rowOffset = int( indirectLayer0WeightsOffset ).add( j.mul( INDIRECT_INPUT_SIZE ) );

					Loop( { start: 0, end: INDIRECT_INPUT_SIZE, type: 'int', name: 'i', condition: '<' }, ( { i } ) => {

						val.addAssign( weightsStorage.element( rowOffset.add( i ) ).mul( activationsStorage.element( indirectA0.add( i ) ) ) );

					} );

					activationsStorage.element( actBase.add( int( actIndirectZ1Offset ) ).add( j ) ).assign( val );
					activationsStorage.element( actBase.add( int( actIndirectA1Offset ) ).add( j ) ).assign( max( val, float( 0.0 ) ) );

				} );

				Loop( { start: 0, end: INDIRECT_OUTPUT_SIZE, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

					const val = weightsStorage.element( int( indirectLayer1BiasesOffset ).add( j ) ).toVar();
					const rowOffset = int( indirectLayer1WeightsOffset ).add( j.mul( iblHiddenSize ) );

					Loop( { start: 0, end: iblHiddenSize, type: 'int', name: 'i', condition: '<' }, ( { i } ) => {

						val.addAssign( weightsStorage.element( rowOffset.add( i ) ).mul( activationsStorage.element( actBase.add( int( actIndirectA1Offset ) ).add( i ) ) ) );

					} );

					activationsStorage.element( actBase.add( int( actIndirectZ2Offset ) ).add( j ) ).assign( val );

				} );

				const indirectDelta2 = actBase.add( int( actIndirectDelta2Offset ) );

				Loop( { start: 0, end: INDIRECT_OUTPUT_SIZE, type: 'int', name: 'c', condition: '<' }, ( { c } ) => {

					const z3_c = activationsStorage.element( actBase.add( int( actIndirectZ2Offset ) ).add( c ) );
					const yHat_c = max( z3_c, float( 0.0 ) );
					const target_c = samplesStorage.element( sampleOffset.add( 28 ).add( c ) );
					const predClamped = max( yHat_c, float( 1e-6 ) );
					const refClamped = max( target_c, float( 1e-6 ) );
					const predLog = pow( predClamped, float( 1.0 / 3.0 ) ).sub( 1.0 ).mul( 3.0 );
					const refLog = pow( refClamped, float( 1.0 / 3.0 ) ).sub( 1.0 ).mul( 3.0 );
					const diff = predLog.sub( refLog );
					sampleLoss.addAssign( diff.abs().mul( iblScale ).div( 3.0 ) );
					const leak = select( z3_c.greaterThan( 0.0 ), float( 1.0 ), float( OUTPUT_CLAMP_GRADIENT_LEAK ) );
					const grad = sign( diff ).mul( pow( predClamped, float( - 2.0 / 3.0 ) ) ).mul( iblScale ).div( 3.0 ).mul( leak );
					activationsStorage.element( indirectDelta2.add( c ) ).assign( grad );

				} );

				Loop( { start: 0, end: INDIRECT_OUTPUT_SIZE, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

					const delta_j = activationsStorage.element( indirectDelta2.add( j ) );
					atomicAdd( gradWeightsAtomic.element( int( indirectLayer1BiasesOffset ).add( j ) ), int( delta_j.mul( float( FIXED_POINT_SCALE ) ) ) );
					const rowOffset = int( indirectLayer1WeightsOffset ).add( j.mul( iblHiddenSize ) );

					Loop( { start: 0, end: iblHiddenSize, type: 'int', name: 'i', condition: '<' }, ( { i } ) => {

						const a1_i = activationsStorage.element( actBase.add( int( actIndirectA1Offset ) ).add( i ) );
						atomicAdd( gradWeightsAtomic.element( rowOffset.add( i ) ), int( delta_j.mul( a1_i ).mul( float( FIXED_POINT_SCALE ) ) ) );

					} );

				} );

				Loop( { start: 0, end: iblHiddenSize, type: 'int', name: 'i', condition: '<' }, ( { i } ) => {

					const gradInput_i = float( 0.0 ).toVar();

					Loop( { start: 0, end: INDIRECT_OUTPUT_SIZE, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

						const delta_j = activationsStorage.element( indirectDelta2.add( j ) );
						const w_ji = weightsStorage.element( int( indirectLayer1WeightsOffset ).add( j.mul( iblHiddenSize ) ).add( i ) );
						gradInput_i.addAssign( delta_j.mul( w_ji ) );

					} );

					const z1_i = activationsStorage.element( actBase.add( int( actIndirectZ1Offset ) ).add( i ) );
					activationsStorage.element( actBase.add( int( actIndirectDelta1Offset ) ).add( i ) ).assign(
						select( z1_i.greaterThan( 0.0 ), gradInput_i, float( 0.0 ) )
					);

				} );

				Loop( { start: 0, end: iblHiddenSize, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

					const delta1_j = activationsStorage.element( actBase.add( int( actIndirectDelta1Offset ) ).add( j ) );
					atomicAdd( gradWeightsAtomic.element( int( indirectLayer0BiasesOffset ).add( j ) ), int( delta1_j.mul( float( FIXED_POINT_SCALE ) ) ) );
					const rowOffset = int( indirectLayer0WeightsOffset ).add( j.mul( INDIRECT_INPUT_SIZE ) );

					Loop( { start: 0, end: INDIRECT_INPUT_SIZE, type: 'int', name: 'i', condition: '<' }, ( { i } ) => {

						const a0_i = activationsStorage.element( indirectA0.add( i ) );
						atomicAdd( gradWeightsAtomic.element( rowOffset.add( i ) ), int( delta1_j.mul( a0_i ).mul( float( FIXED_POINT_SCALE ) ) ) );

					} );

				} );

				Loop( { start: 0, end: 8, type: 'int', name: 'c', condition: '<' }, ( { c } ) => {

					const gradLatent_c = float( 0.0 ).toVar();

					Loop( { start: 0, end: iblHiddenSize, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

						const delta1_j = activationsStorage.element( actBase.add( int( actIndirectDelta1Offset ) ).add( j ) );
						const w_jc = weightsStorage.element( int( indirectLayer0WeightsOffset ).add( j.mul( INDIRECT_INPUT_SIZE ) ).add( c ) );
						gradLatent_c.addAssign( delta1_j.mul( w_jc ) );

					} );

					const curGradZ = activationsStorage.element( actBase.add( int( actGradLatentsOffset ) ).add( c ) );
					activationsStorage.element( actBase.add( int( actGradLatentsOffset ) ).add( c ) ).assign( curGradZ.add( gradLatent_c ) );

				} );

			} );

			// Accumulate loss after all active heads contribute.
			atomicAdd( lossAtomic.element( 0 ), int( sampleLoss.mul( float( FIXED_POINT_SCALE ) ) ) );

			// 15. Scatter Bilinear Latent Gradients
			Loop( { start: 0, end: 8, type: 'int', name: 'c', condition: '<' }, ( { c } ) => {

				const gradZ_c = activationsStorage.element( actBase.add( int( actGradLatentsOffset ) ).add( c ) );
				atomicAdd( gradLatentsAtomic.element( off0.add( c ) ), int( gradZ_c.mul( w0 ).mul( float( FIXED_POINT_SCALE ) ) ) );
				atomicAdd( gradLatentsAtomic.element( off1.add( c ) ), int( gradZ_c.mul( w1 ).mul( float( FIXED_POINT_SCALE ) ) ) );
				atomicAdd( gradLatentsAtomic.element( off2.add( c ) ), int( gradZ_c.mul( w2 ).mul( float( FIXED_POINT_SCALE ) ) ) );
				atomicAdd( gradLatentsAtomic.element( off3.add( c ) ), int( gradZ_c.mul( w3 ).mul( float( FIXED_POINT_SCALE ) ) ) );

			} );

		} );

	} )().compute( batchSize ).setName( 'NeuralAppearanceTrainBatch' );

}

/**
 * Clears the scalar accumulator used by the gradient clipping pass.
 */
function createResetGradientNormComputeNode( gpuModel ) {

	const { gradNormAtomic } = gpuModel;

	return Fn( () => {

		atomicStore( gradNormAtomic.element( 0 ), int( 0 ) );

	} )().compute( 1 ).setName( 'NeuralAppearanceResetGradientNorm' );

}

/**
 * Clears accumulated weight and latent gradients before a scoped optimizer pass.
 */
function createResetGradientsComputeNode( gpuModel ) {

	const {
		layout,
		gradWeightsAtomic,
		gradLatentsAtomic
	} = gpuModel;
	const dispatchCount = layout.totalWeights + layout.totalLatents;

	return Fn( () => {

		const idx = int( instanceIndex );

		If( idx.lessThan( int( layout.totalWeights ) ), () => {

			atomicStore( gradWeightsAtomic.element( idx ), int( 0 ) );

		} ).Else( () => {

			atomicStore( gradLatentsAtomic.element( idx.sub( int( layout.totalWeights ) ) ), int( 0 ) );

		} );

	} )().compute( dispatchCount ).setName( 'NeuralAppearanceResetGradients' );

}

/**
 * Accumulates squared weight and latent gradients for global norm clipping.
 */
function createAccumulateGradientNormComputeNode( gpuModel, { weightOffset = 0, weightCount = null, includeLatents = true } = {} ) {

	const {
		layout,
		gradWeightsAtomic,
		gradLatentsAtomic,
		gradNormAtomic
	} = gpuModel;

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
		weightsStorage,
		gradWeightsAtomic,
		mWeightsStorage,
		vWeightsStorage,
		learningRateUniform,
		stepUniform,
		gradNormAtomic,
		maxGradientNormUniform
	} = gpuModel;

	const resolvedWeightCount = weightCount === null ? layout.totalWeights : weightCount;

	return Fn( () => {

		const idx = int( instanceIndex ).add( int( weightOffset ) );
		const rawGrad = float( atomicLoad( gradWeightsAtomic.element( idx ) ) ).div( float( FIXED_POINT_SCALE ) );
		const grad = rawGrad.mul( computeGradientClipScale( gradNormAtomic, maxGradientNormUniform ) );
		atomicStore( gradWeightsAtomic.element( idx ), int( 0 ) );

		const m = mWeightsStorage.element( idx );
		const v = vWeightsStorage.element( idx );
		const w = weightsStorage.element( idx );

		const nextM = float( beta1 ).mul( m ).add( float( 1.0 - beta1 ).mul( grad ) );
		const nextV = float( beta2 ).mul( v ).add( float( 1.0 - beta2 ).mul( grad ).mul( grad ) );
		mWeightsStorage.element( idx ).assign( nextM );
		vWeightsStorage.element( idx ).assign( nextV );

		const beta1Corr = float( 1.0 ).sub( pow( float( beta1 ), float( stepUniform ) ) );
		const beta2Corr = float( 1.0 ).sub( pow( float( beta2 ), float( stepUniform ) ) );
		const mHat = nextM.div( max( beta1Corr, float( 1e-10 ) ) );
		const vHat = nextV.div( max( beta2Corr, float( 1e-10 ) ) );

		const stepVal = learningRateUniform.mul( mHat ).div( sqrt( max( vHat, float( 0.0 ) ) ).add( float( epsilon ) ) );
		weightsStorage.element( idx ).assign( w.sub( stepVal ) );

	} )().compute( resolvedWeightCount ).setName( 'NeuralAppearanceAdamWeights' );

}

/**
 * Creates the Adam optimizer compute node for multi-mip latent textures.
 */
function createAdamLatentsComputeNode( gpuModel, { beta1 = 0.9, beta2 = 0.999, epsilon = 1e-7 } = {} ) {

	const {
		layout,
		latentsStorage,
		gradLatentsAtomic,
		mLatentsStorage,
		vLatentsStorage,
		learningRateUniform,
		stepUniform,
		gradNormAtomic,
		maxGradientNormUniform
	} = gpuModel;

	const { totalLatents } = layout;

	return Fn( () => {

		const idx = int( instanceIndex );
		const rawGrad = float( atomicLoad( gradLatentsAtomic.element( idx ) ) ).div( float( FIXED_POINT_SCALE ) );
		const grad = rawGrad.mul( computeGradientClipScale( gradNormAtomic, maxGradientNormUniform ) );
		atomicStore( gradLatentsAtomic.element( idx ), int( 0 ) );

		const m = mLatentsStorage.element( idx );
		const v = vLatentsStorage.element( idx );
		const lat = latentsStorage.element( idx );

		const nextM = float( beta1 ).mul( m ).add( float( 1.0 - beta1 ).mul( grad ) );
		const nextV = float( beta2 ).mul( v ).add( float( 1.0 - beta2 ).mul( grad ).mul( grad ) );
		mLatentsStorage.element( idx ).assign( nextM );
		vLatentsStorage.element( idx ).assign( nextV );

		const beta1Corr = float( 1.0 ).sub( pow( float( beta1 ), float( stepUniform ) ) );
		const beta2Corr = float( 1.0 ).sub( pow( float( beta2 ), float( stepUniform ) ) );
		const mHat = nextM.div( max( beta1Corr, float( 1e-10 ) ) );
		const vHat = nextV.div( max( beta2Corr, float( 1e-10 ) ) );

		const stepVal = learningRateUniform.mul( mHat ).div( sqrt( max( vHat, float( 0.0 ) ) ).add( float( epsilon ) ) );
		latentsStorage.element( idx ).assign( lat.sub( stepVal ) );

	} )().compute( totalLatents ).setName( 'NeuralAppearanceAdamLatents' );

}

export {
	createTrainBatchComputeNode,
	createResetGradientNormComputeNode,
	createResetGradientsComputeNode,
	createAccumulateGradientNormComputeNode,
	createAdamWeightsComputeNode,
	createAdamLatentsComputeNode
};
