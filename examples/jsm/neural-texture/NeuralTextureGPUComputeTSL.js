import {
	Fn,
	If,
	Loop,
	atomicAdd,
	atomicLoad,
	float,
	floor,
	fract,
	instanceIndex,
	int,
	max,
	select,
	sin,
	textureLevel,
	vec2
} from 'three/tsl';
import { FIXED_POINT_SCALE, GRADIENT_NORM_SCALE } from '../neural/NeuralGPUTrainingConstants.js';
import { wrapIndexTSL, createAdamComputeNode } from '../neural/NeuralGPUComputeTSL.js';
import { applyChannelActivation, channelActivationDerivativeFromOutput } from '../neural/NeuralOutputActivations.js';

function hash1( seed ) {

	return fract( sin( seed ).mul( 43758.5453123 ) );

}

/**
 * Generates a stratified-random UV in [0,1)^2 for training sample `sampleIdx`,
 * re-jittered every training step via `stepUniform` so the same texel isn't
 * sampled every iteration.
 */
function randomStratifiedUV( sampleIdx, stepUniform, gridSize ) {

	const cellX = sampleIdx.mod( int( gridSize ) );
	const cellY = int( floor( float( sampleIdx ).div( float( gridSize ) ) ) );
	const jitterSeed = float( sampleIdx ).mul( 12.9898 ).add( stepUniform.mul( 78.233 ) );
	const jx = hash1( jitterSeed );
	const jy = hash1( jitterSeed.add( 91.345 ) );
	const u = float( cellX ).add( jx ).div( float( gridSize ) );
	const v = float( cellY ).add( jy ).div( float( gridSize ) );

	return vec2( u, v );

}

/**
 * Creates the training compute node: samples the source texture(s) directly
 * (no teacher-atlas readback needed, since the target is already a static
 * GPU texture), forward-evaluates the multiresolution grid + MLP decoder,
 * computes an L2 loss, and hand-differentiates the backward pass -
 * accumulating gradients atomically exactly like the neural appearance
 * trainer's GPU backward pass. Runs one invocation per batch sample.
 *
 * `sourceTextures` is an array of RGBA textures whose channels are
 * concatenated (in order, up to 4 components each) to form the
 * `outputChannels`-wide training target - e.g. a single albedo texture for
 * the texture-fitting demo, or 5 packed textures for the full-material demo
 * (see NeuralMaterialFormat.js).
 *
 * `gpuModel.layout.channelActivations`, if present, is a flat array (one
 * entry per output channel, `undefined`/omitted entries default to plain
 * linear) naming an output nonlinearity applied to that channel's raw
 * decoder output before the loss/delta are computed - see
 * ../neural/NeuralOutputActivations.js. Used by neural-material to fit each
 * channel's targets in their own natural physical range (bounded reflectance
 * via sigmoid, signed tangent-space offsets via tanh, HDR emission via
 * softplus) instead of forcing every channel through the same unbounded
 * linear output.
 */
function createTextureTrainBatchComputeNode( gpuModel, sourceTextures ) {

	const {
		layout,
		batchSize,
		activationsStorage,
		weightsStorage,
		gradWeightsAtomic,
		latentsStorage,
		gradLatentsAtomic,
		lossAtomic,
		stepUniform
	} = gpuModel;

	const {
		gridLevels,
		channels,
		mlpLayers,
		a0Offset,
		uvInputOffset,
		includeUV,
		layerActs,
		deltaOffsets,
		gradA0Offset,
		activationStride,
		outputChannels,
		channelActivations
	} = layout;

	const gridSize = Math.max( 1, Math.ceil( Math.sqrt( batchSize ) ) );

	return Fn( () => {

		const sampleIdx = int( instanceIndex );
		const uv = randomStratifiedUV( sampleIdx, stepUniform, gridSize );
		const actBase = sampleIdx.mul( int( activationStride ) );

		const targetComponents = [];

		for ( const sourceTexture of sourceTextures ) {

			const sample = textureLevel( sourceTexture, uv, 0 );
			const remaining = outputChannels - targetComponents.length;

			if ( remaining > 0 ) targetComponents.push( sample.x );
			if ( remaining > 1 ) targetComponents.push( sample.y );
			if ( remaining > 2 ) targetComponents.push( sample.z );
			if ( remaining > 3 ) targetComponents.push( sample.w );

		}

		// 1. Bilinear-sample every grid level (wrap addressing) and concatenate
		// their features into a0 - this is the trainable positional encoding.
		const levelTaps = [];

		for ( let g = 0; g < gridLevels.length; g ++ ) {

			const level = gridLevels[ g ];
			const x = uv.x.mul( level.width ).sub( 0.5 );
			const y = uv.y.mul( level.height ).sub( 0.5 );
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

			const off0 = int( level.offset ).add( tapY0.mul( level.width ).add( tapX0 ).mul( channels ) );
			const off1 = int( level.offset ).add( tapY1.mul( level.width ).add( tapX1 ).mul( channels ) );
			const off2 = int( level.offset ).add( tapY2.mul( level.width ).add( tapX2 ).mul( channels ) );
			const off3 = int( level.offset ).add( tapY3.mul( level.width ).add( tapX3 ).mul( channels ) );

			levelTaps.push( { off0, off1, off2, off3, w0, w1, w2, w3 } );

			for ( let c = 0; c < channels; c ++ ) {

				const z_c = latentsStorage.element( off0.add( c ) ).mul( w0 )
					.add( latentsStorage.element( off1.add( c ) ).mul( w1 ) )
					.add( latentsStorage.element( off2.add( c ) ).mul( w2 ) )
					.add( latentsStorage.element( off3.add( c ) ).mul( w3 ) );

				activationsStorage.element( actBase.add( int( a0Offset + g * channels + c ) ) ).assign( z_c );

			}

		}

		// 1b. Optionally append the raw (u, v) sample coordinate after the grid
		// taps, as a direct "skip connection" to exact sub-texel position
		// alongside the (bilinearly-blurred) learned grid encoding. Not
		// trainable, so its backward gradient (computed into gradA0 below along
		// with everything else) is simply never scattered anywhere in step 5.
		if ( includeUV ) {

			activationsStorage.element( actBase.add( int( a0Offset + uvInputOffset ) ) ).assign( uv.x );
			activationsStorage.element( actBase.add( int( a0Offset + uvInputOffset + 1 ) ) ).assign( uv.y );

		}

		// 2. Forward MLP (ReLU hidden layers, linear output).
		for ( let l = 0; l < mlpLayers.length; l ++ ) {

			const layer = mlpLayers[ l ];
			const inBase = actBase.add( int( l === 0 ? a0Offset : layerActs[ l - 1 ].aOffset ) );
			const zBase = actBase.add( int( layerActs[ l ].zOffset ) );
			const aBase = layerActs[ l ].aOffset >= 0 ? actBase.add( int( layerActs[ l ].aOffset ) ) : null;

			Loop( { start: 0, end: layer.outputSize, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

				const val = weightsStorage.element( int( layer.biasesOffset ).add( j ) ).toVar();
				const rowOffset = int( layer.weightsOffset ).add( j.mul( layer.inputSize ) );

				Loop( { start: 0, end: layer.inputSize, type: 'int', name: 'i', condition: '<' }, ( { i } ) => {

					val.addAssign( weightsStorage.element( rowOffset.add( i ) ).mul( activationsStorage.element( inBase.add( i ) ) ) );

				} );

				activationsStorage.element( zBase.add( j ) ).assign( val );
				if ( aBase !== null ) activationsStorage.element( aBase.add( j ) ).assign( max( val, float( 0.0 ) ) );

			} );

		}

		// 3. L2 loss + output delta.
		//
		// Each output channel c may carry its own output nonlinearity (see
		// ../neural/NeuralOutputActivations.js, keyed by NeuralMaterialFormat.
		// js's per-channel `activation`) applied on top of this always-linear
		// decoder's raw `z` - the loss is computed
		// against the *activated* prediction `a = activation(z)` (matching a
		// raw-physical-units target), and the stored delta is the chain-rule
		// product `(a - target) * da/dz`, so everything downstream (step 4,
		// the hand-written backward pass) still just consumes a plain
		// per-output `dL/dz` exactly as it did for the old all-linear output.
		//
		// Deltas/gradients are deliberately kept at raw, un-batch-averaged
		// magnitude here (no division by batchSize). Gradients get quantized
		// to fixed-point integers and atomically summed one sample at a time
		// (`int(value * FIXED_POINT_SCALE)` truncates toward zero *before*
		// accumulating - see createTextureAdam*ComputeNode below) - if each
		// individual sample's contribution were pre-divided by batchSize here,
		// most per-sample gradients would truncate to exactly zero well before
		// the network actually converges (an 8192-sample batch needs a raw
		// error of ~8% just to survive 1e-5 quantization once divided by
		// 8192), which reads as loss plateauing into pure noise instead of
		// decreasing further. Dividing by batchSize only happens once, after
		// the full-precision sum has been accumulated (see invBatchUniform
		// usage in the Adam/gradient-norm kernels below).
		const outZBase = actBase.add( int( layerActs[ layerActs.length - 1 ].zOffset ) );
		const outDeltaBase = actBase.add( int( deltaOffsets[ deltaOffsets.length - 1 ] ) );
		const sampleLoss = float( 0.0 ).toVar();

		for ( let c = 0; c < outputChannels; c ++ ) {

			const activation = channelActivations ? channelActivations[ c ] : undefined;
			const z = activationsStorage.element( outZBase.add( c ) );
			const pred = applyChannelActivation( z, activation );
			const diff = pred.sub( targetComponents[ c ] );
			sampleLoss.addAssign( diff.mul( diff ).mul( 0.5 ) );
			const delta = diff.mul( channelActivationDerivativeFromOutput( pred, activation ) );
			activationsStorage.element( outDeltaBase.add( c ) ).assign( delta );

		}

		atomicAdd( lossAtomic.element( 0 ), int( sampleLoss.mul( float( FIXED_POINT_SCALE ) ) ) );

		// 4. Backward through the MLP layers (output -> input).
		for ( let l = mlpLayers.length - 1; l >= 0; l -- ) {

			const layer = mlpLayers[ l ];
			const deltaBase = actBase.add( int( deltaOffsets[ l ] ) );
			const inBase = actBase.add( int( l === 0 ? a0Offset : layerActs[ l - 1 ].aOffset ) );

			Loop( { start: 0, end: layer.outputSize, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

				const delta_j = activationsStorage.element( deltaBase.add( j ) );
				atomicAdd( gradWeightsAtomic.element( int( layer.biasesOffset ).add( j ) ), int( delta_j.mul( float( FIXED_POINT_SCALE ) ) ) );
				const rowOffset = int( layer.weightsOffset ).add( j.mul( layer.inputSize ) );

				Loop( { start: 0, end: layer.inputSize, type: 'int', name: 'i', condition: '<' }, ( { i } ) => {

					const a_i = activationsStorage.element( inBase.add( i ) );
					atomicAdd( gradWeightsAtomic.element( rowOffset.add( i ) ), int( delta_j.mul( a_i ).mul( float( FIXED_POINT_SCALE ) ) ) );

				} );

			} );

			if ( l > 0 ) {

				const prevZBase = actBase.add( int( layerActs[ l - 1 ].zOffset ) );
				const prevDeltaBase = actBase.add( int( deltaOffsets[ l - 1 ] ) );

				Loop( { start: 0, end: layer.inputSize, type: 'int', name: 'i', condition: '<' }, ( { i } ) => {

					const gradInput_i = float( 0.0 ).toVar();

					Loop( { start: 0, end: layer.outputSize, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

						const delta_j = activationsStorage.element( deltaBase.add( j ) );
						const w_ji = weightsStorage.element( int( layer.weightsOffset ).add( j.mul( layer.inputSize ) ).add( i ) );
						gradInput_i.addAssign( delta_j.mul( w_ji ) );

					} );

					const z_i = activationsStorage.element( prevZBase.add( i ) );
					activationsStorage.element( prevDeltaBase.add( i ) ).assign( select( z_i.greaterThan( 0.0 ), gradInput_i, float( 0.0 ) ) );

				} );

			} else {

				// Backward into gradA0 (the concatenated, un-activated grid features).
				const gradA0Base = actBase.add( int( gradA0Offset ) );

				Loop( { start: 0, end: layer.inputSize, type: 'int', name: 'i', condition: '<' }, ( { i } ) => {

					const gradInput_i = float( 0.0 ).toVar();

					Loop( { start: 0, end: layer.outputSize, type: 'int', name: 'j', condition: '<' }, ( { j } ) => {

						const delta_j = activationsStorage.element( deltaBase.add( j ) );
						const w_ji = weightsStorage.element( int( layer.weightsOffset ).add( j.mul( layer.inputSize ) ).add( i ) );
						gradInput_i.addAssign( delta_j.mul( w_ji ) );

					} );

					activationsStorage.element( gradA0Base.add( i ) ).assign( gradInput_i );

				} );

			}

		}

		// 5. Scatter gradA0 back into the latent grids using the same bilinear
		// taps/weights computed in the forward pass.
		const gradA0Base = actBase.add( int( gradA0Offset ) );

		for ( let g = 0; g < gridLevels.length; g ++ ) {

			const taps = levelTaps[ g ];

			for ( let c = 0; c < channels; c ++ ) {

				const gradZ_c = activationsStorage.element( gradA0Base.add( int( g * channels + c ) ) );
				atomicAdd( gradLatentsAtomic.element( taps.off0.add( c ) ), int( gradZ_c.mul( taps.w0 ).mul( float( FIXED_POINT_SCALE ) ) ) );
				atomicAdd( gradLatentsAtomic.element( taps.off1.add( c ) ), int( gradZ_c.mul( taps.w1 ).mul( float( FIXED_POINT_SCALE ) ) ) );
				atomicAdd( gradLatentsAtomic.element( taps.off2.add( c ) ), int( gradZ_c.mul( taps.w2 ).mul( float( FIXED_POINT_SCALE ) ) ) );
				atomicAdd( gradLatentsAtomic.element( taps.off3.add( c ) ), int( gradZ_c.mul( taps.w3 ).mul( float( FIXED_POINT_SCALE ) ) ) );

			}

		}

	} )().compute( batchSize ).setName( 'NeuralTextureTrainBatch' );

}

/**
 * Accumulates squared weight and latent gradients for global norm clipping.
 * Unlike the neural-appearance version, the raw fixed-point sum is first
 * converted back to a batch-averaged gradient (`.mul(invBatchUniform)`)
 * before squaring - see createTextureTrainBatchComputeNode for why gradients
 * are deposited at raw, un-averaged magnitude.
 */
function createAccumulateGradientNormComputeNode( gpuModel ) {

	const { layout, gradWeightsAtomic, gradLatentsAtomic, gradNormAtomic, invBatchUniform } = gpuModel;
	const { totalWeights, totalLatents } = layout;
	const dispatchCount = totalWeights + totalLatents;

	return Fn( () => {

		const idx = int( instanceIndex );
		const rawGrad = float( 0.0 ).toVar();

		If( idx.lessThan( int( totalWeights ) ), () => {

			rawGrad.assign( float( atomicLoad( gradWeightsAtomic.element( idx ) ) ).div( float( FIXED_POINT_SCALE ) ) );

		} ).Else( () => {

			const latentIdx = idx.sub( int( totalWeights ) );
			rawGrad.assign( float( atomicLoad( gradLatentsAtomic.element( latentIdx ) ) ).div( float( FIXED_POINT_SCALE ) ) );

		} );

		const grad = rawGrad.mul( invBatchUniform );
		atomicAdd( gradNormAtomic.element( 0 ), int( grad.mul( grad ).mul( float( GRADIENT_NORM_SCALE ) ) ) );

	} )().compute( dispatchCount ).setName( 'NeuralTextureAccumulateGradientNorm' );

}

/**
 * Adam step for MLP weights. Batch-averages the raw fixed-point gradient sum
 * (`.mul(invBatchUniform)`) only after it has been fully accumulated, so
 * individual sample contributions never get quantized below the fixed-point
 * resolution before summing.
 */
function createTextureAdamWeightsComputeNode( gpuModel, { beta1 = 0.9, beta2 = 0.999, epsilon = 1e-7 } = {} ) {

	const {
		layout,
		weightsStorage,
		gradWeightsAtomic,
		mWeightsStorage,
		vWeightsStorage,
		learningRateUniform,
		stepUniform,
		gradNormAtomic,
		maxGradientNormUniform,
		invBatchUniform
	} = gpuModel;

	return createAdamComputeNode( {
		valuesStorage: weightsStorage,
		gradAtomic: gradWeightsAtomic,
		mStorage: mWeightsStorage,
		vStorage: vWeightsStorage,
		gradNormAtomic,
		maxGradientNormUniform,
		learningRateUniform,
		stepUniform,
		invBatchUniform,
		count: layout.totalWeights,
		beta1,
		beta2,
		epsilon,
		name: 'NeuralTextureAdamWeights'
	} );

}

/**
 * Adam step for latent grid features - same batch-averaging fix as
 * createTextureAdamWeightsComputeNode above.
 */
function createTextureAdamLatentsComputeNode( gpuModel, { beta1 = 0.9, beta2 = 0.999, epsilon = 1e-7 } = {} ) {

	const {
		layout,
		latentsStorage,
		gradLatentsAtomic,
		mLatentsStorage,
		vLatentsStorage,
		learningRateUniform,
		stepUniform,
		gradNormAtomic,
		maxGradientNormUniform,
		invBatchUniform
	} = gpuModel;

	return createAdamComputeNode( {
		valuesStorage: latentsStorage,
		gradAtomic: gradLatentsAtomic,
		mStorage: mLatentsStorage,
		vStorage: vLatentsStorage,
		gradNormAtomic,
		maxGradientNormUniform,
		learningRateUniform,
		stepUniform,
		invBatchUniform,
		count: layout.totalLatents,
		beta1,
		beta2,
		epsilon,
		name: 'NeuralTextureAdamLatents'
	} );

}

export {
	createTextureTrainBatchComputeNode,
	createAccumulateGradientNormComputeNode,
	createTextureAdamWeightsComputeNode,
	createTextureAdamLatentsComputeNode
};
