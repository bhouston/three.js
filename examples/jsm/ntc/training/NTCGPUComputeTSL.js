import {
	Fn,
	If,
	Loop,
	atomicAdd,
	atomicLoad,
	cos,
	exp,
	float,
	floor,
	fract,
	instanceIndex,
	int,
	log,
	sin,
	textureLevel,
	vec2
} from 'three/tsl';
import { FIXED_POINT_SCALE, GRADIENT_NORM_SCALE } from './NTCGPUTrainingConstants.js';
import {
	wrapIndexTSL,
	forwardDenseLayerTSL,
	accumulateDenseLayerGradTSL,
	backwardDenseLayerReLUTSL,
	createAdamComputeNode
} from './NTCGPUKernelsTSL.js';
import { applyChannelActivation, channelActivationDerivativeFromOutput } from '../NTCOutputActivations.js';
import { QUANTIZATION_SCHEMES } from './NTCQuantization.js';
import { selectFeatureLevelTSL } from '../NTCMipBands.js';
import { applyUvTransformTSL } from '../NTCUvTransform.js';

function hash1( seed ) {

	return fract( sin( seed ).mul( 43758.5453123 ) );

}

/**
 * Generates a stratified-random UV in [0,1)^2 for training sample `sampleIdx`,
 * re-jittered every training step via `stepUniform` so the same texel isn't
 * sampled every iteration.
 *
 * `tileRange` (default 0 - a byte-for-byte no-op, only ever non-zero when
 * `uvTransformNode` is in play, see createTextureTrainBatchComputeNode's doc
 * comment on why) additionally offsets the sample by a random *integer*
 * number of tiles in `[-tileRange, tileRange]` along each axis, re-drawn
 * every sample/step exactly like the fractional jitter above. Both the
 * source-texture ground-truth sample and the (`wrapIndexTSL`-addressed) grid
 * taps already wrap correctly for any real coordinate - see this file's other
 * doc comments - so widening the sampled range costs nothing but a couple
 * extra hashes; it exists purely to fix *coverage*, not correctness: a
 * learned rotation/anisotropic scale generally isn't in `GL(2,Z)`, so folding
 * a single UV tile's transformed image back onto the grid's periodic (`mod
 * 1`) canonical space does not tile it evenly - some grid cells would get
 * systematically over/under-sampled if training only ever drew from one
 * tile. Sampling a handful of tiles and letting the existing wrap addressing
 * do the folding is enough for that non-uniformity to average out over many
 * steps.
 */
function randomStratifiedUV( sampleIdx, stepUniform, gridSize, tileRange = 0 ) {

	const cellX = sampleIdx.mod( int( gridSize ) );
	const cellY = int( floor( float( sampleIdx ).div( float( gridSize ) ) ) );
	const jitterSeed = float( sampleIdx ).mul( 12.9898 ).add( stepUniform.mul( 78.233 ) );
	const jx = hash1( jitterSeed );
	const jy = hash1( jitterSeed.add( 91.345 ) );
	let u = float( cellX ).add( jx ).div( float( gridSize ) );
	let v = float( cellY ).add( jy ).div( float( gridSize ) );

	if ( tileRange > 0 ) {

		const tileSpan = float( 2 * tileRange + 1 );
		const tileX = float( int( hash1( jitterSeed.add( 171.23 ) ).mul( tileSpan ) ) ).sub( tileRange );
		const tileY = float( int( hash1( jitterSeed.add( 254.67 ) ).mul( tileSpan ) ) ).sub( tileRange );
		u = u.add( tileX );
		v = v.add( tileY );

	}

	return vec2( u, v );

}

/**
 * Samples a per-training-sample LOD (mip index) from `seedBase`, following
 * the NVIDIA neural texture compression paper's own batch-LOD distribution
 * (Section 5.1): `LOD = floor(-log4(X))`, `X ~ U(0,1)` - this biases sampling
 * toward finer mips, which is correct since they cover proportionally more
 * texels (a mip is 4x the texel count of the next-coarser one), so a uniform
 * *texel* sampling density across the whole pyramid naturally means most
 * *samples* land at fine LODs. A separate 5% of samples instead draw LOD
 * uniformly across `[0, maxLod]`, so the coarsest mips - which the area-
 * biased distribution alone would sample increasingly rarely as `maxLod`
 * grows - still get enough training signal to actually converge, rather than
 * their feature levels/decoder weights merely drifting from initialization.
 * Both draws (and the 5% selector) are re-derived from `seedBase`, so the
 * whole thing changes every sample/training-step exactly like
 * `randomStratifiedUV`'s jitter does.
 */
function sampleTrainingLod( seedBase, maxLod ) {

	const x = hash1( seedBase ).max( 1e-6 );
	const areaLod = floor( log( x ).div( Math.log( 4 ) ).negate() );
	const uniformLod = floor( hash1( seedBase.add( 131.71 ) ).mul( maxLod + 1 ) );
	const useUniform = hash1( seedBase.add( 257.13 ) ).lessThan( 0.05 );

	return useUniform.select( uniformLod, areaLod ).clamp( 0, maxLod );

}

/**
 * Creates the training compute node: samples the source texture(s) directly
 * (no teacher-atlas readback needed, since the target is already a static
 * GPU texture), forward-evaluates the mip pyramid + MLP decoder, computes an
 * L2 loss, and hand-differentiates the backward pass - accumulating
 * gradients atomically exactly like the neural appearance trainer's GPU
 * backward pass. Runs one invocation per batch sample.
 *
 * Every sample trains against exactly one, stochastically chosen, integer
 * mip level (see `sampleTrainingLod`, sampled across the model's full
 * `maxLod` mip range - see NTCGridPyramidModel.js) - the source texture is
 * sampled at that exact mip, and that LOD is mapped down onto exactly one
 * *stored* grid level (`selectFeatureLevelTSL`, see NTCMipBands.js - the
 * pyramid stores far fewer levels than `maxLod`, each one reused to
 * reconstruct several physical mips, matching the NVIDIA neural texture
 * compression paper's Table 1 rather than a real per-mip GPU mip chain). The
 * decoder's input is that one selected grid level's `channels`-wide bilinear
 * tap plus the normalized LOD itself - the LOD is what lets the *same*
 * stored level's decode differ correctly across the several physical mips it
 * covers (see NTCGridPyramidModel.js's doc comment for why the decoder is
 * always `channels + 1` wide, independent of how many levels are stored).
 * No cross-level blending happens during training - only ever one level is
 * selected per sample - that only happens at inference, via interpolation
 * between two independently-trained neighboring levels (see
 * NTCDecoderTSL.js).
 *
 * `sourceTextures` is an array of RGBA textures whose channels are
 * concatenated (in order, up to 4 components each) to form the
 * `outputChannels`-wide training target - e.g. a single albedo texture for
 * the texture-fitting demo, or 5 packed textures for the full-material demo
 * (see NeuralMaterialFormat.js). Each must have a real mip chain
 * (`generateMipmaps: true` or a manually supplied `.mipmaps`) reaching at
 * least `maxLod` levels deep, since sampling stops each sample's chosen LOD
 * from *this* texture, not from the latent grids.
 *
 * `gpuModel.layout.channelActivations`, if present, is a flat array (one
 * entry per output channel, `undefined`/omitted entries default to plain
 * linear) naming an output nonlinearity applied to that channel's raw
 * decoder output before the loss/delta are computed - see
 * ./NTCOutputActivations.js. Used by neural-material to fit each
 * channel's targets in their own natural physical range (bounded reflectance
 * via sigmoid, signed tangent-space offsets via tanh, HDR emission via
 * softplus) instead of forcing every channel through the same unbounded
 * linear output.
 *
 * `uvTransformState`, when given (see NTCUvTransformGPU.
 * createUvTransformGPUState), applies its live `matrixNode` to a *copy* of
 * `uv` used only for the grid-level taps below (step 1) - never to the
 * ground-truth source-texture sample above, which must stay in the mesh's
 * real UV space (what we're trying to reproduce output *for*, not an input
 * the grid gets to reparameterize). This function's backward pass (step
 * "5.5", after the usual latent-grid gradient scatter) additionally
 * computes the analytic gradient of the loss with respect to
 * `uvTransformState`'s 3 scalars and atomically accumulates it into
 * `uvTransformState.buffers.gradAtomic` - see that step's own doc comment
 * for the derivation. `tileRange`, only meaningful alongside a non-null
 * `uvTransformState` (see `randomStratifiedUV`'s doc comment for why),
 * widens where `uv` itself is drawn from.
 */
function createTextureTrainBatchComputeNode( gpuModel, sourceTextures, uvTransformState = null, tileRange = 0 ) {

	const {
		layout,
		batchSize,
		activationsStorage,
		lossAtomic,
		stepUniform,
		quantization,
		quantizationRangeUniforms
	} = gpuModel;
	const { valuesStorage: weightsStorage, gradAtomic: gradWeightsAtomic } = gpuModel.weightsBuffers;
	const { valuesStorage: latentsStorage, gradAtomic: gradLatentsAtomic } = gpuModel.latentsBuffers;
	// QAT (see NeuralQuantization.js/NTCGPUModel.js): only latents
	// are quantized (`target !== 'latents'` is rejected by
	// resolveQuantizationConfig unless mode is 'none'), and only the forward
	// pass - see the module doc comment in NeuralQuantization.js for why a
	// forward-only Straight-Through-Estimator quantize is enough here. `mode
	// === 'none'` is resolved to a plain passthrough at kernel-*build* time
	// (not per-invocation), so the default training path's node graph is
	// byte-for-byte what it was before QAT existed.
	const quantizeLatent = quantization.mode !== 'none' ?
		QUANTIZATION_SCHEMES[ quantization.mode ].quantizeForwardTSL :
		null;

	const {
		gridLevels,
		channels,
		mlpLayers,
		a0Offset,
		layerActs,
		deltaOffsets,
		gradA0Offset,
		activationStride,
		outputChannels,
		channelActivations,
		mipsPerLevel,
		maxLod
	} = layout;

	const gridSize = Math.max( 1, Math.ceil( Math.sqrt( batchSize ) ) );

	return Fn( () => {

		const sampleIdx = int( instanceIndex );
		const uv = randomStratifiedUV( sampleIdx, stepUniform, gridSize, tileRange );
		// Grid-tap coordinate: `uv` re-aligned by the learned UV transform
		// when one is in play (see this function's doc comment above),
		// otherwise identical to `uv` - the ground-truth sample below always
		// uses plain `uv`, never this.
		const canonicalUv = uvTransformState !== null ? applyUvTransformTSL( uv, uvTransformState.matrixNode ) : uv;
		const actBase = sampleIdx.mul( int( activationStride ) );

		// This sample's stochastically chosen, exact-integer training LOD,
		// across the model's full physical mip range - see
		// sampleTrainingLod's doc comment - and the *stored* grid level it
		// maps onto (see this function's doc comment and NTCMipBands.js).
		const lod = sampleTrainingLod( float( sampleIdx ).mul( 12.9898 ).add( stepUniform.mul( 78.233 ) ), maxLod );
		const selectedLevel = selectFeatureLevelTSL( lod, gridLevels.length, mipsPerLevel );

		const targetComponents = [];

		for ( const sourceTexture of sourceTextures ) {

			const sample = textureLevel( sourceTexture, uv, lod );
			const remaining = outputChannels - targetComponents.length;

			if ( remaining > 0 ) targetComponents.push( sample.x );
			if ( remaining > 1 ) targetComponents.push( sample.y );
			if ( remaining > 2 ) targetComponents.push( sample.z );
			if ( remaining > 3 ) targetComponents.push( sample.w );

		}

		// 1. Bilinear-sample every grid level (wrap addressing), but keep only
		// the one matching `selectedLevel` - every other level's contribution
		// is multiplied by an exact 0/1 selector (`weight`) rather than being
		// omitted from the shader, since which level is selected is a
		// per-invocation runtime value, not something known at kernel-build
		// time (unlike `gridLevels.length` itself, which is why this loop is
		// still unrolled in JS). Accumulated into a single shared,
		// `channels`-wide `a0Vars` (not a per-level slot - see
		// NTCGridPyramidModel.js's doc comment on why the decoder input no
		// longer scales with level count): since `weight` is 1 for exactly
		// one `g` and 0 for every other, the sum equals that one level's
		// value.
		const levelTaps = [];
		const a0Vars = [];
		for ( let c = 0; c < channels; c ++ ) a0Vars.push( float( 0.0 ).toVar() );

		for ( let g = 0; g < gridLevels.length; g ++ ) {

			const level = gridLevels[ g ];
			const x = canonicalUv.x.mul( level.width ).sub( 0.5 );
			const y = canonicalUv.y.mul( level.height ).sub( 0.5 );
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

			const weight = selectedLevel.equal( int( g ) ).select( float( 1 ), float( 0 ) );

			levelTaps.push( { off0, off1, off2, off3, w0, w1, w2, w3, weight } );

			for ( let c = 0; c < channels; c ++ ) {

				const z_c = latentsStorage.element( off0.add( c ) ).mul( w0 )
					.add( latentsStorage.element( off1.add( c ) ).mul( w1 ) )
					.add( latentsStorage.element( off2.add( c ) ).mul( w2 ) )
					.add( latentsStorage.element( off3.add( c ) ).mul( w3 ) );

				// QAT forward quantize (STE) - the backward scatter in step 5
				// below still targets the *raw* latentsStorage taps untouched,
				// exactly as it did before QAT: only this forward-read value
				// changes.
				const quantized_c = quantizeLatent !== null ?
					quantizeLatent( z_c, quantizationRangeUniforms[ g ].min, quantizationRangeUniforms[ g ].max ) :
					z_c;

				a0Vars[ c ].addAssign( quantized_c.mul( weight ) );

			}

		}

		for ( let c = 0; c < channels; c ++ ) {

			activationsStorage.element( actBase.add( int( a0Offset + c ) ) ).assign( a0Vars[ c ] );

		}

		// Append the normalized LOD value as the decoder's final input
		// component (see NTCGridPyramidModel.js's `inputSize = channels + 1`)
		// - necessary because the selected grid level alone doesn't say which
		// of its several covered mips this sample targets; this is what lets
		// the shared decoder disambiguate that, matching the paper's own
		// decoder input layout (Section 4.4: "... and a LOD value").
		activationsStorage.element( actBase.add( int( a0Offset + channels ) ) ).assign( lod.div( Math.max( 1, maxLod ) ) );

		// 2. Forward MLP (ReLU hidden layers, linear output).
		for ( let l = 0; l < mlpLayers.length; l ++ ) {

			const layer = mlpLayers[ l ];
			const inBase = actBase.add( int( l === 0 ? a0Offset : layerActs[ l - 1 ].aOffset ) );
			const zBase = actBase.add( int( layerActs[ l ].zOffset ) );
			const aBase = layerActs[ l ].aOffset >= 0 ? actBase.add( int( layerActs[ l ].aOffset ) ) : null;

			forwardDenseLayerTSL( {
				activationsStorage, weightsStorage,
				inputBase: inBase, inputSize: layer.inputSize, outputSize: layer.outputSize,
				weightsOffset: layer.weightsOffset, biasesOffset: layer.biasesOffset,
				zBase, aBase
			} );

		}

		// 3. L2 loss + output delta.
		//
		// Each output channel c may carry its own output nonlinearity (see
		// ./NTCOutputActivations.js, keyed by NeuralMaterialFormat.
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

			accumulateDenseLayerGradTSL( {
				activationsStorage, gradWeightsAtomic,
				deltaBase, inputBase: inBase, inputSize: layer.inputSize, outputSize: layer.outputSize,
				weightsOffset: layer.weightsOffset, biasesOffset: layer.biasesOffset
			} );

			if ( l > 0 ) {

				const prevZBase = actBase.add( int( layerActs[ l - 1 ].zOffset ) );
				const prevDeltaBase = actBase.add( int( deltaOffsets[ l - 1 ] ) );

				backwardDenseLayerReLUTSL( {
					activationsStorage, weightsStorage,
					deltaBase, deltaSize: layer.outputSize,
					weightsOffset: layer.weightsOffset, prevSize: layer.inputSize,
					prevZBase, outDeltaBase: prevDeltaBase
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
		// taps/weights computed in the forward pass. The chain rule needs the
		// same selection `weight` here too - forward computed `a0_c = weight_g
		// * bilinear(...)`, and `weight_g` doesn't depend on the latents
		// themselves, so `d(a0_c)/d(latent) = weight_g * d(bilinear)/d(latent)`
		// - i.e. exactly `gradZ_c * weight_g` scattered through the same
		// bilinear taps below. Since `weight_g` is 0 for every level but the
		// one `selectedLevel` picked, only that level's latents actually
		// receive gradient this sample - every other level's atomicAdd below
		// contributes exactly 0.
		const gradA0Base = actBase.add( int( gradA0Offset ) );

		for ( let g = 0; g < gridLevels.length; g ++ ) {

			const taps = levelTaps[ g ];

			for ( let c = 0; c < channels; c ++ ) {

				const gradZ_c = activationsStorage.element( gradA0Base.add( c ) ).mul( taps.weight );

				atomicAdd( gradLatentsAtomic.element( taps.off0.add( c ) ), int( gradZ_c.mul( taps.w0 ).mul( float( FIXED_POINT_SCALE ) ) ) );
				atomicAdd( gradLatentsAtomic.element( taps.off1.add( c ) ), int( gradZ_c.mul( taps.w1 ).mul( float( FIXED_POINT_SCALE ) ) ) );
				atomicAdd( gradLatentsAtomic.element( taps.off2.add( c ) ), int( gradZ_c.mul( taps.w2 ).mul( float( FIXED_POINT_SCALE ) ) ) );
				atomicAdd( gradLatentsAtomic.element( taps.off3.add( c ) ), int( gradZ_c.mul( taps.w3 ).mul( float( FIXED_POINT_SCALE ) ) ) );

			}

		}

		// 5.5. Analytic gradient of the loss w.r.t. the UV transform's 3
		// scalars (rotation, log-scale x, log-scale y - see
		// NTCUvTransformGPU.js), when one is in play. Two chain-rule stages:
		//
		//  a) d(a0_c)/d(canonicalUv) - the standard bilinear-sample
		//     coordinate gradient (the same one used to backprop through
		//     `grid_sample` in e.g. PyTorch's Spatial Transformer tutorial):
		//     treating each tap's integer texel index as locally constant
		//     (the same Straight-Through-Estimator-style approximation the
		//     forward pass's own bilinear weights already are, since a
		//     `floor()` has no useful derivative), `d(bilinear)/d(tx) =
		//     (1-ty)*(L10-L00) + ty*(L11-L01)` and symmetrically for `ty` -
		//     `tx`/`ty` are recovered here as `w1+w3`/`w2+w3` (algebraically
		//     equal to the same fractional taps step 1 computed, see its
		//     `w0..w3` - not recomputed as a new floor/fract) - chained
		//     through `d(tx)/d(canonicalUv.x) = level.width` (from step 1's
		//     `x = canonicalUv.x * level.width - 0.5`) and `d(ty)/d(
		//     canonicalUv.y) = level.height` likewise. Reusing `gradA0`
		//     (already computed above, `dL/d(a0_c)`) and `levelTaps`/
		//     `latentsStorage` (from step 1) gives `dL/d(canonicalUv)`
		//     without re-deriving anything step 1/4 didn't already compute.
		//
		//  b) d(canonicalUv)/d(rotation, log-scale) - the analytic Jacobian
		//     of `canonicalUv = M(rotation, exp(logScale)) . uv` (see
		//     NTCUvTransform.composeUvTransformMatrixTSL's `[a,b,c,d,e,f]` =
		//     `[cosθ·sx, -sinθ·sy, sinθ·sx, cosθ·sy, 0, 0]`), evaluated at
		//     this sample's own `uv` (the *original*, untransformed
		//     coordinate - the same one step 1 fed into the transform) and
		//     the transform's current live `rotationNode`/`logScaleX/YNode`
		//     values.
		if ( uvTransformState !== null ) {

			const gradUvX = float( 0.0 ).toVar();
			const gradUvY = float( 0.0 ).toVar();

			for ( let g = 0; g < gridLevels.length; g ++ ) {

				const level = gridLevels[ g ];
				const taps = levelTaps[ g ];
				const tx = taps.w1.add( taps.w3 );
				const ty = taps.w2.add( taps.w3 );

				for ( let c = 0; c < channels; c ++ ) {

					const gradA0_c = activationsStorage.element( gradA0Base.add( c ) ).mul( taps.weight );

					const l00 = latentsStorage.element( taps.off0.add( c ) );
					const l10 = latentsStorage.element( taps.off1.add( c ) );
					const l01 = latentsStorage.element( taps.off2.add( c ) );
					const l11 = latentsStorage.element( taps.off3.add( c ) );

					const dTapDx = float( 1.0 ).sub( ty ).mul( l10.sub( l00 ) ).add( ty.mul( l11.sub( l01 ) ) );
					const dTapDy = float( 1.0 ).sub( tx ).mul( l01.sub( l00 ) ).add( tx.mul( l11.sub( l10 ) ) );

					gradUvX.addAssign( gradA0_c.mul( dTapDx ).mul( level.width ) );
					gradUvY.addAssign( gradA0_c.mul( dTapDy ).mul( level.height ) );

				}

			}

			const theta = uvTransformState.rotationNode;
			const sx = exp( uvTransformState.logScaleXNode );
			const sy = exp( uvTransformState.logScaleYNode );
			const cosT = cos( theta );
			const sinT = sin( theta );

			const dCxDTheta = sinT.negate().mul( sx ).mul( uv.x ).sub( cosT.mul( sy ).mul( uv.y ) );
			const dCyDTheta = cosT.mul( sx ).mul( uv.x ).sub( sinT.mul( sy ).mul( uv.y ) );
			const dCxDLogSx = cosT.mul( sx ).mul( uv.x );
			const dCyDLogSx = sinT.mul( sx ).mul( uv.x );
			const dCxDLogSy = sinT.negate().mul( sy ).mul( uv.y );
			const dCyDLogSy = cosT.mul( sy ).mul( uv.y );

			const gradTheta = gradUvX.mul( dCxDTheta ).add( gradUvY.mul( dCyDTheta ) );
			const gradLogSx = gradUvX.mul( dCxDLogSx ).add( gradUvY.mul( dCyDLogSx ) );
			const gradLogSy = gradUvX.mul( dCxDLogSy ).add( gradUvY.mul( dCyDLogSy ) );

			// Same raw, un-batch-averaged fixed-point atomic accumulation
			// convention as every other gradient in this kernel - see this
			// function's step-3 doc comment for why.
			atomicAdd( uvTransformState.buffers.gradAtomic.element( 0 ), int( gradTheta.mul( float( FIXED_POINT_SCALE ) ) ) );
			atomicAdd( uvTransformState.buffers.gradAtomic.element( 1 ), int( gradLogSx.mul( float( FIXED_POINT_SCALE ) ) ) );
			atomicAdd( uvTransformState.buffers.gradAtomic.element( 2 ), int( gradLogSy.mul( float( FIXED_POINT_SCALE ) ) ) );

		}

	} )().compute( batchSize ).setName( 'NTCTrainBatch' );

}

/**
 * Accumulates squared weight and latent gradients for global norm clipping.
 * Unlike the neural-appearance version, the raw fixed-point sum is first
 * converted back to a batch-averaged gradient (`.mul(invBatchUniform)`)
 * before squaring - see createTextureTrainBatchComputeNode for why gradients
 * are deposited at raw, un-averaged magnitude.
 */
function createAccumulateGradientNormComputeNode( gpuModel ) {

	const { layout, gradNormAtomic, invBatchUniform } = gpuModel;
	const { gradAtomic: gradWeightsAtomic } = gpuModel.weightsBuffers;
	const { gradAtomic: gradLatentsAtomic } = gpuModel.latentsBuffers;
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

	} )().compute( dispatchCount ).setName( 'NTCAccumulateGradientNorm' );

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
		learningRateUniform,
		stepUniform,
		gradNormAtomic,
		maxGradientNormUniform,
		invBatchUniform
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
		invBatchUniform,
		count: layout.totalWeights,
		beta1,
		beta2,
		epsilon,
		name: 'NTCAdamWeights'
	} );

}

/**
 * Adam step for latent grid features - same batch-averaging fix as
 * createTextureAdamWeightsComputeNode above.
 */
function createTextureAdamLatentsComputeNode( gpuModel, { beta1 = 0.9, beta2 = 0.999, epsilon = 1e-7 } = {} ) {

	const {
		layout,
		learningRateUniform,
		stepUniform,
		gradNormAtomic,
		maxGradientNormUniform,
		invBatchUniform
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
		invBatchUniform,
		count: layout.totalLatents,
		beta1,
		beta2,
		epsilon,
		name: 'NTCAdamLatents'
	} );

}

export {
	createTextureTrainBatchComputeNode,
	createAccumulateGradientNormComputeNode,
	createTextureAdamWeightsComputeNode,
	createTextureAdamLatentsComputeNode
};
