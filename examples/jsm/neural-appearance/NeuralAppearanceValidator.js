import { powerLog } from '../neural/NeuralMLP.js';
import {
	getSampleWeightSum,
	cross,
	normalize
} from './NeuralAppearanceModel.js';
import {
	evaluateNeuralAppearanceJson,
	evaluateNeuralAppearanceOutputs,
	evaluateNeuralPrefilteredIBL,
	evaluateNeuralIBLWhiteFurnace,
	integrateNeuralBRDFWhiteFurnace
} from './NeuralAppearanceRuntime.js';

const DEFAULT_PREVIEW_SAMPLE_COUNT = 64;

function evaluateRuntimeValidation( json, samples, previewSampleCount = DEFAULT_PREVIEW_SAMPLE_COUNT ) {

	const invBatch = 1 / Math.max( getSampleWeightSum( samples ), 1 );
	const invAuxBatch = 1 / Math.max( samples.length, 1 );
	let brdfLoss = 0;
	let emissionLoss = 0;
	let emissionCount = 0;
	let opacityLoss = 0;
	let opacityCount = 0;
	let iblLoss = 0;
	let iblQueryLoss = 0;
	let iblIndirectLoss = 0;
	let iblCount = 0;
	let iblQueryCount = 0;
	let iblIndirectCount = 0;
	const whiteFurnace = createDifferenceMetric();
	const prefilteredIBL = createDifferenceMetric();
	const mipConsistency = createDifferenceMetric();
	const framePriors = createFramePriorMetric();
	const angularBins = {
		wi: createAngularBins(),
		wo: createAngularBins()
	};
	const reciprocity = createDifferenceMetric();
	const angularSmoothness = createDifferenceMetric();
	const preview = createRuntimePreview( previewSampleCount, samples.length );

	for ( const sample of samples ) {

		const sampleWeight = sample.weight !== undefined ? sample.weight : 1;
		const outputs = evaluateNeuralAppearanceOutputs( json, sample );
		const prediction = outputs.brdf;
		const target = sample.target;
		const nDotL = Math.max( sample.wi[ 2 ], 0 );
		const directTarget = Array.isArray( sample.directTarget ) ? sample.directTarget : target.map( ( value ) => value * nDotL );
		const directPrediction = prediction.map( ( value ) => value * nDotL );
		const reciprocalPrediction = evaluateNeuralAppearanceJson( json, { ...sample, wi: sample.wo, wo: sample.wi } );
		const perturbedWiPrediction = evaluateNeuralAppearanceJson( json, { ...sample, wi: perturbDirection( sample.wi, 0.02 ) } );
		const perturbedWoPrediction = evaluateNeuralAppearanceJson( json, { ...sample, wo: perturbDirection( sample.wo, 0.02 ) } );
		const nextMipPrediction = evaluateNextMipPrediction( json, sample );

		if ( sampleWeight > 0 ) {

			if ( prediction.every( Number.isFinite ) === false || target.every( Number.isFinite ) === false ) {

				throw new Error( 'THREE.NeuralAppearanceTrainer: Runtime validation produced non-finite values.' );

			}

			for ( let i = 0; i < 3; i ++ ) {

				const pred = Math.max( prediction[ i ], 1e-6 );
				const ref = Math.max( target[ i ], 1e-6 );
				const diff = powerLog( pred, 3 ) - powerLog( ref, 3 );
				brdfLoss += Math.abs( diff ) * sampleWeight * invBatch / 3;

			}

		}

		if ( outputs.emission && sample.emissionTarget ) {

			emissionCount ++;

			for ( let i = 0; i < 3; i ++ ) {

				const pred = Math.max( outputs.emission[ i ], 1e-6 );
				const ref = Math.max( sample.emissionTarget[ i ], 1e-6 );
				emissionLoss += Math.abs( powerLog( pred, 3 ) - powerLog( ref, 3 ) ) * invAuxBatch / 3;

			}

		}

		if ( Number.isFinite( outputs.opacity ) && Number.isFinite( sample.opacityTarget ) ) {

			opacityCount ++;
			opacityLoss += Math.abs( outputs.opacity - sample.opacityTarget ) * invAuxBatch;

		}

		if ( outputs.ibl ) {

			accumulateFramePriorMetric( framePriors, outputs.ibl );
			const predictedWhite = evaluateNeuralIBLWhiteFurnace( json, sample );
			const predictedPrefiltered = evaluateNeuralPrefilteredIBL( json, {
				...sample,
				iblIncoming: [ 1, 1, 1 ],
				iblIrradiance: [ 1, 1, 1 ]
			} );
			const integratedWhite = sample.directionalAlbedo || integrateNeuralBRDFWhiteFurnace( json, sample, 32 );
			accumulateDifferenceMetric( whiteFurnace, predictedWhite, integratedWhite );
			accumulateDifferenceMetric( prefilteredIBL, predictedPrefiltered, integratedWhite );

			if ( Array.isArray( sample.iblDirection ) && Number.isFinite( sample.iblRoughness ) ) {

				const directionError = Math.hypot(
					outputs.ibl.direction[ 0 ] - sample.iblDirection[ 0 ],
					outputs.ibl.direction[ 1 ] - sample.iblDirection[ 1 ],
					outputs.ibl.direction[ 2 ] - sample.iblDirection[ 2 ]
				);
				const queryLoss = ( directionError + Math.abs( outputs.ibl.roughness - sample.iblRoughness ) ) / 2;
				iblQueryLoss += queryLoss * invAuxBatch;
				iblLoss += queryLoss * invAuxBatch;
				iblQueryCount ++;
				iblCount ++;

			}

			if ( Array.isArray( sample.iblIndirectRadiance ) || Array.isArray( sample.iblIndirectIrradiance ) ) {

				const predicted = evaluateNeuralAppearanceOutputs( json, sample );
				const targetRadiance = sample.iblIndirectRadiance || [ 0, 0, 0 ];
				const targetIrradiance = sample.iblIndirectIrradiance || [ 0, 0, 0 ];
				const predictedRadiance = predicted.indirectRadiance || [ 0, 0, 0 ];
				const predictedIrradiance = predicted.indirectIrradiance || [ 0, 0, 0 ];

				for ( let i = 0; i < 3; i ++ ) {

					const radianceLoss = Math.abs( predictedRadiance[ i ] - targetRadiance[ i ] ) * invAuxBatch / 3;
					const irradianceLoss = Math.abs( predictedIrradiance[ i ] - targetIrradiance[ i ] ) * invAuxBatch / 3;
					iblIndirectLoss += radianceLoss + irradianceLoss;
					iblLoss += radianceLoss + irradianceLoss;

				}

				iblIndirectCount ++;
				iblCount ++;

			}

		}

		if ( nextMipPrediction ) accumulateDifferenceMetric( mipConsistency, prediction, nextMipPrediction );

		accumulateDifferenceMetric( reciprocity, prediction, reciprocalPrediction );
		accumulateDifferenceMetric( angularSmoothness, prediction, perturbedWiPrediction );
		accumulateDifferenceMetric( angularSmoothness, prediction, perturbedWoPrediction );

		if ( Array.isArray( sample.directTarget ) && sample.wi[ 2 ] >= 0 && sample.wo[ 2 ] >= 0 ) {

			accumulateAngularError( angularBins.wi, sample.wi[ 2 ], directPrediction, sample.directTarget );
			accumulateAngularError( angularBins.wo, sample.wo[ 2 ], directPrediction, sample.directTarget );

		}

		appendRuntimePreviewSample( preview, sample, directTarget, directPrediction, sampleWeight );

	}

	const auxiliaryLoss = emissionLoss + opacityLoss + iblLoss;

	return {
		loss: brdfLoss + auxiliaryLoss,
		directLoss: brdfLoss,
		brdfLoss,
		emissionLoss: emissionCount > 0 ? emissionLoss : null,
		opacityLoss: opacityCount > 0 ? opacityLoss : null,
		iblLoss: iblCount > 0 ? iblLoss : null,
		iblQueryLoss: iblQueryCount > 0 ? iblQueryLoss : null,
		iblIndirectLoss: iblIndirectCount > 0 ? iblIndirectLoss : null,
		sampleCount: samples.length,
		mipLevels: json.latents.textures[ 0 ].mipmaps.length,
		preview,
		angularBins: {
			wi: finalizeAngularBins( angularBins.wi ),
			wo: finalizeAngularBins( angularBins.wo )
		},
		reciprocity: finalizeDifferenceMetric( reciprocity ),
		angularSmoothness: finalizeDifferenceMetric( angularSmoothness ),
		whiteFurnace: finalizeDifferenceMetric( whiteFurnace ),
		prefilteredIBL: finalizeDifferenceMetric( prefilteredIBL ),
		mipConsistency: finalizeDifferenceMetric( mipConsistency ),
		framePriors: finalizeFramePriorMetric( framePriors )
	};

}

function evaluateNextMipPrediction( json, sample ) {

	const mipmaps = json.latents && json.latents.textures && json.latents.textures[ 0 ] ? json.latents.textures[ 0 ].mipmaps : null;
	if ( ! mipmaps || mipmaps.length < 2 ) return null;

	const mip = Math.min( Math.max( sample.mip || 0, 0 ), mipmaps.length - 1 );
	if ( mip >= mipmaps.length - 1 ) return null;

	return evaluateNeuralAppearanceJson( json, { ...sample, mip: mip + 1, duvDx: null, duvDy: null } );

}

function createFramePriorMetric() {

	return { sampleCount: 0, normalLengthError: 0, roughnessOutOfRange: 0, maxNormalTilt: 0 };

}

function accumulateFramePriorMetric( metric, query ) {

	const normal = query.direction;
	const roughness = query.roughness;
	if ( ! normal || normal.every( Number.isFinite ) === false || Number.isFinite( roughness ) === false ) return;

	const normalLength = Math.hypot( normal[ 0 ], normal[ 1 ], normal[ 2 ] );
	const normalTilt = Math.acos( Math.min( Math.max( Math.abs( normal[ 2 ] ) / Math.max( normalLength, 1e-12 ), 0 ), 1 ) );

	metric.sampleCount ++;
	metric.normalLengthError += Math.abs( normalLength - 1 );
	metric.roughnessOutOfRange += roughness < 0 || roughness > 1 ? 1 : 0;
	metric.maxNormalTilt = Math.max( metric.maxNormalTilt, normalTilt );

}

function finalizeFramePriorMetric( metric ) {

	return {
		sampleCount: metric.sampleCount,
		meanNormalLengthError: metric.sampleCount > 0 ? metric.normalLengthError / metric.sampleCount : null,
		roughnessOutOfRange: metric.roughnessOutOfRange,
		maxNormalTilt: metric.sampleCount > 0 ? metric.maxNormalTilt : null
	};

}

function createDifferenceMetric() {

	return { sampleCount: 0, absoluteDifference: 0, magnitude: 0, maxChannelDifference: 0 };

}

function accumulateDifferenceMetric( metric, a, b ) {

	if ( a.every( Number.isFinite ) === false || b.every( Number.isFinite ) === false ) return;
	metric.sampleCount ++;

	for ( let channel = 0; channel < 3; channel ++ ) {

		const difference = Math.abs( a[ channel ] - b[ channel ] );
		metric.absoluteDifference += difference;
		metric.magnitude += 0.5 * ( Math.abs( a[ channel ] ) + Math.abs( b[ channel ] ) );
		metric.maxChannelDifference = Math.max( metric.maxChannelDifference, difference );

	}

}

function finalizeDifferenceMetric( metric ) {

	return {
		sampleCount: metric.sampleCount,
		meanAbsoluteDifference: metric.sampleCount > 0 ? metric.absoluteDifference / ( metric.sampleCount * 3 ) : null,
		relativeAbsoluteDifference: metric.magnitude > 0 ? metric.absoluteDifference / metric.magnitude : null,
		maxChannelDifference: metric.sampleCount > 0 ? metric.maxChannelDifference : null
	};

}

function perturbDirection( direction, amount ) {

	const reference = Math.abs( direction[ 2 ] ) < 0.9 ? [ 0, 0, 1 ] : [ 1, 0, 0 ];
	const tangent = normalize( cross( direction, reference ) );

	return normalize( [
		direction[ 0 ] + tangent[ 0 ] * amount,
		direction[ 1 ] + tangent[ 1 ] * amount,
		Math.max( direction[ 2 ] + tangent[ 2 ] * amount, 0 )
	] );

}

function createRuntimePreview( maxSampleCount, sourceSampleCount ) {

	const sampleCount = Math.min( Math.max( 0, Math.floor( maxSampleCount ) ), sourceSampleCount );
	const width = sampleCount > 0 ? Math.ceil( Math.sqrt( sampleCount ) ) : 0;

	return {
		width,
		height: width > 0 ? Math.ceil( sampleCount / width ) : 0,
		samples: []
	};

}

function appendRuntimePreviewSample( preview, sample, targetRgb, predictionRgb, weight ) {

	if ( preview.samples.length >= preview.width * preview.height ) return;

	preview.samples.push( {
		uv: sample.uv.slice( 0, 2 ),
		wi: sample.wi.slice( 0, 3 ),
		wo: sample.wo.slice( 0, 3 ),
		mip: sample.mip || 0,
		weight,
		targetRgb: sanitizeRgb( targetRgb ),
		predictionRgb: sanitizeRgb( predictionRgb )
	} );

}

function sanitizeRgb( rgb ) {

	return [
		sanitizeChannel( rgb[ 0 ] ),
		sanitizeChannel( rgb[ 1 ] ),
		sanitizeChannel( rgb[ 2 ] )
	];

}

function sanitizeChannel( value ) {

	return Number.isFinite( value ) ? Math.max( value, 0 ) : 0;

}

function createAngularBins() {

	return [
		{ min: 0, max: 0.05, count: 0, absoluteError: 0, targetMagnitude: 0, maxChannelError: 0 },
		{ min: 0.05, max: 0.2, count: 0, absoluteError: 0, targetMagnitude: 0, maxChannelError: 0 },
		{ min: 0.2, max: 1, count: 0, absoluteError: 0, targetMagnitude: 0, maxChannelError: 0 }
	];

}

function accumulateAngularError( bins, cosine, prediction, target ) {

	const bin = bins.find( ( candidate, index ) => cosine >= candidate.min && ( cosine < candidate.max || index === bins.length - 1 ) );
	if ( bin === undefined || target.every( Number.isFinite ) === false ) return;

	bin.count ++;

	for ( let channel = 0; channel < 3; channel ++ ) {

		const error = Math.abs( prediction[ channel ] - target[ channel ] );
		bin.absoluteError += error;
		bin.targetMagnitude += Math.abs( target[ channel ] );
		bin.maxChannelError = Math.max( bin.maxChannelError, error );

	}

}

function finalizeAngularBins( bins ) {

	return bins.map( ( bin ) => ( {
		min: bin.min,
		max: bin.max,
		count: bin.count,
		meanAbsoluteError: bin.count > 0 ? bin.absoluteError / ( bin.count * 3 ) : null,
		relativeAbsoluteError: bin.targetMagnitude > 0 ? bin.absoluteError / bin.targetMagnitude : null,
		maxChannelError: bin.count > 0 ? bin.maxChannelError : null
	} ) );

}

export {
	evaluateRuntimeValidation,
	createDifferenceMetric,
	accumulateDifferenceMetric,
	finalizeDifferenceMetric,
	createAngularBins,
	accumulateAngularError,
	finalizeAngularBins
};
