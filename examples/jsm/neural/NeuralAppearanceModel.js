import {
	LATENT_CHANNELS,
	DECODER_INPUT_SIZE
} from './NeuralAppearanceFormat.js';
import {
	createMLP,
	forwardMLP,
	backwardMLP,
	zeroGradients,
	applyAdam,
	sigmoid,
	binaryCrossEntropy,
	powerLog
} from './NeuralAppearanceMLP.js';

const OUTPUT_CLAMP_GRADIENT_LEAK = 0.01;

function createModel( options, random ) {

	const decoder = createMLP( DECODER_INPUT_SIZE, [ options.hiddenSize, options.hiddenSize ], 3, random, 'relu', 'linear' );
	const emissionHead = options.outputFeatures && options.outputFeatures.emission ?
		createMLP( LATENT_CHANNELS, [], 3, random, 'relu', 'linear' ) :
		null;
	const opacityHead = options.outputFeatures && options.outputFeatures.opacity ?
		createMLP( LATENT_CHANNELS, [], 1, random, 'relu', 'linear' ) :
		null;
	const rotationWeights = new Array( LATENT_CHANNELS * 12 ).fill( 0 );
	const latentGrids = createLatentMipGrids( options.resolution, options.resolution, random );

	return {
		decoder,
		emissionHead,
		opacityHead,
		rotationWeights,
		rotationGrad: new Array( rotationWeights.length ).fill( 0 ),
		rotationM: new Array( rotationWeights.length ).fill( 0 ),
		rotationV: new Array( rotationWeights.length ).fill( 0 ),
		latentGrid: latentGrids[ 0 ],
		latentGrids
	};

}

function createLatentMipGrids( baseWidth, baseHeight, random ) {

	const grids = [];
	let width = baseWidth;
	let height = baseHeight;

	while ( true ) {

		grids.push( createLatentGrid( width, height, random ) );

		if ( width === 1 && height === 1 ) break;
		width = Math.max( 1, width >> 1 );
		height = Math.max( 1, height >> 1 );

	}

	return grids;

}

function createLatentGrid( width, height, random ) {

	const data = new Array( width * height * LATENT_CHANNELS );

	for ( let i = 0; i < data.length; i ++ ) {

		data[ i ] = ( random() * 2 - 1 ) * 0.25;

	}

	return {
		width,
		height,
		data,
		grad: new Array( data.length ).fill( 0 ),
		m: new Array( data.length ).fill( 0 ),
		v: new Array( data.length ).fill( 0 )
	};

}

function sampleLatents( grid, uv ) {

	const x = uv[ 0 ] * grid.width - 0.5;
	const y = uv[ 1 ] * grid.height - 0.5;
	const x0 = Math.floor( x );
	const y0 = Math.floor( y );
	const tx = x - x0;
	const ty = y - y0;
	const taps = [
		{ x: wrapIndex( x0, grid.width ), y: wrapIndex( y0, grid.height ), weight: ( 1 - tx ) * ( 1 - ty ) },
		{ x: wrapIndex( x0 + 1, grid.width ), y: wrapIndex( y0, grid.height ), weight: tx * ( 1 - ty ) },
		{ x: wrapIndex( x0, grid.width ), y: wrapIndex( y0 + 1, grid.height ), weight: ( 1 - tx ) * ty },
		{ x: wrapIndex( x0 + 1, grid.width ), y: wrapIndex( y0 + 1, grid.height ), weight: tx * ty }
	];
	const output = new Array( LATENT_CHANNELS ).fill( 0 );

	for ( const tap of taps ) {

		const offset = ( tap.y * grid.width + tap.x ) * LATENT_CHANNELS;

		for ( let channel = 0; channel < LATENT_CHANNELS; channel ++ ) {

			output[ channel ] += grid.data[ offset + channel ] * tap.weight;

		}

	}

	return { output, taps };

}

function scatterLatentGradients( grid, latentRun, gradLatents ) {

	for ( const tap of latentRun.taps ) {

		const offset = ( tap.y * grid.width + tap.x ) * LATENT_CHANNELS;

		for ( let channel = 0; channel < LATENT_CHANNELS; channel ++ ) {

			grid.grad[ offset + channel ] += gradLatents[ channel ] * tap.weight;

		}

	}

}

function zeroLatentGradients( grid ) {

	grid.grad.fill( 0 );

}

function applyAdamLatents( grid, learningRate, step ) {

	const beta1 = 0.9;
	const beta2 = 0.999;
	const epsilon = 1e-7;
	const beta1Correction = 1 - Math.pow( beta1, step );
	const beta2Correction = 1 - Math.pow( beta2, step );

	for ( let i = 0; i < grid.data.length; i ++ ) {

		const grad = grid.grad[ i ];
		grid.m[ i ] = beta1 * grid.m[ i ] + ( 1 - beta1 ) * grad;
		grid.v[ i ] = beta2 * grid.v[ i ] + ( 1 - beta2 ) * grad * grad;
		grid.data[ i ] -= learningRate * ( grid.m[ i ] / beta1Correction ) / ( Math.sqrt( grid.v[ i ] / beta2Correction ) + epsilon );

	}

}

function applyAdamRotation( model, learningRate, step ) {

	const beta1 = 0.9;
	const beta2 = 0.999;
	const epsilon = 1e-7;
	const beta1Correction = 1 - Math.pow( beta1, step );
	const beta2Correction = 1 - Math.pow( beta2, step );

	for ( let i = 0; i < model.rotationWeights.length; i ++ ) {

		const grad = model.rotationGrad[ i ];
		model.rotationM[ i ] = beta1 * model.rotationM[ i ] + ( 1 - beta1 ) * grad;
		model.rotationV[ i ] = beta2 * model.rotationV[ i ] + ( 1 - beta2 ) * grad * grad;
		model.rotationWeights[ i ] -= learningRate * ( model.rotationM[ i ] / beta1Correction ) / ( Math.sqrt( model.rotationV[ i ] / beta2Correction ) + epsilon );

	}

}

function forwardDecoderInput( latents, rotationWeights, wi, wo ) {

	const output = latents.slice();
	const frames = [];

	for ( let frame = 0; frame < 2; frame ++ ) {

		const offset = frame * 6;
		const rawN = [
			linearRotationValue( latents, rotationWeights, offset ),
			linearRotationValue( latents, rotationWeights, offset + 1 ),
			linearRotationValue( latents, rotationWeights, offset + 2 ) + 1
		];
		const rawT = [
			linearRotationValue( latents, rotationWeights, offset + 3 ) + 1,
			linearRotationValue( latents, rotationWeights, offset + 4 ),
			linearRotationValue( latents, rotationWeights, offset + 5 )
		];
		const n = normalize( rawN );
		const t = normalize( rawT );
		const rawB = cross( n, t );
		const b = normalize( rawB );

		output.push( dot( wi, t ), dot( wi, b ), dot( wi, n ) );
		output.push( dot( wo, t ), dot( wo, b ), dot( wo, n ) );
		frames.push( { offset, rawN, rawT, rawB, n, t, b } );

	}

	return { output, latents, wi, wo, frames };

}

function backwardDecoderInput( run, gradOutput, rotationWeights ) {

	const gradLatents = gradOutput.slice( 0, LATENT_CHANNELS );
	const gradRotationWeights = new Array( rotationWeights.length ).fill( 0 );

	for ( let frame = 0; frame < run.frames.length; frame ++ ) {

		const frameRun = run.frames[ frame ];
		const inputOffset = LATENT_CHANNELS + frame * 6;
		const gradT = addScaledVectors( run.wi, gradOutput[ inputOffset ], run.wo, gradOutput[ inputOffset + 3 ] );
		const gradB = addScaledVectors( run.wi, gradOutput[ inputOffset + 1 ], run.wo, gradOutput[ inputOffset + 4 ] );
		const gradN = addScaledVectors( run.wi, gradOutput[ inputOffset + 2 ], run.wo, gradOutput[ inputOffset + 5 ] );
		const gradRawB = backwardNormalize( frameRun.rawB, frameRun.b, gradB );
		const gradNormalizedN = addVectors( gradN, cross( frameRun.t, gradRawB ) );
		const gradNormalizedT = addVectors( gradT, cross( gradRawB, frameRun.n ) );
		const gradRawN = backwardNormalize( frameRun.rawN, frameRun.n, gradNormalizedN );
		const gradRawT = backwardNormalize( frameRun.rawT, frameRun.t, gradNormalizedT );
		const gradFrame = [ ...gradRawN, ...gradRawT ];

		for ( let outputIndex = 0; outputIndex < 6; outputIndex ++ ) {

			const rotationOutput = frameRun.offset + outputIndex;
			const weightOffset = rotationOutput * LATENT_CHANNELS;

			for ( let latentIndex = 0; latentIndex < LATENT_CHANNELS; latentIndex ++ ) {

				gradRotationWeights[ weightOffset + latentIndex ] += gradFrame[ outputIndex ] * run.latents[ latentIndex ];
				gradLatents[ latentIndex ] += gradFrame[ outputIndex ] * rotationWeights[ weightOffset + latentIndex ];

			}

		}

	}

	return {
		latents: gradLatents,
		rotationWeights: gradRotationWeights
	};

}

function buildDecoderInput( latents, rotationWeights, wi, wo ) {

	return forwardDecoderInput( latents, rotationWeights, wi, wo ).output;

}

function trainBatch( model, samples, teacher, learningRate, step, maxGradientNorm ) {

	const sampleWeightSum = getSampleWeightSum( samples );
	if ( sampleWeightSum <= 0 ) {

		throw new Error( 'THREE.NeuralAppearanceTrainer: Batch contains no finite, well-conditioned teacher samples.' );

	}

	const invBatch = 1 / sampleWeightSum;
	let loss = 0;

	zeroGradients( model.decoder );
	if ( model.emissionHead ) zeroGradients( model.emissionHead );
	if ( model.opacityHead ) zeroGradients( model.opacityHead );
	for ( const grid of model.latentGrids ) zeroLatentGradients( grid );
	model.rotationGrad.fill( 0 );

	for ( const sample of samples ) {

		const latentGrid = model.latentGrids[ sample.mip || 0 ];
		const latentRun = sampleLatents( latentGrid, sample.uv );
		const latents = latentRun.output;
		const inputRun = forwardDecoderInput( latents, model.rotationWeights, sample.wi, sample.wo );
		const decoderRun = forwardMLP( model.decoder, inputRun.output );
		const rawPrediction = decoderRun.output;
		const prediction = rawPrediction.map( ( value ) => Math.max( 0, value ) );
		const target = sample.target;
		const sampleWeight = sample.weight !== undefined ? sample.weight : 1;
		const gradPrediction = [ 0, 0, 0 ];

		if ( sampleWeight === 0 ) continue;
		if ( prediction.every( Number.isFinite ) === false || target.every( Number.isFinite ) === false ) {

			throw new Error( 'THREE.NeuralAppearanceTrainer: Encountered a non-finite training sample.' );

		}

		for ( let i = 0; i < 3; i ++ ) {

			const pred = Math.max( prediction[ i ], 1e-6 );
			const ref = Math.max( target[ i ], 1e-6 );
			const predLog = powerLog( pred, 3 );
			const refLog = powerLog( ref, 3 );
			const diff = predLog - refLog;
			loss += Math.abs( diff ) * sampleWeight * invBatch / 3;
			gradPrediction[ i ] = Math.sign( diff ) * Math.pow( pred, 1 / 3 - 1 ) * sampleWeight * invBatch / 3;

		}

		const gradRawPrediction = applyOutputClampGradient( rawPrediction, gradPrediction );
		const gradDecoderInput = backwardMLP( model.decoder, decoderRun, gradRawPrediction );
		const inputGradients = backwardDecoderInput( inputRun, gradDecoderInput, model.rotationWeights );

		for ( let i = 0; i < model.rotationGrad.length; i ++ ) {

			model.rotationGrad[ i ] += inputGradients.rotationWeights[ i ];

		}

		scatterLatentGradients( latentGrid, latentRun, inputGradients.latents );

		if ( model.emissionHead && sample.emissionTarget ) {

			const emissionRun = forwardMLP( model.emissionHead, latents );
			const rawEmission = emissionRun.output;
			const emission = rawEmission.map( ( value ) => Math.max( 0, value ) );
			const gradEmission = [ 0, 0, 0 ];
			const emissionWeight = invBatch * sampleWeight / 3;

			for ( let i = 0; i < 3; i ++ ) {

				const pred = Math.max( emission[ i ], 1e-6 );
				const ref = Math.max( sample.emissionTarget[ i ], 1e-6 );
				const diff = powerLog( pred, 3 ) - powerLog( ref, 3 );
				loss += Math.abs( diff ) * emissionWeight;
				gradEmission[ i ] = Math.sign( diff ) * Math.pow( pred, 1 / 3 - 1 ) * emissionWeight;

			}

			const gradRawEmission = applyOutputClampGradient( rawEmission, gradEmission );
			scatterLatentGradients( latentGrid, latentRun, backwardMLP( model.emissionHead, emissionRun, gradRawEmission ) );

		}

		if ( model.opacityHead && Number.isFinite( sample.opacityTarget ) ) {

			const opacityRun = forwardMLP( model.opacityHead, latents );
			const rawOpacity = opacityRun.output[ 0 ];
			const opacity = sigmoid( rawOpacity );
			const targetOpacity = Math.min( Math.max( sample.opacityTarget, 0 ), 1 );
			const opacityWeight = invBatch * sampleWeight;

			loss += binaryCrossEntropy( opacity, targetOpacity ) * opacityWeight;
			scatterLatentGradients( latentGrid, latentRun, backwardMLP( model.opacityHead, opacityRun, [ ( opacity - targetOpacity ) * opacityWeight ] ) );

		}

	}

	clipModelGradients( model, maxGradientNorm );
	for ( const grid of model.latentGrids ) applyAdamLatents( grid, learningRate, step );
	applyAdamRotation( model, learningRate, step );
	applyAdam( model.decoder, learningRate, step );
	if ( model.emissionHead ) applyAdam( model.emissionHead, learningRate, step );
	if ( model.opacityHead ) applyAdam( model.opacityHead, learningRate, step );
	assertModelFinite( model );

	if ( Number.isFinite( loss ) === false ) {

		throw new Error( 'THREE.NeuralAppearanceTrainer: Training produced a non-finite loss.' );

	}

	return loss;

}

function applyOutputClampGradient( rawOutput, gradOutput ) {

	return gradOutput.map( ( gradient, channel ) => rawOutput[ channel ] > 0 ? gradient : gradient * OUTPUT_CLAMP_GRADIENT_LEAK );

}

function clipModelGradients( model, maxNorm ) {

	if ( ! Number.isFinite( maxNorm ) || maxNorm <= 0 ) return;

	const gradients = [ model.rotationGrad ];

	for ( const grid of model.latentGrids ) gradients.push( grid.grad );
	for ( const layer of model.decoder.layers ) gradients.push( layer.gradWeights, layer.gradBiases );
	if ( model.emissionHead ) {

		for ( const layer of model.emissionHead.layers ) gradients.push( layer.gradWeights, layer.gradBiases );

	}

	if ( model.opacityHead ) {

		for ( const layer of model.opacityHead.layers ) gradients.push( layer.gradWeights, layer.gradBiases );

	}

	let squaredNorm = 0;

	for ( const values of gradients ) {

		for ( const value of values ) squaredNorm += value * value;

	}

	const norm = Math.sqrt( squaredNorm );
	if ( ! Number.isFinite( norm ) ) {

		throw new Error( 'THREE.NeuralAppearanceTrainer: Training produced non-finite gradients.' );

	}

	if ( norm <= maxNorm ) return;

	const scale = maxNorm / norm;

	for ( const values of gradients ) {

		for ( let i = 0; i < values.length; i ++ ) values[ i ] *= scale;

	}

}

function assertModelFinite( model ) {

	assertFiniteArray( model.rotationWeights, 'learned-frame weights' );

	for ( const grid of model.latentGrids ) assertFiniteArray( grid.data, 'latent values' );

	for ( const layer of model.decoder.layers ) {

		assertFiniteArray( layer.weights, 'decoder weights' );
		assertFiniteArray( layer.biases, 'decoder biases' );

	}

	for ( const [ label, head ] of [[ 'emission', model.emissionHead ], [ 'opacity', model.opacityHead ]] ) {

		if ( ! head ) continue;

		for ( const layer of head.layers ) {

			assertFiniteArray( layer.weights, `${ label } weights` );
			assertFiniteArray( layer.biases, `${ label } biases` );

		}

	}

}

function assertFiniteArray( values, label ) {

	if ( values.every( Number.isFinite ) === false ) {

		throw new Error( `THREE.NeuralAppearanceTrainer: Training produced non-finite ${ label }.` );

	}

}

function getSampleWeightSum( samples ) {

	let weight = 0;

	for ( const sample of samples ) {

		weight += sample.weight !== undefined ? sample.weight : 1;

	}

	return weight;

}

function backwardNormalize( raw, normalized, gradNormalized ) {

	const inverseLength = 1 / ( Math.hypot( raw[ 0 ], raw[ 1 ], raw[ 2 ] ) || 1 );
	const projectedGradient = dot( normalized, gradNormalized );

	return [
		( gradNormalized[ 0 ] - normalized[ 0 ] * projectedGradient ) * inverseLength,
		( gradNormalized[ 1 ] - normalized[ 1 ] * projectedGradient ) * inverseLength,
		( gradNormalized[ 2 ] - normalized[ 2 ] * projectedGradient ) * inverseLength
	];

}

function addVectors( a, b ) {

	return [ a[ 0 ] + b[ 0 ], a[ 1 ] + b[ 1 ], a[ 2 ] + b[ 2 ] ];

}

function addScaledVectors( a, scaleA, b, scaleB ) {

	return [
		a[ 0 ] * scaleA + b[ 0 ] * scaleB,
		a[ 1 ] * scaleA + b[ 1 ] * scaleB,
		a[ 2 ] * scaleA + b[ 2 ] * scaleB
	];

}

function linearRotationValue( latents, weights, outputIndex ) {

	let value = 0;

	for ( let i = 0; i < latents.length; i ++ ) {

		value += weights[ outputIndex * LATENT_CHANNELS + i ] * latents[ i ];

	}

	return value;

}

function dot( a, b ) {

	return a[ 0 ] * b[ 0 ] + a[ 1 ] * b[ 1 ] + a[ 2 ] * b[ 2 ];

}

function cross( a, b ) {

	return [
		a[ 1 ] * b[ 2 ] - a[ 2 ] * b[ 1 ],
		a[ 2 ] * b[ 0 ] - a[ 0 ] * b[ 2 ],
		a[ 0 ] * b[ 1 ] - a[ 1 ] * b[ 0 ]
	];

}

function normalize( value ) {

	const length = Math.hypot( value[ 0 ], value[ 1 ], value[ 2 ] ) || 1;

	return [ value[ 0 ] / length, value[ 1 ] / length, value[ 2 ] / length ];

}

function wrapIndex( value, size ) {

	return ( ( value % size ) + size ) % size;

}

export {
	createModel,
	createLatentMipGrids,
	createLatentGrid,
	sampleLatents,
	scatterLatentGradients,
	zeroLatentGradients,
	applyAdamLatents,
	applyAdamRotation,
	forwardDecoderInput,
	backwardDecoderInput,
	buildDecoderInput,
	trainBatch,
	applyOutputClampGradient,
	clipModelGradients,
	assertModelFinite,
	assertFiniteArray,
	getSampleWeightSum,
	dot,
	cross,
	normalize,
	wrapIndex
};
