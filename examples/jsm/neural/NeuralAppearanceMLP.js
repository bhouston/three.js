function createMLP( inputSize, hiddenLayers, outputSize, random, hiddenActivation = 'relu', outputActivation = 'linear' ) {

	const sizes = [ inputSize, ...hiddenLayers, outputSize ];
	const layers = [];

	for ( let i = 0; i < sizes.length - 1; i ++ ) {

		const input = sizes[ i ];
		const output = sizes[ i + 1 ];
		const scale = Math.sqrt( 2 / input );
		const weights = new Array( input * output );
		const biases = new Array( output ).fill( 0 );

		for ( let j = 0; j < weights.length; j ++ ) {

			weights[ j ] = ( random() * 2 - 1 ) * scale;

		}

		layers.push( {
			inputSize: input,
			outputSize: output,
			weights,
			biases,
			activation: i === sizes.length - 2 ? outputActivation : hiddenActivation,
			gradWeights: new Array( weights.length ).fill( 0 ),
			gradBiases: new Array( biases.length ).fill( 0 ),
			mWeights: new Array( weights.length ).fill( 0 ),
			vWeights: new Array( weights.length ).fill( 0 ),
			mBiases: new Array( biases.length ).fill( 0 ),
			vBiases: new Array( biases.length ).fill( 0 )
		} );

	}

	return { layers };

}

function forwardMLP( mlp, input ) {

	const activations = [ input.slice() ];
	const preActivations = [];
	let values = input.slice();

	for ( const layer of mlp.layers ) {

		const next = new Array( layer.outputSize );
		const pre = new Array( layer.outputSize );

		for ( let output = 0; output < layer.outputSize; output ++ ) {

			let value = layer.biases[ output ];

			for ( let inputIndex = 0; inputIndex < layer.inputSize; inputIndex ++ ) {

				value += layer.weights[ output * layer.inputSize + inputIndex ] * values[ inputIndex ];

			}

			pre[ output ] = value;
			next[ output ] = activate( value, layer.activation );

		}

		preActivations.push( pre );
		activations.push( next );
		values = next;

	}

	return { activations, preActivations, output: values };

}

function backwardMLP( mlp, run, gradOutput ) {

	let grad = gradOutput.slice();

	for ( let layerIndex = mlp.layers.length - 1; layerIndex >= 0; layerIndex -- ) {

		const layer = mlp.layers[ layerIndex ];
		const input = run.activations[ layerIndex ];
		const pre = run.preActivations[ layerIndex ];
		const gradInput = new Array( layer.inputSize ).fill( 0 );

		if ( layer.gradWeights === undefined ) {

			layer.gradWeights = new Array( layer.weights.length ).fill( 0 );
			layer.gradBiases = new Array( layer.biases.length ).fill( 0 );

		}

		for ( let output = 0; output < layer.outputSize; output ++ ) {

			const delta = grad[ output ] * activateDerivative( pre[ output ], layer.activation );
			layer.gradBiases[ output ] += delta;

			for ( let inputIndex = 0; inputIndex < layer.inputSize; inputIndex ++ ) {

				const weightIndex = output * layer.inputSize + inputIndex;
				layer.gradWeights[ weightIndex ] += delta * input[ inputIndex ];
				gradInput[ inputIndex ] += delta * layer.weights[ weightIndex ];

			}

		}

		grad = gradInput;

	}

	return grad;

}

function zeroGradients( mlp ) {

	for ( const layer of mlp.layers ) {

		layer.gradWeights = new Array( layer.weights.length ).fill( 0 );
		layer.gradBiases = new Array( layer.biases.length ).fill( 0 );

	}

}

function applyAdam( mlp, learningRate, step ) {

	const beta1 = 0.9;
	const beta2 = 0.999;
	const epsilon = 1e-7;
	const beta1Correction = 1 - Math.pow( beta1, step );
	const beta2Correction = 1 - Math.pow( beta2, step );

	for ( const layer of mlp.layers ) {

		for ( let i = 0; i < layer.weights.length; i ++ ) {

			const grad = layer.gradWeights[ i ];
			layer.mWeights[ i ] = beta1 * layer.mWeights[ i ] + ( 1 - beta1 ) * grad;
			layer.vWeights[ i ] = beta2 * layer.vWeights[ i ] + ( 1 - beta2 ) * grad * grad;
			layer.weights[ i ] -= learningRate * ( layer.mWeights[ i ] / beta1Correction ) / ( Math.sqrt( layer.vWeights[ i ] / beta2Correction ) + epsilon );

		}

		for ( let i = 0; i < layer.biases.length; i ++ ) {

			const grad = layer.gradBiases[ i ];
			layer.mBiases[ i ] = beta1 * layer.mBiases[ i ] + ( 1 - beta1 ) * grad;
			layer.vBiases[ i ] = beta2 * layer.vBiases[ i ] + ( 1 - beta2 ) * grad * grad;
			layer.biases[ i ] -= learningRate * ( layer.mBiases[ i ] / beta1Correction ) / ( Math.sqrt( layer.vBiases[ i ] / beta2Correction ) + epsilon );

		}

	}

}

function activate( value, activation ) {

	if ( activation === 'relu' ) return Math.max( 0, value );
	if ( activation === 'leakyRelu' ) return value >= 0 ? value : value * 0.01;
	if ( activation === 'tanh' ) return Math.tanh( value );

	return value;

}

function activateDerivative( value, activation ) {

	if ( activation === 'relu' ) return value > 0 ? 1 : 0;
	if ( activation === 'leakyRelu' ) return value >= 0 ? 1 : 0.01;
	if ( activation === 'tanh' ) {

		const tanh = Math.tanh( value );
		return 1 - tanh * tanh;

	}

	return 1;

}

function sigmoid( value ) {

	return 1 / ( 1 + Math.exp( - value ) );

}

function binaryCrossEntropy( prediction, target ) {

	const clampedPrediction = Math.min( Math.max( prediction, 1e-6 ), 1 - 1e-6 );
	return - target * Math.log( clampedPrediction ) - ( 1 - target ) * Math.log( 1 - clampedPrediction );

}

function powerLog( value, power ) {

	return power * ( Math.pow( value, 1 / power ) - 1 );

}

export {
	createMLP,
	forwardMLP,
	backwardMLP,
	zeroGradients,
	applyAdam,
	activate,
	activateDerivative,
	sigmoid,
	binaryCrossEntropy,
	powerLog
};
