const LINEAR_SCALAR_OUTPUT_HE_SCALE = 0.2;
const LINEAR_RGB_OUTPUT_HE_SCALE = 0.45;
const LINEAR_RGB_OUTPUT_BIAS = 0.3;

function createMLP( inputSize, hiddenLayers, outputSize, random, hiddenActivation = 'relu', outputActivation = 'linear' ) {

	const sizes = [ inputSize, ...hiddenLayers, outputSize ];
	const layers = [];

	for ( let i = 0; i < sizes.length - 1; i ++ ) {

		const input = sizes[ i ];
		const output = sizes[ i + 1 ];
		const isOutputLayer = i === sizes.length - 2;
		const activation = isOutputLayer ? outputActivation : hiddenActivation;
		const isLinearOutput = isOutputLayer && activation === 'linear';
		const isLinearRgb = isLinearOutput && output === 3;
		const scale = Math.sqrt( 2 / input ) * ( isLinearRgb ? LINEAR_RGB_OUTPUT_HE_SCALE : ( isLinearOutput ? LINEAR_SCALAR_OUTPUT_HE_SCALE : 1 ) );
		const weights = new Array( input * output );
		const biases = new Array( output ).fill( isLinearRgb ? LINEAR_RGB_OUTPUT_BIAS : 0 );

		for ( let j = 0; j < weights.length; j ++ ) {

			weights[ j ] = ( random() * 2 - 1 ) * scale;

		}

		layers.push( {
			inputSize: input,
			outputSize: output,
			weights,
			biases,
			activation
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

function activate( value, activation ) {

	if ( activation === 'relu' ) return Math.max( 0, value );
	if ( activation === 'leakyRelu' ) return value >= 0 ? value : value * 0.01;
	if ( activation === 'tanh' ) return Math.tanh( value );

	return value;

}

function sigmoid( value ) {

	return 1 / ( 1 + Math.exp( - value ) );

}

function powerLog( value, power ) {

	return power * ( Math.pow( value, 1 / power ) - 1 );

}

export {
	createMLP,
	forwardMLP,
	activate,
	sigmoid,
	powerLog
};
