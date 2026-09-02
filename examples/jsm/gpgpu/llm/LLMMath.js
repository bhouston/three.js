function linear( input, weight, bias, inputSize, outputSize, target = new Float32Array( outputSize ) ) {

	for ( let output = 0; output < outputSize; output ++ ) {

		let sum = bias !== null ? bias[ output ] : 0;

		for ( let i = 0; i < inputSize; i ++ ) {

			sum += input[ i ] * weight[ i * outputSize + output ];

		}

		target[ output ] = sum;

	}

	return target;

}

function layerNorm( input, weight, bias, epsilon = 1e-5, target = new Float32Array( input.length ) ) {

	let mean = 0;

	for ( let i = 0; i < input.length; i ++ ) mean += input[ i ];

	mean /= input.length;

	let variance = 0;

	for ( let i = 0; i < input.length; i ++ ) {

		const d = input[ i ] - mean;
		variance += d * d;

	}

	const invStd = 1 / Math.sqrt( variance / input.length + epsilon );

	for ( let i = 0; i < input.length; i ++ ) {

		target[ i ] = ( input[ i ] - mean ) * invStd * weight[ i ] + bias[ i ];

	}

	return target;

}

function geluNew( x ) {

	return 0.5 * x * ( 1 + Math.tanh( Math.sqrt( 2 / Math.PI ) * ( x + 0.044715 * x * x * x ) ) );

}

function softmax( values, target = new Float32Array( values.length ) ) {

	let maxValue = - Infinity;

	for ( let i = 0; i < values.length; i ++ ) maxValue = Math.max( maxValue, values[ i ] );

	let sum = 0;

	for ( let i = 0; i < values.length; i ++ ) {

		const value = Math.exp( values[ i ] - maxValue );
		target[ i ] = value;
		sum += value;

	}

	for ( let i = 0; i < target.length; i ++ ) target[ i ] /= sum;

	return target;

}

function sampleTopK( logits, { temperature = 0.8, topK = 40, random = Math.random } = {} ) {

	const k = Math.min( topK, logits.length );
	const candidates = [];

	if ( temperature <= 0 || k === 1 ) {

		let bestIndex = 0;
		let bestValue = logits[ 0 ];

		for ( let i = 1; i < logits.length; i ++ ) {

			if ( logits[ i ] > bestValue ) {

				bestIndex = i;
				bestValue = logits[ i ];

			}

		}

		return bestIndex;

	}

	for ( let i = 0; i < logits.length; i ++ ) {

		const value = logits[ i ] / Math.max( temperature, 1e-6 );

		if ( candidates.length < k ) {

			candidates.push( [ i, value ] );
			candidates.sort( ( a, b ) => b[ 1 ] - a[ 1 ] );

		} else if ( value > candidates[ k - 1 ][ 1 ] ) {

			candidates[ k - 1 ] = [ i, value ];
			candidates.sort( ( a, b ) => b[ 1 ] - a[ 1 ] );

		}

	}

	const values = new Float32Array( candidates.length );

	for ( let i = 0; i < candidates.length; i ++ ) values[ i ] = candidates[ i ][ 1 ];

	const probabilities = softmax( values );
	let r = random();

	for ( let i = 0; i < probabilities.length; i ++ ) {

		r -= probabilities[ i ];
		if ( r <= 0 ) return candidates[ i ][ 0 ];

	}

	return candidates[ candidates.length - 1 ][ 0 ];

}

export { geluNew, layerNorm, linear, sampleTopK, softmax };
