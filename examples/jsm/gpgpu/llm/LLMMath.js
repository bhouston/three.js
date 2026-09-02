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

function rmsNorm( input, weight, epsilon = 1e-5, offsetWeight = false, target = new Float32Array( input.length ) ) {

	let sumSquares = 0;

	for ( let i = 0; i < input.length; i ++ ) sumSquares += input[ i ] * input[ i ];

	const invRms = 1 / Math.sqrt( sumSquares / input.length + epsilon );

	for ( let i = 0; i < input.length; i ++ ) {

		const scale = offsetWeight ? 1 + weight[ i ] : weight[ i ];
		target[ i ] = input[ i ] * invRms * scale;

	}

	return target;

}

function silu( x ) {

	return x / ( 1 + Math.exp( - x ) );

}

function geluPytorchTanh( x ) {

	return geluNew( x );

}

function rotaryAngle( position, freqIndex, rotaryDim, theta ) {

	return position * Math.pow( theta, - 2 * freqIndex / rotaryDim );

}

function applyRoPE( vector, headOffset, rotaryDim, position, theta ) {

	if ( rotaryDim <= 0 ) return vector;

	const half = rotaryDim / 2;
	const rotated = new Float32Array( rotaryDim );

	for ( let i = 0; i < rotaryDim; i ++ ) {

		const x = vector[ headOffset + i ];
		const partner = i < half ? - vector[ headOffset + i + half ] : vector[ headOffset + i - half ];
		const freqIndex = i < half ? i : i - half;
		const angle = rotaryAngle( position, freqIndex, rotaryDim, theta );

		rotated[ i ] = x * Math.cos( angle ) + partner * Math.sin( angle );

	}

	for ( let i = 0; i < rotaryDim; i ++ ) vector[ headOffset + i ] = rotated[ i ];

	return vector;

}

function rmsNormPackedHeads( vector, headCount, headDim, weight, epsilon, offsetWeight ) {

	for ( let head = 0; head < headCount; head ++ ) {

		const offset = head * headDim;
		const slice = vector.subarray( offset, offset + headDim );
		slice.set( rmsNorm( slice, weight, epsilon, offsetWeight ) );

	}

}

function causalAttention( qkv, options ) {

	const {
		headCount,
		position,
		keyCache,
		valueCache,
		headDim = options.hiddenSize / headCount,
		kvHeadCount = headCount,
		ropeTheta = 0,
		rotaryDim = headDim,
		slidingWindow = 0,
		attnScale = 1 / Math.sqrt( headDim ),
		qNormWeight = null,
		kNormWeight = null,
		rmsEpsilon = 1e-6,
		offsetRMSNorm = false
	} = options;
	const qSize = headCount * headDim;
	const kvSize = kvHeadCount * headDim;
	const query = qkv.slice( 0, qSize );
	const key = qkv.slice( qSize, qSize + kvSize );
	const firstToken = slidingWindow > 0 ? Math.max( 0, position - slidingWindow + 1 ) : 0;

	if ( qNormWeight !== null ) rmsNormPackedHeads( query, headCount, headDim, qNormWeight, rmsEpsilon, offsetRMSNorm );
	if ( kNormWeight !== null ) rmsNormPackedHeads( key, kvHeadCount, headDim, kNormWeight, rmsEpsilon, offsetRMSNorm );

	if ( ropeTheta > 0 ) {

		for ( let head = 0; head < headCount; head ++ ) {

			applyRoPE( query, head * headDim, rotaryDim, position, ropeTheta );

		}

		for ( let head = 0; head < kvHeadCount; head ++ ) {

			applyRoPE( key, head * headDim, rotaryDim, position, ropeTheta );

		}

	}

	for ( let dim = 0; dim < kvSize; dim ++ ) {

		keyCache[ position * kvSize + dim ] = key[ dim ];
		valueCache[ position * kvSize + dim ] = qkv[ qSize + kvSize + dim ];

	}

	const output = new Float32Array( qSize );

	for ( let head = 0; head < headCount; head ++ ) {

		const qOffset = head * headDim;
		const kvHead = Math.floor( head * kvHeadCount / headCount );
		const kvOffset = kvHead * headDim;
		const scores = new Float32Array( position - firstToken + 1 );

		for ( let token = firstToken; token <= position; token ++ ) {

			let dot = 0;

			for ( let i = 0; i < headDim; i ++ ) {

				dot += query[ qOffset + i ] * keyCache[ token * kvSize + kvOffset + i ];

			}

			scores[ token - firstToken ] = dot * attnScale;

		}

		const weights = softmax( scores );

		for ( let i = 0; i < headDim; i ++ ) {

			let sum = 0;

			for ( let token = firstToken; token <= position; token ++ ) {

				sum += weights[ token - firstToken ] * valueCache[ token * kvSize + kvOffset + i ];

			}

			output[ qOffset + i ] = sum;

		}

	}

	return output;

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

export {
	applyRoPE,
	causalAttention,
	geluNew,
	geluPytorchTanh,
	layerNorm,
	linear,
	rmsNorm,
	rmsNormPackedHeads,
	rotaryAngle,
	sampleTopK,
	silu,
	softmax
};
