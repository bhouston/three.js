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

		const scale = weight === null ? 1 : ( offsetWeight ? 1 + weight[ i ] : weight[ i ] );
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

function applyRoPE( vector, headOffset, rotaryDim, position, theta, options = {} ) {

	if ( rotaryDim <= 0 ) return vector;

	const half = rotaryDim / 2;
	const freqDim = options.ropeFreqDim || rotaryDim;
	const pairCount = options.ropePairCount !== undefined ? options.ropePairCount : half;
	const rotated = new Float32Array( rotaryDim );

	for ( let i = 0; i < rotaryDim; i ++ ) {

		const x = vector[ headOffset + i ];
		const freqIndex = i < half ? i : i - half;

		if ( freqIndex >= pairCount ) {

			rotated[ i ] = x;
			continue;

		}

		const partner = i < half ? - vector[ headOffset + i + half ] : vector[ headOffset + i - half ];
		const angle = rotaryAngle( position, freqIndex, freqDim, theta );

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
		offsetRMSNorm = false,
		ropeFreqDim = rotaryDim,
		ropePairCount,
		queryOnly = false,
		writeCache = true,
		vNorm = false,
		outputGate = null
	} = options;
	const qSize = headCount * headDim;
	const kvSize = kvHeadCount * headDim;
	const query = qkv.slice( 0, qSize );
	const key = queryOnly ? null : qkv.slice( qSize, qSize + kvSize );
	const firstToken = slidingWindow > 0 ? Math.max( 0, position - slidingWindow + 1 ) : 0;
	const ropeOptions = { ropeFreqDim, ropePairCount };

	if ( qNormWeight !== null ) rmsNormPackedHeads( query, headCount, headDim, qNormWeight, rmsEpsilon, offsetRMSNorm );

	if ( queryOnly === false && kNormWeight !== null ) {

		rmsNormPackedHeads( key, kvHeadCount, headDim, kNormWeight, rmsEpsilon, offsetRMSNorm );

	}

	if ( ropeTheta > 0 ) {

		for ( let head = 0; head < headCount; head ++ ) {

			applyRoPE( query, head * headDim, rotaryDim, position, ropeTheta, ropeOptions );

		}

		if ( queryOnly === false ) {

			for ( let head = 0; head < kvHeadCount; head ++ ) {

				applyRoPE( key, head * headDim, rotaryDim, position, ropeTheta, ropeOptions );

			}

		}

	}

	if ( queryOnly === false && writeCache ) {

		const value = qkv.slice( qSize + kvSize, qSize + 2 * kvSize );

		if ( vNorm ) rmsNormPackedHeads( value, kvHeadCount, headDim, null, rmsEpsilon, false );

		for ( let dim = 0; dim < kvSize; dim ++ ) {

			keyCache[ position * kvSize + dim ] = key[ dim ];
			valueCache[ position * kvSize + dim ] = value[ dim ];

		}

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

	if ( outputGate !== null ) {

		for ( let i = 0; i < output.length; i ++ ) output[ i ] *= sigmoid( outputGate[ i ] );

	}

	return output;

}

function sigmoid( x ) {

	return 1 / ( 1 + Math.exp( - x ) );

}

function softplus( x ) {

	if ( x > 20 ) return x;
	if ( x < - 20 ) return Math.exp( x );
	return Math.log( 1 + Math.exp( x ) );

}

function logitSoftcap( logits, cap ) {

	if ( cap === null || cap === undefined ) return logits;

	for ( let i = 0; i < logits.length; i ++ ) logits[ i ] = cap * Math.tanh( logits[ i ] / cap );

	return logits;

}

function l2norm( vector, offset, size, epsilon = 1e-6 ) {

	let sumSquares = 0;

	for ( let i = 0; i < size; i ++ ) sumSquares += vector[ offset + i ] * vector[ offset + i ];

	const inv = 1 / Math.sqrt( sumSquares + epsilon );

	for ( let i = 0; i < size; i ++ ) vector[ offset + i ] *= inv;

	return vector;

}

function splitHeadGate( packed, headCount, headDim ) {

	const qSize = headCount * headDim;
	const query = new Float32Array( qSize );
	const gate = new Float32Array( qSize );

	for ( let head = 0; head < headCount; head ++ ) {

		const packedOffset = head * headDim * 2;
		const headOffset = head * headDim;
		query.set( packed.subarray( packedOffset, packedOffset + headDim ), headOffset );
		gate.set( packed.subarray( packedOffset + headDim, packedOffset + headDim * 2 ), headOffset );

	}

	return { query, gate };

}

function causalConv1dStep( input, state, weight, kernelSize, activation = 'silu' ) {

	const convDim = input.length;
	const output = new Float32Array( convDim );
	const window = new Float32Array( kernelSize );

	for ( let channel = 0; channel < convDim; channel ++ ) {

		const stateOffset = channel * kernelSize;
		const weightOffset = channel * kernelSize;

		for ( let k = 0; k < kernelSize; k ++ ) window[ k ] = state[ stateOffset + k ];

		for ( let k = 0; k < kernelSize - 1; k ++ ) state[ stateOffset + k ] = window[ k + 1 ];

		state[ stateOffset + kernelSize - 1 ] = input[ channel ];

		let sum = 0;

		for ( let k = 0; k < kernelSize - 1; k ++ ) sum += weight[ weightOffset + k ] * window[ k + 1 ];

		sum += weight[ weightOffset + kernelSize - 1 ] * input[ channel ];
		output[ channel ] = activation === 'silu' ? silu( sum ) : sum;

	}

	return output;

}

function gatedDeltaRuleStep( query, key, value, decay, beta, state, options ) {

	const { numVHeads, keyDim, valueDim } = options;
	const output = new Float32Array( numVHeads * valueDim );

	for ( let head = 0; head < numVHeads; head ++ ) {

		const decayH = decay[ head ];
		const betaH = beta[ head ];
		const qOffset = head * keyDim;
		const vOffset = head * valueDim;
		const stateOffset = head * keyDim * valueDim;

		for ( let v = 0; v < valueDim; v ++ ) {

			let kvMem = 0;

			for ( let k = 0; k < keyDim; k ++ ) {

				const index = stateOffset + k * valueDim + v;
				state[ index ] *= decayH;
				kvMem += state[ index ] * key[ qOffset + k ];

			}

			const delta = ( value[ vOffset + v ] - kvMem ) * betaH;
			let mixed = 0;

			for ( let k = 0; k < keyDim; k ++ ) {

				const index = stateOffset + k * valueDim + v;
				state[ index ] += key[ qOffset + k ] * delta;
				mixed += state[ index ] * query[ qOffset + k ];

			}

			output[ vOffset + v ] = mixed;

		}

	}

	return output;

}

function rmsNormGated( input, gate, weight, headCount, headDim, epsilon = 1e-6 ) {

	const output = new Float32Array( input.length );

	for ( let head = 0; head < headCount; head ++ ) {

		const offset = head * headDim;
		const slice = input.subarray( offset, offset + headDim );
		const normed = rmsNorm( slice, weight, epsilon, false );

		for ( let i = 0; i < headDim; i ++ ) output[ offset + i ] = normed[ i ] * silu( gate[ offset + i ] );

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
	causalConv1dStep,
	gatedDeltaRuleStep,
	geluNew,
	geluPytorchTanh,
	l2norm,
	layerNorm,
	linear,
	logitSoftcap,
	rmsNorm,
	rmsNormGated,
	rmsNormPackedHeads,
	rotaryAngle,
	sampleTopK,
	sigmoid,
	silu,
	softplus,
	softmax,
	splitHeadGate
};
