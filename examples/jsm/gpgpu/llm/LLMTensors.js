/**
 * Shared tensor helpers for Hugging Face causal LM loaders.
 *
 * @three_import import { tensorToFloat32, transpose2D } from 'three/addons/gpgpu/llm/LLMTensors.js';
 */

function transpose2D( data, rows, columns ) {

	const target = new Float32Array( data.length );

	for ( let row = 0; row < rows; row ++ ) {

		for ( let column = 0; column < columns; column ++ ) {

			target[ column * rows + row ] = data[ row * columns + column ];

		}

	}

	return target;

}

function float16ToFloat32( value ) {

	const sign = ( value >> 15 ) & 1;
	const exponent = ( value >> 10 ) & 0x1f;
	const fraction = value & 0x3ff;

	if ( exponent === 0 ) {

		if ( fraction === 0 ) return sign ? - 0 : 0;

		return ( sign ? - 1 : 1 ) * Math.pow( 2, - 14 ) * ( fraction / 1024 );

	}

	if ( exponent === 31 ) {

		return fraction ? NaN : ( sign ? - Infinity : Infinity );

	}

	return ( sign ? - 1 : 1 ) * Math.pow( 2, exponent - 15 ) * ( 1 + fraction / 1024 );

}

function bfloat16ToFloat32( value ) {

	const bits = new Uint32Array( 1 );
	bits[ 0 ] = value << 16;

	return new Float32Array( bits.buffer )[ 0 ];

}

function tensorToFloat32( tensor ) {

	if ( tensor.dtype === 'F32' ) return tensor.data;

	const source = tensor.data;
	const target = new Float32Array( source.length );

	if ( tensor.dtype === 'F16' ) {

		for ( let i = 0; i < source.length; i ++ ) target[ i ] = float16ToFloat32( source[ i ] );

	} else if ( tensor.dtype === 'BF16' ) {

		for ( let i = 0; i < source.length; i ++ ) target[ i ] = bfloat16ToFloat32( source[ i ] );

	} else {

		throw new Error( `LLMTensors: Tensor "${ tensor.name }" uses dtype "${ tensor.dtype }"; only F32, F16, and BF16 are supported.` );

	}

	return target;

}

function packProjections( projections, inputSize ) {

	const outputSizes = projections.map( ( projection ) => projection.length / inputSize );
	const outputSize = outputSizes.reduce( ( sum, size ) => sum + size, 0 );
	const packed = new Float32Array( inputSize * outputSize );

	for ( let row = 0; row < inputSize; row ++ ) {

		let offset = row * outputSize;

		for ( let i = 0; i < projections.length; i ++ ) {

			const size = outputSizes[ i ];
			packed.set( projections[ i ].subarray( row * size, row * size + size ), offset );
			offset += size;

		}

	}

	return packed;

}

function packBiases( biases ) {

	if ( biases.every( ( bias ) => bias === null ) ) return null;

	const parts = biases.map( ( bias ) => bias || new Float32Array( 0 ) );
	const packed = new Float32Array( parts.reduce( ( sum, part ) => sum + part.length, 0 ) );
	let offset = 0;

	for ( const part of parts ) {

		packed.set( part, offset );
		offset += part.length;

	}

	return packed;

}

function prepareGeneration( tokenizer, prompt, maxTokens, maxNewTokens, endOfTextTokenId ) {

	const encoded = tokenizer.encode( prompt );
	const promptBudget = Math.max( 1, maxTokens - 1 );
	const inputTokens = encoded.length === 0 ? [ endOfTextTokenId ] : encoded.slice( - promptBudget );
	const newTokenBudget = Math.max( 0, Math.min( maxNewTokens, maxTokens - inputTokens.length ) );

	return { inputTokens, newTokenBudget };

}

async function fetchJSON( url, label = 'LLM' ) {

	const response = await fetch( url );

	if ( response.ok === false ) {

		throw new Error( `${ label }: Failed to load "${ url }" (${ response.status } ${ response.statusText })` );

	}

	return response.json();

}

function formatBytes( bytes ) {

	if ( bytes < 1024 ) return `${ bytes } B`;
	if ( bytes < 1024 * 1024 ) return `${ ( bytes / 1024 ).toFixed( 1 ) } KB`;
	return `${ ( bytes / ( 1024 * 1024 ) ).toFixed( 1 ) } MB`;

}

function yieldToBrowser() {

	return new Promise( ( resolve ) => setTimeout( resolve, 0 ) );

}

function createProgress( label, onProgress ) {

	return async function report( message ) {

		const line = `${ label }: ${ message }`;
		console.log( line );
		if ( onProgress ) onProgress( line );
		await yieldToBrowser();

	};

}

async function fetchArrayBuffer( url, label = 'LLM', onProgress ) {

	const response = await fetch( url );

	if ( response.ok === false ) {

		throw new Error( `${ label }: Failed to load "${ url }" (${ response.status } ${ response.statusText })` );

	}

	const total = Number( response.headers.get( 'Content-Length' ) ) || 0;
	const report = createProgress( label, onProgress );

	if ( response.body === null || typeof response.body.getReader !== 'function' ) {

		await report( `Downloading ${ url }${ total ? ` (${ formatBytes( total ) })` : '' }...` );
		return response.arrayBuffer();

	}

	const reader = response.body.getReader();
	const chunks = [];
	let received = 0;
	let lastReport = 0;

	await report( `Downloading ${ url }${ total ? ` (${ formatBytes( total ) })` : '' }...` );

	while ( true ) {

		const { done, value } = await reader.read();
		if ( done ) break;

		chunks.push( value );
		received += value.byteLength;

		if ( received - lastReport >= 8 * 1024 * 1024 || ( total > 0 && received === total ) ) {

			lastReport = received;
			const totalText = total > 0 ? ` / ${ formatBytes( total ) }` : '';
			await report( `Downloading weights ${ formatBytes( received ) }${ totalText }` );

		}

	}

	const bytes = new Uint8Array( received );
	let offset = 0;

	for ( let i = 0; i < chunks.length; i ++ ) {

		bytes.set( chunks[ i ], offset );
		offset += chunks[ i ].byteLength;

	}

	await report( `Downloaded ${ formatBytes( received ) }` );
	return bytes.buffer;

}

export {
	bfloat16ToFloat32,
	createProgress,
	fetchArrayBuffer,
	fetchJSON,
	float16ToFloat32,
	formatBytes,
	packBiases,
	packProjections,
	prepareGeneration,
	tensorToFloat32,
	transpose2D,
	yieldToBrowser
};
