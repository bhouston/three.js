import { TSLLinear } from './TSLLinear.js';

function createChunkedLogitLayers( inputNode, weights, chunkSize, name ) {

	const logits = [];
	const hiddenSize = weights.hiddenSize;

	for ( let offset = 0; offset < weights.vocabSize; offset += chunkSize ) {

		const size = Math.min( chunkSize, weights.vocabSize - offset );
		const chunkWeight = new Float32Array( hiddenSize * size );

		for ( let i = 0; i < hiddenSize; i ++ ) {

			const sourceOffset = i * weights.vocabSize + offset;
			chunkWeight.set( weights.logitWeight.subarray( sourceOffset, sourceOffset + size ), i * size );

		}

		logits.push( {
			offset,
			size,
			layer: new TSLLinear( inputNode, chunkWeight, null, hiddenSize, size, {
				name: `${ name }${ offset }`,
				workgroupSize: 256
			} )
		} );

	}

	return logits;

}

async function readChunkedLogits( renderer, chunks, vocabSize ) {

	const logits = new Float32Array( vocabSize );
	const values = await Promise.all( chunks.map( async ( chunk ) => (
		new Float32Array( await renderer.getArrayBufferAsync( chunk.layer.outputAttribute ) )
	) ) );

	for ( let i = 0; i < chunks.length; i ++ ) {

		const chunk = chunks[ i ];
		logits.set( values[ i ].subarray( 0, chunk.size ), chunk.offset );

	}

	return logits;

}

export { createChunkedLogitLayers, readChunkedLogits };
