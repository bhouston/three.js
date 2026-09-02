import { StorageBufferAttribute, WebGPURenderer } from 'three/webgpu';
import { storage } from 'three/tsl';

import { GPT2CPURunner } from '../../../../examples/jsm/gpgpu/llm/GPT2CPURunner.js';
import { GPT2Tokenizer } from '../../../../examples/jsm/gpgpu/llm/GPT2Tokenizer.js';
import { GPT2TSLRunner } from '../../../../examples/jsm/gpgpu/llm/GPT2TSLRunner.js';
import { GPT2Weights } from '../../../../examples/jsm/gpgpu/llm/GPT2Weights.js';
import { geluNew, layerNorm, linear, sampleTopK, softmax } from '../../../../examples/jsm/gpgpu/llm/LLMMath.js';
import { parseSafeTensors } from '../../../../examples/jsm/gpgpu/llm/SafeTensorsLoader.js';
import { TSLAdd } from '../../../../examples/jsm/gpgpu/llm/TSLAdd.js';
import { TSLAttention } from '../../../../examples/jsm/gpgpu/llm/TSLAttention.js';
import { TSLGELU } from '../../../../examples/jsm/gpgpu/llm/TSLGELU.js';
import { TSLLinear } from '../../../../examples/jsm/gpgpu/llm/TSLLinear.js';
import { TSLMLP } from '../../../../examples/jsm/gpgpu/llm/TSLMLP.js';
import { TSLNormalize } from '../../../../examples/jsm/gpgpu/llm/TSLNormalize.js';

function closeArray( assert, actual, expected, epsilon, message ) {

	assert.strictEqual( actual.length, expected.length, `${ message } length` );

	for ( let i = 0; i < expected.length; i ++ ) {

		assert.ok( Math.abs( actual[ i ] - expected[ i ] ) <= epsilon, `${ message }[${ i }]: ${ actual[ i ] } ~= ${ expected[ i ] }` );

	}

}

function mapGelu( values ) {

	const target = new Float32Array( values.length );

	for ( let i = 0; i < values.length; i ++ ) target[ i ] = geluNew( values[ i ] );

	return target;

}

function cpuMLP( input, fcWeight, fcBias, projWeight, projBias, hiddenSize, innerSize ) {

	const hidden = linear( input, fcWeight, fcBias, hiddenSize, innerSize );

	return linear( mapGelu( hidden ), projWeight, projBias, innerSize, hiddenSize );

}

// Spec oracle for causal one-query attention: softmax( Q K^T / sqrt(d) ) V,
// using LLMMath.softmax rather than a transliteration of TSLAttention's loops.
function cpuAttention( qkv, hiddenSize, headCount, keyCache, valueCache, position ) {

	const headSize = hiddenSize / headCount;
	const scale = 1 / Math.sqrt( headSize );

	for ( let dim = 0; dim < hiddenSize; dim ++ ) {

		keyCache[ position * hiddenSize + dim ] = qkv[ hiddenSize + dim ];
		valueCache[ position * hiddenSize + dim ] = qkv[ hiddenSize * 2 + dim ];

	}

	const output = new Float32Array( hiddenSize );

	for ( let head = 0; head < headCount; head ++ ) {

		const headOffset = head * headSize;
		const scores = new Float32Array( position + 1 );

		for ( let token = 0; token <= position; token ++ ) {

			let dot = 0;

			for ( let i = 0; i < headSize; i ++ ) {

				dot += qkv[ headOffset + i ] * keyCache[ token * hiddenSize + headOffset + i ];

			}

			scores[ token ] = dot * scale;

		}

		const weights = softmax( scores );

		for ( let i = 0; i < headSize; i ++ ) {

			let sum = 0;

			for ( let token = 0; token <= position; token ++ ) {

				sum += weights[ token ] * valueCache[ token * hiddenSize + headOffset + i ];

			}

			output[ headOffset + i ] = sum;

		}

	}

	return output;

}

function createSafeTensorsFixture() {

	const header = {
		values: {
			dtype: 'F32',
			shape: [ 2, 2 ],
			data_offsets: [ 0, 16 ]
		}
	};
	const headerBytes = new TextEncoder().encode( JSON.stringify( header ) );
	const buffer = new ArrayBuffer( 8 + headerBytes.length + 16 );
	const view = new DataView( buffer );

	view.setUint32( 0, headerBytes.length, true );
	view.setUint32( 4, 0, true );
	new Uint8Array( buffer, 8, headerBytes.length ).set( headerBytes );

	for ( let i = 0; i < 4; i ++ ) {

		view.setFloat32( 8 + headerBytes.length + i * 4, i + 1, true );

	}

	return buffer;

}

function storageFromArray( array ) {

	const attribute = new StorageBufferAttribute( array, 1 );

	return {
		attribute,
		node: storage( attribute, 'float', array.length ).toReadOnly()
	};

}

async function createRenderer( assert ) {

	const renderer = new WebGPURenderer( { antialias: false } );

	try {

		await renderer.init();

	} catch ( error ) {

		assert.ok( true, `SKIPPED: WebGPU backend is not available (${ error.message })` );
		return null;

	}

	return renderer;

}

async function readOutput( renderer, layer ) {

	return new Float32Array( await renderer.getArrayBufferAsync( layer.outputAttribute ) );

}

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'GPGPU', () => {

		QUnit.module( 'LLM', () => {

			QUnit.test( 'SafeTensorsLoader parses F32 tensors', ( assert ) => {

				const parsed = parseSafeTensors( createSafeTensorsFixture() );

				assert.deepEqual( parsed.tensors.values.shape, [ 2, 2 ], 'shape is parsed' );
				assert.strictEqual( parsed.tensors.values.dtype, 'F32', 'dtype is parsed' );
				assert.deepEqual( Array.from( parsed.tensors.values.data ), [ 1, 2, 3, 4 ], 'data is parsed' );

			} );

			QUnit.test( 'GPT2Tokenizer round-trips model text', async ( assert ) => {

				const tokenizer = await GPT2Tokenizer.fromURLs(
					'/examples/models/llm/tinystories-gpt2-0.1-3m/vocab.json',
					'/examples/models/llm/tinystories-gpt2-0.1-3m/merges.txt'
				);
				const text = 'Once upon a time, a small cat smiled.';

				assert.strictEqual( tokenizer.decode( tokenizer.encode( text ) ), text, 'text round-trips through GPT-2 BPE' );
				assert.strictEqual( tokenizer.endOfTextTokenId, 50256, 'end-of-text token id is available' );

			} );

			QUnit.test( 'LLMMath computes reference operations', ( assert ) => {

				closeArray( assert, linear(
					new Float32Array( [ 1, 2 ] ),
					new Float32Array( [ 3, 4, 5, 6 ] ),
					new Float32Array( [ 7, 8 ] ),
					2,
					2
				), new Float32Array( [ 20, 24 ] ), 1e-6, 'linear' );

				closeArray( assert, linear(
					new Float32Array( [ 1, 2 ] ),
					new Float32Array( [ 3, 4, 5, 6 ] ),
					null,
					2,
					2
				), new Float32Array( [ 13, 16 ] ), 1e-6, 'linear without bias' );

				closeArray( assert, linear(
					new Float32Array( [ 1, 2 ] ),
					new Float32Array( [ 1, 0, 0, 1 ] ),
					new Float32Array( [ 0, 0 ] ),
					2,
					2
				), new Float32Array( [ 1, 2 ] ), 1e-6, 'linear identity' );

				closeArray( assert, softmax( new Float32Array( [ 1, 2, 3 ] ) ), new Float32Array( [ 0.09003057, 0.24472848, 0.66524094 ] ), 1e-6, 'softmax' );
				closeArray( assert, softmax( new Float32Array( [ 1, 1, 1 ] ) ), new Float32Array( [ 1 / 3, 1 / 3, 1 / 3 ] ), 1e-6, 'softmax equal logits' );
				closeArray( assert, softmax( new Float32Array( [ 100, 101, 102 ] ) ), softmax( new Float32Array( [ 0, 1, 2 ] ) ), 1e-6, 'softmax is shift-invariant' );

				let softmaxSum = 0;
				const probabilities = softmax( new Float32Array( [ - 5, 0, 2.5 ] ) );
				for ( let i = 0; i < probabilities.length; i ++ ) softmaxSum += probabilities[ i ];
				assert.ok( Math.abs( softmaxSum - 1 ) < 1e-6, 'softmax sums to 1' );

				assert.ok( Math.abs( geluNew( 0 ) ) < 1e-6, 'gelu_new(0) == 0' );
				assert.ok( Math.abs( geluNew( 1 ) - 0.84119199 ) < 1e-6, 'gelu_new matches reference value' );
				assert.ok( Math.abs( geluNew( 20 ) - 20 ) < 1e-4, 'gelu_new(20) ~= 20' );
				assert.ok( Math.abs( geluNew( - 20 ) ) < 1e-4, 'gelu_new(-20) ~= 0' );

				assert.strictEqual( sampleTopK( new Float32Array( [ 1, 2, 3 ] ), { topK: 1 } ), 2, 'topK=1 returns argmax' );
				assert.strictEqual( sampleTopK( new Float32Array( [ 1, 4, 3 ] ), { temperature: 0, topK: 40 } ), 1, 'temperature=0 returns argmax' );
				assert.strictEqual( sampleTopK( new Float32Array( [ 1, 2, 3, 4 ] ), { topK: 2, temperature: 1, random: () => 0 } ), 3, 'random() == 0 picks the top candidate' );
				assert.strictEqual( sampleTopK( new Float32Array( [ 1, 2, 3, 4 ] ), { temperature: 1e-8, topK: 40, random: () => 0.5 } ), 3, 'near-zero temperature picks argmax' );

				const normalized = layerNorm(
					new Float32Array( [ 1, 2, 3 ] ),
					new Float32Array( [ 1, 1, 1 ] ),
					new Float32Array( [ 0, 0, 0 ] ),
					1e-5
				);

				closeArray( assert, normalized, new Float32Array( [ - 1.2247356, 0, 1.2247356 ] ), 1e-5, 'layerNorm' );

				const affine = layerNorm(
					new Float32Array( [ 1, 2, 3 ] ),
					new Float32Array( [ 2, 0.5, 1 ] ),
					new Float32Array( [ 0.1, - 0.2, 0.3 ] )
				);
				closeArray( assert, affine, new Float32Array( [
					normalized[ 0 ] * 2 + 0.1,
					normalized[ 1 ] * 0.5 - 0.2,
					normalized[ 2 ] * 1 + 0.3
				] ), 1e-5, 'layerNorm applies affine scale and shift' );

			} );

			QUnit.test( 'TSLLinear matches CPU reference', async ( assert ) => {

				const renderer = await createRenderer( assert );
				if ( renderer === null ) return;

				const input = storageFromArray( new Float32Array( [ 1, 2 ] ) );
				const layer = new TSLLinear(
					input.node,
					new Float32Array( [ 3, 4, 5, 6 ] ),
					new Float32Array( [ 7, 8 ] ),
					2,
					2
				);

				layer.compute( renderer );

				closeArray( assert, await readOutput( renderer, layer ), new Float32Array( [ 20, 24 ] ), 1e-5, 'TSLLinear' );
				renderer.dispose();

			} );

			QUnit.test( 'TSLLinear matches CPU reference without bias and for a non-square map', async ( assert ) => {

				const renderer = await createRenderer( assert );
				if ( renderer === null ) return;

				const input = new Float32Array( [ 1, - 1, 0.5 ] );
				const weight = new Float32Array( [ 1, 2, 3, 4, 5, 6 ] );
				const { node } = storageFromArray( input );
				const layer = new TSLLinear( node, weight, null, 3, 2, { workgroupSize: 2 } );

				layer.compute( renderer );

				closeArray( assert, await readOutput( renderer, layer ), new Float32Array( [ 0.5, 1 ] ), 1e-5, 'TSLLinear 3->2, null bias' );
				renderer.dispose();

			} );

			QUnit.test( 'TSLNormalize matches CPU reference', async ( assert ) => {

				const renderer = await createRenderer( assert );
				if ( renderer === null ) return;

				const input = new Float32Array( [ 1, 2, 3 ] );
				const { node } = storageFromArray( input );
				const layer = new TSLNormalize(
					node,
					new Float32Array( [ 1, 1, 1 ] ),
					new Float32Array( [ 0, 0, 0 ] ),
					3,
					{ workgroupSize: 3 }
				);

				layer.compute( renderer );

				closeArray( assert, await readOutput( renderer, layer ), layerNorm( input, new Float32Array( [ 1, 1, 1 ] ), new Float32Array( [ 0, 0, 0 ] ) ), 1e-5, 'TSLNormalize' );
				renderer.dispose();

			} );

			QUnit.test( 'TSLNormalize matches CPU reference with affine parameters', async ( assert ) => {

				const renderer = await createRenderer( assert );
				if ( renderer === null ) return;

				const input = new Float32Array( [ 1, 2, 3, - 1 ] );
				const weight = new Float32Array( [ 2, 0.5, 1, 1.5 ] );
				const bias = new Float32Array( [ 0.1, - 0.2, 0.3, 0 ] );
				const { node } = storageFromArray( input );
				const layer = new TSLNormalize( node, weight, bias, 4, { workgroupSize: 4, epsilon: 1e-5 } );

				layer.compute( renderer );

				closeArray( assert, await readOutput( renderer, layer ), layerNorm( input, weight, bias ), 1e-5, 'TSLNormalize affine' );
				renderer.dispose();

			} );

			QUnit.test( 'TSLGELU matches CPU gelu_new', async ( assert ) => {

				const renderer = await createRenderer( assert );
				if ( renderer === null ) return;

				const input = new Float32Array( [ - 2, - 1, 0, 0.5, 1, 2, 3 ] );
				const { node } = storageFromArray( input );
				const layer = new TSLGELU( node, input.length, { workgroupSize: input.length } );

				layer.compute( renderer );

				const output = await readOutput( renderer, layer );
				closeArray( assert, output, mapGelu( input ), 1e-5, 'TSLGELU' );
				assert.ok( Math.abs( output[ 4 ] - 0.84119199 ) < 1e-5, 'TSLGELU(1) matches published gelu_new(1)' );
				renderer.dispose();

			} );

			QUnit.test( 'TSLAdd matches element-wise CPU add', async ( assert ) => {

				const renderer = await createRenderer( assert );
				if ( renderer === null ) return;

				const a = new Float32Array( [ 1, 2, 3, - 4 ] );
				const b = new Float32Array( [ 4, - 1, 0.5, 4 ] );
				const expected = new Float32Array( [ 5, 1, 3.5, 0 ] );
				const layer = new TSLAdd( storageFromArray( a ).node, storageFromArray( b ).node, 4, { workgroupSize: 4 } );

				layer.compute( renderer );

				closeArray( assert, await readOutput( renderer, layer ), expected, 1e-5, 'TSLAdd' );
				renderer.dispose();

			} );

			QUnit.test( 'TSLMLP matches CPU dense -> gelu_new -> dense', async ( assert ) => {

				const renderer = await createRenderer( assert );
				if ( renderer === null ) return;

				const input = new Float32Array( [ 1, - 1 ] );
				const fcWeight = new Float32Array( [ 0.5, - 0.25, 1, 0.75, 0.5, - 1 ] );
				const fcBias = new Float32Array( [ 0.1, 0, - 0.2 ] );
				const projWeight = new Float32Array( [ 1, 0, 0, 1, 0.5, - 0.5 ] );
				const projBias = new Float32Array( [ 0, 0.25 ] );
				const layer = new TSLMLP(
					storageFromArray( input ).node,
					fcWeight,
					fcBias,
					projWeight,
					projBias,
					2,
					3,
					{ workgroupSize: 3 }
				);

				layer.compute( renderer );

				closeArray( assert, await readOutput( renderer, layer.proj ), cpuMLP( input, fcWeight, fcBias, projWeight, projBias, 2, 3 ), 1e-4, 'TSLMLP' );
				renderer.dispose();

			} );

			QUnit.test( 'TSLAttention matches CPU reference for a one-token and two-token pass', async ( assert ) => {

				const renderer = await createRenderer( assert );
				if ( renderer === null ) return;

				const hiddenSize = 4;
				const headCount = 2;
				const maxTokens = 4;
				const qkvBuffer = new Float32Array( hiddenSize * 3 );
				const { attribute, node } = storageFromArray( qkvBuffer );
				const layer = new TSLAttention( node, hiddenSize, headCount, maxTokens, { workgroupSize: hiddenSize } );
				const keyCache = new Float32Array( hiddenSize * maxTokens );
				const valueCache = new Float32Array( hiddenSize * maxTokens );

				// One token: softmax is a single 1, so the attention output is V.
				qkvBuffer.set( [ 1, 0, 0, 1, 1, 0, 0, 1, 2, 3, 4, 5 ] );
				attribute.needsUpdate = true;
				layer.compute( renderer, 0 );

				const output0 = await readOutput( renderer, layer );
				closeArray(
					assert,
					output0,
					cpuAttention( qkvBuffer.slice(), hiddenSize, headCount, keyCache, valueCache, 0 ),
					1e-4,
					'TSLAttention token 0'
				);
				closeArray( assert, output0, new Float32Array( [ 2, 3, 4, 5 ] ), 1e-4, 'one-token attention returns V' );

				qkvBuffer.set( [ 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1 ] );
				attribute.needsUpdate = true;
				layer.compute( renderer, 1 );

				closeArray(
					assert,
					await readOutput( renderer, layer ),
					cpuAttention( qkvBuffer.slice(), hiddenSize, headCount, keyCache, valueCache, 1 ),
					1e-4,
					'TSLAttention token 1'
				);
				renderer.dispose();

			} );

			QUnit.test( 'CPU TinyStories greedy continuation stays on the prompt', async ( assert ) => {

				const weights = await GPT2Weights.fromURL( '/examples/models/llm/tinystories-gpt2-0.1-3m/' );
				const result = new GPT2CPURunner( weights ).generate( 'Once upon a time,', {
					maxNewTokens: 24,
					temperature: 0,
					topK: 1
				} );

				assert.strictEqual(
					result.text,
					'Once upon a time, there was a little girl named Lily. She loved to play with her toys. One day, she saw a big,'
				);

			} );

			QUnit.test( 'TSL TinyStories greedy continuation matches the CPU runner', async ( assert ) => {

				const renderer = await createRenderer( assert );
				if ( renderer === null ) return;

				const weights = await GPT2Weights.fromURL( '/examples/models/llm/tinystories-gpt2-0.1-3m/' );
				const prompt = 'Once upon a time,';
				const options = { maxNewTokens: 8, temperature: 0, topK: 1 };
				const cpu = new GPT2CPURunner( weights, { maxTokens: 128 } ).generate( prompt, options );
				const gpu = await new GPT2TSLRunner( weights, { maxTokens: 128 } ).generate( renderer, prompt, options );

				assert.strictEqual( gpu.text, cpu.text, 'GPU greedy text matches CPU' );
				assert.deepEqual( gpu.generatedTokens, cpu.generatedTokens, 'GPU token ids match CPU' );
				renderer.dispose();

			} );

		} );

	} );

} );
