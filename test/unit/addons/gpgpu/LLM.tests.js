import { StorageBufferAttribute, WebGPURenderer } from 'three/webgpu';
import { storage } from 'three/tsl';

import { DecoderCPURunner } from '../../../../examples/jsm/gpgpu/llm/DecoderCPURunner.js';
import { DecoderTSLRunner } from '../../../../examples/jsm/gpgpu/llm/DecoderTSLRunner.js';
import { DecoderWeights } from '../../../../examples/jsm/gpgpu/llm/DecoderWeights.js';
import { GPT2Tokenizer } from '../../../../examples/jsm/gpgpu/llm/GPT2Tokenizer.js';
import { architectureFor } from '../../../../examples/jsm/gpgpu/llm/LLMFactory.js';
import { planPromptCache, sharedPrefixLength } from '../../../../examples/jsm/gpgpu/llm/LLMGenerate.js';
import { applyRoPE, causalAttention, gatedDeltaRuleStep, geluNew, layerNorm, linear, logitSoftcap, rmsNorm, sampleTopK, silu, softmax, splitHeadGate } from '../../../../examples/jsm/gpgpu/llm/LLMMath.js';
import { bfloat16ToFloat32, convertAllTensors, float16ToFloat32, tensorToFloat32 } from '../../../../examples/jsm/gpgpu/llm/LLMTensors.js';
import { QwenCPURunner } from '../../../../examples/jsm/gpgpu/llm/QwenCPURunner.js';
import { QwenTSLRunner } from '../../../../examples/jsm/gpgpu/llm/QwenTSLRunner.js';
import { QwenWeights } from '../../../../examples/jsm/gpgpu/llm/QwenWeights.js';
import { parseSafeTensors } from '../../../../examples/jsm/gpgpu/llm/SafeTensorsLoader.js';
import { resolveTensor } from '../../../../examples/jsm/gpgpu/llm/TensorNameMap.js';
import { TSLAdd } from '../../../../examples/jsm/gpgpu/llm/TSLAdd.js';
import { TSLAttention } from '../../../../examples/jsm/gpgpu/llm/TSLAttention.js';
import { TSLGatedMLP } from '../../../../examples/jsm/gpgpu/llm/TSLGatedMLP.js';
import { TSLGELU } from '../../../../examples/jsm/gpgpu/llm/TSLGELU.js';
import { TSLLinear } from '../../../../examples/jsm/gpgpu/llm/TSLLinear.js';
import { TSLMLP } from '../../../../examples/jsm/gpgpu/llm/TSLMLP.js';
import { TSLNormalize } from '../../../../examples/jsm/gpgpu/llm/TSLNormalize.js';
import { TSLRMSNorm } from '../../../../examples/jsm/gpgpu/llm/TSLRMSNorm.js';
import { TSLSiLUMul } from '../../../../examples/jsm/gpgpu/llm/TSLSiLUMul.js';
import { UnigramTokenizer } from '../../../../examples/jsm/gpgpu/llm/UnigramTokenizer.js';

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

function cpuAttentionScores( qkv, hiddenSize, headCount, keyCache, position, maxTokens ) {

	const headSize = hiddenSize / headCount;
	const scale = 1 / Math.sqrt( headSize );
	const scores = new Float32Array( headCount * maxTokens );

	for ( let head = 0; head < headCount; head ++ ) {

		const headOffset = head * headSize;

		for ( let token = 0; token <= position; token ++ ) {

			let dot = 0;

			for ( let i = 0; i < headSize; i ++ ) {

				dot += qkv[ headOffset + i ] * keyCache[ token * hiddenSize + headOffset + i ];

			}

			scores[ head * maxTokens + token ] = dot * scale;

		}

	}

	return scores;

}

async function assertAttentionSequence( assert, renderer, hiddenSize, headCount, maxTokens, sequence, workgroupSize, epsilon = 1e-4 ) {

	const qkvBuffer = new Float32Array( hiddenSize * 3 );
	const { attribute, node } = storageFromArray( qkvBuffer );
	const layer = new TSLAttention( node, hiddenSize, headCount, maxTokens, { workgroupSize } );
	const keyCache = new Float32Array( hiddenSize * maxTokens );
	const valueCache = new Float32Array( hiddenSize * maxTokens );

	for ( let position = 0; position < sequence.length; position ++ ) {

		qkvBuffer.set( sequence[ position ] );
		attribute.needsUpdate = true;
		layer.compute( renderer, position );

		const expected = cpuAttention( qkvBuffer.slice(), hiddenSize, headCount, keyCache, valueCache, position );
		const gpuOutput = await readOutput( renderer, layer );
		const gpuScores = new Float32Array( await renderer.getArrayBufferAsync( layer.scoreAttribute ) );
		const expectedScores = cpuAttentionScores( qkvBuffer, hiddenSize, headCount, keyCache, position, maxTokens );

		closeArray( assert, gpuOutput, expected, epsilon, `attention output token ${ position }` );

		for ( let head = 0; head < headCount; head ++ ) {

			for ( let token = 0; token <= position; token ++ ) {

				const index = head * maxTokens + token;
				assert.ok(
					Math.abs( gpuScores[ index ] - expectedScores[ index ] ) <= epsilon,
					`attention scores head ${ head } token ${ token } @ ${ position }: ${ gpuScores[ index ] } ~= ${ expectedScores[ index ] }`
				);

			}

		}

	}

	renderer.dispose();

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

function fillSin( array, seed ) {

	for ( let i = 0; i < array.length; i ++ ) array[ i ] = Math.sin( seed + i * 0.17 ) * 0.35;

	return array;

}

function makeTensor( name, shape, seed ) {

	const data = fillSin( new Float32Array( shape.reduce( ( product, value ) => product * value, 1 ) ), seed );

	return { name, dtype: 'F32', shape, data };

}

function tinyTokenizer( eos = 0 ) {

	return {
		endOfTextTokenId: eos,
		encode() {

			return [ 1, 2 ];

		},
		decode( ids ) {

			return ids.join( ',' );

		}
	};

}

function createTinyLlama() {

	const hidden = 8;
	const inner = 16;
	const heads = 4;
	const kvHeads = 2;
	const headDim = 2;
	const layers = 2;
	const vocab = 8;
	const qSize = heads * headDim;
	const kvSize = kvHeads * headDim;
	const tensors = {
		'model.embed_tokens.weight': makeTensor( 'embed', [ vocab, hidden ], 0.1 ),
		'model.norm.weight': makeTensor( 'norm', [ hidden ], 1.1 )
	};

	for ( let layer = 0; layer < layers; layer ++ ) {

		const p = `model.layers.${ layer }`;
		tensors[ `${ p }.input_layernorm.weight` ] = makeTensor( 'ln1', [ hidden ], 2 + layer );
		tensors[ `${ p }.post_attention_layernorm.weight` ] = makeTensor( 'ln2', [ hidden ], 3 + layer );
		tensors[ `${ p }.self_attn.q_proj.weight` ] = makeTensor( 'q', [ qSize, hidden ], 4 + layer );
		tensors[ `${ p }.self_attn.k_proj.weight` ] = makeTensor( 'k', [ kvSize, hidden ], 5 + layer );
		tensors[ `${ p }.self_attn.v_proj.weight` ] = makeTensor( 'v', [ kvSize, hidden ], 6 + layer );
		tensors[ `${ p }.self_attn.o_proj.weight` ] = makeTensor( 'o', [ hidden, qSize ], 7 + layer );
		tensors[ `${ p }.mlp.gate_proj.weight` ] = makeTensor( 'gate', [ inner, hidden ], 8 + layer );
		tensors[ `${ p }.mlp.up_proj.weight` ] = makeTensor( 'up', [ inner, hidden ], 9 + layer );
		tensors[ `${ p }.mlp.down_proj.weight` ] = makeTensor( 'down', [ hidden, inner ], 10 + layer );

	}

	return new DecoderWeights( {
		model_type: 'llama',
		hidden_size: hidden,
		intermediate_size: inner,
		num_hidden_layers: layers,
		num_attention_heads: heads,
		num_key_value_heads: kvHeads,
		vocab_size: vocab,
		rms_norm_eps: 1e-5,
		rope_theta: 10000,
		hidden_act: 'silu',
		tie_word_embeddings: true,
		eos_token_id: 0,
		max_position_embeddings: 16
	}, tensors, tinyTokenizer() );

}

function createTinyPhi() {

	const hidden = 8;
	const inner = 8;
	const heads = 2;
	const headDim = 4;
	const layers = 1;
	const vocab = 8;
	const tensors = {
		'model.embed_tokens.weight': makeTensor( 'embed', [ vocab, hidden ], 0.2 ),
		'model.final_layernorm.weight': makeTensor( 'fnw', [ hidden ], 1.2 ),
		'model.final_layernorm.bias': makeTensor( 'fnb', [ hidden ], 1.3 ),
		'lm_head.weight': makeTensor( 'lm', [ vocab, hidden ], 0.4 )
	};
	const p = 'model.layers.0';
	tensors[ `${ p }.input_layernorm.weight` ] = makeTensor( 'lnw', [ hidden ], 2.1 );
	tensors[ `${ p }.input_layernorm.bias` ] = makeTensor( 'lnb', [ hidden ], 2.2 );
	tensors[ `${ p }.self_attn.q_proj.weight` ] = makeTensor( 'q', [ hidden, hidden ], 3.1 );
	tensors[ `${ p }.self_attn.k_proj.weight` ] = makeTensor( 'k', [ hidden, hidden ], 3.2 );
	tensors[ `${ p }.self_attn.v_proj.weight` ] = makeTensor( 'v', [ hidden, hidden ], 3.3 );
	tensors[ `${ p }.self_attn.q_proj.bias` ] = makeTensor( 'qb', [ hidden ], 3.4 );
	tensors[ `${ p }.self_attn.k_proj.bias` ] = makeTensor( 'kb', [ hidden ], 3.5 );
	tensors[ `${ p }.self_attn.v_proj.bias` ] = makeTensor( 'vb', [ hidden ], 3.6 );
	tensors[ `${ p }.self_attn.dense.weight` ] = makeTensor( 'd', [ hidden, hidden ], 3.7 );
	tensors[ `${ p }.self_attn.dense.bias` ] = makeTensor( 'db', [ hidden ], 3.8 );
	tensors[ `${ p }.mlp.fc1.weight` ] = makeTensor( 'fc1', [ inner, hidden ], 4.1 );
	tensors[ `${ p }.mlp.fc1.bias` ] = makeTensor( 'fc1b', [ inner ], 4.2 );
	tensors[ `${ p }.mlp.fc2.weight` ] = makeTensor( 'fc2', [ hidden, inner ], 4.3 );
	tensors[ `${ p }.mlp.fc2.bias` ] = makeTensor( 'fc2b', [ hidden ], 4.4 );

	return new DecoderWeights( {
		model_type: 'phi',
		hidden_size: hidden,
		intermediate_size: inner,
		num_hidden_layers: layers,
		num_attention_heads: heads,
		vocab_size: vocab,
		layer_norm_eps: 1e-5,
		rope_theta: 10000,
		partial_rotary_factor: 0.5,
		eos_token_id: 0,
		max_position_embeddings: 16
	}, tensors, tinyTokenizer() );

}

function createTinyGemma() {

	const hidden = 8;
	const inner = 16;
	const heads = 2;
	const kvHeads = 1;
	const headDim = 8;
	const layers = 2;
	const vocab = 8;
	const qSize = heads * headDim;
	const kvSize = kvHeads * headDim;
	const tensors = {
		'model.embed_tokens.weight': makeTensor( 'embed', [ vocab, hidden ], 0.15 ),
		'model.norm.weight': makeTensor( 'norm', [ hidden ], 1.15 )
	};

	for ( let layer = 0; layer < layers; layer ++ ) {

		const p = `model.layers.${ layer }`;
		tensors[ `${ p }.input_layernorm.weight` ] = makeTensor( 'ln1', [ hidden ], 2.1 + layer );
		tensors[ `${ p }.post_attention_layernorm.weight` ] = makeTensor( 'postA', [ hidden ], 2.2 + layer );
		tensors[ `${ p }.pre_feedforward_layernorm.weight` ] = makeTensor( 'preM', [ hidden ], 2.3 + layer );
		tensors[ `${ p }.post_feedforward_layernorm.weight` ] = makeTensor( 'postM', [ hidden ], 2.4 + layer );
		tensors[ `${ p }.self_attn.q_norm.weight` ] = makeTensor( 'qn', [ headDim ], 2.5 + layer );
		tensors[ `${ p }.self_attn.k_norm.weight` ] = makeTensor( 'kn', [ headDim ], 2.6 + layer );
		tensors[ `${ p }.self_attn.q_proj.weight` ] = makeTensor( 'q', [ qSize, hidden ], 4 + layer );
		tensors[ `${ p }.self_attn.k_proj.weight` ] = makeTensor( 'k', [ kvSize, hidden ], 5 + layer );
		tensors[ `${ p }.self_attn.v_proj.weight` ] = makeTensor( 'v', [ kvSize, hidden ], 6 + layer );
		tensors[ `${ p }.self_attn.o_proj.weight` ] = makeTensor( 'o', [ hidden, qSize ], 7 + layer );
		tensors[ `${ p }.mlp.gate_proj.weight` ] = makeTensor( 'gate', [ inner, hidden ], 8 + layer );
		tensors[ `${ p }.mlp.up_proj.weight` ] = makeTensor( 'up', [ inner, hidden ], 9 + layer );
		tensors[ `${ p }.mlp.down_proj.weight` ] = makeTensor( 'down', [ hidden, inner ], 10 + layer );

	}

	return new DecoderWeights( {
		model_type: 'gemma3_text',
		hidden_size: hidden,
		intermediate_size: inner,
		num_hidden_layers: layers,
		num_attention_heads: heads,
		num_key_value_heads: kvHeads,
		head_dim: headDim,
		vocab_size: vocab,
		rms_norm_eps: 1e-6,
		rope_theta: 1000000,
		rope_local_base_freq: 10000,
		sliding_window: 2,
		layer_types: [ 'sliding_attention', 'full_attention' ],
		hidden_activation: 'gelu_pytorch_tanh',
		query_pre_attn_scalar: headDim,
		eos_token_id: 0,
		max_position_embeddings: 16
	}, tensors, tinyTokenizer() );

}

function createTinyQwenWeights() {

	const hidden = 8;
	const inner = 8;
	const heads = 2;
	const kvHeads = 1;
	const headDim = 4;
	const layers = 2;
	const vocab = 8;
	const qSize = heads * headDim;
	const kvSize = kvHeads * headDim;
	const linHeads = 2;
	const linDim = 4;
	const kernel = 4;
	const convDim = linHeads * linDim * 2 + linHeads * linDim;
	const tensors = {
		'model.language_model.embed_tokens.weight': makeTensor( 'embed', [ vocab, hidden ], 0.12 ),
		'model.language_model.norm.weight': makeTensor( 'norm', [ hidden ], 1.12 )
	};

	const linearPrefix = 'model.language_model.layers.0';
	tensors[ `${ linearPrefix }.input_layernorm.weight` ] = makeTensor( 'ln1', [ hidden ], 2.1 );
	tensors[ `${ linearPrefix }.post_attention_layernorm.weight` ] = makeTensor( 'ln2', [ hidden ], 2.2 );
	tensors[ `${ linearPrefix }.mlp.gate_proj.weight` ] = makeTensor( 'gate', [ inner, hidden ], 8.1 );
	tensors[ `${ linearPrefix }.mlp.up_proj.weight` ] = makeTensor( 'up', [ inner, hidden ], 8.2 );
	tensors[ `${ linearPrefix }.mlp.down_proj.weight` ] = makeTensor( 'down', [ hidden, inner ], 8.3 );
	tensors[ `${ linearPrefix }.linear_attn.in_proj_qkv.weight` ] = makeTensor( 'dqkv', [ convDim, hidden ], 3.1 );
	tensors[ `${ linearPrefix }.linear_attn.in_proj_z.weight` ] = makeTensor( 'dz', [ linHeads * linDim, hidden ], 3.2 );
	tensors[ `${ linearPrefix }.linear_attn.in_proj_b.weight` ] = makeTensor( 'db', [ linHeads, hidden ], 3.3 );
	tensors[ `${ linearPrefix }.linear_attn.in_proj_a.weight` ] = makeTensor( 'da', [ linHeads, hidden ], 3.4 );
	tensors[ `${ linearPrefix }.linear_attn.out_proj.weight` ] = makeTensor( 'do', [ hidden, linHeads * linDim ], 3.5 );
	tensors[ `${ linearPrefix }.linear_attn.conv1d.weight` ] = makeTensor( 'dc', [ convDim, 1, kernel ], 3.6 );
	tensors[ `${ linearPrefix }.linear_attn.A_log` ] = makeTensor( 'alog', [ linHeads ], 0.4 );
	tensors[ `${ linearPrefix }.linear_attn.dt_bias` ] = makeTensor( 'dt', [ linHeads ], 0.2 );
	tensors[ `${ linearPrefix }.linear_attn.norm.weight` ] = makeTensor( 'dn', [ linDim ], 1.05 );

	const fullPrefix = 'model.language_model.layers.1';
	tensors[ `${ fullPrefix }.input_layernorm.weight` ] = makeTensor( 'fln1', [ hidden ], 2.3 );
	tensors[ `${ fullPrefix }.post_attention_layernorm.weight` ] = makeTensor( 'fln2', [ hidden ], 2.4 );
	tensors[ `${ fullPrefix }.mlp.gate_proj.weight` ] = makeTensor( 'fgate', [ inner, hidden ], 8.4 );
	tensors[ `${ fullPrefix }.mlp.up_proj.weight` ] = makeTensor( 'fup', [ inner, hidden ], 8.5 );
	tensors[ `${ fullPrefix }.mlp.down_proj.weight` ] = makeTensor( 'fdown', [ hidden, inner ], 8.6 );
	tensors[ `${ fullPrefix }.self_attn.q_proj.weight` ] = makeTensor( 'q', [ qSize * 2, hidden ], 4.1 );
	tensors[ `${ fullPrefix }.self_attn.k_proj.weight` ] = makeTensor( 'k', [ kvSize, hidden ], 4.2 );
	tensors[ `${ fullPrefix }.self_attn.v_proj.weight` ] = makeTensor( 'v', [ kvSize, hidden ], 4.3 );
	tensors[ `${ fullPrefix }.self_attn.o_proj.weight` ] = makeTensor( 'o', [ hidden, qSize ], 4.4 );
	tensors[ `${ fullPrefix }.self_attn.q_norm.weight` ] = makeTensor( 'qn', [ headDim ], 0.3 );
	tensors[ `${ fullPrefix }.self_attn.k_norm.weight` ] = makeTensor( 'kn', [ headDim ], 0.4 );

	return new QwenWeights( {
		model_type: 'qwen3_5_text',
		hidden_size: hidden,
		intermediate_size: inner,
		num_hidden_layers: layers,
		num_attention_heads: heads,
		num_key_value_heads: kvHeads,
		head_dim: headDim,
		vocab_size: vocab,
		hidden_act: 'silu',
		rms_norm_eps: 1e-6,
		layer_types: [ 'linear_attention', 'full_attention' ],
		linear_conv_kernel_dim: kernel,
		linear_key_head_dim: linDim,
		linear_value_head_dim: linDim,
		linear_num_key_heads: linHeads,
		linear_num_value_heads: linHeads,
		rope_parameters: { rope_theta: 10000, partial_rotary_factor: 0.5 },
		tie_word_embeddings: true,
		eos_token_id: 0,
		max_position_embeddings: 16
	}, tensors, tinyTokenizer() );

}

async function assertCausalSequence( assert, renderer, sequence, options, epsilon = 1e-4 ) {

	const { headCount, maxTokens, workgroupSize } = options;
	const headDim = options.headDim;
	const kvHeadCount = options.kvHeadCount || headCount;
	const qSize = headCount * headDim;
	const kvSize = kvHeadCount * headDim;
	const qkvBuffer = new Float32Array( qSize + 2 * kvSize );
	const { attribute, node } = storageFromArray( qkvBuffer );
	const layer = new TSLAttention( node, qSize, headCount, maxTokens, {
		headDim,
		kvHeadCount,
		ropeTheta: options.ropeTheta || 0,
		rotaryDim: options.rotaryDim,
		slidingWindow: options.slidingWindow || 0,
		attnScale: options.attnScale,
		qNormWeight: options.qNormWeight,
		kNormWeight: options.kNormWeight,
		rmsEpsilon: options.rmsEpsilon,
		offsetRMSNorm: options.offsetRMSNorm,
		workgroupSize
	} );
	const keyCache = new Float32Array( kvSize * maxTokens );
	const valueCache = new Float32Array( kvSize * maxTokens );

	for ( let position = 0; position < sequence.length; position ++ ) {

		qkvBuffer.set( sequence[ position ] );
		attribute.needsUpdate = true;
		layer.compute( renderer, position );

		const expected = causalAttention( qkvBuffer.slice(), {
			headCount,
			headDim,
			kvHeadCount,
			position,
			keyCache,
			valueCache,
			ropeTheta: options.ropeTheta || 0,
			rotaryDim: options.rotaryDim !== undefined ? options.rotaryDim : headDim,
			slidingWindow: options.slidingWindow || 0,
			attnScale: options.attnScale,
			qNormWeight: options.qNormWeight || null,
			kNormWeight: options.kNormWeight || null,
			rmsEpsilon: options.rmsEpsilon,
			offsetRMSNorm: options.offsetRMSNorm === true
		} );

		closeArray( assert, await readOutput( renderer, layer ), expected, epsilon, `causal attention token ${ position }` );

	}

	renderer.dispose();

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

const SMOLLM2_ROOT = '/examples/models/llm/smollm2-135m/';
const GEMMA3_ROOT = '/examples/models/llm/gemma-3-270m/';
const QWEN35_ROOT = '/examples/models/llm/qwen3.5-0.8b/';
const PHI15_ROOT = '/examples/models/llm/phi-1.5/';
const STORY_PROMPT = 'Once upon a time,';
const GREEDY = { maxNewTokens: 8, temperature: 0, topK: 1 };
const GREEDY_SHORT = { maxNewTokens: 4, temperature: 0, topK: 1 };
const PHI15_GREEDY_TEXT = 'Once upon a time, in a small town called Sunnyville,';
const localCheckpoints = new Map();

async function localCheckpointReady( assert, root ) {

	try {

		const configResponse = await fetch( `${ root }config.json` );
		if ( configResponse.ok === false ) {

			assert.ok( true, `SKIPPED: no config.json at ${ root }` );
			return false;

		}

		const weightsResponse = await fetch( `${ root }model.safetensors`, { method: 'HEAD' } );
		if ( weightsResponse.ok ) return true;

		const indexResponse = await fetch( `${ root }model.safetensors.index.json`, { method: 'HEAD' } );
		if ( indexResponse.ok ) return true;

		assert.ok( true, `SKIPPED: no model.safetensors at ${ root }` );
		return false;

	} catch ( error ) {

		assert.ok( true, `SKIPPED: ${ error.message }` );
		return false;

	}

}

async function loadLocalCheckpoint( assert, Loader, root ) {

	if ( localCheckpoints.has( root ) ) return localCheckpoints.get( root );

	if ( await localCheckpointReady( assert, root ) === false ) {

		localCheckpoints.set( root, null );
		return null;

	}

	const loaded = Loader.fromURL( root );
	localCheckpoints.set( root, loaded );
	return loaded;

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

			QUnit.test( 'GPT2Tokenizer keeps hash-character BPE merges', ( assert ) => {

				const tokenizer = new GPT2Tokenizer( {
					'a': 0,
					'#': 1,
					'##': 2,
					'a#': 3,
					'<|endoftext|>': 4
				}, [
					'#version: 0.2',
					'# #',
					'a #'
				] );

				assert.strictEqual( tokenizer.bpe( '##' ), '##', 'hash pairs still merge after the version header' );
				assert.strictEqual( tokenizer.bpe( 'a#' ), 'a#', 'later merges keep their ranks' );
				assert.deepEqual( tokenizer.encode( '##' ), [ 2 ] );

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

			QUnit.test( 'GPT2Tokenizer encodes added special tokens as whole ids', ( assert ) => {

				const tokenizer = new GPT2Tokenizer( {
					'a': 0,
					'<|endoftext|>': 1
				}, [], {
					addedTokens: [
						{ id: 10, content: '<|im_start|>' },
						{ id: 11, content: '<think>' },
						{ id: 12, content: '</think>' }
					]
				} );

				assert.deepEqual( tokenizer.encode( '<|im_start|><think></think>' ), [ 10, 11, 12 ], 'added tokens are not BPE-split' );
				assert.strictEqual( tokenizer.decode( [ 10, 11, 12 ] ), '<|im_start|><think></think>', 'added tokens decode to their content' );

			} );

			QUnit.test( 'QwenWeights formats chat with thinking disabled by default', ( assert ) => {

				const weights = createTinyQwenWeights();
				const messages = [ { role: 'user', text: 'Hi' } ];

				assert.strictEqual(
					weights.formatChat( messages ),
					'<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',
					'default chat prompt closes an empty think block'
				);
				assert.strictEqual(
					weights.formatChat( messages, { enableThinking: true } ),
					'<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n',
					'thinking mode leaves the think block open'
				);

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
				assert.strictEqual( sampleTopK( new Float32Array( [ 5, 4 ] ), { temperature: 0, tokens: [ 0 ], repetitionPenalty: 2 } ), 1, 'repetition penalty downweights seen tokens' );
				assert.strictEqual( sampleTopK( new Float32Array( [ 3, 2, 1 ] ), { temperature: 0, tokens: [ 0, 0 ], frequencyPenalty: 1 } ), 1, 'frequency penalty scales with token count' );
				assert.strictEqual( sampleTopK( new Float32Array( [ 0, 0, 10, 1 ] ), {
					temperature: 0,
					tokens: [ 0, 1, 2, 0, 1 ],
					noRepeatNgramSize: 3
				} ), 3, 'no-repeat n-grams ban the completing token' );
				assert.strictEqual( sampleTopK( new Float32Array( [ 5, 4 ] ), { temperature: 0, tokens: [ 0 ], repetitionPenalty: 1 } ), 0, 'repetitionPenalty=1 leaves logits unchanged' );

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

				const rms = rmsNorm( new Float32Array( [ 1, 2, 3 ] ), new Float32Array( [ 1, 1, 1 ] ) );
				const invRms = 1 / Math.sqrt( 14 / 3 + 1e-5 );
				closeArray( assert, rms, new Float32Array( [ invRms, 2 * invRms, 3 * invRms ] ), 1e-5, 'rmsNorm' );

				const gemmaRms = rmsNorm( new Float32Array( [ 1, 2, 3 ] ), new Float32Array( [ 0.5, 0, - 0.25 ] ), 1e-5, true );
				closeArray( assert, gemmaRms, new Float32Array( [
					invRms * 1.5,
					2 * invRms,
					3 * invRms * 0.75
				] ), 1e-5, 'gemma rmsNorm uses 1 + weight' );

				assert.ok( Math.abs( silu( 0 ) ) < 1e-6, 'silu(0) == 0' );
				assert.ok( Math.abs( silu( 1 ) - 0.731058578 ) < 1e-6, 'silu(1) matches 1/(1+e^-1)' );

				const rope = applyRoPE( new Float32Array( [ 1, 0, 0, 1 ] ), 0, 4, 1, 10000 );
				assert.ok( Math.abs( rope[ 0 ] - Math.cos( 1 ) ) < 1e-5, 'RoPE rotates the first pair using theta^0' );

				assert.strictEqual( architectureFor( { model_type: 'gpt2' } ), 'gpt2' );
				assert.strictEqual( architectureFor( { model_type: 'llama' } ), 'llama' );
				assert.strictEqual( architectureFor( { model_type: 'phi' } ), 'phi' );
				assert.strictEqual( architectureFor( { model_type: 'gemma3_text' } ), 'gemma3' );
				assert.strictEqual( architectureFor( { model_type: 'qwen3_5' } ), 'qwen3_5' );
				assert.strictEqual( architectureFor( { model_type: 'qwen3_5', text_config: { model_type: 'qwen3_5_text' } } ), 'qwen3_5' );
				assert.throws( () => architectureFor( { model_type: 'gemma4' } ), /Unsupported model_type "gemma4"/ );

				const mapped = resolveTensor( {
					'model.layers.0.self_attn.q_proj.weight': { name: 'q' }
				}, 'model.', 'llama', 'attn_q', 0 );
				assert.strictEqual( mapped.name, 'q', 'TensorNameMap resolves Llama HF q_proj aliases' );

				const capped = logitSoftcap( new Float32Array( [ 60, - 60, 0 ] ), 30 );
				assert.ok( Math.abs( capped[ 0 ] - 30 * Math.tanh( 2 ) ) < 1e-5, 'logit softcap saturates large values' );
				assert.ok( Math.abs( capped[ 2 ] ) < 1e-6, 'logit softcap leaves zero at zero' );

				const split = splitHeadGate( new Float32Array( [ 1, 2, 3, 4 ] ), 2, 1 );
				closeArray( assert, split.query, new Float32Array( [ 1, 3 ] ), 1e-6, 'splitHeadGate query' );
				closeArray( assert, split.gate, new Float32Array( [ 2, 4 ] ), 1e-6, 'splitHeadGate gate' );

				const deltaState = new Float32Array( 4 );
				const deltaOut = gatedDeltaRuleStep(
					new Float32Array( [ 1, 0 ] ),
					new Float32Array( [ 0, 1 ] ),
					new Float32Array( [ 2, 3 ] ),
					new Float32Array( [ 0.5 ] ),
					new Float32Array( [ 1 ] ),
					deltaState,
					{ numVHeads: 1, keyDim: 2, valueDim: 2 }
				);
				assert.strictEqual( deltaOut.length, 2, 'gated delta rule writes a value-sized vector' );

				assert.ok( Math.abs( float16ToFloat32( 0x3c00 ) - 1 ) < 1e-6, 'f16 1.0' );
				assert.ok( Math.abs( bfloat16ToFloat32( 0x3f80 ) - 1 ) < 1e-6, 'bf16 1.0' );
				assert.ok( Math.abs( tensorToFloat32( {
					name: 'w',
					dtype: 'BF16',
					data: new Uint16Array( [ 0x4000 ] )
				} )[ 0 ] - 2 ) < 1e-6, 'tensorToFloat32 BF16 2.0' );

			} );

			QUnit.test( 'convertAllTensors reports BF16 progress and rewrites tensors as F32', async ( assert ) => {

				const tensors = {
					small: { name: 'small', dtype: 'BF16', data: new Uint16Array( [ 0x3f80, 0x4000 ] ) },
					left: { name: 'left', dtype: 'F32', data: new Float32Array( [ 9 ] ) }
				};
				const messages = [];

				const count = await convertAllTensors( tensors, ( message ) => messages.push( message ), 'Test' );

				assert.strictEqual( count, 1 );
				assert.strictEqual( tensors.small.dtype, 'F32' );
				assert.ok( Math.abs( tensors.small.data[ 0 ] - 1 ) < 1e-6 );
				assert.ok( Math.abs( tensors.small.data[ 1 ] - 2 ) < 1e-6 );
				assert.strictEqual( tensors.left.dtype, 'F32' );
				assert.ok( messages.some( ( message ) => message.includes( 'Converting BF16' ) ), 'progress mentions BF16 conversion' );

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

				await assertAttentionSequence( assert, renderer, 4, 2, 4, [
					[ 1, 0, 0, 1, 1, 0, 0, 1, 2, 3, 4, 5 ],
					[ 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1 ]
				], 4 );

			} );

			QUnit.test( 'TSLAttention matches CPU reference over a longer cached sequence', async ( assert ) => {

				const renderer = await createRenderer( assert );
				if ( renderer === null ) return;

				await assertAttentionSequence( assert, renderer, 4, 2, 8, [
					[ 1, 0, 0, 1, 1, 0, 0, 1, 2, 3, 4, 5 ],
					[ 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1 ],
					[ 0.5, - 1, 2, 0, 0.25, 0.5, - 0.5, 1, 0, 2, - 1, 3 ],
					[ - 2, 1, 0.5, 0.5, 1, - 1, 0, 0.25, 4, 0, 1, - 2 ]
				], 64 );

			} );

			QUnit.test( 'TSLAttention matches CPU reference for GPT-2-sized heads', async ( assert ) => {

				const renderer = await createRenderer( assert );
				if ( renderer === null ) return;

				const hiddenSize = 128;
				const sequence = [];

				for ( let position = 0; position < 4; position ++ ) {

					const qkv = new Float32Array( hiddenSize * 3 );

					for ( let i = 0; i < qkv.length; i ++ ) qkv[ i ] = Math.sin( position * 19.1 + i * 0.17 ) * 0.35;

					sequence.push( qkv );

				}

				await assertAttentionSequence( assert, renderer, hiddenSize, 2, 8, sequence, 64, 2e-4 );

			} );

			QUnit.test( 'prepareGeneration keeps the prompt when new tokens fill the window', async ( assert ) => {

				const weights = await DecoderWeights.fromURL( '/examples/models/llm/tinystories-gpt2-0.1-3m/' );
				const prompt = 'Once upon a time,';
				const encoded = weights.tokenizer.encode( prompt );
				const { inputTokens, newTokenBudget } = weights.prepareGeneration( prompt, 8, 128 );

				assert.deepEqual( Array.from( inputTokens ), encoded, 'prompt is not sliced to make room for new tokens' );
				assert.strictEqual( newTokenBudget, 8 - encoded.length, 'generation uses leftover context slots' );

				const result = new DecoderCPURunner( weights, { maxTokens: 8 } ).generate( prompt, {
					maxNewTokens: 128,
					temperature: 0,
					topK: 1
				} );

				assert.ok( result.text.startsWith( prompt ), 'decoded output still starts with the prompt' );
				assert.strictEqual( result.generatedTokens.length, 8 - encoded.length );

			} );

			QUnit.test( 'CPU TinyStories greedy continuation stays on the prompt', async ( assert ) => {

				const weights = await DecoderWeights.fromURL( '/examples/models/llm/tinystories-gpt2-0.1-3m/' );
				const result = new DecoderCPURunner( weights ).generate( 'Once upon a time,', {
					maxNewTokens: 24,
					temperature: 0,
					topK: 1
				} );

				assert.strictEqual(
					result.text,
					'Once upon a time, there was a little girl named Lily. She loved to play with her toys. One day, she saw a big,'
				);

			} );

			QUnit.test( 'prompt cache reuses a matching prefix', ( assert ) => {

				assert.strictEqual( sharedPrefixLength( [ 1, 2, 3 ], [ 1, 2, 9 ] ), 2 );
				assert.strictEqual( sharedPrefixLength( [ 1, 2 ], [ 1, 2, 3 ] ), 2 );

				const append = planPromptCache( [ 1, 2, 3 ], new Float32Array( [ 0 ] ), [ 1, 2, 3, 4 ], true );
				assert.strictEqual( append.start, 3, 'append-only starts after the cached prefix' );
				assert.strictEqual( append.reset, false );
				assert.ok( append.logits !== null );

				const rewind = planPromptCache( [ 1, 2, 3, 9 ], new Float32Array( [ 0 ] ), [ 1, 2, 4 ], true );
				assert.strictEqual( rewind.start, 2, 'transformer cache can resume at the first mismatch' );
				assert.strictEqual( rewind.reset, false );

				const recurrent = planPromptCache( [ 1, 2, 3, 9 ], new Float32Array( [ 0 ] ), [ 1, 2, 4 ], false );
				assert.strictEqual( recurrent.reset, true, 'linear-attention cache cannot rewind' );

			} );

			QUnit.test( 'CPU TinyStories reuses the KV cache on a longer prompt', async ( assert ) => {

				const weights = await DecoderWeights.fromURL( '/examples/models/llm/tinystories-gpt2-0.1-3m/' );
				const options = { maxNewTokens: 8, temperature: 0, topK: 1 };
				const runner = new DecoderCPURunner( weights, { maxTokens: 64 } );
				const first = runner.generate( 'Once upon a time,', options );
				const continued = first.text + ' She';
				const second = runner.generate( continued, options );
				const fresh = new DecoderCPURunner( weights, { maxTokens: 64 } ).generate( continued, options );

				assert.strictEqual( second.text, fresh.text, 'cached continuation matches a cold run' );
				assert.ok( second.cachedPromptTokens > 0, 'second generate reused prompt tokens' );

				runner.resetCache();
				const afterReset = runner.generate( continued, options );
				assert.strictEqual( afterReset.cachedPromptTokens, 0, 'resetCache drops the prompt prefix' );
				assert.strictEqual( afterReset.text, fresh.text );

			} );

			QUnit.test( 'TSL TinyStories greedy continuation matches the CPU runner', async ( assert ) => {

				const renderer = await createRenderer( assert );
				if ( renderer === null ) return;

				const weights = await DecoderWeights.fromURL( '/examples/models/llm/tinystories-gpt2-0.1-3m/' );
				const prompt = 'Once upon a time,';
				const options = { maxNewTokens: 8, temperature: 0, topK: 1 };
				const cpu = new DecoderCPURunner( weights, { maxTokens: 128 } ).generate( prompt, options );
				const gpu = await new DecoderTSLRunner( weights, { maxTokens: 128 } ).generate( renderer, prompt, options );

				assert.strictEqual( gpu.text, cpu.text, 'GPU greedy text matches CPU' );
				assert.deepEqual( gpu.generatedTokens, cpu.generatedTokens, 'GPU token ids match CPU' );
				renderer.dispose();

			} );

			QUnit.test( 'CPU GPT-2 greedy continuation stays on the prompt', async ( assert ) => {

				assert.timeout( 60000 );

				const weights = await DecoderWeights.fromURL( '/examples/models/llm/gpt2/' );
				const result = new DecoderCPURunner( weights, { maxTokens: 32 } ).generate( 'Once upon a time,', {
					maxNewTokens: 8,
					temperature: 0,
					topK: 1
				} );

				assert.strictEqual( result.text, 'Once upon a time, the world was a place of great beauty' );

			} );

			QUnit.test( 'TSL GPT-2 greedy continuation matches the CPU runner', async ( assert ) => {

				const renderer = await createRenderer( assert );
				if ( renderer === null ) return;

				assert.timeout( 180000 );

				const weights = await DecoderWeights.fromURL( '/examples/models/llm/gpt2/' );
				const prompt = 'Once upon a time,';
				const options = { maxNewTokens: 8, temperature: 0, topK: 1 };
				const cpu = new DecoderCPURunner( weights, { maxTokens: 32 } ).generate( prompt, options );
				const gpu = await new DecoderTSLRunner( weights, { maxTokens: 32 } ).generate( renderer, prompt, options );

				assert.ok( cpu.text.startsWith( prompt ), 'CPU GPT-2 keeps the prompt' );
				assert.strictEqual( gpu.text, cpu.text, 'GPU GPT-2 greedy text matches CPU' );
				assert.deepEqual( gpu.generatedTokens, cpu.generatedTokens, 'GPU GPT-2 token ids match CPU' );
				renderer.dispose();

			} );

			QUnit.test( 'SmolLM2 loads Llama-style weights from the local checkpoint', async ( assert ) => {

				assert.timeout( 120000 );

				const weights = await loadLocalCheckpoint( assert, DecoderWeights, SMOLLM2_ROOT );
				if ( weights === null ) return;

				assert.strictEqual( architectureFor( weights.config ), 'llama' );
				assert.strictEqual( weights.architecture, 'llama' );
				assert.strictEqual( weights.hiddenSize, 576 );
				assert.strictEqual( weights.innerSize, 1536 );
				assert.strictEqual( weights.layerCount, 30 );
				assert.strictEqual( weights.headCount, 9 );
				assert.strictEqual( weights.kvHeadCount, 3 );
				assert.strictEqual( weights.headDim, 64 );
				assert.strictEqual( weights.qSize, 576 );
				assert.strictEqual( weights.kvSize, 192 );
				assert.strictEqual( weights.vocabSize, 49152 );
				assert.strictEqual( weights.ropeTheta, 100000 );
				assert.strictEqual( weights.mlpActivation, 'silu' );
				assert.strictEqual( weights.endOfTextTokenId, 0 );

				const promptIds = weights.tokenizer.encode( STORY_PROMPT );
				assert.deepEqual( promptIds, [ 6403, 1980, 253, 655, 28 ], 'SmolLM2 encodes the story prompt' );
				assert.strictEqual( weights.tokenizer.decode( promptIds ), STORY_PROMPT, 'SmolLM2 BPE round-trips the prompt' );
				assert.ok( weights.hasTensor( 'layers.0.self_attn.q_proj.weight' ), 'layer 0 Q projection is present' );

			} );

			QUnit.test( 'CPU SmolLM2 greedy continuation stays on the prompt', async ( assert ) => {

				assert.timeout( 180000 );

				const weights = await loadLocalCheckpoint( assert, DecoderWeights, SMOLLM2_ROOT );
				if ( weights === null ) return;

				const result = new DecoderCPURunner( weights, { maxTokens: 32 } ).generate( STORY_PROMPT, GREEDY );

				assert.ok( result.text.startsWith( STORY_PROMPT ), 'decoded SmolLM2 output still starts with the prompt' );
				assert.strictEqual( result.generatedTokens.length, 8, 'SmolLM2 emits eight greedy tokens' );
				assert.strictEqual(
					result.text,
					'Once upon a time, there was a little girl named Lily.'
				);

			} );

			QUnit.test( 'TSL SmolLM2 greedy continuation matches the CPU runner', async ( assert ) => {

				const renderer = await createRenderer( assert );
				if ( renderer === null ) return;

				assert.timeout( 180000 );

				const weights = await loadLocalCheckpoint( assert, DecoderWeights, SMOLLM2_ROOT );
				if ( weights === null ) {

					renderer.dispose();
					return;

				}

				const cpu = new DecoderCPURunner( weights, { maxTokens: 32 } ).generate( STORY_PROMPT, GREEDY );
				const gpu = await new DecoderTSLRunner( weights, { maxTokens: 32 } ).generate( renderer, STORY_PROMPT, GREEDY );

				assert.ok( cpu.text.startsWith( STORY_PROMPT ), 'CPU SmolLM2 keeps the prompt' );
				assert.strictEqual( gpu.text, cpu.text, 'GPU SmolLM2 greedy text matches CPU' );
				assert.deepEqual( gpu.generatedTokens, cpu.generatedTokens, 'GPU SmolLM2 token ids match CPU' );
				renderer.dispose();

			} );

			QUnit.test( 'Gemma 3 270M loads Gemma-style weights from the local checkpoint', async ( assert ) => {

				assert.timeout( 180000 );

				const weights = await loadLocalCheckpoint( assert, DecoderWeights, GEMMA3_ROOT );
				if ( weights === null ) return;

				assert.strictEqual( architectureFor( weights.config ), 'gemma3' );
				assert.strictEqual( weights.architecture, 'gemma3' );
				assert.strictEqual( weights.hiddenSize, 640 );
				assert.strictEqual( weights.innerSize, 2048 );
				assert.strictEqual( weights.layerCount, 18 );
				assert.strictEqual( weights.headCount, 4 );
				assert.strictEqual( weights.kvHeadCount, 1 );
				assert.strictEqual( weights.headDim, 256 );
				assert.strictEqual( weights.qSize, 1024 );
				assert.strictEqual( weights.kvSize, 256 );
				assert.strictEqual( weights.vocabSize, 262144 );
				assert.strictEqual( weights.globalRopeTheta, 1000000 );
				assert.strictEqual( weights.localRopeTheta, 10000 );
				assert.strictEqual( weights.slidingWindow, 512 );
				assert.strictEqual( weights.mlpActivation, 'gelu_pytorch_tanh' );
				assert.strictEqual( weights.endOfTextTokenId, 1 );
				assert.ok( Math.abs( weights.embedScale - Math.sqrt( 640 ) ) < 1e-6, 'embeddings are scaled by sqrt(hidden)' );
				assert.ok( Math.abs( weights.attnScale - 1 / 16 ) < 1e-6, 'attention scale is 1/sqrt(head_dim)' );
				assert.strictEqual( weights.block( 0 ).slidingWindow, 512, 'first layer is sliding-window' );
				assert.strictEqual( weights.block( 5 ).slidingWindow, 0, 'every sixth layer is global' );

				const promptIds = weights.tokenizer.encode( STORY_PROMPT );
				assert.deepEqual( promptIds, [ 2, 14946, 3324, 496, 990, 236764 ], 'Gemma encode prepends BOS and uses SentencePiece BPE' );
				assert.strictEqual( weights.tokenizer.decode( promptIds ), STORY_PROMPT, 'Gemma unigram round-trips the prompt' );
				assert.ok( weights.hasTensor( 'layers.0.self_attn.q_norm.weight' ), 'layer 0 QK-norm is present' );

			} );

			QUnit.test( 'CPU Gemma 3 270M greedy continuation stays on the prompt', async ( assert ) => {

				assert.timeout( 180000 );

				const weights = await loadLocalCheckpoint( assert, DecoderWeights, GEMMA3_ROOT );
				if ( weights === null ) return;

				const result = new DecoderCPURunner( weights, { maxTokens: 32 } ).generate( STORY_PROMPT, GREEDY );

				assert.ok( result.text.startsWith( STORY_PROMPT ), 'decoded Gemma output still starts with the prompt' );
				assert.strictEqual( result.generatedTokens.length, 8, 'Gemma emits eight greedy tokens' );
				assert.strictEqual(
					result.text,
					'Once upon a time, there was a man named John. He'
				);

			} );

			QUnit.test( 'TSL Gemma 3 270M greedy continuation matches the CPU runner', async ( assert ) => {

				const renderer = await createRenderer( assert );
				if ( renderer === null ) return;

				assert.timeout( 300000 );

				const weights = await loadLocalCheckpoint( assert, DecoderWeights, GEMMA3_ROOT );
				if ( weights === null ) {

					renderer.dispose();
					return;

				}

				const cpu = new DecoderCPURunner( weights, { maxTokens: 32 } ).generate( STORY_PROMPT, GREEDY );
				const gpu = await new DecoderTSLRunner( weights, { maxTokens: 32 } ).generate( renderer, STORY_PROMPT, GREEDY );

				assert.ok( cpu.text.startsWith( STORY_PROMPT ), 'CPU Gemma keeps the prompt' );
				assert.strictEqual( gpu.text, cpu.text, 'GPU Gemma greedy text matches CPU' );
				assert.deepEqual( gpu.generatedTokens, cpu.generatedTokens, 'GPU Gemma token ids match CPU' );
				renderer.dispose();

			} );

			QUnit.test( 'TSLRMSNorm matches CPU rmsNorm', async ( assert ) => {

				const renderer = await createRenderer( assert );
				if ( renderer === null ) return;

				const input = new Float32Array( [ 1, 2, 3, - 1 ] );
				const weight = new Float32Array( [ 2, 0.5, 1, 1.5 ] );
				const layer = new TSLRMSNorm( storageFromArray( input ).node, weight, 4, { workgroupSize: 4 } );

				layer.compute( renderer );

				closeArray( assert, await readOutput( renderer, layer ), rmsNorm( input, weight ), 1e-5, 'TSLRMSNorm' );
				renderer.dispose();

			} );

			QUnit.test( 'TSLSiLUMul matches silu(gate) * up', async ( assert ) => {

				const renderer = await createRenderer( assert );
				if ( renderer === null ) return;

				const gate = new Float32Array( [ - 1, 0, 1, 2 ] );
				const up = new Float32Array( [ 0.5, - 2, 3, 0.25 ] );
				const expected = new Float32Array( gate.map( ( value, i ) => silu( value ) * up[ i ] ) );
				const layer = new TSLSiLUMul( storageFromArray( gate ).node, storageFromArray( up ).node, 4, { workgroupSize: 4 } );

				layer.compute( renderer );

				closeArray( assert, await readOutput( renderer, layer ), expected, 1e-5, 'TSLSiLUMul' );
				renderer.dispose();

			} );

			QUnit.test( 'TSLGatedMLP matches CPU SwiGLU', async ( assert ) => {

				const renderer = await createRenderer( assert );
				if ( renderer === null ) return;

				const input = new Float32Array( [ 1, - 0.5 ] );
				const gateWeight = new Float32Array( [ 0.5, - 0.25, 1, 0.75, 0.5, - 1 ] );
				const upWeight = new Float32Array( [ 1, 0, 0, 1, 0.5, - 0.5 ] );
				const downWeight = new Float32Array( [ 1, 0, 0, 0.25, - 0.5, 0.5 ] );
				const gate = linear( input, gateWeight, null, 2, 3 );
				const up = linear( input, upWeight, null, 2, 3 );
				const hidden = new Float32Array( 3 );
				for ( let i = 0; i < 3; i ++ ) hidden[ i ] = silu( gate[ i ] ) * up[ i ];
				const expected = linear( hidden, downWeight, null, 3, 2 );
				const layer = new TSLGatedMLP(
					storageFromArray( input ).node,
					gateWeight,
					upWeight,
					downWeight,
					2,
					3,
					{ workgroupSize: 3 }
				);

				layer.compute( renderer );

				closeArray( assert, await readOutput( renderer, layer.down ), expected, 1e-4, 'TSLGatedMLP SwiGLU' );
				renderer.dispose();

			} );

			QUnit.test( 'TSLAttention matches GQA without RoPE', async ( assert ) => {

				const renderer = await createRenderer( assert );
				if ( renderer === null ) return;

				await assertCausalSequence( assert, renderer, [
					fillSin( new Float32Array( 16 ), 0.3 ),
					fillSin( new Float32Array( 16 ), 1.1 )
				], { headCount: 4, kvHeadCount: 2, headDim: 2, maxTokens: 4, workgroupSize: 16 } );

			} );

			QUnit.test( 'TSLAttention matches RoPE multi-head attention', async ( assert ) => {

				const renderer = await createRenderer( assert );
				if ( renderer === null ) return;

				await assertCausalSequence( assert, renderer, [
					fillSin( new Float32Array( 12 ), 0.2 ),
					fillSin( new Float32Array( 12 ), 0.8 ),
					fillSin( new Float32Array( 12 ), 1.4 )
				], { headCount: 2, kvHeadCount: 2, headDim: 2, maxTokens: 8, ropeTheta: 10000, workgroupSize: 16 } );

			} );

			QUnit.test( 'TSLAttention matches grouped-query RoPE', async ( assert ) => {

				const renderer = await createRenderer( assert );
				if ( renderer === null ) return;

				await assertCausalSequence( assert, renderer, [
					fillSin( new Float32Array( 16 ), 0.5 ),
					fillSin( new Float32Array( 16 ), 1.5 )
				], { headCount: 4, kvHeadCount: 2, headDim: 2, maxTokens: 4, ropeTheta: 10000, rotaryDim: 2, workgroupSize: 16 } );

			} );

			QUnit.test( 'tiny Llama greedy GPU matches CPU', async ( assert ) => {

				const renderer = await createRenderer( assert );
				if ( renderer === null ) return;

				const weights = createTinyLlama();
				const options = { maxNewTokens: 4, temperature: 0, topK: 1 };
				const cpu = new DecoderCPURunner( weights, { maxTokens: 8 } ).generate( 'hello', options );
				const gpu = await new DecoderTSLRunner( weights, { maxTokens: 8 } ).generate( renderer, 'hello', options );

				assert.deepEqual( gpu.generatedTokens, cpu.generatedTokens, 'tiny Llama GPU tokens match CPU' );
				assert.strictEqual( gpu.text, cpu.text );
				renderer.dispose();

			} );

			QUnit.test( 'tiny Phi greedy GPU matches CPU', async ( assert ) => {

				const renderer = await createRenderer( assert );
				if ( renderer === null ) return;

				const weights = createTinyPhi();
				const options = { maxNewTokens: 4, temperature: 0, topK: 1 };
				const cpu = new DecoderCPURunner( weights, { maxTokens: 8 } ).generate( 'hello', options );
				const gpu = await new DecoderTSLRunner( weights, { maxTokens: 8 } ).generate( renderer, 'hello', options );

				assert.deepEqual( gpu.generatedTokens, cpu.generatedTokens, 'tiny Phi GPU tokens match CPU' );
				assert.strictEqual( gpu.text, cpu.text );
				renderer.dispose();

			} );

			QUnit.test( 'Phi-1.5 loads Phi-style weights from the local checkpoint', async ( assert ) => {

				assert.timeout( 180000 );

				const weights = await loadLocalCheckpoint( assert, DecoderWeights, PHI15_ROOT );
				if ( weights === null ) return;

				assert.strictEqual( architectureFor( weights.config ), 'phi' );
				assert.strictEqual( weights.architecture, 'phi' );
				assert.strictEqual( weights.hiddenSize, 2048 );
				assert.strictEqual( weights.innerSize, 8192 );
				assert.strictEqual( weights.layerCount, 24 );
				assert.strictEqual( weights.headCount, 32 );
				assert.strictEqual( weights.kvHeadCount, 32 );
				assert.strictEqual( weights.headDim, 64 );
				assert.strictEqual( weights.qSize, 2048 );
				assert.strictEqual( weights.kvSize, 2048 );
				assert.strictEqual( weights.vocabSize, 51200 );
				assert.strictEqual( weights.ropeTheta, 10000 );
				assert.strictEqual( weights.rotaryDim, 32 );
				assert.strictEqual( weights.endOfTextTokenId, 50256 );

				const promptIds = weights.tokenizer.encode( STORY_PROMPT );
				assert.deepEqual( promptIds, [ 7454, 2402, 257, 640, 11 ], 'Phi-1.5 encodes the story prompt with CodeGen BPE' );
				assert.strictEqual( weights.tokenizer.decode( promptIds ), STORY_PROMPT, 'Phi-1.5 BPE round-trips the prompt' );
				assert.ok( weights.tensors[ 'model.layers.0.self_attn.q_proj.weight' ], 'layer 0 Q projection is present' );
				assert.ok( weights.tensors[ 'lm_head.weight' ], 'untied LM head is present' );

			} );

			QUnit.test( 'CPU Phi-1.5 greedy continuation stays on the prompt', async ( assert ) => {

				assert.timeout( 300000 );

				const weights = await loadLocalCheckpoint( assert, DecoderWeights, PHI15_ROOT );
				if ( weights === null ) return;

				const result = new DecoderCPURunner( weights, { maxTokens: 32 } ).generate( STORY_PROMPT, GREEDY );

				assert.ok( result.text.startsWith( STORY_PROMPT ), 'decoded Phi-1.5 output still starts with the prompt' );
				assert.strictEqual( result.generatedTokens.length, 8, 'Phi-1.5 emits eight greedy tokens' );
				assert.strictEqual(
					result.text,
					PHI15_GREEDY_TEXT
				);

			} );

			QUnit.test( 'TSL Phi-1.5 greedy continuation matches the CPU runner', async ( assert ) => {

				const renderer = await createRenderer( assert );
				if ( renderer === null ) return;

				assert.timeout( 300000 );

				const weights = await loadLocalCheckpoint( assert, DecoderWeights, PHI15_ROOT );
				if ( weights === null ) {

					renderer.dispose();
					return;

				}

				const cpu = new DecoderCPURunner( weights, { maxTokens: 32 } ).generate( STORY_PROMPT, GREEDY );
				const gpu = await new DecoderTSLRunner( weights, { maxTokens: 32 } ).generate( renderer, STORY_PROMPT, GREEDY );

				assert.ok( cpu.text.startsWith( STORY_PROMPT ), 'CPU Phi-1.5 keeps the prompt' );
				assert.strictEqual( gpu.text, cpu.text, 'GPU Phi-1.5 greedy text matches CPU' );
				assert.deepEqual( gpu.generatedTokens, cpu.generatedTokens, 'GPU Phi-1.5 token ids match CPU' );
				renderer.dispose();

			} );

			QUnit.test( 'UnigramTokenizer encodes Hugging Face BPE vocabs', ( assert ) => {

				const tokenizer = new UnigramTokenizer( {
					model: {
						type: 'BPE',
						byte_fallback: true,
						unk_token: '<unk>',
						vocab: {
							'<unk>': 0,
							'<eos>': 1,
							'<bos>': 2,
							'▁': 3,
							'h': 4,
							'e': 5,
							'l': 6,
							'o': 7,
							'he': 8,
							'▁he': 9,
							'll': 10,
							'o▁': 11
						},
						merges: [
							[ 'h', 'e' ],
							[ 'l', 'l' ],
							[ '▁', 'he' ]
						]
					}
				}, { bos_token_id: 2, eos_token_id: 1, add_bos_token: true } );

				assert.strictEqual( tokenizer.useBpe, true, 'dict vocab selects BPE' );
				assert.deepEqual( tokenizer.encode( 'he' ), [ 2, 8 ], 'BPE merges letters and prepends BOS' );
				assert.strictEqual( tokenizer.decode( [ 2, 9, 10 ] ), 'hell', 'BPE decode skips BOS and restores text' );

			} );

			QUnit.test( 'UnigramTokenizer round-trips metaspace text and prepends BOS', ( assert ) => {

				const tokenizer = new UnigramTokenizer( {
					model: {
						type: 'Unigram',
						unk_id: 0,
						vocab: [
							[ '<unk>', 0 ],
							[ '<eos>', 0 ],
							[ '<bos>', 0 ],
							[ '▁hello', - 1 ],
							[ '▁world', - 2 ],
							[ '▁', - 4 ]
						]
					}
				}, { bos_token_id: 2, eos_token_id: 1, add_bos_token: true } );

				assert.deepEqual( tokenizer.encode( 'hello world' ), [ 2, 3, 4 ], 'encode prepends BOS and splits on metaspace' );
				assert.strictEqual( tokenizer.decode( [ 2, 3, 4 ] ), 'hello world', 'decode skips BOS and restores spaces' );

			} );

			QUnit.test( 'TSLAttention matches sliding-window GQA', async ( assert ) => {

				const renderer = await createRenderer( assert );
				if ( renderer === null ) return;

				await assertCausalSequence( assert, renderer, [
					fillSin( new Float32Array( 16 ), 0.4 ),
					fillSin( new Float32Array( 16 ), 1.4 ),
					fillSin( new Float32Array( 16 ), 2.4 )
				], { headCount: 4, kvHeadCount: 2, headDim: 2, maxTokens: 8, slidingWindow: 2, workgroupSize: 16 } );

			} );

			QUnit.test( 'TSLAttention matches QK-norm and RoPE', async ( assert ) => {

				const renderer = await createRenderer( assert );
				if ( renderer === null ) return;

				await assertCausalSequence( assert, renderer, [
					fillSin( new Float32Array( 16 ), 0.7 ),
					fillSin( new Float32Array( 16 ), 1.7 )
				], {
					headCount: 2,
					kvHeadCount: 1,
					headDim: 4,
					maxTokens: 4,
					ropeTheta: 10000,
					qNormWeight: new Float32Array( [ 0.5, - 0.25, 0.1, 0 ] ),
					kNormWeight: new Float32Array( [ 0.2, 0.3, - 0.1, 0.4 ] ),
					offsetRMSNorm: true,
					rmsEpsilon: 1e-6,
					workgroupSize: 16
				} );

			} );

			QUnit.test( 'tiny Gemma 3 greedy GPU matches CPU', async ( assert ) => {

				const renderer = await createRenderer( assert );
				if ( renderer === null ) return;

				const weights = createTinyGemma();
				const options = { maxNewTokens: 4, temperature: 0, topK: 1 };
				const cpu = new DecoderCPURunner( weights, { maxTokens: 8 } ).generate( 'hello', options );
				const gpu = await new DecoderTSLRunner( weights, { maxTokens: 8 } ).generate( renderer, 'hello', options );

				assert.strictEqual( weights.block( 0 ).slidingWindow, 2, 'first layer is sliding-window' );
				assert.strictEqual( weights.block( 1 ).slidingWindow, 0, 'second layer is global' );
				assert.deepEqual( gpu.generatedTokens, cpu.generatedTokens, 'tiny Gemma GPU tokens match CPU' );
				assert.strictEqual( gpu.text, cpu.text );
				renderer.dispose();

			} );

			QUnit.test( 'Qwen3.5 0.8B loads hybrid weights from the local checkpoint', async ( assert ) => {

				assert.timeout( 300000 );

				const weights = await loadLocalCheckpoint( assert, QwenWeights, QWEN35_ROOT );
				if ( weights === null ) return;

				assert.strictEqual( architectureFor( weights.config ), 'qwen3_5' );
				assert.strictEqual( weights.architecture, 'qwen3_5' );
				assert.strictEqual( weights.hiddenSize, 1024 );
				assert.strictEqual( weights.innerSize, 3584 );
				assert.strictEqual( weights.layerCount, 24 );
				assert.strictEqual( weights.headCount, 8 );
				assert.strictEqual( weights.kvHeadCount, 2 );
				assert.strictEqual( weights.headDim, 256 );
				assert.strictEqual( weights.qSize, 2048 );
				assert.strictEqual( weights.kvSize, 512 );
				assert.strictEqual( weights.vocabSize, 248320 );
				assert.strictEqual( weights.linearKeyHeads, 16 );
				assert.strictEqual( weights.linearValueHeads, 16 );
				assert.strictEqual( weights.linearKeyDim, 128 );
				assert.strictEqual( weights.rotaryDim, 64 );
				assert.strictEqual( weights.mlpActivation, 'silu' );
				assert.strictEqual( weights.block( 0 ).layerType, 'linear_attention', 'first layer is Gated DeltaNet' );
				assert.strictEqual( weights.block( 3 ).layerType, 'full_attention', 'every fourth layer is gated attention' );
				assert.ok( weights.hasTensor( 'layers.0.linear_attn.in_proj_qkv.weight' ), 'linear attention QKV is present' );
				assert.ok( weights.hasTensor( 'layers.3.self_attn.q_norm.weight' ), 'full attention QK-norm is present' );

				const promptIds = weights.tokenizer.encode( STORY_PROMPT );
				assert.deepEqual( promptIds, [ 12162, 5028, 264, 854, 11 ], 'Qwen BPE matches the checkpoint tokenizer' );
				assert.strictEqual( weights.tokenizer.decode( promptIds ), STORY_PROMPT, 'Qwen BPE round-trips the prompt' );

				const chat = weights.formatChat( [ { role: 'user', text: 'Hi' } ] );
				const chatIds = weights.tokenizer.encode( chat );
				assert.strictEqual( chatIds[ 0 ], 248045, 'chat prompt starts with im_start' );
				assert.ok( chatIds.includes( 248068 ), 'disabled thinking includes the open think token' );
				assert.ok( chatIds.includes( 248069 ), 'disabled thinking includes the close think token' );
				assert.strictEqual( weights.tokenizer.decode( chatIds ), chat, 'Qwen chat template round-trips' );
				assert.deepEqual( weights.stopTokenIds, [ 248044, 248046 ], 'Qwen stops on endoftext and im_end' );

			} );

			QUnit.test( 'CPU Qwen3.5 0.8B greedy continuation stays on the prompt', async ( assert ) => {

				assert.timeout( 300000 );

				const weights = await loadLocalCheckpoint( assert, QwenWeights, QWEN35_ROOT );
				if ( weights === null ) return;

				const result = new QwenCPURunner( weights, { maxTokens: 32 } ).generate( STORY_PROMPT, GREEDY_SHORT );

				assert.ok( result.text.startsWith( STORY_PROMPT ), 'decoded Qwen output still starts with the prompt' );
				assert.ok( result.generatedTokens.length > 0, 'Qwen emits greedy tokens' );

			} );

			QUnit.test( 'TSL Qwen3.5 0.8B greedy continuation matches the CPU runner', async ( assert ) => {

				const renderer = await createRenderer( assert );
				if ( renderer === null ) return;

				assert.timeout( 300000 );

				const weights = await loadLocalCheckpoint( assert, QwenWeights, QWEN35_ROOT );
				if ( weights === null ) {

					renderer.dispose();
					return;

				}

				const cpu = new QwenCPURunner( weights, { maxTokens: 32 } ).generate( STORY_PROMPT, GREEDY_SHORT );
				const gpu = await new QwenTSLRunner( weights, { maxTokens: 32 } ).generate( renderer, STORY_PROMPT, GREEDY_SHORT );

				assert.ok( cpu.text.startsWith( STORY_PROMPT ), 'CPU Qwen keeps the prompt' );
				assert.strictEqual( gpu.text, cpu.text, 'GPU Qwen greedy text matches CPU' );
				assert.deepEqual( gpu.generatedTokens, cpu.generatedTokens, 'GPU Qwen token ids match CPU' );
				renderer.dispose();

			} );

			QUnit.test( 'tiny Qwen3.5 greedy GPU matches CPU', async ( assert ) => {

				const renderer = await createRenderer( assert );
				if ( renderer === null ) return;

				const weights = createTinyQwenWeights();
				const options = { maxNewTokens: 4, temperature: 0, topK: 1 };
				const cpu = new QwenCPURunner( weights, { maxTokens: 8 } ).generate( 'hello', options );
				const gpu = await new QwenTSLRunner( weights, { maxTokens: 8 } ).generate( renderer, 'hello', options );

				assert.strictEqual( weights.block( 0 ).layerType, 'linear_attention' );
				assert.strictEqual( weights.block( 1 ).layerType, 'full_attention' );
				assert.deepEqual( gpu.generatedTokens, cpu.generatedTokens, 'tiny Qwen GPU tokens match CPU' );
				assert.strictEqual( gpu.text, cpu.text );
				renderer.dispose();

			} );

		} );

	} );

} );
