import { DecoderCPURunner } from './DecoderCPURunner.js';
import { architectureFor } from './DecoderRecipe.js';
import { DecoderTSLRunner } from './DecoderTSLRunner.js';
import { DecoderWeights } from './DecoderWeights.js';
import { loadHFModelBundle, normalizeRoot } from './HFModelBundle.js';
import { createProgress } from './LLMTensors.js';
import { QwenCPURunner } from './QwenCPURunner.js';
import { QwenTSLRunner } from './QwenTSLRunner.js';
import { QwenWeights } from './QwenWeights.js';

const MODEL_CATALOG = [
	{
		id: 'tinystories',
		name: 'TinyStories GPT-2 3M',
		url: './models/llm/tinystories-gpt2-0.1-3m/',
		prompt: 'Once upon a time,',
		note: 'Children\'s stories. Fits in a few megabytes.'
	},
	{
		id: 'gpt2',
		name: 'GPT-2 124M',
		url: './models/llm/gpt2/',
		prompt: 'Once upon a time,',
		note: 'Classic dense GPT-2. About 500 MB of float32 weights.'
	},
	{
		id: 'smollm2',
		name: 'SmolLM2 135M',
		url: 'https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/',
		localUrl: './models/llm/smollm2-135m/',
		prompt: 'Once upon a time,',
		note: 'Llama-style: RMSNorm, RoPE, grouped-query attention, SwiGLU. ~270 MB BF16 from Hugging Face.'
	},
	{
		id: 'gemma3-270m',
		name: 'Gemma 3 270M',
		url: 'https://huggingface.co/google/gemma-3-270m/resolve/main/',
		localUrl: './models/llm/gemma-3-270m/',
		prompt: 'Once upon a time,',
		note: 'Google Gemma 3 270M (sliding-window GQA, QK-norm, GeGLU). Gated on Hugging Face — accept the license and put config.json, tokenizer.json, and model.safetensors in examples/models/llm/gemma-3-270m/. About 540 MB BF16.'
	},
	{
		id: 'qwen3.5-0.8b',
		name: 'Qwen3.5 0.8B',
		url: 'https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/',
		localUrl: './models/llm/qwen3.5-0.8b/',
		prompt: 'Once upon a time,',
		note: 'Qwen3.5 0.8B hybrid: Gated DeltaNet linear attention plus gated full attention. About 1.8 GB BF16. Text-only decode; vision tensors are skipped. Thinking is off by default so replies skip the <think> block.'
	},
	{
		id: 'phi-1.5',
		name: 'Phi-1.5 1.3B',
		url: 'https://huggingface.co/microsoft/phi-1_5/resolve/main/',
		localUrl: './models/llm/phi-1.5/',
		prompt: 'Once upon a time,',
		note: 'Microsoft Phi-1.5 (LayerNorm, partial RoPE, parallel attention + MLP). About 2.8 GB FP16 from Hugging Face.'
	}
];

async function loadWeights( baseURL, options = {} ) {

	const report = createProgress( 'LLMFactory', options.onProgress );

	try {

		const bundle = await loadHFModelBundle( baseURL, { ...options, label: 'LLMFactory' } );
		await report( `Using ${ bundle.architecture } loader for model_type "${ bundle.rawConfig.model_type }"` );

		if ( bundle.recipe.graph === 'qwen35' ) return QwenWeights.fromBundle( bundle, options );

		return DecoderWeights.fromBundle( bundle, options );

	} catch ( error ) {

		throw new Error( `LLMFactory: failed to load "${ normalizeRoot( baseURL ) }": ${ error.message }` );

	}

}

async function createTSLRunner( baseURL, options = {} ) {

	const report = createProgress( 'LLMFactory', options.onProgress );
	const weights = await loadWeights( baseURL, options );
	await report( `Building ${ weights.architecture } GPU runner (${ weights.layerCount } layers, vocab ${ weights.vocabSize })...` );

	const runner = weights.recipe.graph === 'qwen35'
		? new QwenTSLRunner( weights, options )
		: new DecoderTSLRunner( weights, options );

	await report( 'GPU runner ready' );
	return runner;

}

async function createCPURunner( baseURL, options = {} ) {

	const weights = await loadWeights( baseURL, options );

	if ( weights.recipe.graph === 'qwen35' ) return new QwenCPURunner( weights, options );

	return new DecoderCPURunner( weights, options );

}

export {
	MODEL_CATALOG,
	architectureFor,
	createCPURunner,
	createTSLRunner,
	loadWeights
};
