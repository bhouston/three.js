import { fetchJSON } from './LLMTensors.js';
import { GPT2CPURunner } from './GPT2CPURunner.js';
import { GPT2TSLRunner } from './GPT2TSLRunner.js';
import { GPT2Weights } from './GPT2Weights.js';
import { LlamaCPURunner } from './LlamaCPURunner.js';
import { LlamaTSLRunner } from './LlamaTSLRunner.js';
import { LlamaWeights } from './LlamaWeights.js';
import { PhiCPURunner } from './PhiCPURunner.js';
import { PhiTSLRunner } from './PhiTSLRunner.js';
import { PhiWeights } from './PhiWeights.js';

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
	}
];

function normalizeRoot( baseURL ) {

	return baseURL.endsWith( '/' ) ? baseURL : `${ baseURL }/`;

}

function architectureFor( config ) {

	const type = config.model_type;

	if ( type === 'gpt2' ) return 'gpt2';
	if ( type === 'llama' || type === 'mistral' || type === 'qwen2' || type === 'gemma' || type === 'gemma2' ) return 'llama';
	if ( type === 'phi' ) return 'phi';

	throw new Error( `LLMFactory: Unsupported model_type "${ type }".` );

}

async function loadWeights( baseURL ) {

	const config = await fetchJSON( `${ normalizeRoot( baseURL ) }config.json`, 'LLMFactory' );
	const architecture = architectureFor( config );

	if ( architecture === 'gpt2' ) return GPT2Weights.fromURL( baseURL );
	if ( architecture === 'phi' ) return PhiWeights.fromURL( baseURL );

	return LlamaWeights.fromURL( baseURL );

}

async function createTSLRunner( baseURL, options ) {

	const weights = await loadWeights( baseURL );

	if ( weights.architecture === 'gpt2' ) return new GPT2TSLRunner( weights, options );
	if ( weights.architecture === 'phi' ) return new PhiTSLRunner( weights, options );

	return new LlamaTSLRunner( weights, options );

}

async function createCPURunner( baseURL, options ) {

	const weights = await loadWeights( baseURL );

	if ( weights.architecture === 'gpt2' ) return new GPT2CPURunner( weights, options );
	if ( weights.architecture === 'phi' ) return new PhiCPURunner( weights, options );

	return new LlamaCPURunner( weights, options );

}

export {
	MODEL_CATALOG,
	architectureFor,
	createCPURunner,
	createTSLRunner,
	loadWeights
};
