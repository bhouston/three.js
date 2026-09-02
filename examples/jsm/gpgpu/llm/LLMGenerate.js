import { sampleTopK } from './LLMMath.js';

function sharedPrefixLength( a = [], b = [] ) {

	const n = Math.min( a.length, b.length );
	let i = 0;

	while ( i < n && a[ i ] === b[ i ] ) i ++;

	return i;

}

function prepareGenerationFromTokens( tokens, maxTokens, maxNewTokens, endOfTextTokenId ) {

	const promptBudget = Math.max( 1, maxTokens - 1 );
	const inputTokens = tokens.length === 0 ? [ endOfTextTokenId ] : tokens.slice( - promptBudget );
	const newTokenBudget = Math.max( 0, Math.min( maxNewTokens, maxTokens - inputTokens.length ) );

	return { inputTokens, newTokenBudget };

}

function resolvePromptTokens( runner, prompt, options ) {

	const maxNewTokens = options.maxNewTokens || 32;

	if ( options.inputTokens ) {

		return prepareGenerationFromTokens(
			options.inputTokens,
			runner.maxTokens,
			maxNewTokens,
			runner.weights.endOfTextTokenId
		);

	}

	return runner.weights.prepareGeneration( prompt, runner.maxTokens, maxNewTokens );

}

function planPromptCache( cachedTokens, cachedLogits, inputTokens, rewindable ) {

	const prefix = sharedPrefixLength( cachedTokens, inputTokens );
	const hasLogits = cachedLogits !== null && cachedLogits !== undefined;
	const appendOnly = prefix > 0 && prefix === cachedTokens.length && hasLogits;

	if ( appendOnly ) {

		return { start: prefix, logits: cachedLogits, reset: false, reused: prefix };

	}

	if ( rewindable && prefix > 0 ) {

		const start = prefix >= inputTokens.length ? inputTokens.length - 1 : prefix;
		return { start, logits: null, reset: false, reused: start };

	}

	return { start: 0, logits: null, reset: true, reused: 0 };

}

function finishGeneration( runner, weights, allTokens, generatedTokens, logits, extra ) {

	runner._cacheTokens = allTokens.slice();
	runner._cacheLogits = logits;

	return {
		tokens: allTokens,
		generatedTokens,
		text: weights.tokenizer.decode( allTokens ),
		generatedText: weights.tokenizer.decode( generatedTokens ),
		cachedPromptTokens: extra.reused,
		promptTokens: extra.promptTokens,
		... extra.rest
	};

}

function generateSync( runner, prompt, options = {}, controls ) {

	const { rewindable, resetCache, forwardToken } = controls;
	const { inputTokens, newTokenBudget } = resolvePromptTokens( runner, prompt, options );
	const plan = planPromptCache( runner._cacheTokens || [], runner._cacheLogits, inputTokens, rewindable );

	if ( plan.reset ) resetCache();

	if ( options.onPrefill ) {

		options.onPrefill( {
			cachedPromptTokens: plan.reused,
			promptTokens: inputTokens.length
		} );

	}

	const allTokens = inputTokens.slice();
	const generatedTokens = [];
	let logits = plan.logits;

	for ( let i = plan.start; i < inputTokens.length; i ++ ) {

		logits = forwardToken( inputTokens[ i ], i );

	}

	for ( let i = 0; i < newTokenBudget; i ++ ) {

		const nextToken = sampleTopK( logits, { ...options, tokens: allTokens } );

		if ( nextToken === runner.weights.endOfTextTokenId ) break;

		allTokens.push( nextToken );
		generatedTokens.push( nextToken );
		logits = forwardToken( nextToken, allTokens.length - 1 );

	}

	return finishGeneration( runner, runner.weights, allTokens, generatedTokens, logits, {
		reused: plan.reused,
		promptTokens: inputTokens.length,
		rest: {}
	} );

}

async function generateAsync( runner, prompt, options = {}, controls ) {

	const { rewindable, resetCache, computeToken, readLogits } = controls;
	const { inputTokens, newTokenBudget } = resolvePromptTokens( runner, prompt, options );
	const plan = planPromptCache( runner._cacheTokens || [], runner._cacheLogits, inputTokens, rewindable );
	const signal = options.signal;

	if ( plan.reset ) resetCache();

	if ( options.onPrefill ) {

		options.onPrefill( {
			cachedPromptTokens: plan.reused,
			promptTokens: inputTokens.length
		} );

	}

	const allTokens = inputTokens.slice();
	const generatedTokens = [];
	let logits = plan.logits;

	for ( let i = plan.start; i < inputTokens.length; i ++ ) {

		if ( signal !== undefined && signal.aborted ) break;

		await computeToken( inputTokens[ i ], i );
		if ( i === inputTokens.length - 1 ) logits = await readLogits();

	}

	for ( let i = 0; i < newTokenBudget; i ++ ) {

		if ( signal !== undefined && signal.aborted ) break;
		if ( logits === null ) break;

		const nextToken = sampleTopK( logits, { ...options, tokens: allTokens } );

		if ( nextToken === runner.weights.endOfTextTokenId ) break;

		allTokens.push( nextToken );
		generatedTokens.push( nextToken );

		if ( options.onToken ) {

			options.onToken( runner.weights.tokenizer.decode( allTokens ), nextToken );

		}

		await computeToken( nextToken, allTokens.length - 1 );
		logits = await readLogits();

	}

	return finishGeneration( runner, runner.weights, allTokens, generatedTokens, logits, {
		reused: plan.reused,
		promptTokens: inputTokens.length,
		rest: { aborted: signal !== undefined && signal.aborted }
	} );

}

export {
	generateAsync,
	generateSync,
	planPromptCache,
	prepareGenerationFromTokens,
	sharedPrefixLength
};
