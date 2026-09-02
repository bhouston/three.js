/**
 * Minimal GPT-2 byte-level BPE tokenizer.
 *
 * @three_import import { GPT2Tokenizer } from 'three/addons/gpgpu/llm/GPT2Tokenizer.js';
 */
class GPT2Tokenizer {

	constructor( vocab, merges, options = {} ) {

		this.encoder = vocab;
		this.decoder = [];
		this.cache = new Map();
		this.byteEncoder = bytesToUnicode();
		this.byteDecoder = {};
		this.unknownToken = options.unknownToken || '<|endoftext|>';
		this.endOfTextToken = options.endOfTextToken || '<|endoftext|>';
		this.endOfTextTokenId = this.encoder[ this.endOfTextToken ];
		this.tokenPattern = options.tokenPattern || GPT2_TOKEN_PATTERN;

		for ( const token in vocab ) {

			this.decoder[ vocab[ token ] ] = token;

		}

		for ( const key in this.byteEncoder ) {

			this.byteDecoder[ this.byteEncoder[ key ] ] = Number( key );

		}

		this.bpeRanks = new Map();

		for ( let i = 0; i < merges.length; i ++ ) {

			const merge = merges[ i ].trim();

			if ( merge === '' || merge.startsWith( '#version' ) ) continue;

			const pair = merge.split( /\s+/ );
			this.bpeRanks.set( pair.join( '\u0000' ), this.bpeRanks.size );

		}

	}

	static async fromURLs( vocabURL, mergesURL, options ) {

		const [ vocabResponse, mergesResponse ] = await Promise.all( [
			fetch( vocabURL ),
			fetch( mergesURL )
		] );

		if ( vocabResponse.ok === false ) {

			throw new Error( `GPT2Tokenizer: Failed to load "${ vocabURL }" (${ vocabResponse.status } ${ vocabResponse.statusText })` );

		}

		if ( mergesResponse.ok === false ) {

			throw new Error( `GPT2Tokenizer: Failed to load "${ mergesURL }" (${ mergesResponse.status } ${ mergesResponse.statusText })` );

		}

		return new GPT2Tokenizer( await vocabResponse.json(), ( await mergesResponse.text() ).split( /\r?\n/ ), options );

	}

	encode( text ) {

		const tokens = [];
		const matches = text.match( this.tokenPattern ) || [];

		for ( const match of matches ) {

			const encoded = byteEncode( match, this.byteEncoder );
			const bpeTokens = this.bpe( encoded ).split( ' ' );

			for ( const token of bpeTokens ) {

				const id = this.encoder[ token ];
				tokens.push( id === undefined ? this.encoder[ this.unknownToken ] : id );

			}

		}

		return tokens;

	}

	decode( tokenIds ) {

		let text = '';

		for ( const id of tokenIds ) {

			text += this.decoder[ id ] || '';

		}

		const bytes = [];

		for ( const char of text ) {

			const byte = this.byteDecoder[ char ];
			if ( byte !== undefined ) bytes.push( byte );

		}

		return new TextDecoder( 'utf-8', { fatal: false } ).decode( new Uint8Array( bytes ) );

	}

	bpe( token ) {

		if ( this.cache.has( token ) ) return this.cache.get( token );

		let word = Array.from( token );
		let pairs = getPairs( word );

		if ( pairs.size === 0 ) return token;

		while ( true ) {

			let bestPair = null;
			let bestRank = Infinity;

			for ( const pair of pairs ) {

				const rank = this.bpeRanks.get( pair );

				if ( rank !== undefined && rank < bestRank ) {

					bestPair = pair;
					bestRank = rank;

				}

			}

			if ( bestPair === null ) break;

			const [ first, second ] = bestPair.split( '\u0000' );
			const nextWord = [];
			let i = 0;

			while ( i < word.length ) {

				if ( word[ i ] === first && i < word.length - 1 && word[ i + 1 ] === second ) {

					nextWord.push( first + second );
					i += 2;

				} else {

					nextWord.push( word[ i ] );
					i ++;

				}

			}

			word = nextWord;

			if ( word.length === 1 ) break;

			pairs = getPairs( word );

		}

		const value = word.join( ' ' );
		this.cache.set( token, value );

		return value;

	}

}

const GPT2_TOKEN_PATTERN = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
const QWEN_TOKEN_PATTERN = /(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

function bytesToUnicode() {

	const bs = [];
	const cs = [];

	for ( let i = 33; i <= 126; i ++ ) bs.push( i );
	for ( let i = 161; i <= 172; i ++ ) bs.push( i );
	for ( let i = 174; i <= 255; i ++ ) bs.push( i );

	for ( const b of bs ) cs.push( b );

	let n = 0;

	for ( let b = 0; b < 256; b ++ ) {

		if ( bs.includes( b ) === false ) {

			bs.push( b );
			cs.push( 256 + n );
			n ++;

		}

	}

	const table = {};

	for ( let i = 0; i < bs.length; i ++ ) {

		table[ bs[ i ] ] = String.fromCodePoint( cs[ i ] );

	}

	return table;

}

function byteEncode( text, byteEncoder ) {

	const bytes = new TextEncoder().encode( text );
	let result = '';

	for ( const byte of bytes ) {

		result += byteEncoder[ byte ];

	}

	return result;

}

function getPairs( word ) {

	const pairs = new Set();

	for ( let i = 0; i < word.length - 1; i ++ ) {

		pairs.add( `${ word[ i ] }\u0000${ word[ i + 1 ] }` );

	}

	return pairs;

}

export { GPT2Tokenizer, GPT2_TOKEN_PATTERN, QWEN_TOKEN_PATTERN };
