import { triangleWaveEncode } from '../neural/NeuralGridModel.js';

/**
 * CPU-side computation of the UV-derived array appended after the frame-
 * projected direction dots in the decoder/IBL/indirect-probe head inputs
 * (see NeuralAppearanceModel.forwardDecoderInput/forwardIBLInput/
 * buildIndirectProbeInput) - shared by NeuralAppearanceModel.js (training-time
 * CPU probes, e.g. decorrelateLinearRgbHead) and NeuralAppearanceRuntime.js
 * (JSON-manifest CPU inference), so the two can't silently disagree on what
 * `inputEncoding` produces. Mirrors the training kernel's own three-way
 * branch in NeuralAppearanceGPUComputeTSL.js:
 * - 'positional': `peOctaves` octaves of NTC-style tiled positional encoding
 *   (see NeuralGridModel.triangleWaveEncode) - the historical-only behavior.
 * - 'raw': the raw (u, v) coordinate, unencoded.
 * - 'none' (default/fallback, including 0-octave 'positional'): no UV-derived
 *   input at all - `[]`, reproducing this model's pre-`inputEncoding` output
 *   byte-for-byte.
 */
function computeUVEncoding( inputEncoding, uv, peOctaves ) {

	if ( inputEncoding === 'positional' ) return triangleWaveEncode( uv[ 0 ], uv[ 1 ], peOctaves );
	if ( inputEncoding === 'raw' ) return [ uv[ 0 ], uv[ 1 ] ];

	return [];

}

export { computeUVEncoding };
