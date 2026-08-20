/**
 * Shared math/formatting for reporting a trained model's footprint: the
 * multiresolution latent grid, and one or more MLP heads decoding it (a
 * single decoder for neural-texture/neural-material, or a main decoder plus
 * extra IBL/indirect heads for neural-appearance). Every example built its
 * own copy of the MLP weight-count formula and byte formatter; this module
 * is the one place that math lives now.
 */

function formatBytes( bytes ) {

	if ( bytes < 1024 ) return `${bytes} B`;
	if ( bytes < 1024 * 1024 ) return `${( bytes / 1024 ).toFixed( 1 )} KiB`;
	return `${( bytes / ( 1024 * 1024 ) ).toFixed( 1 )} MiB`;

}

// Weight + bias parameter count for a fully-connected MLP with `hiddenLayers`
// hidden layers of width `hiddenSize`, mapping `inputSize` to `outputSize`.
function computeMLPParamCount( { inputSize, hiddenSize, hiddenLayers = 2, outputSize } ) {

	if ( hiddenLayers <= 0 ) return inputSize * outputSize + outputSize;

	let count = inputSize * hiddenSize + hiddenSize;
	for ( let i = 1; i < hiddenLayers; i ++ ) count += hiddenSize * hiddenSize + hiddenSize;
	count += hiddenSize * outputSize + outputSize;

	return count;

}

// Total latent texel count across a multiresolution grid's per-level square
// resolutions (see NeuralGridModel.js's computeGridLevels).
function computeGridLatentTexels( resolutions ) {

	return resolutions.reduce( ( sum, resolution ) => sum + resolution * resolution, 0 );

}

// Builds a single "total (grid [...] + head + head...)" summary line shared
// across the three examples' "network size" panels.
//
//   grid: { label, bytes }
//   heads: [ { label, bytes } ]
function formatModelSizeSummary( { grid, heads } ) {

	const totalBytes = grid.bytes + heads.reduce( ( sum, head ) => sum + head.bytes, 0 );
	const headsText = heads.map( ( head ) => `${formatBytes( head.bytes )} ${head.label}` ).join( ' + ' );

	return `${formatBytes( totalBytes )} (${formatBytes( grid.bytes )} grid [${grid.label}] + ${headsText})`;

}

export { formatBytes, computeMLPParamCount, computeGridLatentTexels, formatModelSizeSummary };
