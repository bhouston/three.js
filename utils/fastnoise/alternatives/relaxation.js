/**
 * Alternating spatial/temporal high-pass projections with histogram equalization.
 * CPU only; periodic boundaries; output layout is z, y, x.
 */
export function generate( { size = 128, depth = 32, seed = 1, iterations = 8 } = {} ) {

	if ( ! Number.isInteger( size ) || size < 2 || ! Number.isInteger( depth ) || depth < 2 || ! Number.isInteger( iterations ) || iterations < 0 ) {

		throw new RangeError( 'size and depth must be integers >= 2; iterations must be an integer >= 0.' );

	}

	const plane = size * size, count = plane * depth;
	const values = new Float32Array( count );
	const scratch = new Float32Array( count ), blurred = new Float32Array( count );
	const bins = 8192, histogram = new Uint32Array( bins ), cdf = new Float32Array( bins );
	let state = seed >>> 0;
	for ( let i = 0; i < count; i ++ ) {

		state = ( state + 0x6D2B79F5 ) >>> 0;
		let t = Math.imul( state ^ state >>> 15, state | 1 );
		t ^= t + Math.imul( t ^ t >>> 7, t | 61 );
		values[ i ] = Math.min( 1 - 2 ** - 24, ( ( t ^ t >>> 14 ) >>> 0 ) / 4294967296 );

	}

	// Equalizing after each projection restores the marginal distribution. Linear
	// interpolation within histogram bins avoids quantizing to only 8192 values.
	function equalize() {

		let min = Infinity, max = - Infinity;
		for ( let i = 0; i < count; i ++ ) {

			min = Math.min( min, scratch[ i ] ); max = Math.max( max, scratch[ i ] );

		}

		const scale = ( bins - 1 ) / ( max - min || 1 );
		histogram.fill( 0 );
		for ( let i = 0; i < count; i ++ ) histogram[ Math.floor( ( scratch[ i ] - min ) * scale ) ] ++;
		let total = 0;
		for ( let b = 0; b < bins; b ++ ) {

			cdf[ b ] = total / count; total += histogram[ b ];

		}

		for ( let i = 0; i < count; i ++ ) {

			const position = ( scratch[ i ] - min ) * scale;
			const bin = Math.floor( position );
			values[ i ] = Math.min( 1 - 2 ** - 24, cdf[ bin ] + ( position - bin ) * histogram[ bin ] / count );

		}

	}

	for ( let iteration = 0; iteration < iterations; iteration ++ ) {

		// Separable [1, 4, 6, 4, 1]/16 Gaussian approximation, spatial only.
		for ( let z = 0; z < depth; z ++ ) for ( let y = 0; y < size; y ++ ) for ( let x = 0; x < size; x ++ ) {

			const base = z * plane + y * size, i = base + x;
			blurred[ i ] = ( values[ base + ( x + size - 2 ) % size ] + 4 * values[ base + ( x + size - 1 ) % size ] + 6 * values[ i ] + 4 * values[ base + ( x + 1 ) % size ] + values[ base + ( x + 2 ) % size ] ) / 16;

		}

		for ( let z = 0; z < depth; z ++ ) for ( let y = 0; y < size; y ++ ) for ( let x = 0; x < size; x ++ ) {

			const base = z * plane + x, i = base + y * size;
			const smooth = ( blurred[ base + ( y + size - 2 ) % size * size ] + 4 * blurred[ base + ( y + size - 1 ) % size * size ] + 6 * blurred[ i ] + 4 * blurred[ base + ( y + 1 ) % size * size ] + blurred[ base + ( y + 2 ) % size * size ] ) / 16;
			scratch[ i ] = values[ i ] - 0.95 * smooth;

		}

		equalize();
		// A separate temporal projection encourages blue spectra along each pixel's
		// time sequence, without treating all 3D neighbors as a single distance.
		for ( let z = 0; z < depth; z ++ ) for ( let p = 0; p < plane; p ++ ) {

			const i = z * plane + p;
			const smooth = ( values[ ( z + depth - 2 ) % depth * plane + p ] + 4 * values[ ( z + depth - 1 ) % depth * plane + p ] + 6 * values[ i ] + 4 * values[ ( z + 1 ) % depth * plane + p ] + values[ ( z + 2 ) % depth * plane + p ] ) / 16;
			scratch[ i ] = values[ i ] - 0.95 * smooth;

		}

		equalize();

	}

	return values;

}
