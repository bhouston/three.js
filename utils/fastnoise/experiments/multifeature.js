/**
 * Experimental multifeature projected gradient noise, periodic z/y/x layout.
 * Cosine features approximate integrated threshold discrepancy rather than
 * cancelling only the stored scalar values. CPU only, no dependencies.
 */
export function generate( { size = 128, depth = 32, seed = 1, iterations = 48, features = 3, step = 0.3, temporalWeight = 1, initial = null } = {} ) {

	if ( ! Number.isInteger( size ) || size < 2 || ! Number.isInteger( depth ) || depth < 2 || ! Number.isInteger( iterations ) || iterations < 0 || ! Number.isInteger( features ) || features < 1 || ! Number.isFinite( step ) || step <= 0 || ! Number.isFinite( temporalWeight ) || temporalWeight < 0 ) throw new RangeError( 'Invalid multifeature generator parameters.' );
	const plane = size * size, count = plane * depth;
	if ( initial !== null && ( ! ( initial instanceof Float32Array ) || initial.length !== count || ! initial.every( v => Number.isFinite( v ) && v >= 0 && v < 1 ) ) ) throw new RangeError( 'initial must contain size*size*depth values in [0,1).' );
	const values = initial === null ? new Float32Array( count ) : initial.slice();
	const gradient = new Float32Array( count ), field = new Float32Array( count ), horizontal = new Float32Array( count );
	const bins = 65536, histogram = new Uint32Array( bins ), offsets = new Uint32Array( bins );
	const lutSize = 4096, lookup = new Uint16Array( count );
	const feature = new Float32Array( lutSize ), derivative = new Float32Array( lutSize );
	let state = seed >>> 0;
	if ( initial === null ) for ( let i = 0; i < count; i ++ ) {

		state = ( state + 0x6D2B79F5 ) >>> 0;
		let t = Math.imul( state ^ state >>> 15, state | 1 );
		t ^= t + Math.imul( t ^ t >>> 7, t | 61 );
		values[ i ] = Math.min( 1 - 2 ** - 24, ( ( t ^ t >>> 14 ) >>> 0 ) / 4294967296 );

	}

	// Stable bucket ordering restores an exact discrete global uniform marginal.
	function project() {

		let min = Infinity, max = - Infinity;
		for ( let i = 0; i < count; i ++ ) {

			min = Math.min( min, values[ i ] ); max = Math.max( max, values[ i ] );

		}

		const scale = ( bins - 1 ) / ( max - min || 1 );
		histogram.fill( 0 );
		for ( let i = 0; i < count; i ++ ) histogram[ Math.floor( ( values[ i ] - min ) * scale ) ] ++;
		let sum = 0;
		for ( let b = 0; b < bins; b ++ ) {

			offsets[ b ] = sum; sum += histogram[ b ];

		}

		for ( let i = 0; i < count; i ++ ) values[ i ] = ( offsets[ Math.floor( ( values[ i ] - min ) * scale ) ] ++ + 0.5 ) / count;

	}

	project();
	for ( let iteration = 0; iteration < iterations; iteration ++ ) {

		gradient.fill( 0 );
		for ( let i = 0; i < count; i ++ ) lookup[ i ] = Math.min( lutSize - 1, Math.floor( values[ i ] * lutSize ) );
		for ( let k = 1; k <= features; k ++ ) {

			// Neumann cosine basis of the Brownian-bridge threshold covariance.
			// E[cos(k*pi*U)] = 0 for every integer k and uniform U.
			for ( let b = 0; b < lutSize; b ++ ) {

				const angle = Math.PI * k * ( b + 0.5 ) / lutSize;
				feature[ b ] = Math.cos( angle );
				derivative[ b ] = - Math.sin( angle ) / ( Math.PI * k );

			}

			for ( let i = 0; i < count; i ++ ) field[ i ] = feature[ lookup[ i ] ];
			for ( let z = 0; z < depth; z ++ ) for ( let y = 0; y < size; y ++ ) for ( let x = 0; x < size; x ++ ) {

				const base = z * plane + y * size, i = base + x;
				horizontal[ i ] = ( field[ base + ( x + size - 2 ) % size ] + 4 * field[ base + ( x + size - 1 ) % size ] + 6 * field[ i ] + 4 * field[ base + ( x + 1 ) % size ] + field[ base + ( x + 2 ) % size ] ) / 16;

			}

			for ( let z = 0; z < depth; z ++ ) for ( let y = 0; y < size; y ++ ) for ( let x = 0; x < size; x ++ ) {

				const base = z * plane + x, p = y * size + x, i = z * plane + p;
				const spatial = ( horizontal[ base + ( y + size - 2 ) % size * size ] + 4 * horizontal[ base + ( y + size - 1 ) % size * size ] + 6 * horizontal[ i ] + 4 * horizontal[ base + ( y + 1 ) % size * size ] + horizontal[ base + ( y + 2 ) % size * size ] ) / 16;
				const temporal = ( field[ ( z + depth - 2 ) % depth * plane + p ] + 4 * field[ ( z + depth - 1 ) % depth * plane + p ] + 6 * field[ i ] + 4 * field[ ( z + 1 ) % depth * plane + p ] + field[ ( z + 2 ) % depth * plane + p ] ) / 16;
				gradient[ i ] += ( spatial + temporalWeight * temporal ) * derivative[ lookup[ i ] ];

			}

		}

		for ( let i = 0; i < count; i ++ ) values[ i ] -= step * gradient[ i ];
		project();

	}

	return values;

}
