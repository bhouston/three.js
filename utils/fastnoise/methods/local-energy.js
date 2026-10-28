/**
 * Cheap, filter-aware quadratic swap optimizer. This is an experimental
 * surrogate, NOT FAST: FAST minimizes error for threshold (Heaviside)
 * integrands, whereas this method minimizes filtered scalar-value energy.
 * https://www.ea.com/seed/news/spatio-temporal-sampling
 * https://arxiv.org/abs/2310.15364
 *
 * The spatial kernel is the autocorrelation of a separable 3x3 binomial
 * filter; the temporal kernel is the autocorrelation of a finite EMA.
 * Their SUM encourages both spatial and temporal quality independently.
 * It does not optimize their composition or the full threshold distribution.
 * A cached convolution makes each accepted swap O(25 + 2 * temporalRadius).
 * Threshold masks can therefore remain much worse than FAST despite good
 * scalar-value error. Evaluate multiple thresholds and temporal filters.
 */
export function generate( { size = 128, depth = 32, seed = 1, passes = 4, temporalWeight = 1, temporalAlpha = 0.25, temporalRadius = 4 } = {} ) {

	if ( ! Number.isInteger( size ) || size < 1 || ! Number.isInteger( depth ) || depth < 1 ) throw new RangeError( 'size and depth must be positive integers.' );
	if ( ! Number.isInteger( passes ) || passes < 0 || ! Number.isInteger( temporalRadius ) || temporalRadius < 0 ) throw new RangeError( 'passes and temporalRadius must be nonnegative integers.' );
	if ( ! Number.isFinite( temporalWeight ) || temporalWeight < 0 || ! Number.isFinite( temporalAlpha ) || temporalAlpha <= 0 || temporalAlpha > 1 ) throw new RangeError( 'Invalid temporal filter.' );

	const plane = size * size;
	const count = plane * depth;
	const values = new Float32Array( count );
	const potential = new Float64Array( count );
	let state = seed >>> 0;
	const random = () => {

		state = ( state + 0x6D2B79F5 ) >>> 0;
		let t = state;
		t = Math.imul( t ^ t >>> 15, t | 1 );
		t ^= t + Math.imul( t ^ t >>> 7, t | 61 );
		return ( ( t ^ t >>> 14 ) >>> 0 ) / 4294967296;

	};
	const wrap = ( v, n ) => ( v % n + n ) % n;

	// One jittered sample per temporal stratum at every pixel, independently
	// permuted. Optimization can break this per-pixel stratification, but swaps
	// preserve the complete volume's original value distribution exactly.
	const order = new Uint32Array( depth );
	for ( let p = 0; p < plane; p ++ ) {

		for ( let z = 0; z < depth; z ++ ) order[ z ] = z;
		for ( let z = depth - 1; z > 0; z -- ) {

			const j = Math.floor( random() * ( z + 1 ) );
			const t = order[ z ];
			order[ z ] = order[ j ];
			order[ j ] = t;

		}
		for ( let z = 0; z < depth; z ++ ) values[ z * plane + p ] = Math.min( ( order[ z ] + random() ) / depth, 1 - 2 ** - 24 );

	}

	const stencil = [];
	const spatial = new Float64Array( plane );
	const temporal = new Float64Array( depth );
	const binomial = [ 1 / 6, 4 / 6, 1, 4 / 6, 1 / 6 ];
	for ( let dy = - 2; dy <= 2; dy ++ ) {

		for ( let dx = - 2; dx <= 2; dx ++ ) {

			const weight = binomial[ dx + 2 ] * binomial[ dy + 2 ];
			stencil.push( [ dx, dy, 0, weight ] );
			spatial[ wrap( dy, size ) * size + wrap( dx, size ) ] += weight;

		}

	}
	const impulse = Array.from( { length: temporalRadius + 1 }, ( _, i ) => ( 1 - temporalAlpha ) ** i );
	let normalization = 0;
	for ( const v of impulse ) normalization += v * v;
	for ( let dz = - temporalRadius; dz <= temporalRadius; dz ++ ) {

		let weight = 0;
		for ( let i = 0; i + Math.abs( dz ) < impulse.length; i ++ ) weight += impulse[ i ] * impulse[ i + Math.abs( dz ) ];
		weight *= temporalWeight / normalization;
		stencil.push( [ 0, 0, dz, weight ] );
		temporal[ wrap( dz, depth ) ] += weight;

	}

	// Lookup tables avoid modulo in the optimization's innermost loops.
	const lookup = stencil.map( ( [ dx, dy, dz, weight ] ) => ( {
		x: Int32Array.from( { length: size }, ( _, i ) => wrap( i + dx, size ) ),
		y: Int32Array.from( { length: size }, ( _, i ) => wrap( i + dy, size ) * size ),
		z: Int32Array.from( { length: depth }, ( _, i ) => wrap( i + dz, depth ) * plane ),
		weight
	} ) );
	function update( x, y, z, delta ) {

		for ( let k = 0; k < lookup.length; k ++ ) {

			const s = lookup[ k ];
			potential[ s.x[ x ] + s.y[ y ] + s.z[ z ] ] += delta * s.weight;

		}

	}
	for ( let z = 0; z < depth; z ++ ) {

		for ( let y = 0; y < size; y ++ ) {

			for ( let x = 0; x < size; x ++ ) update( x, y, z, values[ z * plane + y * size + x ] - 0.5 );

		}

	}

	const self = spatial[ 0 ] + temporal[ 0 ];
	for ( let step = 0; step < passes * count; step ++ ) {

		const a = Math.floor( random() * count );
		const b = Math.floor( random() * count );
		const az = Math.floor( a / plane ), bz = Math.floor( b / plane );
		const ay = Math.floor( a % plane / size ), by = Math.floor( b % plane / size );
		const ax = a % size, bx = b % size;
		const dx = wrap( bx - ax, size ), dy = wrap( by - ay, size ), dz = wrap( bz - az, depth );
		const coupling = ( dz === 0 ? spatial[ dy * size + dx ] : 0 ) + ( dx === 0 && dy === 0 ? temporal[ dz ] : 0 );
		const delta = values[ b ] - values[ a ];
		// Exact change in v^T K v for swapping a and b, including overlap.
		if ( delta * ( potential[ a ] - potential[ b ] + delta * ( self - coupling ) ) < - 1e-12 ) {

			const t = values[ a ];
			values[ a ] = values[ b ];
			values[ b ] = t;
			update( ax, ay, az, delta );
			update( bx, by, bz, - delta );

		}

	}
	return values;

}
