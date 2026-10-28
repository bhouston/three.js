/**
 * Temporal-stratum-preserving threshold optimization.
 * Each pixel receives one jittered sample from each equal temporal stratum.
 * Swaps only reorder times at the SAME pixel. The optimizer cannot trade
 * full-cycle nonlinear quadrature away to improve short-window quality.
 * Spatial and temporal binary-threshold quadratic energies are summed.
 * This constraint may hurt spatial quality and short temporal windows;
 * full-cycle convergence alone does not prove superior rendering quality.
 * initialVolume optionally supplies a sample set per pixel to reorder.
 * It is copied, and only stratified inputs have the strata guarantee.
 * jitter=0 uses bin midpoints; jitter=1 uses full within-bin randomization.
 * Based on the cached local-energy method, itself inspired by FAST:
 * https://arxiv.org/abs/2310.15364
 */
export function generate( { size = 128, depth = 32, seed = 1, passes = 24, temporalWeight = 1, temporalAlpha = 0.1, temporalRadius = 12, thresholdCount = 31, initialVolume = null, jitter = 1 } = {} ) {

	if ( ! Number.isInteger( size ) || size < 1 || ! Number.isInteger( depth ) || depth < 1 ) throw new RangeError( 'size and depth must be positive integers.' );
	if ( ! Number.isInteger( passes ) || passes < 0 || ! Number.isInteger( temporalRadius ) || temporalRadius < 0 ) throw new RangeError( 'passes and temporalRadius must be nonnegative integers.' );
	if ( ! Number.isInteger( thresholdCount ) || thresholdCount < 0 ) throw new RangeError( 'thresholdCount must be a nonnegative integer.' );
	if ( ! Number.isFinite( temporalWeight ) || temporalWeight < 0 || ! Number.isFinite( temporalAlpha ) || temporalAlpha <= 0 || temporalAlpha > 1 ) throw new RangeError( 'Invalid temporal filter.' );

	const plane = size * size;
	const count = plane * depth;
	if ( ! Number.isFinite( jitter ) || jitter < 0 || jitter > 1 ) throw new RangeError( 'jitter must be between zero and one.' );
	if ( initialVolume !== null && initialVolume.length !== count ) throw new RangeError( 'initialVolume has incorrect length.' );
	const values = new Float32Array( count );
	const channels = Math.max( 1, thresholdCount );
	const potential = new Float64Array( count * channels );
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
	// permuted. Optimization only reorders times at the SAME pixel, preserving
	// each pixel's complete sample set and hence full-cycle quadrature.
	const order = new Uint32Array( depth );
	for ( let p = 0; p < plane; p ++ ) {

		for ( let z = 0; z < depth; z ++ ) order[ z ] = z;
		for ( let z = depth - 1; z > 0; z -- ) {

			const j = Math.floor( random() * ( z + 1 ) );
			const t = order[ z ];
			order[ z ] = order[ j ];
			order[ j ] = t;

		}

		for ( let z = 0; z < depth; z ++ ) values[ z * plane + p ] = Math.min( ( order[ z ] + 0.5 + jitter * ( random() - 0.5 ) ) / depth, 1 - 2 ** - 24 );

	}

	if ( initialVolume !== null ) {

		for ( let i = 0; i < count; i ++ ) {

			if ( ! Number.isFinite( initialVolume[ i ] ) || initialVolume[ i ] < 0 || initialVolume[ i ] >= 1 ) throw new RangeError( 'initialVolume values must be in [0,1).' );
			values[ i ] = initialVolume[ i ];

		}

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
	function update( x, y, z, delta, lower = 0, upper = channels ) {

		for ( let k = 0; k < lookup.length; k ++ ) {

			const s = lookup[ k ];
			const base = ( s.x[ x ] + s.y[ y ] + s.z[ z ] ) * channels;
			const change = delta * s.weight;
			for ( let c = lower; c < upper; c ++ ) potential[ base + c ] += change;

		}

	}

	for ( let z = 0; z < depth; z ++ ) {

		for ( let y = 0; y < size; y ++ ) {

			for ( let x = 0; x < size; x ++ ) {

				const value = values[ z * plane + y * size + x ];
				if ( thresholdCount === 0 ) update( x, y, z, value - 0.5 );
				else update( x, y, z, 1, Math.floor( value * channels + 0.5 ) );

			}

		}

	}

	const self = spatial[ 0 ] + temporal[ 0 ];
	for ( let step = 0; step < passes * count; step ++ ) {

		const a = Math.floor( random() * count );
		const b = Math.floor( random() * depth ) * plane + a % plane;
		const az = Math.floor( a / plane ), bz = Math.floor( b / plane );
		const ay = Math.floor( a % plane / size ), by = Math.floor( b % plane / size );
		const ax = a % size, bx = b % size;
		const dx = wrap( bx - ax, size ), dy = wrap( by - ay, size ), dz = wrap( bz - az, depth );
		const coupling = ( dz === 0 ? spatial[ dy * size + dx ] : 0 ) + ( dx === 0 && dy === 0 ? temporal[ dz ] : 0 );
		let delta = values[ b ] - values[ a ];
		let lower = 0, upper = channels;
		if ( thresholdCount > 0 ) {

			const ca = Math.floor( values[ a ] * channels + 0.5 );
			const cb = Math.floor( values[ b ] * channels + 0.5 );
			lower = Math.min( ca, cb );
			upper = Math.max( ca, cb );
			delta = cb < ca ? 1 : - 1;

		}

		// Exact change in the sum of binary-mask quadratic energies.
		let energy = 0;
		for ( let c = lower; c < upper; c ++ ) energy += delta * ( potential[ a * channels + c ] - potential[ b * channels + c ] + delta * ( self - coupling ) );
		if ( energy < - 1e-12 ) {

			const t = values[ a ];
			values[ a ] = values[ b ];
			values[ b ] = t;
			update( ax, ay, az, delta, lower, upper );
			update( bx, by, bz, - delta, lower, upper );

		}

	}

	return values;

}

