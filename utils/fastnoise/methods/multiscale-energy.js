/** Experimental multi-scale CDF energy descent; exact histogram-preserving swaps. */
export function generate( { size = 128, depth = 32, seed = 1, passes = 4, temporalWeight = 1, temporalAlpha = 0.1, thresholdCount = 31, initialVolume = null, spatialSigmas = [ 0.7, 1 ], onProgress = null } = {} ) {

	if ( ! Number.isInteger( size ) || size < 1 || ! Number.isInteger( depth ) || depth < 1 ) throw new RangeError( 'size and depth must be positive integers.' );
	if ( ! Number.isInteger( passes ) || passes < 0 ) throw new RangeError( 'passes must be a nonnegative integer.' );
	if ( ! Number.isInteger( thresholdCount ) || thresholdCount < 0 ) throw new RangeError( 'thresholdCount must be a nonnegative integer.' );
	if ( ! Number.isFinite( temporalWeight ) || temporalWeight < 0 || ! Number.isFinite( temporalAlpha ) || temporalAlpha <= 0 || temporalAlpha > 1 ) throw new RangeError( 'Invalid temporal filter.' );

	const plane = size * size;
	const count = plane * depth;
	if ( initialVolume !== null && ( initialVolume.length !== count || Array.from( initialVolume ).some( v => ! Number.isFinite( v ) || v < 0 || v >= 1 ) ) ) throw new RangeError( 'Invalid initialVolume.' );
	if ( ! Array.isArray( spatialSigmas ) || spatialSigmas.length === 0 || spatialSigmas.some( s => ! Number.isFinite( s ) || s <= 0 ) ) throw new RangeError( 'Invalid spatialSigmas.' );
	const values = initialVolume === null ? new Float32Array( count ) : Float32Array.from( initialVolume );
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
	// permuted. Optimization can break this per-pixel stratification, but swaps
	// preserve the complete volume's original value distribution exactly.
	const order = new Uint32Array( depth );
	for ( let p = 0; initialVolume === null && p < plane; p ++ ) {

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
	// Mixture of Gaussian autocorrelation kernels; normalized per scale.
	for ( const sigma of spatialSigmas ) {

		const radius = Math.ceil( sigma * 3 );
		const filter = Array.from( { length: radius * 2 + 1 }, ( _, i ) => Math.exp( - ( ( i - radius ) ** 2 ) / ( 2 * sigma * sigma ) ) );
		const correlation = new Float64Array( 4 * radius + 1 );
		for ( let d = - 2 * radius; d <= 2 * radius; d ++ ) {

			for ( let i = 0; i < filter.length; i ++ ) if ( i + d >= 0 && i + d < filter.length ) correlation[ d + 2 * radius ] += filter[ i ] * filter[ i + d ];

		}
		const selfWeight = correlation[ 2 * radius ] ** 2;
		for ( let dy = - 2 * radius; dy <= 2 * radius; dy ++ ) {

			for ( let dx = - 2 * radius; dx <= 2 * radius; dx ++ ) {

				const weight = correlation[ dx + 2 * radius ] * correlation[ dy + 2 * radius ] / selfWeight / spatialSigmas.length;
				spatial[ wrap( dy, size ) * size + wrap( dx, size ) ] += weight;

			}

		}

	}
	// Exact autocorrelation of a cyclic EMA, including the infinite periodic tail.
	const impulse = Array.from( { length: depth }, ( _, i ) => ( 1 - temporalAlpha ) ** i );
	let normalization = 0;
	for ( const v of impulse ) normalization += v * v;
	for ( let dz = 0; dz < depth; dz ++ ) {

		let weight = 0;
		for ( let i = 0; i < depth; i ++ ) weight += impulse[ i ] * impulse[ ( i + dz ) % depth ];
		temporal[ dz ] = temporalWeight * weight / normalization;

	}
	// Consolidate periodic aliases so self and pair couplings stay exact.
	for ( let y = 0; y < size; y ++ ) for ( let x = 0; x < size; x ++ ) if ( spatial[ y * size + x ] !== 0 ) stencil.push( [ x, y, 0, spatial[ y * size + x ] ] );
	for ( let z = 0; z < depth; z ++ ) if ( temporal[ z ] !== 0 ) stencil.push( [ 0, 0, z, temporal[ z ] ] );

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
	let accepted = 0, energyChange = 0;
	for ( let step = 0; step < passes * count; step ++ ) {

		if ( onProgress && step % count === 0 ) onProgress( { pass: step / count, accepted, energyChange } );

		const a = Math.floor( random() * count );
		const b = Math.floor( random() * count );
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

			accepted ++;
			energyChange += 2 * energy;
			const t = values[ a ];
			values[ a ] = values[ b ];
			values[ b ] = t;
			update( ax, ay, az, delta, lower, upper );
			update( bx, by, bz, - delta, lower, upper );

		}

	}

	if ( onProgress ) onProgress( { pass: passes, accepted, energyChange } );
	return values;

}
