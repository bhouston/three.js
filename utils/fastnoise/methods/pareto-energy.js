/**
 * Experimental Pareto refinement: each accepted swap cannot increase spatial or
 * temporal aggregate CDF energy, or any selected continuous feature's energy.
 * This guards these exact objectives only, not all held-out rendering metrics.
 * Two Float64 caches cost 2 * voxelCount * (thresholdCount + features) * 8 bytes.
 */
export function generate( { size = 128, depth = 32, seed = 1, passes = 4, temporalWeight = 1, temporalAlpha = 0.1, thresholdCount = 31, initialVolume = null, spatialSigmas = [ 1 ], featureGuards = [ 'identity', 'squared', 'fourth', 'cosine' ], onProgress = null } = {} ) {

	if ( ! Number.isInteger( size ) || size < 1 || ! Number.isInteger( depth ) || depth < 1 ) throw new RangeError( 'size and depth must be positive integers.' );
	if ( ! Number.isInteger( passes ) || passes < 0 ) throw new RangeError( 'passes must be a nonnegative integer.' );
	if ( ! Number.isInteger( thresholdCount ) || thresholdCount < 0 ) throw new RangeError( 'thresholdCount must be a nonnegative integer.' );
	if ( ! Number.isFinite( temporalWeight ) || temporalWeight < 0 || ! Number.isFinite( temporalAlpha ) || temporalAlpha <= 0 || temporalAlpha > 1 ) throw new RangeError( 'Invalid temporal filter.' );

	const plane = size * size;
	const count = plane * depth;
	if ( initialVolume !== null && ( initialVolume.length !== count || Array.from( initialVolume ).some( v => ! Number.isFinite( v ) || v < 0 || v >= 1 ) ) ) throw new RangeError( 'Invalid initialVolume.' );
	if ( ! Array.isArray( spatialSigmas ) || spatialSigmas.length === 0 || spatialSigmas.some( s => ! Number.isFinite( s ) || s <= 0 ) ) throw new RangeError( 'Invalid spatialSigmas.' );
	const values = initialVolume === null ? new Float32Array( count ) : Float32Array.from( initialVolume );
	if ( thresholdCount < 1 ) throw new RangeError( 'thresholdCount must be positive.' );
	const featureFunctions = { identity: x => x, squared: x => x * x, fourth: x => x ** 4, cosine: x => Math.cos( 2 * Math.PI * x ) };
	if ( ! Array.isArray( featureGuards ) || featureGuards.some( name => ! Object.hasOwn( featureFunctions, name ) ) ) throw new RangeError( 'Invalid featureGuards.' );
	const features = featureGuards.map( name => featureFunctions[ name ] );
	const channels = thresholdCount + features.length;
	const potentials = [ new Float64Array( count * channels ), new Float64Array( count * channels ) ];
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
	const wrapIndex = ( x, y, z ) => wrap( z, depth ) * plane + wrap( y, size ) * size + wrap( x, size );
	const kernels = [ [], [] ];
	for ( let y = 0; y < size; y ++ ) for ( let x = 0; x < size; x ++ ) if ( spatial[ y * size + x ] !== 0 ) kernels[ 0 ].push( [ x, y, 0, spatial[ y * size + x ] ] );
	for ( let z = 0; z < depth; z ++ ) if ( temporal[ z ] !== 0 ) kernels[ 1 ].push( [ 0, 0, z, temporal[ z ] ] );
	const lookup = kernels.map( kernel => kernel.map( ( [ dx, dy, dz, weight ] ) => ( {
		x: Int32Array.from( { length: size }, ( _, i ) => wrap( i + dx, size ) ),
		y: Int32Array.from( { length: size }, ( _, i ) => wrap( i + dy, size ) * size ),
		z: Int32Array.from( { length: depth }, ( _, i ) => wrap( i + dz, depth ) * plane ), weight
	} ) ) );
	function update( family, x, y, z, delta, lower, upper ) {

		const potential = potentials[ family ];
		for ( const entry of lookup[ family ] ) {

			const base = ( entry.x[ x ] + entry.y[ y ] + entry.z[ z ] ) * channels;
			for ( let c = lower; c < upper; c ++ ) potential[ base + c ] += delta * entry.weight;

		}

	}
	for ( let z = 0; z < depth; z ++ ) for ( let y = 0; y < size; y ++ ) for ( let x = 0; x < size; x ++ ) {

		const value = values[ wrapIndex( x, y, z ) ];
		for ( let family = 0; family < 2; family ++ ) {

			update( family, x, y, z, 1, Math.floor( value * thresholdCount + 0.5 ), thresholdCount );
			for ( let f = 0; f < features.length; f ++ ) update( family, x, y, z, features[ f ]( value ), thresholdCount + f, thresholdCount + f + 1 );

		}

	}
	let accepted = 0;
	const objectiveChanges = [ new Float64Array( features.length + 1 ), new Float64Array( features.length + 1 ) ];
	const changes = [ new Float64Array( features.length + 1 ), new Float64Array( features.length + 1 ) ];
	const featureDeltas = new Float64Array( features.length );
	function report( pass ) {

		if ( onProgress ) onProgress( { pass, accepted, objectives: [ 'cdf', ...featureGuards ], objectiveChanges: objectiveChanges.map( row => Array.from( row ) ) } );

	}
	for ( let step = 0; step < passes * count; step ++ ) {

		if ( step % count === 0 ) report( step / count );
		const a = Math.floor( random() * count ), b = Math.floor( random() * count );
		const az = Math.floor( a / plane ), bz = Math.floor( b / plane );
		const ay = Math.floor( a % plane / size ), by = Math.floor( b % plane / size );
		const ax = a % size, bx = b % size;
		const dx = wrap( bx - ax, size ), dy = wrap( by - ay, size ), dz = wrap( bz - az, depth );
		const ca = Math.floor( values[ a ] * thresholdCount + 0.5 ), cb = Math.floor( values[ b ] * thresholdCount + 0.5 );
		const lower = Math.min( ca, cb ), upper = Math.max( ca, cb ), delta = cb < ca ? 1 : - 1;
		for ( let f = 0; f < features.length; f ++ ) featureDeltas[ f ] = features[ f ]( values[ b ] ) - features[ f ]( values[ a ] );
		let allowed = true, totalChange = 0;
		for ( let family = 0; family < 2; family ++ ) {

			const potential = potentials[ family ];
			const self = family === 0 ? spatial[ 0 ] : temporal[ 0 ];
			const coupling = family === 0 ? ( dz === 0 ? spatial[ dy * size + dx ] : 0 ) : ( dx === 0 && dy === 0 ? temporal[ dz ] : 0 );
			let cdfChange = 0;
			for ( let c = lower; c < upper; c ++ ) cdfChange += 2 * delta * ( potential[ a * channels + c ] - potential[ b * channels + c ] + delta * ( self - coupling ) );
			changes[ family ][ 0 ] = cdfChange;
			if ( cdfChange > 1e-12 ) allowed = false;
			totalChange += cdfChange;
			for ( let f = 0; f < features.length; f ++ ) {

				const d = featureDeltas[ f ], c = thresholdCount + f;
				const change = 2 * d * ( potential[ a * channels + c ] - potential[ b * channels + c ] + d * ( self - coupling ) );
				changes[ family ][ f + 1 ] = change;
				if ( change > 1e-12 ) allowed = false;
				totalChange += change;

			}

		}
		if ( ! allowed || totalChange >= - 1e-12 ) continue;
		accepted ++;
		[ values[ a ], values[ b ] ] = [ values[ b ], values[ a ] ];
		for ( let family = 0; family < 2; family ++ ) {

			for ( let c = 0; c <= features.length; c ++ ) objectiveChanges[ family ][ c ] += changes[ family ][ c ];
			update( family, ax, ay, az, delta, lower, upper );
			update( family, bx, by, bz, - delta, lower, upper );
			for ( let f = 0; f < features.length; f ++ ) {

				update( family, ax, ay, az, featureDeltas[ f ], thresholdCount + f, thresholdCount + f + 1 );
				update( family, bx, by, bz, - featureDeltas[ f ], thresholdCount + f, thresholdCount + f + 1 );

			}

		}

	}
	report( passes );
	return values;

}
