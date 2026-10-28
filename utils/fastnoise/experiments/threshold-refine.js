/** Experimental multi-integrand, multi-filter quadratic swap refinement. */
export function generate( { size = 128, depth = 32, seed = 1, passes = 16, initial,
	thresholdCount = 15, scalarWeight = 4, thresholdWeight = 1, smoothWeight = 0.25,
	temporalWeight = 1, jointWeight = 0.5, preserveColumns = true, verify = false } = {} ) {

	if ( ! Number.isInteger( size ) || size < 1 || ! Number.isInteger( depth ) || depth < 1 ) throw new RangeError( 'Invalid dimensions.' );
	if ( ! Number.isInteger( passes ) || passes < 0 || ! Number.isInteger( thresholdCount ) || thresholdCount < 1 ) throw new RangeError( 'Invalid iteration or threshold count.' );
	for ( const weight of [ scalarWeight, thresholdWeight, smoothWeight, temporalWeight, jointWeight ] ) {

		if ( ! Number.isFinite( weight ) || weight < 0 ) throw new RangeError( 'Weights must be finite and nonnegative.' );

	}

	const plane = size * size, count = plane * depth, channels = thresholdCount + 3;
	if ( initial !== undefined && ( ! ( initial instanceof Float32Array ) || initial.length !== count || initial.some( v => ! Number.isFinite( v ) || v < 0 || v >= 1 ) ) ) throw new RangeError( 'initial must be a matching Float32Array in [0,1).' );
	if ( verify && count > 128 ) throw new RangeError( 'Verification is only for tiny volumes.' );
	let state = seed >>> 0;
	function random() {

		state = ( state + 0x6D2B79F5 ) >>> 0;
		let t = state;
		t = Math.imul( t ^ t >>> 15, t | 1 );
		t ^= t + Math.imul( t ^ t >>> 7, t | 61 );
		return ( ( t ^ t >>> 14 ) >>> 0 ) / 4294967296;

	}

	const values = initial === undefined ? new Float32Array( count ) : initial.slice();
	if ( initial === undefined ) {

		// Stratified initialization preserves every pixel's temporal histogram
		// initially; unconstrained global swaps subsequently trade that property
		// against the complete objective. No FAST generation is required.
		const order = new Uint32Array( depth );
		for ( let p = 0; p < plane; p ++ ) {

			for ( let z = 0; z < depth; z ++ ) order[ z ] = z;
			for ( let z = depth - 1; z > 0; z -- ) {

				const j = Math.floor( random() * ( z + 1 ) );
				[ order[ z ], order[ j ] ] = [ order[ j ], order[ z ] ];

			}

			for ( let z = 0; z < depth; z ++ ) values[ z * plane + p ] = Math.min( ( order[ z ] + random() ) / depth, 1 - 2 ** - 24 );

		}

	}

	const weights = Float64Array.from( { length: channels }, ( _, c ) => c < thresholdCount ? thresholdWeight / thresholdCount : c === thresholdCount ? scalarWeight : smoothWeight );
	function feature( v, c ) {

		if ( c < thresholdCount ) return v < ( c + 0.5 ) / thresholdCount ? 1 : 0;
		if ( c === thresholdCount ) return v - 0.5;
		return 0.5 * ( c === thresholdCount + 1 ? Math.cos( 2 * Math.PI * v ) : Math.sin( 2 * Math.PI * v ) );

	}

	const wrap = ( v, n ) => ( v % n + n ) % n;
	const kernel = new Float64Array( count );
	function add( x, y, z, w ) {

		kernel[ wrap( z, depth ) * plane + wrap( y, size ) * size + wrap( x, size ) ] += w;

	}

	// Sum of positive semidefinite autocorrelation kernels. The first is the
	// 3x3 binomial reconstruction filter, the second a finite EMA (alpha .25),
	// and the third a small joint space-time box reconstruction filter.
	const spatial = [ 1 / 6, 2 / 3, 1, 2 / 3, 1 / 6 ];
	for ( let y = - 2; y <= 2; y ++ ) {

		for ( let x = - 2; x <= 2; x ++ ) add( x, y, 0, spatial[ x + 2 ] * spatial[ y + 2 ] );

	}

	const radius = 7, rho = 0.75;
	let norm = 0;
	for ( let j = 0; j <= radius; j ++ ) norm += rho ** ( 2 * j );
	for ( let z = - radius; z <= radius; z ++ ) {

		let w = 0;
		for ( let j = 0; j + Math.abs( z ) <= radius; j ++ ) w += rho ** ( 2 * j + Math.abs( z ) );
		add( 0, 0, z, temporalWeight * w / norm );

	}

	for ( let z = - 1; z <= 1; z ++ ) {

		for ( let y = - 1; y <= 1; y ++ ) {

			for ( let x = - 1; x <= 1; x ++ ) add( x, y, z, jointWeight * ( x === 0 ? 1 : 0.5 ) * ( y === 0 ? 1 : 0.5 ) * ( z === 0 ? 1 : 0.5 ) );

		}

	}

	const lookup = [];
	for ( let i = 0; i < count; i ++ ) {

		if ( kernel[ i ] === 0 ) continue;
		const dx = i % size, dy = Math.floor( i % plane / size ), dz = Math.floor( i / plane );
		lookup.push( {
			x: Int32Array.from( { length: size }, ( _, x ) => wrap( x + dx, size ) ),
			y: Int32Array.from( { length: size }, ( _, y ) => wrap( y + dy, size ) * size ),
			z: Int32Array.from( { length: depth }, ( _, z ) => wrap( z + dz, depth ) * plane ),
			weight: kernel[ i ]
		} );

	}

	const potential = new Float64Array( count * channels );
	const delta = new Float64Array( channels );
	function update( index, differences, sign ) {

		const x = index % size, y = Math.floor( index % plane / size ), z = Math.floor( index / plane );
		for ( const s of lookup ) {

			const base = ( s.x[ x ] + s.y[ y ] + s.z[ z ] ) * channels, factor = s.weight * sign;
			for ( let c = 0; c < channels; c ++ ) potential[ base + c ] += differences[ c ] * factor;

		}

	}

	for ( let i = 0; i < count; i ++ ) {

		for ( let c = 0; c < channels; c ++ ) delta[ c ] = feature( values[ i ], c );
		update( i, delta, 1 );

	}

	function directEnergy() {

		let energy = 0;
		for ( let i = 0; i < count; i ++ ) {

			const x = i % size, y = Math.floor( i % plane / size ), z = Math.floor( i / plane );
			for ( const s of lookup ) {

				const j = s.x[ x ] + s.y[ y ] + s.z[ z ];
				for ( let c = 0; c < channels; c ++ ) energy += weights[ c ] * feature( values[ i ], c ) * feature( values[ j ], c ) * s.weight;

			}

		}

		return energy;

	}

	let previous = verify ? directEnergy() : 0;
	for ( let step = 0; step < passes * count; step ++ ) {

		const a = Math.floor( random() * count );
		const b = preserveColumns ? a % plane + Math.floor( random() * depth ) * plane : Math.floor( random() * count );
		const dx = wrap( b % size - a % size, size );
		const dy = wrap( Math.floor( b % plane / size ) - Math.floor( a % plane / size ), size );
		const dz = wrap( Math.floor( b / plane ) - Math.floor( a / plane ), depth );
		const correction = kernel[ 0 ] - kernel[ dz * plane + dy * size + dx ];
		let change = 0;
		for ( let c = 0; c < channels; c ++ ) {

			const d = delta[ c ] = feature( values[ b ], c ) - feature( values[ a ], c );
			change += weights[ c ] * d * ( potential[ a * channels + c ] - potential[ b * channels + c ] + d * correction );

		}

		if ( change >= - 1e-12 ) continue;
		[ values[ a ], values[ b ] ] = [ values[ b ], values[ a ] ];
		update( a, delta, 1 );
		update( b, delta, - 1 );
		if ( verify ) {

			const next = directEnergy();
			if ( Math.abs( next - previous - 2 * change ) > 1e-8 * Math.max( 1, Math.abs( next ) ) ) throw new Error( 'Cached swap delta disagrees with independent quadratic energy.' );
			previous = next;

		}

	}

	return values;

}


