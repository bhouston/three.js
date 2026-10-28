// Experimental extensions of the existing GPU generator, kept out of its default path.
export function specialize( source, { identityWeight = 0, frequencyWeight = 0, mixedWeight = 0, mixedKernel = 'truncated', swapFloor = 1, stratified = false, annealTemperature = 0, annealSteps = 1500 } = {} ) {

	if ( ! Number.isFinite( identityWeight ) || identityWeight < 0 || ! Number.isFinite( mixedWeight ) || mixedWeight < 0 || mixedWeight > 1 ) throw new RangeError( 'Invalid objective weights.' );
	if ( ! Number.isFinite( frequencyWeight ) || frequencyWeight < 0 || ! Number.isInteger( swapFloor ) || swapFloor < 1 ) throw new RangeError( 'Invalid feature/suppression weights.' );
	if ( ! [ 'truncated', 'binomial' ].includes( mixedKernel ) ) throw new RangeError( 'Unknown mixed kernel.' );
	if ( ! Number.isFinite( annealTemperature ) || annealTemperature < 0 || ! Number.isInteger( annealSteps ) || annealSteps < 1 ) throw new RangeError( 'Invalid annealing schedule.' );
	const distance = 'fn K2( a: f32, b: f32 ) -> f32 { return - abs( a - b ); }';
	const combine = 'return ( k === 0 ? fx * fy * sepWeight : 0 ) + ( i === 0 && j === 0 ? fz * ( 1 - sepWeight ) : 0 );';
	if ( ! source.includes( distance ) || ! source.includes( combine ) ) throw new Error( 'Generator source changed; review experimental specialization.' );
	if ( identityWeight || frequencyWeight ) source = source.replace( distance, `fn K2( a: f32, b: f32 ) -> f32 { let d = a - b; return - abs( d ) - ${identityWeight.toFixed( 8 )} * d * d - ${frequencyWeight.toFixed( 8 )} * ( 1.0 - cos( 6.28318530718 * d ) ); }` );
	if ( swapFloor > 1 ) {

		const acceptance = '( r % params.suppression ) == 0u';
		if ( ! source.includes( acceptance ) ) throw new Error( 'Swap acceptance source changed.' );
		source = source.replace( acceptance, `( r % max( params.suppression, ${swapFloor}u ) ) == 0u` );

	}

	if ( annealTemperature ) {

		const acceptance = 'if ( i < j && l < 0.0 &&';
		if ( ! source.includes( acceptance ) ) throw new Error( 'Annealing source changed.' );
		source = source.replace( acceptance, `let temperature = ${annealTemperature.toFixed( 8 )} * max( 0.0, 1.0 - f32( params.iteration ) / ${annealSteps.toFixed( 1 )} );
	let bound = - temperature * log( max( 0.0000001, f32( wang_hash_uint( r ^ 29389741u ) ) / 4294967296.0 ) );
	if ( i < j && l < bound &&` );

	}

	if ( stratified ) {

		const begin = source.indexOf( '// init: every slice' );
		const end = source.indexOf( 'const adapter =', begin );
		const partner = 'return vec3u( highIndex | lr, index.z );';
		if ( begin < 0 || end < 0 || ! source.includes( partner ) ) throw new Error( 'Initialization/pairing source changed.' );
		source = source.slice( 0, begin ) + `// Experimental per-pixel Latin hypercube, swaps restricted to time.
const N = S * S * Z, values = new Float32Array( N );
for ( let pixel = 0; pixel < S * S; pixel ++ ) {
 const perm = Array.from( { length: Z }, ( _, i ) => i );
 for ( let z = Z - 1; z > 0; z -- ) { const j = Math.floor( rand() * ( z + 1 ) ); [ perm[z], perm[j] ] = [ perm[j], perm[z] ]; }
 for ( let z = 0; z < Z; z ++ ) values[z*S*S+pixel] = Math.min( ( perm[z] + rand() ) / Z, 1 - 2 ** -24 );
}
` + source.slice( end );
		source = source.replace( partner, 'return vec3u( index.xy, index.z ^ ( wang_hash_uint( key_in.x ^ index.x ^ ( index.y * 1973u ) ) & ( Z - 1u ) ) );' );

	}

	if ( mixedWeight ) {

		const mixed = mixedKernel === 'binomial' ? 'const b = d => Math.abs(d) === 0 ? 6/16 : Math.abs(d) === 1 ? 4/16 : Math.abs(d) === 2 ? 1/16 : 0; const mixed = b(i)*b(j)*b(Math.min(k,Z-k));' : 'const mixed = Math.abs( i ) <= 2 && Math.abs( j ) <= 2 && Math.min( k, Z - k ) <= 2 ? fx * fy * fz : 0;';
		source = source.replace( combine, `const separate = ( k === 0 ? fx * fy * sepWeight : 0 ) + ( i === 0 && j === 0 ? fz * ( 1 - sepWeight ) : 0 );
	${mixed}
	return ( 1 - ${mixedWeight} ) * separate + ${mixedWeight} * mixed;` );

	}

	return source;

}
