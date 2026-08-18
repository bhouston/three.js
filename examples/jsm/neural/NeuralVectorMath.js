// Plain-array 3-vector math shared by every CPU-side piece of neural-appearance
// that isn't running inside a TSL compute kernel: the JS-side model forward
// pass, direction sampling, and teacher-atlas readback. Vectors here are
// ordinary `[x, y, z]` arrays, not `THREE.Vector3` or TSL nodes - deliberately
// so, since these run in hot per-sample loops where allocating a Vector3 per
// call would be wasteful.

function dot( a, b ) {

	return a[ 0 ] * b[ 0 ] + a[ 1 ] * b[ 1 ] + a[ 2 ] * b[ 2 ];

}

function cross( a, b ) {

	return [
		a[ 1 ] * b[ 2 ] - a[ 2 ] * b[ 1 ],
		a[ 2 ] * b[ 0 ] - a[ 0 ] * b[ 2 ],
		a[ 0 ] * b[ 1 ] - a[ 1 ] * b[ 0 ]
	];

}

function normalize( value ) {

	const length = Math.hypot( value[ 0 ], value[ 1 ], value[ 2 ] ) || 1;

	return [ value[ 0 ] / length, value[ 1 ] / length, value[ 2 ] / length ];

}

export { dot, cross, normalize };
