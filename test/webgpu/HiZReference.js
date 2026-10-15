/** Independent CPU oracle. Last child owns the trailing row/column at odd sizes. */
export function reduceDepth( source, width, height, reversedDepth ) {

	const levels = [ { width, height, data: source.slice() } ];
	while ( width > 1 || height > 1 ) {

		const previous = levels[ levels.length - 1 ];
		const nextWidth = Math.max( 1, Math.floor( width / 2 ) );
		const nextHeight = Math.max( 1, Math.floor( height / 2 ) );
		const data = new Float32Array( nextWidth * nextHeight );
		for ( let y = 0; y < nextHeight; y ++ ) {

			for ( let x = 0; x < nextWidth; x ++ ) {

				let value = reversedDepth ? Infinity : - Infinity;
				const endX = x === nextWidth - 1 ? width : ( x + 1 ) * 2;
				const endY = y === nextHeight - 1 ? height : ( y + 1 ) * 2;
				for ( let sy = y * 2; sy < endY; sy ++ ) {

					for ( let sx = x * 2; sx < endX; sx ++ ) {

						const sample = previous.data[ sy * width + sx ];
						value = reversedDepth ? Math.min( value, sample ) : Math.max( value, sample );

					}

				}

				data[ y * nextWidth + x ] = value;

			}

		}

		width = nextWidth;
		height = nextHeight;
		levels.push( { width, height, data } );

	}

	return levels;

}

/** Sampled screen-space ray oracle, not a replacement for SSRNode's ray marcher.
 * Checks a fixed-depth ray against tiny point surfaces, with a finite thickness.
 * Hi-Z may reject a sample only when the farthest surface cannot meet the ray.
 */
export function traceSamples( levels, samples, depth, reversedDepth, hierarchical ) {

	const thickness = 1e-6;
	for ( let i = 0; i < samples.length; i ++ ) {

		let [ x, y ] = samples[ i ];
		let rejected = false;
		if ( hierarchical ) {

			for ( let level = 1; level < levels.length; level ++ ) {

				const mip = levels[ level ];
				x = Math.min( Math.floor( x / 2 ), mip.width - 1 );
				y = Math.min( Math.floor( y / 2 ), mip.height - 1 );
				const bound = mip.data[ y * mip.width + x ];
				if ( reversedDepth ? depth < bound - thickness : depth > bound + thickness ) {

					rejected = true;
					break;

				}

			}

		}

		if ( rejected ) continue;
		const [ sx, sy ] = samples[ i ];
		if ( Math.abs( levels[ 0 ].data[ sy * levels[ 0 ].width + sx ] - depth ) <= thickness ) return i;

	}

	return - 1;

}
