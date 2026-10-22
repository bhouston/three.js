import { DataUtils, Vector3 } from '/build/three.webgpu.js';

// CPU inspection of the eight shader inputs at a selected receiver position.
// The independent segment/box test labels false visibility estimates.
export function inspectQuery( probes, pixels, position, normal, camera, blockers ) {

	const min = probes.boundingBox.min;
	const res = probes.resolution;
	const spacing = probes.boundingBox.getSize( new Vector3() ).divide( res.clone().subScalar( 1 ) );
	const clearance = Math.min( ...spacing.toArray() );
	const biased = position.clone().addScaledVector( normal, clearance * probes.normalBias ).addScaledVector( camera.clone().sub( position ).normalize(), clearance * probes.viewBias );
	const gridPosition = biased.clone().sub( min ).divide( spacing ).clamp( new Vector3(), res.clone().subScalar( 1 ) );
	const base = gridPosition.clone().floor().min( res.clone().subScalar( 2 ) );
	const alpha = gridPosition.clone().sub( base );
	const target = probes._distanceTarget;
	const tile = target.width / 32;
	const resolution = tile - 2;
	const sample = ( direction, index ) => {

		const e = direction.clone().divideScalar( Math.abs( direction.x ) + Math.abs( direction.y ) + Math.abs( direction.z ) );
		let x = e.x, y = e.y;
		if ( e.z < 0 ) {

			x = ( 1 - Math.abs( e.y ) ) * ( e.x < 0 ? - 1 : 1 ); y = ( 1 - Math.abs( e.x ) ) * ( e.y < 0 ? - 1 : 1 );

		}

		x = ( index % 32 ) * tile + 0.5 + ( x * 0.5 + 0.5 ) * resolution;
		y = Math.floor( index / 32 ) * tile + 0.5 + ( y * 0.5 + 0.5 ) * resolution;
		const ix = Math.floor( x ), iy = Math.floor( y );
		const fx = x - ix, fy = y - iy;
		const result = [ 0, 0 ];
		for ( let dy = 0; dy < 2; dy ++ ) for ( let dx = 0; dx < 2; dx ++ ) for ( let c = 0; c < 2; c ++ ) {

			const index = ( ( iy + dy ) * target.width + ix + dx ) * 2 + c;
			result[ c ] += DataUtils.fromHalfFloat( pixels[ index ] ) * ( dx ? fx : 1 - fx ) * ( dy ? fy : 1 - fy );

		}

		return result;

	};

	const intersects = ( from, to, box ) => {

		const transform = p => {

			const d = p.clone().sub( new Vector3( ...box.center ) );
			return [ d.x * Math.cos( box.angle ) - d.z * Math.sin( box.angle ), d.y, d.x * Math.sin( box.angle ) + d.z * Math.cos( box.angle ) ];

		};

		const a = transform( from ), b = transform( to );
		let near = 0.0001, far = 0.9999;
		for ( let i = 0; i < 3; i ++ ) {

			const d = b[ i ] - a[ i ];
			if ( Math.abs( d ) < 1e-9 ) {

				if ( Math.abs( a[ i ] ) > box.half[ i ] ) return false; continue;

			}

			const t0 = ( - box.half[ i ] - a[ i ] ) / d, t1 = ( box.half[ i ] - a[ i ] ) / d;
			near = Math.max( near, Math.min( t0, t1 ) ); far = Math.min( far, Math.max( t0, t1 ) );

		}

		return far >= near;

	};

	const corners = [];
	for ( let i = 0; i < 8; i ++ ) {

		const offset = new Vector3( i & 1, ( i >> 1 ) & 1, ( i >> 2 ) & 1 );
		const cell = base.clone().add( offset );
		const p = cell.clone().multiply( spacing ).add( min );
		const toReceiver = biased.clone().sub( p );
		const distance = toReceiver.length();
		const index = cell.x + cell.z * res.x + cell.y * res.x * res.z;
		const moments = sample( toReceiver.clone().normalize(), index );
		const variance = Math.max( 1e-6, moments[ 1 ] - moments[ 0 ] ** 2 );
		const delta = Math.max( 0, distance / probes._distanceScale - moments[ 0 ] - 0.03 );
		const visibility = ( variance / ( variance + delta ** 2 ) ) ** 3;
		const wrap = ( normal.dot( p.clone().sub( position ).normalize() ) * 0.5 + 0.5 ) ** 2 + 0.2;
		const blend = offset.toArray().map( ( v, j ) => v ? alpha.getComponent( j ) : 1 - alpha.getComponent( j ) ).reduce( ( a, b ) => a * b, 1 );
		corners.push( { position: p.toArray(), meanDistance: moments[ 0 ] * probes._distanceScale, distance, variance: variance * probes._distanceScale ** 2, visibility, weight: blend * wrap * visibility, exactBlocked: blockers.some( b => intersects( p, biased, b ) ) } );

	}

	return { position: position.toArray(), normal: normal.toArray(), biased: biased.toArray(), corners };

}
