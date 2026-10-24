/**
 * Independent CPU geometry reference for screen-available diffuse transport.
 * Suitable for browser import, or serializing this function with toString().
 * No GPU work is submitted. Call after throughput measurement and GPU drain.
 *
 * Buffers may be flat RGBA arrays or { data, width, height, origin }, with origin
 * 'top-left' by default ('bottom-left' is also supported). raw/filtered RGB is
 * conditional irradiance and alpha is coverage, as in SSGINode.outputCoverage;
 * set buffer.premultiplied=true if RGB already includes coverage. Source radiance
 * RGB must be linear outgoing radiance, with no receiver albedo or tonemapping.
 *
 * This tests actual triangle visibility, independently of the screen-space
 * integrator. Its lighting reference still reuses screen radiance: it assumes
 * diffuse direction-independent emission and does not path trace full lighting,
 * hidden/offscreen illumination, or multiple bounces. Primary/screen visibility honors material sidedness; secondary triangles block
 * from either side. Backfaces contribute coverage but use backfaceLighting for
 * irradiance, matching the screen integrator. Static mesh normals are
 * interpolated; skinning/normal maps/alpha masks/transmission are not modeled.
 */
export async function ssgiGeometryReference( {
	scene, model = scene, meshes: suppliedMeshes, camera, width, height, radius,
	gridWidth = 32, gridHeight = 24, rays = 128, backfaceLighting = 0,
	radiance = null, raw = null, filtered = null,
	threeModule = '/src/Three.WebGPU.js'
} ) {

	const { DoubleSide, Matrix3, Matrix4, Raycaster, Triangle, Vector2, Vector3, WebGPUCoordinateSystem } = await import( threeModule );
	model?.updateMatrixWorld( true );
	camera.updateMatrixWorld( true );
	const meshes = suppliedMeshes || [];
	if ( suppliedMeshes ) for ( const mesh of suppliedMeshes ) mesh.updateWorldMatrix( true, false );
	if ( ! suppliedMeshes ) model.traverseVisible( object => {

		if ( object.isMesh ) meshes.push( object );

	} );

	const materials = new Map();
	for ( const mesh of meshes ) for ( const material of Array.isArray( mesh.material ) ? mesh.material : [ mesh.material ] ) {

		if ( material && ! materials.has( material ) ) materials.set( material, material.side );

	}

	const primary = new Raycaster();
	const secondary = new Raycaster();
	const screen = new Raycaster();
	const ndc = new Vector2();
	const epsilon = Math.max( radius * 1e-5, 1e-7 );
	const cameraSpace = new Vector3();
	const projected = new Vector3();
	const instanceMatrix = new Matrix4();
	const worldMatrix = new Matrix4();
	const normalMatrix = new Matrix3();
	const depthMinimum = camera.coordinateSystem === WebGPUCoordinateSystem ? 0 : - 1;
	const coverage = new Array( gridWidth * gridHeight ).fill( 0 );
	const irradiance = radiance ? new Array( coverage.length * 3 ).fill( 0 ) : null;
	const valid = new Array( coverage.length ).fill( false );
	const points = new Array( coverage.length ).fill( null );
	const comparisons = {};
	for ( const [ name, buffer ] of [[ 'raw', raw ], [ 'filtered', filtered ]] ) {

		if ( buffer ) comparisons[ name ] = { coverageSquaredError: 0, coverageAbsoluteError: 0, irradianceSquaredError: 0, count: 0 };

	}

	const sampleBuffer = ( buffer, u, v ) => {

		const data = buffer.data || buffer;
		const bw = buffer.width || width, bh = buffer.height || height;
		const x = Math.min( bw - 1, Math.max( 0, Math.floor( u * bw ) ) );
		let y = Math.min( bh - 1, Math.max( 0, Math.floor( v * bh ) ) );
		if ( buffer.origin === 'bottom-left' ) y = bh - 1 - y;

		return Array.from( data.slice( 4 * ( y * bw + x ), 4 * ( y * bw + x ) + 4 ) );

	};

	const inClip = point => {

		projected.copy( point ).project( camera );
		return projected.z >= depthMinimum && projected.z <= 1 && Math.abs( projected.x ) <= 1 && Math.abs( projected.y ) <= 1;

	};

	const cameraHit = raycaster => raycaster.intersectObjects( meshes, false ).find( hit => inClip( hit.point ) );
	const secondaryHit = () => {

		// Preserve rendered sidedness for primary and screen visibility. Inward
		// rooms often use BackSide: making those primary rays double-sided sees
		// the culled outside front wall instead of the rendered interior.
		// Secondary triangles still block from either side, including backfaces.
		for ( const material of materials.keys() ) material.side = DoubleSide;
		try {

			return secondary.intersectObjects( meshes, false )[ 0 ];

		} finally {

			for ( const [ material, side ] of materials ) material.side = side;

		}

	};

	const worldNormal = ( hit, direction ) => {

		const attribute = hit.object.geometry.attributes.normal;
		const face = hit.face;
		const normal = attribute && hit.barycoord ? Triangle.getInterpolatedAttribute( attribute, face.a, face.b, face.c, hit.barycoord, new Vector3() ) : face.normal.clone();
		worldMatrix.copy( hit.object.matrixWorld );
		if ( hit.instanceId !== undefined ) {

			hit.object.getMatrixAt( hit.instanceId, instanceMatrix );
			worldMatrix.multiply( instanceMatrix );

		}

		normal.applyNormalMatrix( normalMatrix.getNormalMatrix( worldMatrix ) );
		if ( normal.dot( direction ) > 0 ) normal.negate();

		return normal;

	};

	const radicalInverse = value => {

		let result = 0, scale = 0.5;
		while ( value > 0 ) {

			result += ( value & 1 ) * scale; value >>>= 1; scale *= 0.5;

		}

		return result;

	};

	const started = Date.now();
	try {

		for ( let gy = 0; gy < gridHeight; gy ++ ) for ( let gx = 0; gx < gridWidth; gx ++ ) {

			const index = gy * gridWidth + gx;
			const u = ( gx + 0.5 ) / gridWidth, v = ( gy + 0.5 ) / gridHeight;
			primary.setFromCamera( ndc.set( u * 2 - 1, 1 - v * 2 ), camera );
			const receiver = cameraHit( primary );
			if ( ! receiver ) continue;

			valid[ index ] = true;
			points[ index ] = receiver.point.toArray();
			const normal = worldNormal( receiver, primary.ray.direction );
			const tangent = new Vector3().crossVectors( Math.abs( normal.z ) < 0.9 ? new Vector3( 0, 0, 1 ) : new Vector3( 0, 1, 0 ), normal ).normalize();
			const bitangent = new Vector3().crossVectors( normal, tangent );
			const origin = receiver.point.clone().addScaledVector( normal, epsilon );
			let available = 0;
			for ( let ray = 0; ray < rays; ray ++ ) {

				// Cosine hemisphere sampling: each ray has equal irradiance weight.
				const radial = Math.sqrt( ( ray + 0.5 ) / rays );
				const angle = 2 * Math.PI * radicalInverse( ray );
				const direction = tangent.clone().multiplyScalar( radial * Math.cos( angle ) ).addScaledVector( bitangent, radial * Math.sin( angle ) ).addScaledVector( normal, Math.sqrt( 1 - radial * radial ) );
				secondary.set( origin, direction );
				secondary.near = epsilon;
				secondary.far = radius;
				const hit = secondaryHit();
				if ( ! hit || hit.point.distanceToSquared( receiver.point ) > radius * radius || ! inClip( hit.point ) ) continue;

				// Read the same finest pixel that supplies screen radiance. A separate
				// camera ray establishes that its first surface is this world hit.
				const hu = projected.x * 0.5 + 0.5, hv = 0.5 - projected.y * 0.5;
				const px = Math.min( width - 1, Math.floor( hu * width ) ), py = Math.min( height - 1, Math.floor( hv * height ) );
				screen.setFromCamera( ndc.set( ( px + 0.5 ) / width * 2 - 1, 1 - ( py + 0.5 ) / height * 2 ), camera );
				const seen = cameraHit( screen );
				if ( ! seen || seen.object !== hit.object || seen.instanceId !== hit.instanceId ) continue;

				cameraSpace.copy( hit.point ).applyMatrix4( camera.matrixWorldInverse );
				const footprint = Math.abs( cameraSpace.z ) * Math.max( 2 / ( Math.abs( camera.projectionMatrix.elements[ 0 ] ) * width ), 2 / ( Math.abs( camera.projectionMatrix.elements[ 5 ] ) * height ) );
				if ( seen.point.distanceTo( hit.point ) > footprint * Math.SQRT2 * 1.5 + epsilon * 2 ) continue;

				available ++;
				if ( radiance ) {

					const color = sampleBuffer( radiance, hu, hv );
					// Use the side that the source pixel actually rendered. A known
					// backface still consumes visibility coverage but contributes no
					// outgoing light toward the receiver at backfaceLighting=0.
					const sourceNormal = worldNormal( seen, screen.ray.direction );
					const facing = sourceNormal.dot( direction ) < 0 ? 1 : backfaceLighting;
					for ( let channel = 0; channel < 3; channel ++ ) irradiance[ 3 * index + channel ] += Math.PI * color[ channel ] * facing / rays;

				}

			}

			coverage[ index ] = available / rays;
			for ( const [ name, buffer ] of [[ 'raw', raw ], [ 'filtered', filtered ]] ) {

				if ( ! buffer ) continue;

				const value = sampleBuffer( buffer, u, v );
				const statistics = comparisons[ name ];
				const error = value[ 3 ] - coverage[ index ];
				statistics.coverageSquaredError += error * error;
				statistics.coverageAbsoluteError += Math.abs( error );
				if ( irradiance ) for ( let channel = 0; channel < 3; channel ++ ) {

					const difference = value[ channel ] * ( buffer.premultiplied ? 1 : value[ 3 ] ) - irradiance[ index * 3 + channel ];
					statistics.irradianceSquaredError += difference * difference;

				}

				statistics.count ++;

			}

		}

	} finally {

		for ( const [ material, side ] of materials ) material.side = side;

	}

	for ( const statistics of Object.values( comparisons ) ) {

		const count = Math.max( statistics.count, 1 );
		statistics.coverageRMSE = Math.sqrt( statistics.coverageSquaredError / count );
		statistics.coverageMAE = statistics.coverageAbsoluteError / count;
		if ( irradiance ) statistics.irradianceRMSE = Math.sqrt( statistics.irradianceSquaredError / ( count * 3 ) );

	}

	return { referenceVersion: 'sidedness-and-source-facing-v2', gridWidth, gridHeight, rays, radius, valid, points, coverage, irradiance, comparisons, milliseconds: Date.now() - started, backfaceLighting, method: 'CPU triangle first-hit, rendered primary sidedness, double-sided secondary blocking, cosine hemisphere; screen radiance approximation, no full path-traced lighting' };

}
