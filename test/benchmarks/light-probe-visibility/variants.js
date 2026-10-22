const exactTest = `
		visibility = float( 1 );
		for ( const blocker of grid.blockers ) {
			const c = Math.cos( blocker.angle ), s = Math.sin( blocker.angle );
			const start = probePosition.sub( vec3( ...blocker.center ) );
			const end = biased.sub( vec3( ...blocker.center ) );
			const origin = vec3( start.x.mul( c ).sub( start.z.mul( s ) ), start.y, start.x.mul( s ).add( start.z.mul( c ) ) ).toVar();
			const finish = vec3( end.x.mul( c ).sub( end.z.mul( s ) ), end.y, end.x.mul( s ).add( end.z.mul( c ) ) );
			const d = finish.sub( origin ).toVar();
			const inv = vec3( signNotZero( d.x ), signNotZero( d.y ), signNotZero( d.z ) ).div( d.abs().max( vec3( 1e-7 ) ) );
			const t0 = vec3( ...blocker.half ).negate().sub( origin ).mul( inv );
			const t1 = vec3( ...blocker.half ).sub( origin ).mul( inv );
			const lo = t0.min( t1 ), hi = t0.max( t1 );
			const near = lo.x.max( lo.y ).max( lo.z ).max( 0.0001 );
			const far = hi.x.min( hi.y ).min( hi.z ).min( 0.9999 );
			visibility = visibility.mul( select( far.greaterThanEqual( near ), float( 0 ), float( 1 ) ) );
		}
`;
export function patch( source, variant ) {

	if ( variant.startsWith( 'slack-' ) ) source = source.replace( 'const VISIBILITY_SLACK = 0.03;', `const VISIBILITY_SLACK = ${Number( variant.slice( 6 ) )};` );
	if ( variant === 'distance-32' ) source = source.replace( 'export const DISTANCE_RESOLUTION = 16;', 'export const DISTANCE_RESOLUTION = 32;' );
	if ( variant === 'crush' ) source = source.replace(
		'const weight = blend.x.mul( blend.y ).mul( blend.z ).mul( wrap ).mul( visibility ).toVar();',
		`const confidence = wrap.mul( visibility ).toVar();
		const crushed = select( confidence.lessThan( 0.2 ), confidence.mul( confidence.mul( confidence ).div( 0.04 ) ), confidence );
		const weight = blend.x.mul( blend.y ).mul( blend.z ).mul( crushed ).toVar();` );
	if ( variant === 'exact' || variant === 'region-mask' ) {

		source = source.replace( 'const visibility = chebyshevVisibility(', 'let visibility = chebyshevVisibility(' );
		source = source.replace( '\n\t\t// Soft backface weight:', ( variant === 'exact' ? exactTest : '\n\t\tconst regionAngle = ( grid.blockers[ 0 ] || { angle: 0 } ).angle;\n\t\tconst regionNormal = vec3( Math.cos( regionAngle ), 0, - Math.sin( regionAngle ) );\n\t\tvisibility = visibility.mul( select( probePosition.dot( regionNormal ).mul( positionWorld.dot( regionNormal ) ).greaterThanEqual( 0 ), float( 1 ), float( 0 ) ) );\n' ) + '\n\t\t// Soft backface weight:' );
		source = source.replace( 'const grid = {', 'const grid = {\n\t\t\tblockers: light._testBlockers || [],' );

	}

	return source;

}

// Independent diagnostic: build the directional irradiance atlas by cosine
// sampling captured radiance, bypassing its low-order SH representation.
export function patchBake( source, variant ) {

	if ( variant !== 'direct-convolution' ) return source;
	const helper = `
function directIrradianceNode( faces, size, batchStart, batchCount ) {
	return Fn( () => {
		const tile = octTileTexel( ivec2( screenCoordinate.xy ), IRRADIANCE_RESOLUTION );
		const slot = tile.probeIndex.sub( batchStart ).toVar();
		Discard( slot.lessThan( 0 ).or( slot.greaterThanEqual( batchCount ) ) );
		const e = vec2( float( tile.x ).add( 0.5 ), float( tile.y ).add( 0.5 ) ).div( IRRADIANCE_RESOLUTION ).mul( 2 ).sub( 1 );
		const n = octDecode( e ).toVar();
		const axis = select( n.x.abs().lessThan( 0.9 ), vec3( 1, 0, 0 ), vec3( 0, 1, 0 ) );
		const tangent = n.cross( axis ).normalize().toVar();
		const bitangent = n.cross( tangent ).toVar();
		const sum = vec3( 0 ).toVar();
		Loop( 256, ( { i } ) => {
			const u = float( i ).add( 0.5 ).div( 256 );
			const radius = u.sqrt();
			const phi = float( i ).mul( GOLDEN_ANGLE );
			const direction = tangent.mul( radius.mul( phi.cos() ) ).add( bitangent.mul( radius.mul( phi.sin() ) ) ).add( n.mul( u.oneMinus().sqrt() ) ).toVar();
			const { coord } = captureTexel( direction, size, slot );
			sum.addAssign( faces.load( coord ).rgb );
		} );
		return vec4( sum.mul( Math.PI / 256 ), 1 );
	} )();
}
`;
	source = source.replace( 'function irradianceNode(', helper + '\nfunction irradianceNode(' );
	source = source.replace( 'this._updateIrradiance( renderer, start, end );', '// Direct irradiance was written during each capture batch.' );
	source = source.replace( '\n\t\t\tif ( recordDistances === true ) {', `
			if ( this._directMaterial === undefined || this._directSize !== _faceSize ) {
				if ( this._directMaterial ) this._directMaterial.dispose();
				this._directMaterial = new NodeMaterial();
				this._directMaterial.outputNode = directIrradianceNode( _faceNode, _faceSize, _batchStartUniform, _batchCountUniform );
				this._directMaterial.depthTest = false;
				this._directMaterial.depthWrite = false;
				this._directSize = _faceSize;
			}
			_batchCountUniform.value = batchCount;
			const output = this._irradianceTarget;
			const firstRow = Math.floor( batchStart / DISTANCE_COLUMNS );
			const lastRow = Math.floor( ( batchStart + batchCount - 1 ) / DISTANCE_COLUMNS );
			output.viewport.set( 0, firstRow * IRRADIANCE_TILE, output.width, ( lastRow - firstRow + 1 ) * IRRADIANCE_TILE );
			renderer.setRenderTarget( output );
			_quad.material = this._directMaterial;
			_quad.render( renderer );

			if ( recordDistances === true ) {` );
	source = source.replace( '\n\t\tsuper.dispose();', '\n\t\tif ( this._directMaterial ) this._directMaterial.dispose();\n\t\tsuper.dispose();' );
	return source;

}
