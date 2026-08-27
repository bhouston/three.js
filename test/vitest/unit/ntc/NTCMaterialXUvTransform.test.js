import { describe, expect, it } from 'vitest';
import { Matrix3, Vector2 } from 'three';
import { findImageNode, inferAlbedoUvTransform, inferUvTransformFromImageNode } from '../../../../examples/jsm/ntc/training/NTCMaterialXUvTransform.js';

// Minimal duck-typed stand-ins for the `MaterialXNode`/`MaterialXDocument`
// API this module reads (`getChildByName`/`children`/`hasReference`/
// `referencePath`/`isConst`/`getVector`/`materialX.getMaterialXNode`) - see
// examples/jsm/loaders/materialx/MaterialXDocument.js. Building a real
// MaterialX document just to test the pure graph-walk/math logic here would
// couple this test to the XML parser instead of the heuristic itself.

function constInput( name, valueString ) {

	return {
		name,
		isConst: true,
		hasReference: false,
		referencePath: null,
		value: valueString,
		getVector() {

			return valueString.split( /[,\s]+/ ).filter( ( s ) => s !== '' ).map( Number );

		}
	};

}

function refInput( name, referencePath ) {

	return {
		name,
		isConst: false,
		hasReference: true,
		referencePath,
		value: null,
		getVector() {

			return [];

		}
	};

}

function makeNode( element, nodePath, children = [] ) {

	return {
		element,
		nodePath,
		children,
		getChildByName( name ) {

			return children.find( ( child ) => child.name === name );

		}
	};

}

function makeMaterialX( nodesByPath, uvSpace = 'bottom-left' ) {

	return {
		uvSpace,
		getMaterialXNode( path ) {

			return nodesByPath[ path ] || null;

		}
	};

}

// A nodegraph's `<output name="out" nodename="...">` is a pure passthrough:
// its own connection lives directly on itself (`nodeName`/`referencePath`),
// not on a child `<input>` - see `resolveUpstreamNode`'s doc comment.
function makeOutputNode( nodePath, referencePath ) {

	return {
		element: 'output',
		nodePath,
		hasReference: true,
		referencePath,
		children: [],
		getChildByName() {

			return undefined;

		}
	};

}

describe( 'Addons > NTC > NTCMaterialXUvTransform', () => {

	it( 'returns identity when the surface shader has no base_color input', () => {

		const materialX = makeMaterialX( {} );
		const surfaceShaderNode = makeNode( 'standard_surface', 'surface', [] );

		const matrix = inferAlbedoUvTransform( materialX, surfaceShaderNode );

		expect( matrix.equals( new Matrix3() ) ).toBe( true );

	} );

	it( 'returns identity when no image/tiledimage node feeds base_color', () => {

		const constantColorNode = makeNode( 'constant', 'const1', [ constInput( 'value', '1, 0, 0' ) ] );
		const nodesByPath = { const1: constantColorNode };
		const materialX = makeMaterialX( nodesByPath );

		const surfaceShaderNode = makeNode( 'standard_surface', 'surface', [ refInput( 'base_color', 'const1' ) ] );

		const matrix = inferAlbedoUvTransform( materialX, surfaceShaderNode );

		expect( matrix.equals( new Matrix3() ) ).toBe( true );

	} );

	it( 'a plain <image> with a default texcoord (no upstream transform) yields identity', () => {

		const imageNode = makeNode( 'image', 'image1', [] ); // no 'texcoord' input at all
		const nodesByPath = { image1: imageNode };
		const materialX = makeMaterialX( nodesByPath );

		const matrix = inferUvTransformFromImageNode( materialX, imageNode );

		expect( matrix.equals( new Matrix3() ) ).toBe( true );

	} );

	it( 'finds the <image> node feeding base_color and infers a <rotate2d> transform', () => {

		const texcoordNode = makeNode( 'texcoord', 'uv0', [] );
		const rotateNode = makeNode( 'rotate2d', 'rotate1', [
			refInput( 'in', 'uv0' ),
			constInput( 'amount', '90' )
		] );
		const imageNode = makeNode( 'image', 'image1', [ refInput( 'texcoord', 'rotate1' ) ] );

		const nodesByPath = { uv0: texcoordNode, rotate1: rotateNode, image1: imageNode };
		const materialX = makeMaterialX( nodesByPath );

		const found = findImageNode( materialX, refInput( 'base_color', 'image1' ), 16 );
		expect( found ).toBe( imageNode );

		const surfaceShaderNode = makeNode( 'standard_surface', 'surface', [ refInput( 'base_color', 'image1' ) ] );
		const matrix = inferAlbedoUvTransform( materialX, surfaceShaderNode );

		// mx_rotate2d(90deg): x' = cos(90)*x + sin(90)*y = y ; y' = cos(90)*y - sin(90)*x = -x
		const p = new Vector2( 1, 0 ).applyMatrix3( matrix );
		expect( p.x ).toBeCloseTo( 0, 10 );
		expect( p.y ).toBeCloseTo( - 1, 10 );

	} );

	it( 'mirrors examples/materialx/standard_surface_rotate2d_test.mtlx - base_color exposed via a nodegraph <output> passthrough', () => {

		// <standard_surface><input name="base_color" nodegraph="rotate2d_test" output="out"/>
		// <nodegraph name="rotate2d_test">
		//   <texcoord name="texcoord1"/>
		//   <rotate2d name="rotate2d_1"><input name="in" nodename="texcoord1"/><input name="amount" value="45.0"/></rotate2d>
		//   <image name="rotated_image"><input name="texcoord" nodename="rotate2d_1"/></image>
		//   <output name="out" nodename="rotated_image"/>
		// </nodegraph>
		const texcoordNode = makeNode( 'texcoord', 'rotate2d_test/texcoord1', [] );
		const rotateNode = makeNode( 'rotate2d', 'rotate2d_test/rotate2d_1', [
			refInput( 'in', 'rotate2d_test/texcoord1' ),
			constInput( 'amount', '45.0' )
		] );
		const imageNode = makeNode( 'image', 'rotate2d_test/rotated_image', [
			refInput( 'texcoord', 'rotate2d_test/rotate2d_1' )
		] );
		const outputNode = makeOutputNode( 'rotate2d_test/out', 'rotate2d_test/rotated_image' );

		const nodesByPath = {
			'rotate2d_test/texcoord1': texcoordNode,
			'rotate2d_test/rotate2d_1': rotateNode,
			'rotate2d_test/rotated_image': imageNode,
			'rotate2d_test/out': outputNode
		};
		const materialX = makeMaterialX( nodesByPath );

		const surfaceShaderNode = makeNode( 'standard_surface', 'surface_shader1', [
			refInput( 'base_color', 'rotate2d_test/out' )
		] );

		const found = findImageNode( materialX, surfaceShaderNode.getChildByName( 'base_color' ), 16 );
		expect( found ).toBe( imageNode );

		const matrix = inferAlbedoUvTransform( materialX, surfaceShaderNode );
		expect( matrix.equals( new Matrix3() ) ).toBe( false ); // must not silently fall back to identity

		const p = new Vector2( 1, 0 ).applyMatrix3( matrix );
		const radians = 45 * Math.PI / 180;
		expect( p.x ).toBeCloseTo( Math.cos( radians ), 10 );
		expect( p.y ).toBeCloseTo( - Math.sin( radians ), 10 );

	} );

	it( 'a <place2d> node with all-constant inputs matches the mx_place2d srt formula', () => {

		const texcoordNode = makeNode( 'texcoord', 'uv0', [] );
		const placeNode = makeNode( 'place2d', 'place1', [
			refInput( 'texcoord', 'uv0' ),
			constInput( 'pivot', '0.5, 0.5' ),
			constInput( 'scale', '2, 2' ),
			constInput( 'rotate', '0' ),
			constInput( 'offset', '0.25, 0.1' )
			// operationorder omitted -> defaults to 0 (srt)
		] );
		const imageNode = makeNode( 'image', 'image1', [ refInput( 'texcoord', 'place1' ) ] );

		const nodesByPath = { uv0: texcoordNode, place1: placeNode, image1: imageNode };
		const materialX = makeMaterialX( nodesByPath );

		const matrix = inferUvTransformFromImageNode( materialX, imageNode );

		// srt: rotate2d((uv - pivot) / scale, 0) - offset + pivot, no rotation
		// => (uv - pivot) / scale - offset + pivot
		const uv = new Vector2( 1, 0 );
		const expected = new Vector2(
			( uv.x - 0.5 ) / 2 - 0.25 + 0.5,
			( uv.y - 0.5 ) / 2 - 0.1 + 0.5
		);

		const actual = uv.clone().applyMatrix3( matrix );
		expect( actual.x ).toBeCloseTo( expected.x, 10 );
		expect( actual.y ).toBeCloseTo( expected.y, 10 );

	} );

	it( 'concatenates a multiply (scale) followed by an add (offset) in graph-evaluation order', () => {

		const texcoordNode = makeNode( 'texcoord', 'uv0', [] );
		const multiplyNode = makeNode( 'multiply', 'mul1', [
			refInput( 'in1', 'uv0' ),
			constInput( 'in2', '2, 2' )
		] );
		const addNode = makeNode( 'add', 'add1', [
			refInput( 'in1', 'mul1' ),
			constInput( 'in2', '0.1, 0.2' )
		] );
		const imageNode = makeNode( 'image', 'image1', [ refInput( 'texcoord', 'add1' ) ] );

		const nodesByPath = { uv0: texcoordNode, mul1: multiplyNode, add1: addNode, image1: imageNode };
		const materialX = makeMaterialX( nodesByPath );

		const matrix = inferUvTransformFromImageNode( materialX, imageNode );

		// Evaluation order is texcoord -> multiply -> add, i.e. final = uv*2 + offset.
		const p = new Vector2( 1, 0 ).applyMatrix3( matrix );
		expect( p.x ).toBeCloseTo( 2.1, 10 );
		expect( p.y ).toBeCloseTo( 0.2, 10 );

	} );

	it( 'aborts to identity when a transform node parameter is not a hardcoded constant', () => {

		const texcoordNode = makeNode( 'texcoord', 'uv0', [] );
		const computedAmountNode = makeNode( 'time', 'time1', [] );
		const rotateNode = makeNode( 'rotate2d', 'rotate1', [
			refInput( 'in', 'uv0' ),
			refInput( 'amount', 'time1' ) // driven by another node - out of scope
		] );
		const imageNode = makeNode( 'image', 'image1', [ refInput( 'texcoord', 'rotate1' ) ] );

		const nodesByPath = { uv0: texcoordNode, time1: computedAmountNode, rotate1: rotateNode, image1: imageNode };
		const materialX = makeMaterialX( nodesByPath );

		const matrix = inferUvTransformFromImageNode( materialX, imageNode );

		expect( matrix.equals( new Matrix3() ) ).toBe( true );

	} );

	it( 'aborts to identity for an unrecognized node category in the chain', () => {

		const texcoordNode = makeNode( 'texcoord', 'uv0', [] );
		const noiseNode = makeNode( 'noise2d', 'noise1', [ refInput( 'texcoord', 'uv0' ) ] );
		const imageNode = makeNode( 'image', 'image1', [ refInput( 'texcoord', 'noise1' ) ] );

		const nodesByPath = { uv0: texcoordNode, noise1: noiseNode, image1: imageNode };
		const materialX = makeMaterialX( nodesByPath );

		const matrix = inferUvTransformFromImageNode( materialX, imageNode );

		expect( matrix.equals( new Matrix3() ) ).toBe( true );

	} );

	// The following three mirror examples/materialx/standard_surface_
	// scale2d_test.mtlx / standard_surface_scale_rotate2d_test.mtlx /
	// standard_surface_rotate_scale2d_test.mtlx - a non-uniform scale (3, 1),
	// isolated and then chained both ways with a 30deg rotate2d, so that a
	// commutation bug in how nodes are accumulated (order matters whenever
	// the scale is non-uniform) would be caught here rather than only
	// visually, the way the original place2d uvSpace bug was.

	function makeScaleNode( nodePath, inputReferencePath ) {

		return makeNode( 'multiply', nodePath, [
			refInput( 'in1', inputReferencePath ),
			constInput( 'in2', '3.0, 1.0' )
		] );

	}

	function makeRotateNode( nodePath, inputReferencePath, amountDegrees ) {

		return makeNode( 'rotate2d', nodePath, [
			refInput( 'in', inputReferencePath ),
			constInput( 'amount', String( amountDegrees ) )
		] );

	}

	it( 'a lone non-uniform <multiply> (scale) node is recognized in isolation', () => {

		const texcoordNode = makeNode( 'texcoord', 'uv0', [] );
		const scaleNode = makeScaleNode( 'scale1', 'uv0' );
		const imageNode = makeNode( 'image', 'image1', [ refInput( 'texcoord', 'scale1' ) ] );

		const nodesByPath = { uv0: texcoordNode, scale1: scaleNode, image1: imageNode };
		const materialX = makeMaterialX( nodesByPath );

		const matrix = inferUvTransformFromImageNode( materialX, imageNode );

		const p1 = new Vector2( 1, 0 ).applyMatrix3( matrix );
		const p2 = new Vector2( 0, 1 ).applyMatrix3( matrix );
		expect( p1.x ).toBeCloseTo( 3, 10 );
		expect( p1.y ).toBeCloseTo( 0, 10 );
		expect( p2.x ).toBeCloseTo( 0, 10 );
		expect( p2.y ).toBeCloseTo( 1, 10 );

	} );

	it( 'accumulates scale-then-rotate and rotate-then-scale into different (non-commuting) matrices', () => {

		const texcoordNode = makeNode( 'texcoord', 'uv0', [] );

		// texcoord -> scale(3,1) -> rotate(30deg) -> image
		const scaleThenRotateScale = makeScaleNode( 'scale1', 'uv0' );
		const scaleThenRotateRotate = makeRotateNode( 'rotate1', 'scale1', 30 );
		const scaleThenRotateImage = makeNode( 'image', 'image1', [ refInput( 'texcoord', 'rotate1' ) ] );
		const scaleThenRotateMaterialX = makeMaterialX( {
			uv0: texcoordNode, scale1: scaleThenRotateScale, rotate1: scaleThenRotateRotate, image1: scaleThenRotateImage
		} );

		// texcoord -> rotate(30deg) -> scale(3,1) -> image
		const rotateThenScaleRotate = makeRotateNode( 'rotate2', 'uv0', 30 );
		const rotateThenScaleScale = makeScaleNode( 'scale2', 'rotate2' );
		const rotateThenScaleImage = makeNode( 'image', 'image2', [ refInput( 'texcoord', 'scale2' ) ] );
		const rotateThenScaleMaterialX = makeMaterialX( {
			uv0: texcoordNode, rotate2: rotateThenScaleRotate, scale2: rotateThenScaleScale, image2: rotateThenScaleImage
		} );

		const scaleThenRotateMatrix = inferUvTransformFromImageNode( scaleThenRotateMaterialX, scaleThenRotateImage );
		const rotateThenScaleMatrix = inferUvTransformFromImageNode( rotateThenScaleMaterialX, rotateThenScaleImage );

		const radians = 30 * Math.PI / 180;
		const ca = Math.cos( radians );
		const sa = Math.sin( radians );

		// scale-then-rotate: rotate2d(scale(uv)) = rotate2d(3u, v)
		const expectedScaleThenRotate = new Vector2( ca * 3, - sa * 3 );
		// rotate-then-scale: scale(rotate2d(uv)) = (3 * rotate2d(uv).x, rotate2d(uv).y)
		const expectedRotateThenScale = new Vector2( 3 * ca, - sa );

		const p1 = new Vector2( 1, 0 ).applyMatrix3( scaleThenRotateMatrix );
		const p2 = new Vector2( 1, 0 ).applyMatrix3( rotateThenScaleMatrix );

		expect( p1.x ).toBeCloseTo( expectedScaleThenRotate.x, 10 );
		expect( p1.y ).toBeCloseTo( expectedScaleThenRotate.y, 10 );
		expect( p2.x ).toBeCloseTo( expectedRotateThenScale.x, 10 );
		expect( p2.y ).toBeCloseTo( expectedRotateThenScale.y, 10 );

		// The whole point: these two chains produce genuinely different
		// results because scale is non-uniform (3, 1) - if node order were
		// ignored (or accumulated in the wrong order), these would collapse
		// to the same matrix. (p1.x/p2.x coincide at this particular sample
		// point/angle - 3*cos(30deg) either way - but y does not: -3*sin(30deg)
		// vs -sin(30deg).)
		expect( p1.y ).not.toBeCloseTo( p2.y, 6 );

	} );

	it( 'honors a document\'s uvSpace: \'top-left\' - folds in the implicit texcoord Y-flip as the innermost step', () => {

		const texcoordNode = makeNode( 'texcoord', 'uv0', [] );
		const rotateNode = makeRotateNode( 'rotate1', 'uv0', 90 );
		const imageNode = makeNode( 'image', 'image1', [ refInput( 'texcoord', 'rotate1' ) ] );

		const nodesByPath = { uv0: texcoordNode, rotate1: rotateNode, image1: imageNode };

		const bottomLeftMatrix = inferUvTransformFromImageNode( makeMaterialX( nodesByPath, 'bottom-left' ), imageNode );
		const topLeftMatrix = inferUvTransformFromImageNode( makeMaterialX( nodesByPath, 'top-left' ), imageNode );

		// bottom-left (MaterialX's own native convention, no flip): rotate2d(90deg, (1,0)) = (0, -1) - see the earlier plain-rotate2d test.
		const pBottomLeft = new Vector2( 1, 0 ).applyMatrix3( bottomLeftMatrix );
		expect( pBottomLeft.x ).toBeCloseTo( 0, 10 );
		expect( pBottomLeft.y ).toBeCloseTo( - 1, 10 );

		// top-left: rotate2d(90deg, flipY(1,0)) = rotate2d(90deg, (1,1)) = (1, -1).
		const pTopLeft = new Vector2( 1, 0 ).applyMatrix3( topLeftMatrix );
		expect( pTopLeft.x ).toBeCloseTo( 1, 10 );
		expect( pTopLeft.y ).toBeCloseTo( - 1, 10 );

	} );

} );
