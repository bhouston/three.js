import {
	ArrayCamera,
	BackSide,
	Box3,
	DataTexture,
	DepthTexture,
	DoubleSide,
	FloatType,
	FrontSide,
	HalfFloatType,
	Light,
	LinearFilter,
	MathUtils,
	MeshBasicNodeMaterial,
	NearestFilter,
	NodeMaterial,
	PerspectiveCamera,
	QuadMesh,
	RenderTarget,
	RenderTarget3D,
	RedFormat,
	RGBAFormat,
	RGFormat,
	SphericalHarmonics3,
	Vector3,
	Vector4
} from 'three/webgpu';

import {
	array,
	Discard,
	float,
	Fn,
	frontFacing,
	getShIrradianceAt,
	int,
	ivec2,
	ivec3,
	Loop,
	perspectiveDepthToViewZ,
	screenCoordinate,
	select,
	texture,
	texture3D,
	uniform,
	vec2,
	vec3,
	vec4
} from 'three/tsl';

import { LightProbeGridNode, ATLAS_PADDING, DISTANCE_COLUMNS, DISTANCE_RESOLUTION, DISTANCE_TILE, IRRADIANCE_RESOLUTION, IRRADIANCE_TILE, octDecode, octTileTexel, packGridSH, unpackGridSH } from '../tsl/lighting/LightProbeGridNode.js';
import { replaceSunLights, restoreSunLights } from './LightProbeGridUtils.js';

// Probes captured per render. Each probe adds six sub-cameras to the capture
// camera, so this trades per-render overhead against uniform buffer size.
const PROBES_PER_BATCH = 8;

// Probe classification: directions tested, and the fraction of back faces seen
// above which a probe counts as inside geometry.
const CLASSIFY_SAMPLES = 64;
const INSIDE_FRACTION = 0.25;

// A probe closer to a surface than this many capture near planes saw through it
// in its radiance capture, so it is hidden like a probe inside geometry. The
// classification capture uses a much closer near plane to measure this.
const CLEARANCE_NEAR_PLANES = 1.5;
const CLASSIFY_NEAR_SCALE = 0.01;

// Golden-angle increment for the equal-area Fibonacci sphere.
const GOLDEN_ANGLE = Math.PI * ( 3.0 - Math.sqrt( 5.0 ) );

// Per-face basis of the capture cameras: forward, right and up. A texel at
// NDC ( x, y ) of face f looks along forward + x * right + y * up.
const FACE_FORWARD = [[ 1, 0, 0 ], [ - 1, 0, 0 ], [ 0, 1, 0 ], [ 0, - 1, 0 ], [ 0, 0, 1 ], [ 0, 0, - 1 ]];
const FACE_RIGHT = [[ 0, 0, 1 ], [ 0, 0, - 1 ], [ - 1, 0, 0 ], [ - 1, 0, 0 ], [ - 1, 0, 0 ], [ 1, 0, 0 ]];
const FACE_UP = [[ 0, 1, 0 ], [ 0, 1, 0 ], [ 0, 0, - 1 ], [ 0, 0, 1 ], [ 0, 1, 0 ], [ 0, 1, 0 ]];

// Shared fullscreen-quad for the bake passes.
const _quad = /*@__PURE__*/ new QuadMesh();

// Reusable temp objects.
const _position = /*@__PURE__*/ new Vector3();
const _target = /*@__PURE__*/ new Vector3();
const _size = /*@__PURE__*/ new Vector3();
const _copyRegion = /*@__PURE__*/ new Box3();

// Bake materials, shared across grids so the shaders compile once, not per bake.
let _shMaterial = null;
let _shFaceSize = - 1;
let _faceNode = null;
let _weightsNode = null;
let _batchNode = null;
let _batchStartUniform = null;
let _batchCountUniform = null;
let _depthNode = null;
let _classifyNode = null;
let _classifyTarget = null;
let _classifyMaterials = null;
let _nearUniform = null;
let _farUniform = null;
let _clearanceUniform = null;
let _distanceScaleUniform = null;
let _distanceMaterial = null;
let _fractionMaterial = null;
let _resolutionUniform = null;
let _sliceZUniform = null;
let _repackMaterials = null;

// Bake render targets, pooled by size so rebakes don't churn allocations.
let _faceTarget = null;
let _faceSize = - 1;
let _batchTarget = null;
let _batchProbes = - 1;
let _batchColumns = - 1;

// The capture camera renders every face of a probe batch in one pass. It is
// never recreated: the camera uniform arrays reference its sub-camera matrices.
let _captureCamera = null;

/**
 * Returns the face block count for a face size: faces are projected in blocks
 * of 8 rows so each probe spreads over many fragments.
 *
 * @private
 * @param {number} size - The face resolution.
 * @return {number} The number of row blocks per face.
 */
function getFaceBlocks( size ) {

	return size % 8 === 0 ? size / 8 : 1;

}

/**
 * Builds the SH projection weights for a face size: for every face texel and
 * coefficient, the L2 SH basis in the texel's direction times the texel's exact
 * solid angle. Texel ( i, j ) of face f and coefficient c is stored at
 * ( f * size + i, c * size + j ).
 *
 * @private
 * @param {number} size - The face resolution.
 * @return {DataTexture} The weight texture.
 */
function createProjectionWeights( size ) {

	const width = 6 * size;
	const data = new Float32Array( width * 9 * size );
	const texelSize = 2 / size;
	const dir = new Vector3();
	const basis = new Array( 9 );

	// Signed area of the projected face region from its center to ( x, y ).
	const area = ( x, y ) => Math.atan2( x * y, Math.sqrt( x * x + y * y + 1 ) );

	for ( let f = 0; f < 6; f ++ ) {

		for ( let j = 0; j < size; j ++ ) {

			for ( let i = 0; i < size; i ++ ) {

				// Rows run top to bottom, so NDC y decreases with j.
				const x0 = i * texelSize - 1, x1 = x0 + texelSize;
				const y0 = 1 - j * texelSize, y1 = y0 - texelSize;
				const solidAngle = Math.abs( area( x0, y0 ) - area( x0, y1 ) - area( x1, y0 ) + area( x1, y1 ) );

				const x = x0 + texelSize * 0.5, y = y0 - texelSize * 0.5;
				dir.fromArray( FACE_FORWARD[ f ] );
				dir.x += x * FACE_RIGHT[ f ][ 0 ] + y * FACE_UP[ f ][ 0 ];
				dir.y += x * FACE_RIGHT[ f ][ 1 ] + y * FACE_UP[ f ][ 1 ];
				dir.z += x * FACE_RIGHT[ f ][ 2 ] + y * FACE_UP[ f ][ 2 ];
				dir.normalize();

				SphericalHarmonics3.getBasisAt( dir, basis );

				for ( let c = 0; c < 9; c ++ ) data[ ( c * size + j ) * width + f * size + i ] = basis[ c ] * solidAngle;

			}

		}

	}

	const weights = new DataTexture( data, width, 9 * size, RedFormat, FloatType );
	weights.needsUpdate = true;

	return weights;

}

/**
 * Returns the output node for the spherical-harmonic projection pass. Each
 * fragment of a batch row integrates one SH coefficient over one row block of a
 * captured cube face, using the precomputed projection weights. Columns are
 * ordered `( coefficient * 6 + face ) * blocks + block`; the repack pass sums the
 * faces and blocks.
 *
 * @private
 * @param {Node} faces - The face atlas texture node.
 * @param {Node} weights - The projection weight texture node.
 * @param {number} size - The face resolution.
 * @param {Node<int>} batchStart - The probe index of the first batch row.
 * @return {Node<vec4>} The projected coefficient for one face block.
 */
function projectSHNode( faces, weights, size, batchStart ) {

	const blocks = getFaceBlocks( size );
	const rows = size / blocks;

	return Fn( () => {

		const column = int( screenCoordinate.x );
		const block = column.mod( blocks );
		const coefFace = column.div( blocks );
		const coefIndex = coefFace.div( 6 );
		const face = coefFace.mod( 6 );
		const slot = int( screenCoordinate.y ).sub( batchStart );
		const rowStart = block.mul( rows );

		const faceOrigin = ivec2( face.mul( size ), slot.mul( size ).add( rowStart ) ).toVar();
		const weightOrigin = ivec2( face.mul( size ), coefIndex.mul( size ).add( rowStart ) ).toVar();
		const accum = vec3( 0.0 ).toVar();

		Loop( rows, size, ( { i, j } ) => {

			const texel = ivec2( j, i );
			accum.addAssign( faces.load( faceOrigin.add( texel ) ).rgb.mul( weights.load( weightOrigin.add( texel ) ).r ) );

		} );

		return vec4( accum, 1.0 );

	} )();

}

/**
 * Returns the repack output node for one of the seven SH textures. It sums the
 * per-face-block projections of the 9 coefficients from the batch texture for
 * the probe at the current texel and packs the four floats stored by this
 * texture index.
 *
 * @private
 * @param {Node} batch - The batch texture node holding projected coefficients.
 * @param {number} textureIndex - The output texture index (0–6).
 * @param {number} blocks - The number of row blocks per face.
 * @param {Node<vec3>} resolution - The probe grid resolution uniform.
 * @param {Node<int>} sliceZ - The current Z slice being written.
 * @return {Node<vec4>} The packed texel.
 */
function repackNode( batch, textureIndex, blocks, resolution, sliceZ ) {

	return Fn( () => {

		const ix = int( screenCoordinate.x );
		const iy = int( screenCoordinate.y );

		// Batch rows follow the bake order (X, then Z, then Y).
		const nx = int( resolution.x );
		const nz = int( resolution.z );
		const probeIndex = ix.add( sliceZ.mul( nx ) ).add( iy.mul( nx ).mul( nz ) ).toVar();

		const coefficient = ( c ) => {

			let sum = batch.load( ivec2( c * 6 * blocks, probeIndex ) );
			for ( let k = 1; k < 6 * blocks; k ++ ) sum = sum.add( batch.load( ivec2( c * 6 * blocks + k, probeIndex ) ) );
			return sum;

		};

		return packGridSH( coefficient, textureIndex );

	} )();

}

/**
 * Returns the capture texel and face coordinates a direction falls on.
 *
 * @private
 * @param {Node<vec3>} dir - The unit direction.
 * @param {number} size - The face resolution.
 * @param {Node<int>} slot - The probe's batch slot.
 * @return {Object} The texel `coord` and the face coordinates `fx`, `fy`.
 */
function captureTexel( dir, size, slot ) {

	const toVec3 = ( v ) => vec3( ...v );

	const a = dir.abs();
	const face = select( a.x.greaterThanEqual( a.y ).and( a.x.greaterThanEqual( a.z ) ),
		select( dir.x.greaterThanEqual( 0.0 ), int( 0 ), int( 1 ) ),
		select( a.y.greaterThanEqual( a.z ),
			select( dir.y.greaterThanEqual( 0.0 ), int( 2 ), int( 3 ) ),
			select( dir.z.greaterThanEqual( 0.0 ), int( 4 ), int( 5 ) ) ) ).toVar();

	const forward = array( FACE_FORWARD.map( toVec3 ) ).element( face );
	const right = array( FACE_RIGHT.map( toVec3 ) ).element( face );
	const up = array( FACE_UP.map( toVec3 ) ).element( face );

	const invForward = dir.dot( forward ).reciprocal();
	const fx = dir.dot( right ).mul( invForward ).toVar();
	const fy = dir.dot( up ).mul( invForward ).toVar();

	// Rows run top to bottom, so NDC y decreases with the row.
	const i = int( fx.add( 1.0 ).mul( 0.5 * size ).floor() ).clamp( 0, size - 1 );
	const j = int( float( 1.0 ).sub( fy ).mul( 0.5 * size ).floor() ).clamp( 0, size - 1 );

	return { coord: ivec2( face.mul( size ).add( i ), slot.mul( size ).add( j ) ), fx, fy };

}

/**
 * Returns the distance from a probe to the nearest surface in the classification
 * capture along a direction, and the capture texel it was read from.
 *
 * @private
 * @param {Node} depth - The classification capture's depth.
 * @param {Node<vec3>} dir - The unit direction.
 * @param {number} size - The face resolution.
 * @param {Node<int>} slot - The probe's batch slot.
 * @param {Node<float>} near - The capture near plane.
 * @param {Node<float>} far - The capture far plane.
 * @return {Object} The texel `coord` and the `distance`.
 */
function captureDistance( depth, dir, size, slot, near, far ) {

	const { coord, fx, fy } = captureTexel( dir, size, slot );
	const viewZ = perspectiveDepthToViewZ( depth.load( coord ).r, near, far );

	// Depth along the face axis to distance along the direction.
	return { coord, distance: viewZ.negate().mul( fx.mul( fx ).add( fy.mul( fy ) ).add( 1.0 ).sqrt() ) };

}

/**
 * Returns the output node for the back-face pass: one fragment per probe that
 * counts how many of 64 directions over an equal-area Fibonacci sphere hit a
 * back face in the classification capture, and finds the nearest hit.
 *
 * @private
 * @param {Node} flags - The classification capture's back-face flags.
 * @param {Node} depth - The classification capture's depth.
 * @param {number} size - The face resolution.
 * @param {Object} uniforms - The batch and camera uniforms.
 * @return {Node<vec4>} The back-face fraction in x and the nearest hit distance in y.
 */
function backFaceFractionNode( flags, depth, size, { batchStart, near, far } ) {

	return Fn( () => {

		const slot = int( screenCoordinate.y ).sub( batchStart ).toVar();
		const backFaces = float( 0.0 ).toVar();
		const nearest = far.toVar();

		Loop( CLASSIFY_SAMPLES, ( { i } ) => {

			const fi = float( i );
			const z = float( 1.0 ).sub( fi.mul( 2.0 ).add( 1.0 ).div( CLASSIFY_SAMPLES ) );
			const r = z.mul( z ).oneMinus().max( 0.0 ).sqrt();
			const phi = fi.mul( GOLDEN_ANGLE );
			const { coord, distance } = captureDistance( depth, vec3( r.mul( phi.cos() ), z, r.mul( phi.sin() ) ).toVar(), size, slot, near, far );

			// Background texels hold the clear color, so only count geometry hits.
			const hit = distance.lessThan( far.mul( 0.99 ) );
			backFaces.addAssign( select( hit, flags.load( coord ).r, float( 0.0 ) ) );
			nearest.assign( nearest.min( distance ) );

		} );

		return vec4( backFaces.div( CLASSIFY_SAMPLES ), nearest, 0.0, 1.0 );

	} )();

}

/**
 * Returns the output node for the distance moments pass, which reads the
 * classification capture. Each fragment is one texel of a probe's octahedral
 * distance tile, including its mirrored border. It averages the normalized
 * distance and squared distance to the nearest surface over 4 directions inside
 * the texel.
 *
 * A probe that sees back faces in more than a quarter of the directions is
 * inside geometry, and one closer to a surface than the clearance saw through it
 * in its radiance capture. Their moments are written as zero distance, so they
 * are hidden from every receiver instead of blending in light from both sides of
 * a surface.
 *
 * @private
 * @param {Node} depth - The classification capture's depth.
 * @param {Node} batch - The batch texture holding each probe's back-face fraction.
 * @param {number} fractionColumn - The batch column of the back-face fraction.
 * @param {number} size - The face resolution.
 * @param {Object} uniforms - The batch, camera and distance scale uniforms.
 * @return {Node<vec4>} The mean and mean squared normalized distance.
 */
function distanceMomentsNode( depth, batch, fractionColumn, size, { batchStart, batchCount, near, far, distanceScale, clearance } ) {

	const R = DISTANCE_RESOLUTION;

	return Fn( () => {

		// Tiles are in bake order. Texels of probes outside the batch keep their data.

		const tile = octTileTexel( ivec2( screenCoordinate.xy ), R );
		const probeIndex = tile.probeIndex.toVar();
		const slot = probeIndex.sub( batchStart ).toVar();

		Discard( slot.lessThan( 0 ).or( slot.greaterThanEqual( batchCount ) ) );

		const classification = batch.load( ivec2( fractionColumn, probeIndex ) );
		const inside = classification.x.greaterThan( INSIDE_FRACTION ).or( classification.y.lessThan( clearance ) );

		const x = tile.x.toVar();
		const y = tile.y.toVar();

		const sum = vec2( 0.0 ).toVar();

		for ( const dy of [ 0.25, 0.75 ] ) {

			for ( const dx of [ 0.25, 0.75 ] ) {

				const e = vec2( float( x ).add( dx ), float( y ).add( dy ) ).div( R ).mul( 2.0 ).sub( 1.0 );
				const { distance } = captureDistance( depth, octDecode( e ).toVar(), size, slot, near, far );
				const normalized = distance.div( distanceScale ).min( 1.0 );

				sum.addAssign( vec2( normalized, normalized.mul( normalized ) ) );

			}

		}

		return vec4( select( inside, vec2( 0.0 ), sum.mul( 0.25 ) ), 0.0, 1.0 );

	} )();

}

/**
 * Returns the output node for the irradiance pass. Each fragment is one texel of
 * a probe's octahedral irradiance tile, including its mirrored border: the
 * probe's SH irradiance for the normal direction of that texel.
 *
 * @private
 * @param {Node} atlas - The SH atlas texture node.
 * @param {Vector3} resolution - The probe grid resolution.
 * @param {Node<int>} start - The first probe index to write.
 * @param {Node<int>} end - The exclusive end probe index to write.
 * @return {Node<vec4>} The irradiance.
 */
function irradianceNode( atlas, resolution, start, end ) {

	const { x: nx, z: nz } = resolution;
	const R = IRRADIANCE_RESOLUTION;

	return Fn( () => {

		const tile = octTileTexel( ivec2( screenCoordinate.xy ), R );
		const probeIndex = tile.probeIndex.toVar();

		Discard( probeIndex.lessThan( start ).or( probeIndex.greaterThanEqual( end ) ) );

		// Bake order is X, then Z, then Y.
		const ix = probeIndex.mod( nx );
		const iz = probeIndex.div( nx ).mod( nz );
		const iy = probeIndex.div( nx * nz );

		const texels = [];

		for ( let t = 0; t < 7; t ++ ) {

			texels.push( atlas.load( ivec3( ix, iy, iz.add( t * ( nz + 2 * ATLAS_PADDING ) + ATLAS_PADDING ) ) ) );

		}

		const e = vec2( float( tile.x ).add( 0.5 ), float( tile.y ).add( 0.5 ) ).div( R ).mul( 2.0 ).sub( 1.0 );
		const irradiance = getShIrradianceAt( octDecode( e ), unpackGridSH( texels ) ).max( vec3( 0.0 ) );

		return vec4( irradiance, 1.0 );

	} )();

}

/**
 * Render object function of the classification capture: opaque meshes are drawn
 * with the classification material for their side, everything else is skipped.
 *
 * @private
 */
function classifyRenderObject( object, scene, camera, geometry, material, group, lightsNode, clippingContext ) {

	if ( object.isMesh !== true || material.transparent === true ) return;

	this.renderObject( object, scene, camera, geometry, _classifyMaterials[ material.side ], group, lightsNode, clippingContext, 'lightProbeClassify' );

}

/**
 * Creates the material that replaces meshes of the given side in the
 * classification capture. It draws both sides and flags the side the original
 * material does not show, which is what a probe inside a closed mesh sees.
 *
 * @private
 * @param {number} side - The side of the replaced material.
 * @return {NodeMaterial} The classification material.
 */
function createClassifyMaterial( side ) {

	let backFace = float( 0.0 );

	if ( side === FrontSide ) backFace = select( frontFacing, float( 0.0 ), float( 1.0 ) );
	else if ( side === BackSide ) backFace = select( frontFacing, float( 1.0 ), float( 0.0 ) );

	const material = new MeshBasicNodeMaterial();
	material.side = DoubleSide;
	material.outputNode = vec4( backFace, 0.0, 0.0, 1.0 );

	return material;

}

/**
 * Lazily creates the shared capture camera: six 90° sub-cameras per batch slot,
 * each rendering into its own tile of the face atlas.
 *
 * @private
 * @param {number} cubemapSize - Resolution of each cubemap face.
 * @param {number} near - Capture near plane.
 * @param {number} far - Capture far plane.
 */
function ensureCaptureCamera( cubemapSize, near, far ) {

	if ( _captureCamera === null ) {

		const cameras = [];

		for ( let i = 0; i < PROBES_PER_BATCH * 6; i ++ ) {

			const camera = new PerspectiveCamera( 90, 1, near, far );
			camera.viewport = new Vector4();
			camera.up.fromArray( FACE_UP[ i % 6 ] );
			cameras.push( camera );

		}

		_captureCamera = new ArrayCamera( cameras );

	}

	for ( let i = 0; i < _captureCamera.cameras.length; i ++ ) {

		const camera = _captureCamera.cameras[ i ];

		if ( camera.near !== near || camera.far !== far ) {

			camera.near = near;
			camera.far = far;
			camera.updateProjectionMatrix();

		}

		camera.viewport.set( ( i % 6 ) * cubemapSize, Math.floor( i / 6 ) * cubemapSize, cubemapSize, cubemapSize );

	}

}

/**
 * Sets the near plane of every capture sub-camera.
 *
 * @private
 * @param {number} near - The near plane.
 */
function setCaptureNear( near ) {

	for ( const camera of _captureCamera.cameras ) {

		camera.near = near;
		camera.updateProjectionMatrix();

	}

}

/**
 * Lazily pools the shared face atlas and batch render targets, recreating them
 * only when their dimensions change.
 *
 * @private
 * @param {number} cubemapSize - Resolution of each cubemap face.
 * @param {number} totalProbes - Number of probes (batch target height).
 */
function ensureBakeTargets( cubemapSize, totalProbes ) {

	if ( _faceTarget === null || _faceSize !== cubemapSize ) {

		if ( _faceTarget !== null ) _faceTarget.dispose();

		// One row of six face tiles per batch slot.
		_faceTarget = new RenderTarget( 6 * cubemapSize, PROBES_PER_BATCH * cubemapSize, {
			type: HalfFloatType,
			minFilter: NearestFilter,
			magFilter: NearestFilter,
			generateMipmaps: false
		} );

		// The classification capture: back-face flags and depth for visibility.
		if ( _classifyTarget !== null ) _classifyTarget.dispose();

		_classifyTarget = new RenderTarget( 6 * cubemapSize, PROBES_PER_BATCH * cubemapSize, {
			minFilter: NearestFilter,
			magFilter: NearestFilter,
			generateMipmaps: false,
			depthTexture: new DepthTexture( 6 * cubemapSize, PROBES_PER_BATCH * cubemapSize )
		} );

		_faceSize = cubemapSize;

	}

	// One row per probe: 9 coefficients x 6 faces x face blocks.
	const columns = 9 * 6 * getFaceBlocks( cubemapSize );

	if ( _batchTarget === null || _batchProbes !== totalProbes || _batchColumns !== columns ) {

		if ( _batchTarget !== null ) _batchTarget.dispose();

		// One more column holds each probe's back-face fraction.
		_batchTarget = new RenderTarget( columns + 1, totalProbes, {
			type: FloatType,
			format: RGBAFormat,
			minFilter: NearestFilter,
			magFilter: NearestFilter,
			depthBuffer: false
		} );

		_batchProbes = totalProbes;
		_batchColumns = columns;

	}

}

/**
 * Lazily builds the shared bake materials and rebinds them to the current
 * face/batch textures. The projection weights and the SH projection and
 * repack materials are rebuilt only when the face size changes.
 *
 * @private
 * @param {number} cubemapSize - Resolution of each cubemap face.
 * @param {Texture} faceMap - The current face atlas texture.
 * @param {Texture} batchMap - The current batch render target texture.
 * @param {Texture} classifyMap - The current classification capture texture.
 * @param {DepthTexture} depthMap - The current classification capture depth texture.
 */
function ensureBakeMaterials( cubemapSize, faceMap, batchMap, classifyMap, depthMap ) {

	if ( _faceNode === null ) {

		_faceNode = texture( faceMap );
		_batchNode = texture( batchMap );
		_classifyNode = texture( classifyMap );
		_depthNode = texture( depthMap );
		_classifyMaterials = [ FrontSide, BackSide, DoubleSide ].map( createClassifyMaterial );
		_batchStartUniform = uniform( 0, 'int' );
		_batchCountUniform = uniform( 0, 'int' );
		_nearUniform = uniform( 0.1 );
		_farUniform = uniform( 100 );
		_clearanceUniform = uniform( 0 );
		_distanceScaleUniform = uniform( 1 );
		_resolutionUniform = uniform( new Vector3() );
		_sliceZUniform = uniform( 0, 'int' );

	} else {

		_faceNode.value = faceMap;
		_batchNode.value = batchMap;
		_classifyNode.value = classifyMap;
		_depthNode.value = depthMap;

	}

	if ( _shMaterial === null || _shFaceSize !== cubemapSize ) {

		if ( _shMaterial !== null ) {

			_shMaterial.dispose();
			_distanceMaterial.dispose();
			_fractionMaterial.dispose();
			_weightsNode.value.dispose();
			for ( const material of _repackMaterials ) material.dispose();

		}

		const weights = createProjectionWeights( cubemapSize );

		if ( _weightsNode === null ) _weightsNode = texture( weights );
		else _weightsNode.value = weights;

		_shMaterial = new NodeMaterial();
		_shMaterial.outputNode = projectSHNode( _faceNode, _weightsNode, cubemapSize, _batchStartUniform );
		_shMaterial.depthTest = false;
		_shMaterial.depthWrite = false;
		_shFaceSize = cubemapSize;

		_fractionMaterial = new NodeMaterial();
		_fractionMaterial.outputNode = backFaceFractionNode( _classifyNode, _depthNode, cubemapSize, {
			batchStart: _batchStartUniform,
			near: _nearUniform,
			far: _farUniform
		} );
		_fractionMaterial.depthTest = false;
		_fractionMaterial.depthWrite = false;

		_distanceMaterial = new NodeMaterial();
		_distanceMaterial.outputNode = distanceMomentsNode( _depthNode, _batchNode, 9 * 6 * getFaceBlocks( cubemapSize ), cubemapSize, {
			batchStart: _batchStartUniform,
			batchCount: _batchCountUniform,
			near: _nearUniform,
			far: _farUniform,
			distanceScale: _distanceScaleUniform,
			clearance: _clearanceUniform
		} );
		_distanceMaterial.depthTest = false;
		_distanceMaterial.depthWrite = false;

		_repackMaterials = [];

		for ( let t = 0; t < 7; t ++ ) {

			const material = new NodeMaterial();
			material.outputNode = repackNode( _batchNode, t, getFaceBlocks( cubemapSize ), _resolutionUniform, _sliceZUniform );
			material.depthTest = false;
			material.depthWrite = false;
			_repackMaterials.push( material );

		}

	}

}

/**
 * A 3D grid of L2 Spherical Harmonic irradiance probes that provides
 * position-dependent diffuse global illumination.
 *
 * This is the {@link WebGPURenderer} version of `LightProbeGrid`. The grid is a
 * {@link Light}, so adding it to the scene applies its baked irradiance to every
 * lit node material automatically. When using {@link WebGLRenderer}, import the
 * grid from `LightProbeGridWebGL.js` instead.
 *
 * The baked data is stored in a single RGBA `RenderTarget3D` atlas that packs
 * the nine L2 SH coefficients into seven sub-volumes stacked along Z. Baking is
 * fully GPU-resident: cubemap rendering, SH projection, and texture packing all
 * happen on the GPU with zero CPU readback.
 *
 * @augments Light
 * @three_import import { LightProbeGrid } from 'three/addons/lighting/LightProbeGrid.js';
 */
class LightProbeGrid extends Light {

	/**
	 * Constructs a new irradiance probe grid.
	 *
	 * The volume is centered at the object's position.
	 *
	 * @param {number} [width=1] - Full width of the volume along X.
	 * @param {number} [height=1] - Full height of the volume along Y.
	 * @param {number} [depth=1] - Full depth of the volume along Z.
	 * @param {number} [widthProbes] - Number of probes along X. Defaults to `Math.max( 2, Math.round( width ) + 1 )`.
	 * @param {number} [heightProbes] - Number of probes along Y. Defaults to `Math.max( 2, Math.round( height ) + 1 )`.
	 * @param {number} [depthProbes] - Number of probes along Z. Defaults to `Math.max( 2, Math.round( depth ) + 1 )`.
	 */
	constructor( width = 1, height = 1, depth = 1, widthProbes, heightProbes, depthProbes ) {

		super( 0xffffff, 1 );

		/**
		 * This flag can be used for type testing.
		 *
		 * @type {boolean}
		 * @readonly
		 * @default true
		 */
		this.isLightProbeGrid = true;

		this.type = 'LightProbeGrid';

		/**
		 * The full width of the volume along X.
		 *
		 * @type {number}
		 */
		this.width = width;

		/**
		 * The full height of the volume along Y.
		 *
		 * @type {number}
		 */
		this.height = height;

		/**
		 * The full depth of the volume along Z.
		 *
		 * @type {number}
		 */
		this.depth = depth;

		/**
		 * The number of probes along each axis.
		 *
		 * @type {Vector3}
		 */
		this.resolution = new Vector3(
			widthProbes !== undefined ? widthProbes : Math.max( 2, Math.round( width ) + 1 ),
			heightProbes !== undefined ? heightProbes : Math.max( 2, Math.round( height ) + 1 ),
			depthProbes !== undefined ? depthProbes : Math.max( 2, Math.round( depth ) + 1 )
		);

		/**
		 * The world-space bounding box for the grid. Updated automatically
		 * by {@link LightProbeGrid#bake}.
		 *
		 * @type {Box3}
		 */
		this.boundingBox = new Box3();

		/**
		 * Distance in world units over which the grid contribution fades out
		 * past the volume boundary. `0` applies the contribution everywhere
		 * (clamped), which matches a single-volume setup. Use a small positive
		 * value to blend multiple overlapping grids.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.falloff = 0;

		/**
		 * Whether shading weighs each probe by whether it can see the surface, using
		 * distance moments recorded during the bake. This keeps light from leaking
		 * through walls between probes, at the cost of more texture reads per
		 * fragment. It can be changed at any time.
		 *
		 * @type {boolean}
		 * @default false
		 */
		this.visibility = false;

		/**
		 * With {@link LightProbeGrid#visibility}, how far the shaded point is moved
		 * along its normal before the probe lookup, as a fraction of the smallest
		 * probe spacing.
		 *
		 * @type {number}
		 * @default 0.08
		 */
		this.normalBias = 0.08;

		/**
		 * With {@link LightProbeGrid#visibility}, how far the shaded point is moved
		 * toward the camera before the probe lookup, as a fraction of the smallest
		 * probe spacing.
		 *
		 * @type {number}
		 * @default 0.02
		 */
		this.viewBias = 0.02;

		/**
		 * The single RGBA atlas 3D texture storing all seven packed SH
		 * sub-volumes stacked along Z. It is all zeros until baked.
		 *
		 * @type {Data3DTexture}
		 */
		this.texture = null;

		/**
		 * Internal render target for GPU-resident baking.
		 *
		 * @private
		 * @type {?RenderTarget3D}
		 * @default null
		 */
		this._renderTarget = null;

		// Indirect captures read a snapshot while the live atlas is updated in place.
		this._snapshotTarget = null;

		// Per-probe distance moments for visibility, and their normalization scale.
		this._distanceTarget = null;
		this._distanceScale = 1;

		// Per-probe irradiance for visibility shading, evaluated from the SH.
		this._irradianceTarget = null;
		this._irradianceSnapshotTarget = null;
		this._irradianceMaterial = null;
		this._bouncePass = - 1;

		// While baking, the grid lights its own captures: with zero intensity in the
		// direct pass and from the snapshot in indirect passes. Keeping the grid
		// itself in the scene, rather than swapping in a separate light, leaves the
		// scene's lights unchanged, so materials are not rebuilt between bake passes
		// and the main view.
		this._captureIntensity = null;
		this._captureSnapshot = false;

		this.updateBoundingBox();

		// The atlases exist from the start so the light's bindings never change.
		this._ensureTextures();

	}

	/**
	 * Returns the world-space position of the probe at grid indices (ix, iy, iz).
	 *
	 * @param {number} ix - X index.
	 * @param {number} iy - Y index.
	 * @param {number} iz - Z index.
	 * @param {Vector3} target - The target vector.
	 * @return {Vector3} The world-space position.
	 */
	getProbePosition( ix, iy, iz, target ) {

		const pos = this.position;
		const res = this.resolution;
		const w = this.width, h = this.height, d = this.depth;

		target.set(
			res.x > 1 ? pos.x - w / 2 + ix * w / ( res.x - 1 ) : pos.x,
			res.y > 1 ? pos.y - h / 2 + iy * h / ( res.y - 1 ) : pos.y,
			res.z > 1 ? pos.z - d / 2 + iz * d / ( res.z - 1 ) : pos.z
		);

		return target;

	}

	/**
	 * Updates the world-space bounding box from the current position and size.
	 */
	updateBoundingBox() {

		_size.set( this.width, this.height, this.depth );
		this.boundingBox.setFromCenterAndSize( this.position, _size );

		// Distance moments are normalized by twice the cell diagonal, which covers
		// every receiver a probe is blended into.

		const res = this.resolution;
		_size.divide( _position.set( Math.max( 1, res.x - 1 ), Math.max( 1, res.y - 1 ), Math.max( 1, res.z - 1 ) ) );
		this._distanceScale = 2 * _size.length();

	}

	/**
	 * Bakes probes by rendering cubemaps at each probe position and
	 * projecting to L2 SH. Optionally iterates additional passes to capture
	 * indirect bounces: each extra pass samples the previous pass's data as
	 * indirect light, accumulating one bounce per extra pass.
	 *
	 * Use `start` and `count` to bake a range and publish its cells immediately.
	 * Indices advance along X, then Z, then Y, filling horizontal layers from bottom
	 * to top. For incremental indirect bounces, finish the whole grid for `pass: 0`,
	 * then repeat with `pass: 1`, etc. Start each pass at index 0 to snapshot the
	 * previous pass before updating its cells.
	 *
	 * Shadow-casting instances of `SunLight` are temporarily replaced with
	 * equivalent directional lights, since their view-fitted shadow cascades
	 * cannot be frozen across probe renders.
	 *
	 * @param {WebGPURenderer} renderer - The renderer.
	 * @param {Scene} scene - The scene to render.
	 * @param {Object} [options] - Bake options.
	 * @param {number} [options.cubemapSize=8] - Resolution of each cubemap face.
	 * @param {number} [options.near=0.1] - Near plane for the cube camera.
	 * @param {number} [options.far=100] - Far plane for the cube camera.
	 * @param {number} [options.bounces=0] - Additional bounce passes. Only available when baking the whole grid.
	 * @param {number} [options.start=0] - Index of the first probe to bake.
	 * @param {number} [options.count] - Number of probes to bake. Defaults to the remaining probes.
	 * @param {number} [options.pass=0] - Starting pass. Zero captures direct light; later passes sample the previous pass. Ranged calls require `bounces: 0`.
	 */
	bake( renderer, scene, options = {} ) {

		// The bake is node based, so it needs a WebGPURenderer.
		if ( renderer.isWebGPURenderer !== true ) {

			throw new Error( 'THREE.LightProbeGrid: .bake() requires a WebGPURenderer. For WebGLRenderer, use LightProbeGridWebGL.' );

		}

		// The bake issues GPU work immediately, so the renderer must be ready.
		if ( renderer.initialized === false ) {

			throw new Error( 'THREE.LightProbeGrid: .bake() called before the renderer is initialized. Use "await renderer.init();" first.' );

		}

		const res = this.resolution;
		const totalProbes = res.x * res.y * res.z;
		const {
			cubemapSize = 8,
			near = 0.1,
			far = 100,
			bounces = 0,
			start = 0,
			count = totalProbes - start,
			pass: firstPass = 0
		} = options;
		const end = start + count;

		if ( ! Number.isInteger( start ) || ! Number.isInteger( count ) || start < 0 || count < 0 || end > totalProbes ) {

			throw new RangeError( 'THREE.LightProbeGrid: Invalid probe range.' );

		}

		if ( ! Number.isInteger( firstPass ) || firstPass < 0 || ! Number.isInteger( bounces ) || bounces < 0 ) {

			throw new RangeError( 'THREE.LightProbeGrid: Pass and bounce counts must be non-negative integers.' );

		}

		if ( bounces > 0 && count !== totalProbes ) {

			throw new RangeError( 'THREE.LightProbeGrid: For ranged baking, use pass instead of bounces.' );

		}

		if ( count === 0 ) return;

		if ( firstPass > 0 && start > 0 && this._bouncePass !== firstPass ) {

			throw new Error( 'THREE.LightProbeGrid: Start each indirect pass at probe 0.' );

		}

		this._ensureTextures();
		this.updateBoundingBox();

		// The grid samples its atlases during the capture, so they must exist as
		// render targets before they are bound. Otherwise they are created as plain
		// textures and replaced when first written, leaving the bindings stale.
		renderer.initRenderTarget( this._renderTarget );
		renderer.initRenderTarget( this._snapshotTarget );

		// Bind the pooled bake resources to the current textures.

		ensureCaptureCamera( cubemapSize, near, far );
		ensureBakeTargets( cubemapSize, totalProbes );
		ensureBakeMaterials( cubemapSize, _faceTarget.texture, _batchTarget.texture, _classifyTarget.texture, _classifyTarget.depthTexture );

		_nearUniform.value = near * CLASSIFY_NEAR_SCALE;
		_clearanceUniform.value = near * CLEARANCE_NEAR_PLANES;
		_farUniform.value = far;
		_distanceScaleUniform.value = this._distanceScale;
		_resolutionUniform.value.copy( res );

		// Save renderer / scene state to restore after the bake.

		const currentRenderTarget = renderer.getRenderTarget();
		const currentActiveCubeFace = renderer.getActiveCubeFace();
		const currentActiveMipmapLevel = renderer.getActiveMipmapLevel();
		const currentAutoClear = renderer.autoClear;
		const currentXrEnabled = renderer.xr.enabled;
		const currentInspectorEnabled = renderer.inspector.enabled;
		const currentMatrixWorldAutoUpdate = scene.matrixWorldAutoUpdate;
		const currentVisible = this.visible;
		const currentParent = this.parent;
		const renderTarget = this._renderTarget;
		const currentViewport = renderTarget.viewport.clone();
		const shadowStates = [];
		let replacedSunLights = null;

		try {

			renderer.inspector.enabled = false;
			renderer.xr.enabled = false;

			// The grid lights the indirect passes, so it must be a visible part of
			// the scene. A grid outside the scene is added for the bake.

			let root = this;
			while ( root.parent !== null ) root = root.parent;
			if ( root !== scene ) scene.add( this );

			this.visible = true;

			// Scene is static during the bake: update once, disable auto-update.

			if ( currentMatrixWorldAutoUpdate === true ) {

				scene.updateMatrixWorld( true );
				scene.matrixWorldAutoUpdate = false;

			}

			replacedSunLights = replaceSunLights( scene );

			// Render each shadow map once, not once per cube face.

			scene.traverse( ( object ) => {

				if ( object.isLight && object.castShadow && object.shadow ) {

					const shadow = object.shadow;
					shadowStates.push( { shadow, autoUpdate: shadow.autoUpdate } );
					shadow.autoUpdate = false;
					shadow.needsUpdate = true;

				}

			} );

			for ( let pass = firstPass; pass <= firstPass + bounces; pass ++ ) {

				this._updateSnapshot( renderer, pass, start );
				// Geometry is static during the bake, so distances are recorded once.
				this._captureProbes( renderer, scene, start, end, pass === 0 );
				this._repackProbes( renderer, start, end );
				this._updateIrradiance( renderer, start, end );

			}

		} finally {

			// Restore renderer / scene state (pooled targets and materials kept).

			renderTarget.viewport.copy( currentViewport );
			renderer.setRenderTarget( currentRenderTarget, currentActiveCubeFace, currentActiveMipmapLevel );
			renderer.autoClear = currentAutoClear;
			renderer.xr.enabled = currentXrEnabled;
			scene.matrixWorldAutoUpdate = currentMatrixWorldAutoUpdate;

			for ( const { shadow, autoUpdate } of shadowStates ) shadow.autoUpdate = autoUpdate;

			if ( replacedSunLights !== null ) restoreSunLights( scene, replacedSunLights );

			this.visible = currentVisible;
			this._captureIntensity = null;
			this._captureSnapshot = false;

			if ( this.parent !== currentParent ) {

				if ( currentParent !== null ) currentParent.add( this );
				else this.removeFromParent();

			}

			renderer.inspector.enabled = currentInspectorEnabled;

		}

	}

	/**
	 * Selects what the grid contributes to its own captures. The direct pass
	 * contributes nothing. Each indirect pass snapshots the previous pass before
	 * its first range overwrites the live atlas, then samples the snapshot.
	 *
	 * @private
	 * @param {WebGPURenderer} renderer - The renderer.
	 * @param {number} pass - The bounce pass.
	 * @param {number} start - The first probe index.
	 */
	_updateSnapshot( renderer, pass, start ) {

		const renderTarget = this._renderTarget;

		if ( pass === 0 ) {

			if ( start === 0 ) this._bouncePass = - 1;

			this._captureIntensity = 0;
			this._captureSnapshot = false;
			return;

		}

		if ( start === 0 ) {

			_copyRegion.min.set( 0, 0, 0 );
			_copyRegion.max.set( renderTarget.width, renderTarget.height, renderTarget.depth );
			renderer.copyTextureToTexture( renderTarget.texture, this._snapshotTarget.texture, _copyRegion );

			const irradianceTarget = this._irradianceTarget;
			_copyRegion.max.set( irradianceTarget.width, irradianceTarget.height, 1 );
			renderer.copyTextureToTexture( irradianceTarget.texture, this._irradianceSnapshotTarget.texture, _copyRegion );
			this._bouncePass = pass;

		}

		this._captureIntensity = this.intensity;
		this._captureSnapshot = true;

	}

	/**
	 * Captures the cube faces of probe batches in single passes and projects
	 * their SH coefficients into the batch target, one row per probe in bake order.
	 *
	 * @private
	 * @param {WebGPURenderer} renderer - The renderer.
	 * @param {Scene} scene - The scene to capture.
	 * @param {number} start - The first probe index.
	 * @param {number} end - The exclusive end probe index.
	 * @param {boolean} recordDistances - Whether to record the probes' distance moments.
	 */
	_captureProbes( renderer, scene, start, end, recordDistances ) {

		// The radiance capture near plane, restored after each classification capture.
		const near = _captureCamera.cameras[ 0 ].near;

		const { x: nx, z: nz } = this.resolution;
		const probesPerLayer = nx * nz;
		const cameras = _captureCamera.cameras;
		const currentRenderObjectFunction = renderer.getRenderObjectFunction();

		for ( let batchStart = start; batchStart < end; batchStart += PROBES_PER_BATCH ) {

			const batchCount = Math.min( PROBES_PER_BATCH, end - batchStart );

			for ( let slot = 0; slot < PROBES_PER_BATCH; slot ++ ) {

				const probeIndex = batchStart + slot;

				if ( slot < batchCount ) {

					const ix = probeIndex % nx;
					const iy = Math.floor( probeIndex / probesPerLayer );
					const iz = Math.floor( probeIndex / nx ) % nz;

					this.getProbePosition( ix, iy, iz, _position );

				}

				for ( let face = 0; face < 6; face ++ ) {

					const camera = cameras[ slot * 6 + face ];

					// Unused slots of a partial batch draw nothing. The camera count
					// stays fixed so the capture pipelines are reused.
					if ( slot < batchCount ) {

						camera.layers.set( 0 );
						camera.position.copy( _position );
						camera.lookAt( _target.fromArray( FACE_FORWARD[ face ] ).add( _position ) );
						camera.updateMatrixWorld();

					} else {

						camera.layers.disableAll();

					}

				}

			}

			renderer.autoClear = true;
			renderer.setRenderTarget( _faceTarget );
			renderer.render( scene, _captureCamera );

			renderer.autoClear = false;
			_batchStartUniform.value = batchStart;
			_batchTarget.viewport.set( 0, batchStart, _batchColumns, batchCount );
			renderer.setRenderTarget( _batchTarget );
			_quad.material = _shMaterial;
			_quad.render( renderer );

			if ( recordDistances === true ) {

				// The tile rows covering this batch; the shader skips other probes.

				const distanceTarget = this._distanceTarget;
				const firstRow = Math.floor( batchStart / DISTANCE_COLUMNS );
				const lastRow = Math.floor( ( batchStart + batchCount - 1 ) / DISTANCE_COLUMNS );

				// Classification capture: every mesh drawn double sided with back-face
				// flags. Its depth sees through nothing, like a double-sided ray cast.

				// A much closer near plane, so surfaces the radiance capture clipped
				// away are seen.

				setCaptureNear( _nearUniform.value );

				renderer.setRenderObjectFunction( classifyRenderObject );
				renderer.autoClear = true;
				renderer.setRenderTarget( _classifyTarget );
				renderer.render( scene, _captureCamera );
				renderer.autoClear = false;
				renderer.setRenderObjectFunction( currentRenderObjectFunction );

				setCaptureNear( near );

				_batchTarget.viewport.set( _batchColumns, batchStart, 1, batchCount );
				renderer.setRenderTarget( _batchTarget );
				_quad.material = _fractionMaterial;
				_quad.render( renderer );

				_batchCountUniform.value = batchCount;
				distanceTarget.viewport.set( 0, firstRow * DISTANCE_TILE, distanceTarget.width, ( lastRow - firstRow + 1 ) * DISTANCE_TILE );
				renderer.setRenderTarget( distanceTarget );
				_quad.material = _distanceMaterial;
				_quad.render( renderer );

			}

		}

	}

	/**
	 * Packs a probe range into the live atlas, including its boundary padding.
	 *
	 * @private
	 * @param {WebGPURenderer} renderer - The renderer.
	 * @param {number} start - The first probe index.
	 * @param {number} end - The exclusive end probe index.
	 */
	_repackProbes( renderer, start, end ) {

		const { x: nx, y: ny, z: nz } = this.resolution;
		const probesPerLayer = nx * nz;
		const startY = Math.floor( start / probesPerLayer );
		const endY = Math.floor( end / probesPerLayer );
		const paddedSlices = nz + 2 * ATLAS_PADDING;
		const renderTarget = this._renderTarget;

		// Map the horizontal bake range to contiguous rows in each Z slice.
		for ( let iz = 0; iz < nz; iz ++ ) {

			const sliceStart = startY * nx + MathUtils.clamp( start % probesPerLayer - iz * nx, 0, nx );
			const sliceEnd = endY * nx + MathUtils.clamp( end % probesPerLayer - iz * nx, 0, nx );

			for ( let probeIndex = sliceStart; probeIndex < sliceEnd; ) {

				const ix = probeIndex % nx;
				const iy = Math.floor( probeIndex / nx );

				// Coalesce complete rows within a slice into one rectangle.
				const width = Math.min( nx - ix, sliceEnd - probeIndex );
				let height = 1;

				if ( width === nx ) {

					height = Math.min( ny - iy, Math.floor( ( sliceEnd - probeIndex ) / nx ) );

				}

				renderTarget.viewport.set( ix, iy, width, height );
				_sliceZUniform.value = iz;

				for ( let t = 0; t < 7; t ++ ) {

					_quad.material = _repackMaterials[ t ];
					const base = t * paddedSlices;

					renderer.setRenderTarget( renderTarget, base + ATLAS_PADDING + iz );
					_quad.render( renderer );

					if ( iz === 0 ) {

						renderer.setRenderTarget( renderTarget, base );
						_quad.render( renderer );

					}

					if ( iz === nz - 1 ) {

						renderer.setRenderTarget( renderTarget, base + ATLAS_PADDING + nz );
						_quad.render( renderer );

					}

				}

				probeIndex += width * height;

			}

		}

	}

	/**
	 * Evaluates the irradiance tiles of a probe range from the live SH atlas.
	 *
	 * @private
	 * @param {WebGPURenderer} renderer - The renderer.
	 * @param {number} start - The first probe index.
	 * @param {number} end - The exclusive end probe index.
	 */
	_updateIrradiance( renderer, start, end ) {

		if ( this._irradianceMaterial === null ) {

			this._irradianceStart = uniform( 0, 'int' );
			this._irradianceEnd = uniform( 0, 'int' );

			this._irradianceMaterial = new NodeMaterial();
			this._irradianceMaterial.outputNode = irradianceNode( texture3D( this.texture ), this.resolution.clone(), this._irradianceStart, this._irradianceEnd );
			this._irradianceMaterial.depthTest = false;
			this._irradianceMaterial.depthWrite = false;

		}

		// The tile rows covering the range; the shader skips other probes.

		const target = this._irradianceTarget;
		const firstRow = Math.floor( start / DISTANCE_COLUMNS );
		const lastRow = Math.floor( ( end - 1 ) / DISTANCE_COLUMNS );

		this._irradianceStart.value = start;
		this._irradianceEnd.value = end;

		target.viewport.set( 0, firstRow * IRRADIANCE_TILE, target.width, ( lastRow - firstRow + 1 ) * IRRADIANCE_TILE );
		renderer.setRenderTarget( target );
		_quad.material = this._irradianceMaterial;
		_quad.render( renderer );

	}

	/**
	 * Ensures the atlas and snapshot 3D textures exist with the correct dimensions.
	 *
	 * @private
	 */
	_ensureTextures() {

		if ( this._renderTarget !== null ) return;

		const res = this.resolution;
		const nx = res.x, ny = res.y, nz = res.z;

		// Atlas depth: 7 sub-volumes, each with ATLAS_PADDING slices at both ends.
		const atlasDepth = 7 * ( nz + 2 * ATLAS_PADDING );

		const options = {
			type: HalfFloatType,
			format: RGBAFormat,
			minFilter: LinearFilter,
			magFilter: LinearFilter,
			generateMipmaps: false,
			depthBuffer: false
		};

		this._renderTarget = new RenderTarget3D( nx, ny, atlasDepth, options );
		this._snapshotTarget = new RenderTarget3D( nx, ny, atlasDepth, options );

		// Irradiance per probe for visibility shading, as octahedral tiles in bake order.

		const irradianceRows = Math.ceil( nx * ny * nz / DISTANCE_COLUMNS );
		const irradianceOptions = {
			type: HalfFloatType,
			minFilter: LinearFilter,
			magFilter: LinearFilter,
			generateMipmaps: false,
			depthBuffer: false
		};

		this._irradianceTarget = new RenderTarget( DISTANCE_COLUMNS * IRRADIANCE_TILE, irradianceRows * IRRADIANCE_TILE, irradianceOptions );
		this._irradianceSnapshotTarget = new RenderTarget( DISTANCE_COLUMNS * IRRADIANCE_TILE, irradianceRows * IRRADIANCE_TILE, irradianceOptions );

		// Distance moments per probe, as octahedral tiles in bake order.

		const rows = Math.ceil( nx * ny * nz / DISTANCE_COLUMNS );

		this._distanceTarget = new RenderTarget( DISTANCE_COLUMNS * DISTANCE_TILE, rows * DISTANCE_TILE, {
			type: HalfFloatType,
			format: RGFormat,
			minFilter: LinearFilter,
			magFilter: LinearFilter,
			generateMipmaps: false,
			depthBuffer: false
		} );

		this.texture = this._renderTarget.texture;

	}

	/**
	 * Frees GPU resources. The grid can be baked again afterwards.
	 */
	dispose() {

		// The targets are kept so the light's bindings stay valid.

		this._renderTarget.dispose();
		this._snapshotTarget.dispose();
		this._distanceTarget.dispose();
		this._irradianceTarget.dispose();
		this._irradianceSnapshotTarget.dispose();

		if ( this._irradianceMaterial !== null ) {

			this._irradianceMaterial.dispose();
			this._irradianceMaterial = null;

		}

		this._bouncePass = - 1;

		super.dispose();

	}

}

LightProbeGrid.registerNode( LightProbeGridNode );

export { LightProbeGrid };
