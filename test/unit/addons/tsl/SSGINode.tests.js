import { DataTexture, DataUtils, FloatType, NodeMaterial, PerspectiveCamera, QuadMesh, RendererUtils, Vector2, Vector3 } from 'three/webgpu';
import { context, float, texture, uint } from 'three/tsl';
import { ssgi, ssgiSectorCoverage } from '../../../../examples/jsm/tsl/display/SSGINode.js';
import { getSharedRenderer, gpuTest } from '../../src/nodes/tsl/gpu-test-utils.js';

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'TSL', () => {

		QUnit.module( 'SSGINode', () => {

			gpuTest( 'cosine-weighted sector coverage is normalized and additive', ( { assert } ) => {

				assert.closeAbs( ssgiSectorCoverage( uint( 0 ) ), float( 0 ), 1e-6, 'No sectors means no coverage' );
				assert.closeAbs( ssgiSectorCoverage( uint( 0xFFFFFFFF ) ), float( 1 ), 1e-5, 'A complete hemisphere has unit coverage' );
				assert.closeAbs( ssgiSectorCoverage( uint( 0xFFFF ) ), float( 0.5 ), 1e-5, 'Half the hemisphere has half coverage' );
				assert.closeAbs( ssgiSectorCoverage( uint( 0x55555555 ) ).add( ssgiSectorCoverage( uint( 0xAAAAAAAA ) ) ), float( 1 ), 1e-5, 'Disjoint masks partition the same integral' );
				assert.greaterThan( ssgiSectorCoverage( uint( 1 << 15 ) ), ssgiSectorCoverage( uint( 1 ) ), 'Near-normal sectors carry more cosine weight than grazing sectors' );

			} );

			QUnit.test( 'tilted plane does not produce striped self-occlusion at half resolution', async assert => {

				const renderer = await getSharedRenderer( 'webgpu' );
				if ( ! renderer ) {

					assert.ok( true, 'SKIPPED: WebGPU unavailable' );
					return;

				}

				const state = RendererUtils.saveRendererState( renderer );
				const previousContext = renderer.contextNode;
				const previousSize = renderer.getSize( new Vector2() );
				const previousRatio = renderer.getPixelRatio();
				const width = 512, height = 384;
				const camera = new PerspectiveCamera( 50, width / height, 0.1, 100 );
				camera.coordinateSystem = renderer.coordinateSystem;
				camera.updateProjectionMatrix();
				const normal = new Vector3( 0.35, 0.2, 1 ).normalize();
				const point = new Vector3();
				const beauty = new Float32Array( width * height * 4 );
				const depth = new Float32Array( beauty.length );
				const normals = new Float32Array( beauty.length );
				for ( let y = 0; y < height; y ++ ) {

					for ( let x = 0; x < width; x ++ ) {

						const i = ( y * width + x ) * 4;
						point.set( ( x + 0.5 ) / width * 2 - 1, 1 - ( y + 0.5 ) / height * 2, 0.5 ).applyMatrix4( camera.projectionMatrixInverse );
						point.multiplyScalar( - 5 / normal.dot( point ) ).applyMatrix4( camera.projectionMatrix );
						depth[ i ] = point.z;
						beauty[ i + 3 ] = 1;
						normals.set( normal.toArray().map( v => Math.round( ( v * 0.5 + 0.5 ) * 255 ) / 255 * 2 - 1 ).concat( 1 ), i );

					}

				}

				const textures = [ beauty, depth, normals ].map( data => {

					const result = new DataTexture( data, width, height );
					result.type = FloatType;
					result.needsUpdate = true;
					return result;

				} );
				const node = ssgi( ...textures.map( value => texture( value ) ), camera, { outputCoverage: true } );
				node.resolutionScale = 0.5;
				node.useTemporalFiltering = false;
				node.useScreenSpaceSampling.value = false;
				node.sliceCount.value = 2;
				node.stepCount.value = 8;
				node.radius.value = 3;
				node.thickness.value = 0.1;
				const material = new NodeMaterial();
				material.fragmentNode = node.getGINode();
				try {

					renderer.setPixelRatio( 1 );
					renderer.setSize( width, height );
					renderer.contextNode = context();
					await new Promise( resolve => requestAnimationFrame( resolve ) );
					renderer.setRenderTarget( null );
					new QuadMesh( material ).render( renderer );
					const target = node.getRenderTarget();
					const pixels = await renderer.readRenderTargetPixelsAsync( target, 0, 0, target.width, target.height, 1 );
					const coverage = Array.from( pixels ).filter( ( v, i ) => i % 4 === 3 ).map( DataUtils.fromHalfFloat );
					const mean = coverage.reduce( ( sum, value ) => sum + value, 0 ) / coverage.length;
					// Sector quantization leaves a small residual. Unsnapped reconstruction
					// produces about 0.013 mean and 0.037 peak coverage on this isolated plane.
					assert.ok( mean < 0.006, `Mean false coverage stays below 0.006 (actual ${ mean })` );
					assert.ok( coverage.every( value => Number.isFinite( value ) && value >= 0 && value < 0.03 ), 'No large false-occlusion bands' );

				} finally {

					node.dispose(); material.dispose(); textures.forEach( value => value.dispose() );
					renderer.contextNode = previousContext;
					renderer.setPixelRatio( previousRatio );
					renderer.setSize( previousSize.x, previousSize.y );
					RendererUtils.restoreRendererState( renderer, state );

				}

			} );

			QUnit.test( 'lighting coverage is independent of radiance, intensity and AO', async assert => {

				const renderer = await getSharedRenderer( 'webgpu' );
				if ( ! renderer ) {

					assert.ok( true, 'SKIPPED: WebGPU unavailable' );
					return;

				}

				const state = RendererUtils.saveRendererState( renderer );
				const previousContext = renderer.contextNode;
				const previousSize = renderer.getSize( new Vector2() );
				const previousRatio = renderer.getPixelRatio();
				const size = 16;
				const makeTexture = callback => {

					const data = new Float32Array( size * size * 4 );
					for ( let y = 0; y < size; y ++ ) {

						for ( let x = 0; x < size; x ++ ) data.set( callback( x, y ), 4 * ( x + size * y ) );

					}

					const result = new DataTexture( data, size, size );
					result.type = FloatType;
					result.needsUpdate = true;
					return result;

				};

				const inside = ( x, y ) => x >= 4 && x < 12 && y >= 4 && y < 12;
				const source = makeTexture( () => [ 2, 1, 0.5, 1 ] );
				const depth = makeTexture( ( x, y ) => [ inside( x, y ) ? 0.8 : 0.4, 0, 0, 0 ] );
				const normal = makeTexture( ( x, y ) => [ 0, 0, inside( x, y ) ? 1 : - 1, 0 ] );
				const camera = new PerspectiveCamera( 50, 1, 0.1, 10 );
				camera.updateMatrixWorld();
				const node = ssgi( texture( source ), texture( depth ), texture( normal ), camera, { outputCoverage: true } );
				node.sliceCount.value = 2;
				node.stepCount.value = 8;
				node.radius.value = 1;
				node.useTemporalFiltering = false;
				const material = new NodeMaterial();
				material.fragmentNode = node.getGINode();
				const quad = new QuadMesh( material );
				const render = async () => {

					await new Promise( resolve => requestAnimationFrame( resolve ) );
					renderer.setRenderTarget( null );
					quad.render( renderer );
					const target = node.getRenderTarget();
					const data = await renderer.readRenderTargetPixelsAsync( target, 0, 0, target.width, target.height, 1 );
					return Array.from( data, DataUtils.fromHalfFloat );

				};

				const setSource = ( rgb, validity ) => {

					for ( let i = 0; i < source.image.data.length; i += 4 ) source.image.data.set( [ ...rgb, validity ], i );
					source.needsUpdate = true;

				};

				try {

					renderer.setPixelRatio( 1 );
					renderer.setSize( size, size );
					renderer.contextNode = context();
					const lit = await render();
					const coverage = lit.filter( ( v, i ) => i % 4 === 3 );
					assert.ok( coverage.some( v => v > 0 ), 'Screen-space geometry produces nonzero lighting coverage' );
					assert.ok( lit.some( ( v, i ) => i % 4 !== 3 && v > 0 ), 'Facing visible sources produce irradiance' );
					assert.ok( lit.every( v => Number.isFinite( v ) ), 'HDR irradiance and coverage stay finite' );
					assert.ok( coverage.every( v => v >= 0 && v <= 1 ), 'Coverage stays in [0, 1]' );
					setSource( [ 0, 0, 0 ], 1 );
					const black = await render();
					assert.deepEqual( black.filter( ( v, i ) => i % 4 === 3 ), coverage, 'Valid black samples consume the same coverage' );
					assert.ok( black.every( ( v, i ) => i % 4 === 3 || v === 0 ), 'Black lighting produces zero irradiance' );
					setSource( [ 2, 1, 0.5 ], 1 );
					node.giIntensity.value = 0;
					node.aoIntensity.value = 4;
					const zeroGain = await render();
					assert.deepEqual( zeroGain.filter( ( v, i ) => i % 4 === 3 ), coverage, 'Neither GI gain nor AO power changes coverage' );
					assert.ok( zeroGain.every( ( v, i ) => i % 4 === 3 || v === 0 ), 'Zero GI gain affects RGB only' );
					node.giIntensity.value = 1;
					setSource( [ 2, 1, 0.5 ], 0 );
					const invalid = await render();
					assert.ok( invalid.every( v => v === 0 ), 'Rejected history contributes neither lighting nor coverage' );
					setSource( [ 2, 1, 0.5 ], 1 );
					node.resolutionScale = 0.5;
					const halfResolution = await render();
					assert.strictEqual( halfResolution.length, 8 * 8 * 4, 'Resolution scale resizes the irradiance/coverage target' );
					assert.ok( halfResolution.some( ( v, i ) => i % 4 === 3 && v > 0 ), 'Half-resolution tracing retains valid coverage' );
					for ( let i = 0; i < depth.image.data.length; i += 4 ) depth.image.data[ i ] = 1;
					depth.needsUpdate = true;
					const background = await render();
					assert.ok( background.every( v => v === 0 ), 'Background pixels clear both irradiance and coverage' );


				} finally {

					node.dispose();
					material.dispose();
					for ( const value of [ source, depth, normal ] ) value.dispose();
					renderer.contextNode = previousContext;
					renderer.setPixelRatio( previousRatio );
					renderer.setSize( previousSize.x, previousSize.y );
					RendererUtils.restoreRendererState( renderer, state );

				}

			} );

		} );

	} );

} );
