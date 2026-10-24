import { HalfFloatType, Mesh, MeshStandardNodeMaterial, MeshPhysicalNodeMaterial, PlaneGeometry, RenderTarget, Scene, DataTexture, DataUtils, FloatType, NodeMaterial, PerspectiveCamera, QuadMesh, RendererUtils, Vector2, Vector3 } from 'three/webgpu';
import { Fn, Loop, context, float, texture, uint, vec2, vec3, vec4, indirectDiffuse, indirectSpecular } from 'three/tsl';
import { supportsUnlayeredDiffuse } from '../../../../examples/jsm/tsl/utils/UnlayeredDiffuseScope.js';
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
				assert.closeAbs( ssgiSectorCoverage( uint( 0x00FFFF00 ) ), float( 0.5 ), 1e-5, 'The central half of a frontal slice covers half its projected solid angle' );
				assert.greaterThan( ssgiSectorCoverage( uint( 1 << 8 ) ), ssgiSectorCoverage( uint( 1 << 15 ) ), 'The spherical area Jacobian suppresses sectors near the view axis' );

			} );

			gpuTest( 'sector integration matches independent spherical quadrature', ( { assert } ) => {

				// Midpoint quadrature of n dot omega times the spherical area measure,
				// rather than the closed-form primitive used by the shader.
				const quadrature = ( mask, n, projectedLength ) => {

					let integral = 0;
					const samplesPerSector = 256;
					const step = Math.PI / ( 32 * samplesPerSector );
					for ( let sector = 0; sector < 32; sector ++ ) {

						if ( ( ( mask >>> sector ) & 1 ) === 0 ) continue;
						for ( let j = 0; j < samplesPerSector; j ++ ) {

							const beta = - Math.PI / 2 + ( sector + ( j + 0.5 ) / samplesPerSector ) * Math.PI / 32;
							integral += projectedLength * Math.cos( beta ) * Math.abs( Math.sin( beta + n ) ) * step;

						}

					}

					return integral;

				};

				for ( const [ mask, n, length ] of [
					[ 0x00FFFF00, 0, 1 ],
					[ 0x55555555, 0.8, 0.75 ],
					[ 0x12345678, - 1.2, 0.4 ],
					[ 0xFFFFFFFF, 0.7, 0.9 ],
					[ 0xFFFFFFFF, 2.1, 0.9 ],
					[ 0xFFFFFFFF, - 2.1, 0.9 ],
					[ 0xFFFFFFFF, 0, 0 ]
				] ) {

					assert.closeAbs( ssgiSectorCoverage( uint( mask ), float( n ), float( length ) ), float( quadrature( mask, n, length ) ), 2e-5, `Mask ${ mask.toString( 16 ) }, normal angle ${ n }, projection ${ length }` );

				}

			} );

			gpuTest( 'tilted white furnace is normalized over view-aligned slice orientations', ( { assert } ) => {

				for ( const tilt of [ 0, 0.5, 1.2, 1.5 ] ) {

					const average = Fn( () => {

						const sum = float( 0 ).toVar();
						Loop( 256, ( { i } ) => {

							const phi = float( i ).add( 0.5 ).mul( Math.PI / 256 );
							const tangent = phi.cos().mul( Math.sin( tilt ) );
							const projectedLength = tangent.mul( tangent ).add( Math.cos( tilt ) ** 2 ).sqrt().toConst();
							const n = float( Math.cos( tilt ) ).div( projectedLength ).clamp( - 1, 1 ).acos().mul( tangent.sign() );
							sum.addAssign( ssgiSectorCoverage( uint( 0xFFFFFFFF ), n, projectedLength ) );

						} );
						return sum.div( 256 );

					} )();
					assert.closeAbs( average, float( 1 ), 2e-4, `Unit incoming radiance integrates to pi irradiance at normal tilt ${ tilt }` );

				}

			} );

			gpuTest( 'two-slice normalization preserves a matched screen and fallback furnace', ( { assert } ) => {

				for ( const tilt of [ 0.5, 1.2, 1.5 ] ) {

					for ( let phase = 0; phase < 8; phase ++ ) {

						const partition = Fn( () => {

							const observed = float( 0 ).toVar();
							const fallback = float( 0 ).toVar();
							const full = float( 0 ).toVar();
							const normalization = float( 0 ).toVar();
							Loop( 2, ( { i } ) => {

								const phi = float( i ).add( phase / 8 ).mul( Math.PI / 2 );
								const tangent = phi.cos().mul( Math.sin( tilt ) );
								const length = tangent.mul( tangent ).add( Math.cos( tilt ) ** 2 ).sqrt().toConst();
								const n = float( Math.cos( tilt ) ).div( length ).clamp( - 1, 1 ).acos().mul( tangent.sign() ).toConst();
								const a = n.abs();
								normalization.addAssign( length.mul( a.cos().add( a.mul( a.sin() ) ) ) );
								// Complementary directional radiance fields: their sum is
								// uniform even though neither field is uniform on its own.
								observed.addAssign( ssgiSectorCoverage( uint( 0x12345678 ), n, length ) );
								fallback.addAssign( ssgiSectorCoverage( uint( 0xEDCBA987 ), n, length ) );
								full.addAssign( ssgiSectorCoverage( uint( 0xFFFFFFFF ), n, length ) );

							} );
							return vec2( observed.add( fallback ).div( normalization ), full.div( normalization ) );

						} )();
						assert.closeAbs( partition.x, float( 1 ), 2e-5, `Complementary masks partition unity at tilt ${ tilt }, phase ${ phase }` );
						assert.closeAbs( partition.y, float( 1 ), 2e-5, `Fully observed furnace stays normalized with only two slices at tilt ${ tilt }, phase ${ phase }` );

					}

				}

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
					// Quantized normals and discrete depth can leave a small residual;
					// a coplanar surface must not create large false-occlusion bands.
					assert.ok( mean < 0.006, `Mean false coverage stays below 0.006 (actual ${ mean })` );
					assert.ok( coverage.every( value => Number.isFinite( value ) && value >= 0 && value < 0.03 ), `No large false-occlusion bands (peak ${ Math.max( ...coverage ) })` );

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


			QUnit.test( 'fallback-only AO preserves visible SSGI in the physical lighting model', async assert => {

				const renderer = await getSharedRenderer( 'webgpu' );
				if ( ! renderer ) {

					assert.ok( true, 'SKIPPED: WebGPU unavailable' );
					return;

				}

				const state = RendererUtils.saveRendererState( renderer );
				const oldContext = renderer.contextNode;
				const scene = new Scene();
				const camera = new PerspectiveCamera( 50, 1, 0.1, 10 );
				camera.position.z = 2;
				const geometry = new PlaneGeometry( 4, 4 );
				const material = new MeshStandardNodeMaterial( { roughness: 1, metalness: 0 } );
				scene.add( new Mesh( geometry, material ) );
				const target = new RenderTarget( 8, 8, { type: HalfFloatType } );
				const screen = () => vec4( vec3( 2 * Math.PI ), 0.5 );
				const render = async scoped => {

					const scopeAO = scoped && supportsUnlayeredDiffuse( scene.children[ 0 ].material );
					const fallback = () => vec4( vec3( Math.PI * ( scopeAO ? 0.25 : 1 ) ), 1 );
					renderer.contextNode = context( {
						getAO: input => scopeAO ? input : input !== null ? input.mul( 0.25 ) : float( 0.25 ),
						indirectDiffuseNode: indirectDiffuse( [ screen, fallback ] ),
						indirectSpecularNode: indirectSpecular( [] )
					} );
					material.needsUpdate = true;
					renderer.setRenderTarget( target );
					renderer.render( scene, camera );
					return Array.from( await renderer.readRenderTargetPixelsAsync( target, 4, 4, 1, 1 ), DataUtils.fromHalfFloat )[ 0 ];

				};

				try {

					const all = await render( false );
					const scoped = await render( true );
					assert.ok( all > 0.01, 'The actual physical material receives the controlled irradiance' );
					assert.ok( Math.abs( scoped / all - 3 ) < 0.02, 'Visible SSGI retains its energy while fallback alone receives quarter AO' );
					for ( const layers of [ { clearcoat: 1 }, { sheen: 1 }, { clearcoatNode: float( 1 ) }, { sheenNode: vec3( 1 ) } ] ) {

						const layered = new MeshPhysicalNodeMaterial( { roughness: 0.4, ...layers } );
						scene.children[ 0 ].material = layered;
						try {

							const legacy = await render( false );
							const guarded = await render( true );
							assert.strictEqual( guarded, legacy, `Unsupported ${ Object.keys( layers )[ 0 ] } retains legacy whole-indirect AO` );

						} finally {

							layered.dispose();

						}

					}


				} finally {

					renderer.contextNode = oldContext;
					geometry.dispose(); material.dispose(); target.dispose();
					RendererUtils.restoreRendererState( renderer, state );

				}

			} );


			QUnit.test( 'unlayered diffuse scope conservatively rejects unsupported materials', assert => {

				assert.ok( supportsUnlayeredDiffuse( { isMeshStandardMaterial: true } ), 'Standard opaque material supported' );
				for ( const overrides of [ { clearcoat: 1 }, { sheen: 1 }, { transmission: 0.5 }, { clearcoatNode: {} }, { sheenNode: {} }, { transmissionNode: {} }, { transparent: true }, { fragmentNode: {} }, { outputNode: {} }, { backdropNode: {} }, { lightsNode: {} } ] ) {

					assert.notOk( supportsUnlayeredDiffuse( { isMeshStandardNodeMaterial: true, ...overrides } ), `${ Object.keys( overrides )[ 0 ] } keeps legacy integration` );

				}

			} );

		} );

	} );

} );
