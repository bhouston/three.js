import { DataUtils, Mesh, MeshBasicNodeMaterial, OrthographicCamera, PerspectiveCamera, PlaneGeometry, RenderPipeline, RendererUtils, Scene, Vector2, WebGPUCoordinateSystem } from 'three/webgpu';
import { mrt, output, pass, vec4, velocity } from 'three/tsl';
import { traa } from '../../../../examples/jsm/tsl/display/TRAANode.js';
import { temporalReproject } from '../../../../examples/jsm/tsl/display/TemporalReprojectNode.js';
import { getProjectionJitter } from '../../../../examples/jsm/tsl/utils/TAAUtils.js';
import { getSharedRenderer } from '../../src/nodes/tsl/gpu-test-utils.js';

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'TSL', () => {

		QUnit.module( 'TemporalPipeline', () => {

			QUnit.test( 'jitter preserves custom projection and tiled camera state', assert => {

				for ( const camera of [ new PerspectiveCamera(), new OrthographicCamera() ] ) {

					camera.setViewOffset( 128, 128, 32, 16, 64, 64 );
					camera.projectionMatrix.elements[ 4 ] += 0.125;
					camera.projectionMatrixInverse.copy( camera.projectionMatrix ).invert();
					const original = camera.projectionMatrix.clone();
					const view = { ...camera.view };
					const node = traa( vec4( 1 ), null, null, camera );
					node._velocityNode = velocity;
					try {

						node.setViewOffset( 64, 64 );
						const jitter = getProjectionJitter( camera, original, new Vector2() );
						assert.ok( Math.abs( jitter.y - 1 / ( 6 * 64 ) ) < 1e-12, 'Centered Halton offset produces expected UV raster displacement' );
						assert.deepEqual( camera.view, view, 'Tile settings survive jitter' );
						assert.ok( velocity.projectionMatrix.equals( original ), 'Velocity uses the original custom projection' );
						node.clearViewOffset();
						assert.ok( camera.projectionMatrix.equals( original ), 'Custom projection restored exactly' );
						assert.ok( camera.projectionMatrixInverse.equals( original.clone().invert() ), 'Inverse projection restored' );
						assert.deepEqual( camera.view, view, 'Tile settings survive cleanup' );
						assert.strictEqual( velocity.projectionMatrix, null, 'Velocity override cleared' );

					} finally {

						node.dispose();

					}

				}

			} );

			QUnit.test( 'TRAA owns jitter through an upstream temporal denoiser', async assert => {

				const renderer = await getSharedRenderer( 'webgpu' );
				if ( ! renderer ) {

					assert.ok( true, 'SKIPPED: WebGPU unavailable' );
					return;

				}

				const state = RendererUtils.saveRendererState( renderer );
				const oldSize = renderer.getSize( new Vector2() );
				const oldRatio = renderer.getPixelRatio();
				const camera = new PerspectiveCamera( 50, 1, 0.1, 100 );
				camera.position.z = 2;
				camera.coordinateSystem = WebGPUCoordinateSystem;
				camera.updateProjectionMatrix();
				const original = camera.projectionMatrix.clone();
				const scene = new Scene();
				const geometry = new PlaneGeometry( 10, 10 );
				const material = new MeshBasicNodeMaterial();
				scene.add( new Mesh( geometry, material ) );
				const source = pass( scene, camera );
				source.setMRT( mrt( { output, normal: vec4( 0.5, 0.5, 1, 1 ), velocity } ) );
				const depth = source.getTextureNode( 'depth' );
				const motion = source.getTextureNode( 'velocity' );
				const temporal = temporalReproject( source, depth, source.getTextureNode( 'normal' ), motion, camera, { accumulate: true } );
				const aa = traa( temporal, depth, motion, camera );
				const pipeline = new RenderPipeline( renderer, aa );
				const projections = [];
				scene.onAfterRender = () => projections.push( camera.projectionMatrix.clone() );
				const copy = renderer.copyTextureToTexture;
				let resolveCopies = 0;
				renderer.copyTextureToTexture = function ( source, ...args ) {

					if ( source === temporal._resolveRenderTarget.texture || source === aa._resolveRenderTarget.texture ) resolveCopies ++;
					return copy.call( this, source, ...args );

				};

				try {

					renderer.setPixelRatio( 1 );
					renderer.setSize( 16, 16 );
					for ( let frame = 0; frame < 4; frame ++ ) {

						await new Promise( resolve => requestAnimationFrame( resolve ) );
						pipeline.render();
						assert.deepEqual( camera.projectionMatrix.elements, original.elements, 'Projection restored after rendering' );
						assert.strictEqual( velocity.projectionMatrix, null, 'Velocity override restored after rendering' );

					}

					assert.strictEqual( pipeline._contextData.renderPipelineState.viewOffsetOwner, aa, 'AA owns jitter despite upstream temporal denoiser' );
					assert.strictEqual( aa._jitterIndex, 3, 'Jitter advances on each frame after pipeline compilation' );
					assert.ok( ! projections.at( - 1 ).equals( original ), 'Geometry renders with jitter' );
					assert.ok( ! projections.at( - 1 ).equals( projections.at( - 2 ) ), 'Geometry receives a different jitter each frame' );
					assert.strictEqual( resolveCopies, 0, 'Temporal resolve and TRAA retain color history without copies' );
					const target = source.renderTarget;
					const index = target.textures.indexOf( source.getTexture( 'velocity' ) );
					const data = await renderer.readRenderTargetPixelsAsync( target, 8, 8, 1, 1, index );
					assert.ok( Array.from( data, DataUtils.fromHalfFloat ).slice( 0, 2 ).every( v => Math.abs( v ) < 1e-5 ), 'Static geometry has zero motion despite changing jitter' );

				} finally {

					renderer.copyTextureToTexture = copy;
					pipeline.dispose();
					aa.dispose();
					temporal.dispose();
					source.dispose();
					geometry.dispose();
					material.dispose();
					renderer.setPixelRatio( oldRatio );
					renderer.setSize( oldSize.x, oldSize.y );
					RendererUtils.restoreRendererState( renderer, state );

				}

			} );

		} );

	} );

} );
