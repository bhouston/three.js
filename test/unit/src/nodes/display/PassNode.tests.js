import { HalfFloatType, UnsignedByteType } from '../../../../../src/constants.js';
import { PerspectiveCamera } from '../../../../../src/cameras/PerspectiveCamera.js';
import { Scene } from '../../../../../src/scenes/Scene.js';
import { pass } from '../../../../../src/nodes/display/PassNode.js';
import { context } from '../../../../../src/nodes/core/ContextNode.js';

export default QUnit.module( 'Nodes', () => {

	QUnit.module( 'Display', () => {

		QUnit.module( 'PassNode', () => {

			QUnit.test( 'nested passes isolate their contexts', ( assert ) => {

				const camera = new PerspectiveCamera();
				const outer = pass( new Scene(), camera );
				const inner = pass( new Scene(), camera );
				const plain = pass( new Scene(), camera );
				outer.contextNode = context( { outerOnly: true, shared: 'outer' } );
				inner.contextNode = context( { innerOnly: true, shared: 'inner' } );

				let renderTarget = null, mrt = null;
				const contexts = [];
				const renderer = {
					contextNode: null,
					getOutputRenderTarget: () => null,
					getDrawingBufferSize: size => size.set( 16, 16 ),
					getRenderTarget: () => renderTarget,
					setRenderTarget: value => {

						renderTarget = value;

					},
					getMRT: () => mrt,
					setMRT: value => {

						mrt = value;

					},
					render: scene => {

						const currentContext = renderer.contextNode;
						contexts.push( currentContext.getFlowContextData() );

						if ( scene === outer.scene ) inner.updateBefore( { renderer } );
						if ( scene === inner.scene ) plain.updateBefore( { renderer } );

						assert.strictEqual( renderer.contextNode, currentContext, 'A nested render restores the calling pass context.' );

					}
				};

				// Repeat with a new renderer context to exercise cached pass contexts too.
				for ( const value of [ 'first', 'second' ] ) {

					const rendererContext = context( { rendererOnly: value, shared: 'renderer' } );
					renderer.contextNode = rendererContext;
					contexts.length = 0;
					outer.updateBefore( { renderer } );

					assert.deepEqual( contexts, [
						{ rendererOnly: value, shared: 'outer', outerOnly: true },
						{ rendererOnly: value, shared: 'inner', innerOnly: true },
						{ rendererOnly: value, shared: 'renderer' }
					], 'Each pass inherits renderer values and only its own overrides.' );
					assert.strictEqual( renderer.contextNode, rendererContext, 'The outer pass restores the renderer context.' );

				}

			} );


			QUnit.test( 'output type', ( assert ) => {

				const renderer = { samples: 0, reversedDepthBuffer: false, getOutputBufferType: () => HalfFloatType };
				const scene = new Scene();
				const camera = new PerspectiveCamera();

				const defaultPass = pass( scene, camera );
				defaultPass.setup( { renderer } );

				assert.strictEqual( defaultPass.renderTarget.texture.type, HalfFloatType, 'Defaults to the output buffer type of the renderer.' );

				const bytePass = pass( scene, camera, { type: UnsignedByteType } );
				bytePass.setup( { renderer } );

				assert.strictEqual( bytePass.renderTarget.texture.type, UnsignedByteType, 'Honors the type option.' );

			} );

		} );

	} );

} );
