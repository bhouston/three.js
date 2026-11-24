import { Object3D, Texture } from 'three/webgpu';
import { texture } from 'three/tsl';
import { gaussianBlur } from '../../../../examples/jsm/tsl/display/GaussianBlurNode.js';
import { RendererInspector } from '../../../../examples/jsm/inspector/RendererInspector.js';

function createInspector( results = [] ) {

	let timers = 0;
	const inspector = new RendererInspector();
	inspector.setRenderer( {
		_initialized: true,
		_nodes: { nodeFrame: { frameId: 1 } },
		getRenderTarget: () => null,
		backend: {
			beginTimer: () => results[ timers ++ ] ?? null,
			endTimer: result => Promise.resolve( result )
		}
	} );
	inspector.begin();
	return { inspector, timerCount: () => timers };

}

function render( inspector, renderGroup, name = 'Pass', timestamp = 10, cpu = 2 ) {

	const uid = `r:${ inspector.currentFrame.renders.length }:0:f${ inspector.currentFrame.frameId }`;
	const scene = { name, renderGroup };
	inspector.beginRender( uid, scene, {}, null );
	const stats = inspector.currentRender;
	inspector.finishRender();
	stats.timestamp = timestamp;
	stats.cpu = cpu;
	return stats;

}

async function finish( inspector ) {

	const frame = inspector.currentFrame;
	inspector.finish();
	await Promise.resolve();
	return frame;

}

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'Inspector', () => {

		QUnit.module( 'RendererInspector', () => {

			QUnit.test( 'Captures membership by identity, including interleaved and equally named groups', async assert => {

				const { inspector, timerCount } = createInspector();
				const first = { name: 'Blur' };
				const second = { name: 'Blur' };
				const a = render( inspector, first, 'Horizontal' );
				const b = render( inspector, second );
				const c = render( inspector, first, 'Vertical' );
				const ungrouped = render( inspector, null );
				const group = a.parent;

				assert.deepEqual( group.children, [ a, c ] );
				assert.notStrictEqual( group, b.parent );
				assert.strictEqual( ungrouped.parent, inspector.currentFrame );
				assert.strictEqual( timerCount(), 4, 'Only actual calls allocate timers' );

				a.scene.renderGroup = second;
				a.scene.name = 'Changed';
				assert.strictEqual( a.parent, group, 'Changing a shared scene does not change recorded membership' );
				assert.strictEqual( a.name, 'Horizontal' );

				await finish( inspector );
				inspector.getRenderer()._nodes.nodeFrame.frameId ++;
				inspector.begin();
				const next = render( inspector, first );
				assert.strictEqual( next.parent.cid, group.cid, 'Group identity is stable across frames' );
				assert.notStrictEqual( next.parent, group, 'Each frame has independent statistics' );
				await finish( inspector );

			} );

			QUnit.test( 'Unions overlapping GPU members without counting gaps or extra group timers', async assert => {

				const { inspector, timerCount } = createInspector( [
					{ start: 0, end: 4, delta: 4 },
					{ start: 7, end: 9, delta: 2 },
					{ start: 2, end: 6, delta: 4 },
					{ start: 10, end: 12, delta: 2 }
				] );
				const group = { name: 'Blur' };
				const a = render( inspector, group, 'Horizontal', 10, 2 );
				render( inspector, null, 'Unrelated', 15, 1 );
				render( inspector, group, 'Vertical', 20, 2 );
				render( inspector, group, 'Resolve', 30, 2 );
				const frame = await finish( inspector );

				assert.strictEqual( a.parent.gpu, 8, 'GPU cost is the union of member intervals' );
				assert.strictEqual( frame.gpuBusy, 10, 'The group bounding span does not fill idle gaps' );
				assert.strictEqual( a.parent.cpu, 6, 'CPU cost sums member calls, excluding gaps' );
				assert.strictEqual( a.parent.cpuFrameStart, 0 );
				assert.strictEqual( a.parent.cpuFrameEnd, 22, 'The diagram spans all members, including gaps' );
				assert.strictEqual( a.parent.gpuFrameEnd, 12 );
				assert.strictEqual( timerCount(), 4 );

			} );

			QUnit.test( 'Retains execution ancestry across nested render and compute calls', async assert => {

				const { inspector } = createInspector( [
					{ start: 0, end: 20, delta: 20 },
					{ start: 2, end: 6, delta: 4 },
					{ start: 3, end: 5, delta: 2 },
					{ start: 3.5, end: 4.5, delta: 1 }
				] );
				inspector.beginRender( 'r:0:0:f1', { name: 'Scene', renderGroup: { name: 'Outline' } }, {}, null );
				const outer = inspector.currentRender;
				inspector.beginRender( 'r:1:0:f1', { name: 'Face', renderGroup: { name: 'Point Shadow' } }, {}, {} );
				const shadow = inspector.currentRender;
				inspector.beginCompute( 'c:0:0:f1', { name: 'Compute' } );
				const compute = inspector.currentCompute;
				inspector.beginRender( 'r:2:0:f1', { name: 'Nested in compute' }, {}, {} );
				inspector.finishRender();
				assert.strictEqual( inspector.currentContext, compute );
				assert.strictEqual( inspector.currentRender, shadow, 'Restores render ancestry through a compute call' );
				inspector.finishCompute();
				assert.strictEqual( inspector.currentContext, shadow );
				inspector.finishRender();
				assert.strictEqual( inspector.currentRender, outer );
				assert.strictEqual( inspector.currentContext, outer );
				assert.strictEqual( shadow.executionParent, outer );
				assert.strictEqual( shadow.parent.parent, outer );
				inspector.finishRender();
				assert.strictEqual( inspector.currentContext, inspector.currentFrame );
				const frame = await finish( inspector );
				assert.strictEqual( frame.gpuBusy, 20, 'Nested calls are not counted twice' );

			} );

			QUnit.test( 'Preserves overdraw for grouped canvas renders and ignores calls outside a frame', async assert => {

				const { inspector, timerCount } = createInspector();
				let overlays = 0;
				inspector.overdraw = true;
				inspector._renderOverdraw = () => overlays ++;
				render( inspector, { name: 'Canvas' } );
				assert.strictEqual( overlays, 1 );
				const frame = await finish( inspector );
				inspector.beginRender( 'r:1:0:f1', { name: 'Outside' }, {}, null );
				inspector.finishRender();
				assert.strictEqual( frame.renders.length, 1 );
				assert.strictEqual( timerCount(), 1 );

			} );

			QUnit.test( 'Supports GPU durations without timestamps and unavailable GPU timing', async assert => {

				const { inspector } = createInspector( [
					{ start: null, end: null, delta: 3 },
					{ start: null, end: null, delta: 4 },
					null
				] );
				const group = { name: 'WebGL' };
				const a = render( inspector, group );
				render( inspector, group );
				const unavailable = render( inspector, { name: 'Unavailable' } );
				const frame = await finish( inspector );
				assert.strictEqual( a.parent.gpu, 7 );
				assert.strictEqual( a.parent.gpuFrameStart, null );
				assert.strictEqual( unavailable.parent.gpuNotAvailable, true );
				assert.strictEqual( frame.gpuBusy, 7 );

			} );

			QUnit.test( 'Blur passes declare membership without an installed inspector', assert => {

				const calls = [];
				const renderer = {
					getRenderTarget: () => null,
					getActiveCubeFace: () => 0,
					getActiveMipmapLevel: () => 0,
					getRenderObjectFunction: () => null,
					getPixelRatio: () => 1,
					getMRT: () => null,
					getClearColor: color => color.set( 0 ),
					getClearAlpha: () => 1,
					getScissorTest: () => false,
					setRenderTarget: () => {},
					setRenderObjectFunction: () => {},
					setPixelRatio: () => {},
					setMRT: () => {},
					setClearColor: () => {},
					setScissorTest: () => {},
					render: scene => calls.push( { name: scene.name, renderGroup: scene.renderGroup } )
				};
				const input = texture( new Texture( { width: 4, height: 4 } ) );
				const first = gaussianBlur( input );
				const second = gaussianBlur( input );

				// Material setup and drawing are unnecessary for checking the pass declarations.

				for ( const node of [ first, second ] ) {

					node._hMaterial = { vertexNode: null };
					node._vMaterial = { vertexNode: null };
					node.updateBefore( { renderer } );

				}

				assert.deepEqual( calls.map( call => call.name ), [ 'Horizontal', 'Vertical', 'Horizontal', 'Vertical' ] );
				assert.strictEqual( calls[ 0 ].renderGroup, calls[ 1 ].renderGroup );
				assert.notStrictEqual( calls[ 0 ].renderGroup, calls[ 2 ].renderGroup );
				assert.strictEqual( calls[ 0 ].renderGroup.name, 'Gaussian Blur' );

			} );

			QUnit.test( 'Cloning preserves group membership without serialization or scene parenting', assert => {

				const scene = new Object3D();
				scene.renderGroup = { name: 'Scene passes' };
				const clone = scene.clone();
				assert.strictEqual( clone.renderGroup, scene.renderGroup );
				assert.strictEqual( clone.parent, null );
				assert.strictEqual( scene.toJSON().object.renderGroup, undefined );

			} );

		} );

	} );

} );
