import { HalfFloatType, PerspectiveCamera, Scene, Texture, Vector2 } from 'three/webgpu';
import { pass, texture, vec3 } from 'three/tsl';
import PowerLobeSSRNode, { powerLobeSSR } from '../../../../examples/jsm/tsl/display/PowerLobeSSRNode.js';
import { ssr } from '../../../../examples/jsm/tsl/display/SSRNode.js';

function fixture() {

	const camera = new PerspectiveCamera( 50, 1, 0.1, 50 );
	const opaque = pass( new Scene(), camera );
	opaque.transparent = false;
	const color = opaque.getTextureNode( 'output' );
	const depth = opaque.getTextureNode( 'depth' );
	const normal = vec3( 0, 0, 1 );
	return { camera, opaque, color, depth, normal, node: powerLobeSSR( color, depth, normal ) };

}

// Exercises lifecycle ordering without creating or making claims about a GPU context.
function recordingRenderer( events ) {

	let target = null;
	return {
		autoClear: true,
		getDrawingBufferSize: size => size.copy( new Vector2( 64, 48 ) ),
		getRenderTarget: () => target,
		getActiveCubeFace: () => 0,
		getActiveMipmapLevel: () => 0,
		getRenderObjectFunction: () => null,
		getPixelRatio: () => 1,
		getMRT: () => null,
		getClearColor: color => color.set( 0 ),
		getClearAlpha: () => 1,
		getScissorTest: () => false,
		setMRT: () => {},
		setRenderObjectFunction: () => {},
		setClearColor: () => {},
		setPixelRatio: () => {},
		setScissorTest: () => {},
		setRenderTarget: value => {

			target = value;
			events.push( { type: 'target', target } );

		}
	};

}

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'TSL', () => {

		QUnit.module( 'PowerLobeSSRNode', () => {

			QUnit.test( 'camera inference, explicit camera and missing-camera guard', assert => {

				const { node, camera, opaque, color, depth, normal } = fixture();
				assert.ok( node instanceof PowerLobeSSRNode );
				assert.strictEqual( node.camera, camera, 'Pass camera is inferred' );
				assert.strictEqual( node.colorNode, color, 'Opaque source identity is retained' );
				assert.strictEqual( node.depthNode, depth, 'Depth source identity is retained' );
				assert.notOk( opaque.transparent, 'Construction does not alter opaque pass configuration' );
				opaque.transparent = true;
				assert.throws( () => powerLobeSSR( color, depth, normal ), /opaque color pass/, 'Transparent final beauty cannot become reflection history' );
				opaque.transparent = false;
				const explicitCamera = new PerspectiveCamera();
				const explicit = powerLobeSSR( color, depth, normal, { camera: explicitCamera } );
				assert.strictEqual( explicit.camera, explicitCamera, 'Explicit camera takes precedence' );
				const external = texture( new Texture() );
				assert.throws( () => powerLobeSSR( external, depth, normal ), /requires a camera/, 'External textures require an explicit camera' );
				node.dispose();
				explicit.dispose();
				external.value.dispose();
				opaque.dispose();

			} );

			QUnit.test( 'weighted distance, incident coverage and metadata remain distinct scheduled HDR outputs', assert => {

				const { node, opaque } = fixture();
				const outputs = [ node.getTextureNode(), node.getIncidentNode(), node.getMetadataNode(), node.getHitNode() ];
				assert.strictEqual( new Set( outputs ).size, 4, 'Each contract has a separate texture node' );
				assert.strictEqual( new Set( outputs.map( output => output.value ) ).size, 4, 'Outputs cannot alias one alpha channel' );
				for ( const output of outputs ) {

					assert.strictEqual( output.passNode, node, 'Sampling every output schedules the producing pass' );
					assert.strictEqual( output.value.type, HalfFloatType, 'All output contracts preserve HDR precision' );

				}

				assert.deepEqual( outputs.map( output => output.value ), node.getRenderTarget().textures, 'Compatibility output is attachment zero; auxiliary outputs follow' );
				assert.strictEqual( node.getTextureNode(), outputs[ 0 ], 'Bindings remain stable across calls' );
				node.dispose();
				opaque.dispose();

			} );

			QUnit.test( 'history reset restarts deterministic sampling without changing comparison settings', assert => {

				const { node, opaque } = fixture();
				node.seed.value = 123;
				node.rayCount.value = 4;
				node.steps.value = 32;
				node._frame.value = 37;
				node._ready.value = true;
				const outputs = [ node.getTextureNode(), node.getIncidentNode(), node.getMetadataNode(), node.getHitNode() ];
				node.resetHistory();
				assert.notOk( node._ready.value, 'Stale history becomes invalid immediately' );
				assert.strictEqual( node._frame.value, 0, 'Noise sequence restarts reproducibly' );
				assert.deepEqual( [ node.seed.value, node.rayCount.value, node.steps.value ], [ 123, 4, 32 ], 'Estimator settings survive a history reset' );
				assert.deepEqual( [ node.getTextureNode(), node.getIncidentNode(), node.getMetadataNode(), node.getHitNode() ], outputs, 'Reset preserves all output bindings' );
				node.dispose();
				opaque.dispose();

			} );

			QUnit.test( 'trace reads old opaque history before copying the current opaque frame', assert => {

				const { node, opaque } = fixture();
				const events = [];
				const renderer = recordingRenderer( events );
				node._quad.render = () => events.push( { type: 'draw', material: node._quad.material, ready: node._ready.value } );
				const frame = { renderer, updateBeforeNode: passNode => events.push( { type: 'dependency', passNode } ) };
				node.updateBefore( frame );
				const draws = events.filter( event => event.type === 'draw' );
				assert.strictEqual( events[ 0 ].type, 'dependency', 'Opaque producer updates before either draw' );
				assert.strictEqual( events[ 0 ].passNode, opaque );
				assert.deepEqual( draws.map( draw => draw.material ), [ node._material, node._copyMaterial ], 'Trace runs before the history copy' );
				assert.deepEqual( draws.map( draw => draw.ready ), [ false, false ], 'First-frame history stays invalid throughout tracing and copying' );
				assert.ok( node._ready.value, 'History becomes usable after the copy' );
				assert.strictEqual( node._frame.value, 1, 'Noise advances once per completed update' );
				assert.deepEqual( [ node.getRenderTarget().width, node.getRenderTarget().height ], [ 64, 48 ], 'Trace follows the drawing buffer' );
				assert.strictEqual( renderer.getRenderTarget(), null, 'Renderer target is restored' );
				events.length = 0;
				node.updateBefore( frame );
				assert.ok( events.find( event => event.type === 'draw' ).ready, 'The next trace can read the completed previous frame' );
				node.resolutionScale = 0.5;
				events.length = 0;
				node.updateBefore( frame );
				assert.notOk( events.find( event => event.type === 'draw' ).ready, 'Changing trace resolution rejects stale history' );
				assert.deepEqual( [ node.getRenderTarget().width, node.getRenderTarget().height ], [ 32, 24 ], 'Trace resolution scales independently' );
				assert.deepEqual( [ node._history.width, node._history.height ], [ 64, 48 ], 'Opaque radiance history stays at source resolution' );
				node.camera.position.x = 3;
				node.camera.updateMatrixWorld( true );
				events.length = 0;
				node.updateBefore( frame );
				assert.notOk( events.find( event => event.type === 'draw' ).ready, 'A camera cut rejects history before tracing' );
				node.resetHistory();
				events.length = 0;
				node.updateBefore( frame );
				assert.notOk( events.find( event => event.type === 'draw' ).ready, 'Explicit reset forces fallback on the next trace' );
				node.dispose();
				opaque.dispose();

			} );

			QUnit.test( 'unsupported logarithmic depth fails explicitly before building a shader', assert => {

				const { node, opaque } = fixture();
				assert.throws( () => node.setup( { renderer: { logarithmicDepthBuffer: true } } ), /does not support logarithmic depth/ );
				node.dispose();
				opaque.dispose();

			} );

			QUnit.test( 'disposal releases owned resources and preserves caller textures', assert => {

				const { node, opaque, color, depth } = fixture();
				let externalDisposals = 0;
				color.value.addEventListener( 'dispose', () => externalDisposals ++ );
				depth.value.addEventListener( 'dispose', () => externalDisposals ++ );
				const owned = [ node.getRenderTarget(), node._history, node._material, node._copyMaterial ];
				const counts = owned.map( () => 0 );
				owned.forEach( ( resource, i ) => resource.addEventListener( 'dispose', () => counts[ i ] ++ ) );
				node.dispose();
				assert.deepEqual( counts, [ 1, 1, 1, 1 ], 'Targets and materials each receive disposal' );
				assert.strictEqual( externalDisposals, 0, 'Caller-owned opaque textures remain usable' );
				opaque.dispose();

			} );

			QUnit.test( 'experimental opt-in leaves stock SSR defaults intact', assert => {

				const { node, opaque, color, depth, normal } = fixture();
				const stock = ssr( color, depth, normal );
				assert.notOk( stock.stochastic, 'Stock factory still selects mirror/blur' );
				assert.notOk( stock.reflectNonMetals, 'Stock metal-only default is preserved' );
				assert.strictEqual( stock.resolutionScale, 1, 'Stock trace resolution remains full size' );
				assert.strictEqual( node.steps.value, 16, 'Alternative starts with bounded 16-step traversal' );
				assert.strictEqual( node.rayCount.value, 1, 'Alternative starts with one ray for comparisons' );
				assert.strictEqual( node.maxLuminance.value, 0, 'Alternative has no implicit radiance clamp' );
				stock.dispose();
				node.dispose();
				opaque.dispose();

			} );

		} );

	} );

} );
