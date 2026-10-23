import { AmbientLight, ArrayCamera, BoxGeometry, Frustum, Group, LOD, MathUtils, Matrix4, Mesh, MeshBasicMaterial, PerspectiveCamera, Scene, SphereGeometry } from 'three';
import { FrustumArray } from '../../../../src/math/FrustumArray.js';
import { SceneSpatialIndex } from '../../../../examples/jsm/utils/SceneSpatialIndex.js';

const geometry = new SphereGeometry( 1, 8, 4 );
const material = new MeshBasicMaterial();

function makeCamera( x = 0 ) {

	const camera = new PerspectiveCamera( 50, 1, 1, 120 );
	camera.position.set( x, 2, 20 );
	camera.updateMatrixWorld();
	return camera;

}

function makeFrustum( camera ) {

	return new Frustum().setFromProjectionMatrix( new Matrix4().multiplyMatrices( camera.projectionMatrix, camera.matrixWorldInverse ) );

}

// Simulate the renderer's visibility/layers/child traversal, including its final sphere test.
function collect( root, camera, frustum, indexed = true ) {

	const result = [];
	function visit( object ) {

		if ( object.visible === false ) return;
		if ( object.isMesh && object.layers.test( camera.layers ) && ( ! object.frustumCulled || object.intersectsFrustum( frustum ) ) ) result.push( object.id );
		const children = indexed && object.getRenderChildren ? object.getRenderChildren( camera, frustum ) : object.children;
		for ( const child of children ) visit( child );

	}

	visit( root );
	return result;

}

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'Utils', () => {

		QUnit.module( 'SceneSpatialIndex', () => {

			QUnit.test( 'flat and nonspatial nested scenes match ordinary traversal across cameras', ( assert ) => {

				MathUtils.seededRandom( 150 );
				for ( const nested of [ false, true ] ) {

					const scene = new Scene();
					const groups = [ scene ];
					if ( nested ) {

						for ( let i = 0; i < 7; i ++ ) {

							const group = new Group();
							group.position.set( i * 2, 0, - i );
							group.rotation.y = i * 0.17;
							group.scale.set( 1.1, 0.8, 1.2 );
							groups[ i % groups.length ].add( group );
							groups.push( group );

						}

					}

					for ( let i = 0; i < 256; i ++ ) {

						const mesh = new Mesh( geometry, material );
						mesh.position.set( MathUtils.seededRandom() * 400 - 200, 0, MathUtils.seededRandom() * 400 - 200 );
						groups[ i % groups.length ].add( mesh );

					}

					const ownership = new Map();
					scene.traverse( object => ownership.set( object, { parent: object.parent, children: object.children.slice() } ) );
					const index = new SceneSpatialIndex( scene );
					for ( const hierarchical of [ false, true ] ) {

						index.hierarchical = hierarchical;
						for ( let i = 0; i < 12; i ++ ) {

							const camera = makeCamera( i * 13 - 70 );
							camera.rotation.y = i * Math.PI / 6;
							camera.updateMatrixWorld();
							const frustum = makeFrustum( camera );
							assert.deepEqual( collect( scene, camera, frustum ), collect( scene, camera, frustum, false ), `nested=${ nested }, hierarchical=${ hierarchical }, camera=${ i }: same draw order` );

						}

					}

					for ( const [ object, original ] of ownership ) {

						assert.strictEqual( object.parent, original.parent, 'Parent preserved' );
						assert.deepEqual( object.children, original.children, 'Authored child order preserved' );

					}

					index.dispose();

				}

			} );

			QUnit.test( 'visibility and layer changes keep ordinary ancestor semantics', ( assert ) => {

				const root = new Group();
				const hidden = new Group();
				const layers = new Group();
				const direct = new Mesh( geometry, material );
				const underHidden = new Mesh( geometry, material );
				const underLayers = new Mesh( geometry, material );
				root.add( hidden, layers, direct );
				hidden.add( underHidden );
				layers.add( underLayers );
				layers.layers.set( 2 );
				const camera = makeCamera();
				const frustum = makeFrustum( camera );
				const index = new SceneSpatialIndex( root );
				hidden.visible = false;
				direct.layers.set( 1 );
				assert.deepEqual( collect( root, camera, frustum ), [ underLayers.id ], 'Hidden ancestors prune; mismatched parent layers do not prune matching descendants' );
				hidden.visible = true;
				camera.layers.enable( 1 );
				assert.deepEqual( collect( root, camera, frustum ), collect( root, camera, frustum, false ), 'Visibility and layer changes do not require a spatial rebuild' );
				index.dispose();

			} );

			QUnit.test( 'array-camera union retains objects visible to either eye', ( assert ) => {

				const root = new Group();
				for ( const x of [ - 80, 0, 80, 200 ] ) {

					const mesh = new Mesh( geometry, material );
					mesh.position.x = x;
					root.add( mesh );

				}

				const camera = new ArrayCamera( [ makeCamera( - 80 ), makeCamera( 80 ) ] );
				const frustum = new FrustumArray().setFromArrayCamera( camera );
				const index = new SceneSpatialIndex( root );
				for ( const hierarchical of [ false, true ] ) {

					index.hierarchical = hierarchical;
					assert.deepEqual( collect( root, camera, frustum ), [ root.children[ 0 ].id, root.children[ 2 ].id ], 'Disconnected frustum union has no false negatives' );

				}

				index.dispose();

			} );

			QUnit.test( 'rebuild refreshes moved, replaced, added and removed objects', ( assert ) => {

				const root = new Group();
				const moved = new Mesh( geometry, material );
				moved.position.x = 300;
				const removed = new Mesh( geometry, material );
				root.add( moved, removed );
				const index = new SceneSpatialIndex( root );
				const camera = makeCamera();
				const frustum = makeFrustum( camera );
				assert.deepEqual( collect( root, camera, frustum ), [ removed.id ], 'Initial selection' );
				moved.position.x = 0;
				moved.geometry = new BoxGeometry( 2, 2, 2 );
				root.remove( removed );
				const added = new Mesh( geometry, material );
				root.add( added );
				index.rebuild();
				assert.deepEqual( collect( root, camera, frustum ), [ moved.id, added.id ], 'Rebuild refreshes bounds and membership in source order' );
				index.enabled = false;
				assert.strictEqual( root.getRenderChildren( camera, frustum ), root.children, 'Disabled index uses ordinary children' );
				index.dispose();
				assert.strictEqual( root.getRenderChildren, undefined, 'Disposal removes owned hook' );
				assert.deepEqual( collect( root, camera, frustum ), [ moved.id, added.id ], 'Ordinary traversal remains valid' );
				moved.geometry.dispose();

			} );

			QUnit.test( 'lights, uncullable meshes, LOD and custom subtrees retain ordinary traversal', ( assert ) => {

				const root = new Group();
				const light = new AmbientLight();
				const uncullable = new Mesh( geometry, material );
				uncullable.frustumCulled = false;
				uncullable.position.x = 500;
				const lod = new LOD();
				const lodMesh = new Mesh( geometry, material );
				lod.addLevel( lodMesh, 0 );
				lod.position.x = 500;
				const custom = new Group();
				const customHook = () => custom.children;
				custom.getRenderChildren = customHook;
				custom.add( new Mesh( geometry, material ) );
				const outside = new Mesh( geometry, material );
				outside.position.x = 500;
				root.add( outside, light, uncullable, lod, custom );
				const index = new SceneSpatialIndex( root );
				const camera = makeCamera();
				assert.deepEqual( root.getRenderChildren( camera, makeFrustum( camera ) ), [ light, uncullable, lod, custom ], 'Fallback objects retain source order while indexed outside mesh is pruned' );
				assert.strictEqual( lod.getRenderChildren, undefined, 'LOD subtree remains under normal renderer control' );
				assert.strictEqual( custom.getRenderChildren, customHook, 'Existing subtree hook is preserved' );
				assert.deepEqual( collect( root, camera, makeFrustum( camera ) ), collect( root, camera, makeFrustum( camera ), false ), 'Fallback and indexed objects produce same visible meshes' );
				const replacement = () => [];
				root.getRenderChildren = replacement;
				index.dispose();
				assert.strictEqual( root.getRenderChildren, replacement, 'Disposal leaves a subsequently installed hook intact' );
				assert.strictEqual( custom.getRenderChildren, customHook, 'Disposal preserves existing subtree hook' );

			} );

		} );

	} );

} );

