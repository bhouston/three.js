import { AnimationClip, Box3, BoxGeometry, BufferGeometry, DataUtils, Float16BufferAttribute, Int16BufferAttribute, Uint16BufferAttribute, Group, InstancedMesh, LOD, Mesh, MeshBasicMaterial, ShaderMaterial, SkinnedMesh, Vector3, VectorKeyframeTrack } from 'three';
import { ClippingGroup } from 'three/webgpu';
import { batchStaticMeshes } from '../../../../examples/jsm/utils/batchStaticMeshes.js';

function mesh( material, x = 0 ) {

	const geometry = new BoxGeometry( 0.2, 0.2, 0.2 );
	geometry.clearGroups();
	geometry.setAttribute( 'uv1', geometry.attributes.uv.clone() );
	const object = new Mesh( geometry, material );
	object.position.x = x;
	return object;

}

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'Utils', () => {

		QUnit.module( 'batchStaticMeshes', () => {

			QUnit.test( 'preserves root-relative geometry, UVs and render flags; cleanup restores sources', assert => {

				const root = new Group(), material = new MeshBasicMaterial();
				root.position.set( 5, 2, 3 ); root.rotation.y = 0.3; root.scale.set( 2, 1, 3 );
				const a = mesh( material, 0 ), b = mesh( material, 0.3 );
				for ( const object of [ a, b ] ) {

					object.castShadow = true; object.receiveShadow = true; object.layers.set( 2 ); object.renderOrder = 7;

				}

				b.geometry.attributes.uv1.array.fill( 0.25 );
				root.add( a, b ); root.updateMatrixWorld( true );
				const expected = new Box3().setFromObject( root );
				const uv = [ ...a.geometry.attributes.uv1.array, ...b.geometry.attributes.uv1.array ];
				const result = batchStaticMeshes( root, { cellSize: 10 } );
				assert.strictEqual( result.batchCount, 1, 'Compatible meshes share one batch' );
				assert.strictEqual( result.sourceMeshes.length, 2, 'Both sources retained' );
				const batch = result.group.children[ 0 ];
				assert.deepEqual( Array.from( batch.geometry.attributes.uv1.array ), uv, 'Atlas UVs survive merging exactly' );
				root.updateMatrixWorld( true );
				const actual = new Box3().setFromObject( batch );
				assert.ok( actual.min.distanceTo( expected.min ) < 1e-5 && actual.max.distanceTo( expected.max ) < 1e-5, 'World geometry survives a translated, rotated and scaled root' );
				assert.ok( batch.castShadow && batch.receiveShadow && batch.layers.mask === a.layers.mask && batch.renderOrder === 7, 'Render flags preserved' );
				assert.ok( ! a.visible && ! b.visible, 'Original draws hidden' );
				root.position.x += 10; root.updateMatrixWorld( true );
				assert.ok( new Box3().setFromObject( batch ).min.distanceTo( expected.min.clone().add( new Vector3( 10, 0, 0 ) ) ) < 1e-5, 'Root can move after batching' );
				let geometryDisposals = 0, materialDisposals = 0;
				batch.geometry.addEventListener( 'dispose', () => geometryDisposals ++ ); material.addEventListener( 'dispose', () => materialDisposals ++ );
				result.dispose(); result.dispose();
				assert.ok( a.visible && b.visible && result.group.parent === null, 'Cleanup restores originals and removes batches' );
				assert.strictEqual( geometryDisposals, 1, 'Owned geometry disposed once' );
				assert.strictEqual( materialDisposals, 0, 'Shared material remains caller-owned' );

			} );

			QUnit.test( 'promotes quantized transforms and preserves half-float UV values', assert => {

				const geometry = new BufferGeometry();
				geometry.setAttribute( 'position', new Uint16BufferAttribute( [ 0, 0, 0, 65535, 0, 0, 0, 65535, 0 ], 3, true ) );
				geometry.setAttribute( 'normal', new Int16BufferAttribute( [ 0, 0, 32767, 0, 0, 32767, 0, 0, 32767 ], 3, true ) );
				geometry.setAttribute( 'uv1', new Float16BufferAttribute( [ 0.25, 0.75, 1, 0, 0, 1 ].map( DataUtils.toHalfFloat ), 2 ) );
				const root = new Group(), material = new MeshBasicMaterial();
				const a = new Mesh( geometry, material ), b = new Mesh( geometry, material );
				a.position.x = 0.3; b.position.x = 1.7; root.add( a, b );
				const expected = new Box3().setFromObject( root );
				const result = batchStaticMeshes( root, { cellSize: 10 } );
				const batch = result.group.children[ 0 ]; root.updateMatrixWorld( true );
				assert.ok( batch.geometry.attributes.position.array instanceof Float32Array, 'Transformed quantized positions use float storage' );
				const actual = new Box3().setFromObject( batch );
				assert.ok( actual.min.distanceTo( expected.min ) < 1e-6 && actual.max.distanceTo( expected.max ) < 1e-6, 'Fractional transforms survive quantized input' );
				assert.strictEqual( batch.geometry.attributes.uv1.getX( 0 ), 0.25, 'Half-float UV X retains its decoded value' );
				assert.strictEqual( batch.geometry.attributes.uv1.getY( 0 ), 0.75, 'Half-float UV Y retains its decoded value' );
				assert.strictEqual( batch.geometry.attributes.normal.getZ( 0 ), 1, 'Quantized normals retain their decoded direction' );
				assert.ok( geometry.attributes.position.array instanceof Uint16Array && geometry.attributes.uv1.isFloat16BufferAttribute, 'Original attributes are untouched' ); result.dispose();

			} );

			QUnit.test( 'retains spatial cells and ancestor group ordering', assert => {

				const root = new Group(), material = new MeshBasicMaterial();
				root.add( mesh( material, 0 ), mesh( material, 0.2 ), mesh( material, 20 ), mesh( material, 20.2 ) );
				const result = batchStaticMeshes( root, { cellSize: 3 } );
				assert.strictEqual( result.batchCount, 2, 'Distant clusters keep separate culling bounds' ); result.dispose();
				const ordered = new Group(); ordered.renderOrder = 5; ordered.layers.set( 2 ); ordered.add( mesh( material ), mesh( material, 0.2 ) ); root.add( ordered );
				const orderedResult = batchStaticMeshes( root, { cellSize: 100 } );
				assert.strictEqual( orderedResult.batchCount, 2, 'Different ancestor group order remains separate' );
				assert.ok( orderedResult.group.children.some( child => child.isGroup && child.renderOrder === 5 && child.layers.mask === ordered.layers.mask ), 'Order and layer conditions retained in the batch hierarchy' ); orderedResult.dispose();

			} );

			QUnit.test( 'leaves animated and unsupported meshes untouched', assert => {

				const root = new Group(), material = new MeshBasicMaterial();
				const a = mesh( material ), b = mesh( material, 0.2 ); root.add( a, b );
				const animated = new Group(); animated.name = 'animated'; animated.add( mesh( material ), mesh( material, 0.2 ) ); root.add( animated );
				const clip = new AnimationClip( 'move', 1, [ new VectorKeyframeTrack( 'animated.position', [ 0, 1 ], [ 0, 0, 0, 1, 0, 0 ] ) ] );
				const transparent = mesh( new MeshBasicMaterial( { transparent: true } ) ), custom = mesh( new ShaderMaterial() ), hooked = mesh( material ), mirrored = mesh( material ), morphed = mesh( material );
				hooked.onBeforeRender = () => {};

				mirrored.scale.x = - 1; morphed.geometry.morphAttributes.position = [ morphed.geometry.attributes.position.clone() ];
				const skinned = new SkinnedMesh( a.geometry, material ), instanced = new InstancedMesh( a.geometry, material, 1 );
				const lod = new LOD(), clipping = new ClippingGroup();
				lod.addLevel( mesh( material ) ); clipping.add( mesh( material ), mesh( material ) ); root.add( lod, clipping );
				const unsupported = [ transparent, custom, hooked, mirrored, morphed, skinned, instanced ]; root.add( ...unsupported );
				const result = batchStaticMeshes( root, { cellSize: 10, animations: [ clip ] } );
				assert.strictEqual( result.sourceMeshes.length, 2, 'Only the two supported static meshes are batched' );
				assert.ok( unsupported.every( object => object.visible ) && animated.children.every( object => object.visible ), 'Unsupported objects and animated descendants remain unchanged' ); result.dispose();
				assert.ok( lod.children.every( object => object.visible ) && clipping.children.every( object => object.visible ), 'LOD and clipping hierarchies remain unchanged' );

			} );

			QUnit.test( 'respects filters, hidden objects and compatibility boundaries', assert => {

				const root = new Group(), material = new MeshBasicMaterial();
				const a = mesh( material ), b = mesh( material ), c = mesh( material ), hidden = mesh( material );
				hidden.visible = false; c.receiveShadow = true; root.add( a, b, c, hidden );
				const filtered = batchStaticMeshes( root, { filter: object => object !== b } );
				assert.strictEqual( filtered.batchCount, 0, 'Filter and render flags keep incompatible singletons separate' ); filtered.dispose();
				const result = batchStaticMeshes( root );
				assert.strictEqual( result.sourceMeshes.length, 2, 'Compatible visible pair is batched' );
				assert.ok( c.visible && ! hidden.visible, 'Different flags and pre-existing hidden state survive' ); result.dispose();
				assert.notOk( hidden.visible, 'Cleanup retains pre-existing hidden state' );
				assert.throws( () => batchStaticMeshes( root, { cellSize: 0 } ), RangeError, 'Invalid cells rejected' );

			} );

		} );

	} );

} );
