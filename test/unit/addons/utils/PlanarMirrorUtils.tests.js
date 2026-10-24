import { BackSide, BoxGeometry, BufferGeometry, Float32BufferAttribute, Mesh, MeshStandardMaterial, PlaneGeometry, SphereGeometry, Vector3 } from 'three/webgpu';
import { getPlanarMirrorPlane } from '../../../../examples/jsm/utils/PlanarMirrorUtils.js';

QUnit.module( 'Addons / Utils / PlanarMirrorUtils', () => {

	QUnit.test( 'Metallic planes qualify regardless of color and roughness', assert => {

		const geometry = new PlaneGeometry( 4, 4 );
		const material = new MeshStandardMaterial( { color: 0x880022, metalness: 0.95, roughness: 0.9 } );
		const mesh = new Mesh( geometry, material );
		mesh.position.set( 1, 2, 3 );
		mesh.rotation.y = Math.PI / 3;
		mesh.scale.set( 2, 3, 0.5 );
		mesh.updateMatrixWorld();
		const plane = getPlanarMirrorPlane( mesh, new Vector3( 10, 2, 10 ) );
		assert.ok( plane, 'Rough, colored metal qualifies' );
		assert.ok( Math.abs( plane.distanceToPoint( mesh.position ) ) < 1e-6, 'Plane follows the world transform' );
		assert.ok( plane.normal.dot( new Vector3( 0, 0, 1 ).transformDirection( mesh.matrixWorld ) ) > 0.999, 'Plane has the transformed surface orientation' );
		material.metalness = 0.89;
		assert.strictEqual( getPlanarMirrorPlane( mesh, new Vector3() ), null, 'Low metalness is excluded' );
		material.dispose(); geometry.dispose();

	} );

	QUnit.test( 'Thin floors face the scene; solid and curved objects are excluded', assert => {

		const material = new MeshStandardMaterial( { metalness: 1 } );
		const floorGeometry = new BoxGeometry( 10, 0.1, 10 );
		const sphereGeometry = new SphereGeometry( 2 );
		const beamGeometry = new BoxGeometry( 100, 0.1, 0.1 );
		const boxGeometry = new BoxGeometry( 2, 2, 2 );
		const floor = new Mesh( floorGeometry, material );
		floor.updateMatrixWorld();
		const plane = getPlanarMirrorPlane( floor, new Vector3( 0, 2, 0 ) );
		assert.ok( plane, 'Thin metallic slab qualifies' );
		assert.ok( plane.normal.y > 0.999 && Math.abs( plane.constant + 0.05 ) < 1e-6, 'Selects the upper floor surface' );
		for ( const geometry of [ sphereGeometry, beamGeometry, boxGeometry ] ) {

			assert.strictEqual( getPlanarMirrorPlane( new Mesh( geometry, material ), new Vector3() ), null, 'Rejects non-planar solid geometry' );
			geometry.dispose();

		}

		material.side = BackSide;
		assert.strictEqual( getPlanarMirrorPlane( floor, new Vector3() ), null, 'Back-side-only surfaces are excluded' );
		floorGeometry.dispose(); material.dispose();

	} );

	QUnit.test( 'Degenerate and unsupported meshes are excluded', assert => {

		const geometry = new BufferGeometry();
		geometry.setAttribute( 'position', new Float32BufferAttribute( [ 0, 0, 0, 1, 0, 0, 2, 0, 0 ], 3 ) );
		const material = new MeshStandardMaterial( { metalness: 1 } );
		const mesh = new Mesh( geometry, material );
		assert.strictEqual( getPlanarMirrorPlane( mesh, new Vector3() ), null, 'Collinear triangles do not form a mirror plane' );
		mesh.material = [ material ];
		assert.strictEqual( getPlanarMirrorPlane( mesh, new Vector3() ), null, 'Multiple material meshes are excluded' );
		geometry.dispose(); material.dispose();

	} );

} );
