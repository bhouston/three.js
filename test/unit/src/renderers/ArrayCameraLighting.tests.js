import { ArrayCamera } from '../../../../src/cameras/ArrayCamera.js';
import { PerspectiveCamera } from '../../../../src/cameras/PerspectiveCamera.js';
import { RectAreaLight } from '../../../../src/lights/RectAreaLight.js';
import { Matrix4 } from '../../../../src/math/Matrix4.js';
import { Plane } from '../../../../src/math/Plane.js';
import { Vector3 } from '../../../../src/math/Vector3.js';
import { ClippingGroup } from '../../../../src/objects/ClippingGroup.js';
import { Scene } from '../../../../src/scenes/Scene.js';
import ClippingContext from '../../../../src/renderers/common/ClippingContext.js';
import RectAreaLightNode from '../../../../src/nodes/lighting/RectAreaLightNode.js';

QUnit.module( 'Renderers', () => {

	QUnit.module( 'ArrayCamera lighting', () => {

		QUnit.test( 'rectangle axes match ordinary camera views', ( assert ) => {

			const light = new RectAreaLight( 0xffffff, 1, 6, 2 );
			light.rotation.set( 0.3, 0.7, - 0.2 );
			light.scale.set( 2, 3, 4 );
			light.updateMatrixWorld();
			const node = new RectAreaLightNode( light );
			const camera = new ArrayCamera();
			camera.position.set( 7, 2, - 3 );
			camera.rotation.set( 0.2, 0.4, 0.1 );
			camera.updateMatrixWorld();
			node.update( { camera } );
			const width = node.halfWidth.value.clone();
			const height = node.halfHeight.value.clone();
			assert.ok( Math.abs( width.length() - 3 ) < 1e-12, 'width keeps its physical size' );
			assert.ok( Math.abs( height.length() - 1 ) < 1e-12, 'height keeps its physical size' );

			for ( let i = 0; i < 6; i ++ ) {

				const subCamera = new PerspectiveCamera();
				subCamera.position.copy( camera.position );
				subCamera.rotation.set( i * 0.4, i * 0.7, i * 0.2 );
				subCamera.updateMatrixWorld();
				node.update( { camera: subCamera } );
				const rotation = new Matrix4().extractRotation( subCamera.matrixWorldInverse );
				assert.ok( width.clone().applyMatrix4( rotation ).distanceTo( node.halfWidth.value ) < 1e-12, 'view ' + i + ' width matches' );
				assert.ok( height.clone().applyMatrix4( rotation ).distanceTo( node.halfHeight.value ) < 1e-12, 'view ' + i + ' height matches' );

			}

		} );

		QUnit.test( 'nested clipping preserves world half-spaces across views', ( assert ) => {

			const scene = new Scene();
			const root = new ClippingContext();
			const parent = new ClippingGroup();
			parent.clippingPlanes = [ new Plane( new Vector3( 1, 2, 3 ).normalize(), - 2 ) ];
			const child = new ClippingGroup();
			child.clipIntersection = true;
			child.clippingPlanes = [ new Plane( new Vector3( - 2, 1, 1 ).normalize(), 3 ) ];
			const camera = new ArrayCamera();
			camera.position.set( 8, - 4, 3 );
			camera.rotation.set( 0.4, 0.7, - 0.2 );
			camera.updateMatrixWorld();
			root.updateGlobal( scene, camera );
			const context = root.getGroupContext( parent ).getGroupContext( child );
			const worldPoint = new Vector3( 4, - 1, 2 );

			for ( let i = 0; i < 6; i ++ ) {

				const subCamera = new PerspectiveCamera();
				subCamera.position.copy( camera.position );
				subCamera.rotation.set( i * 0.3, i * 0.8, i * 0.1 );
				subCamera.updateMatrixWorld();
				const positionView = worldPoint.clone().applyMatrix4( subCamera.matrixWorldInverse );
				const reconstructed = positionView.clone().applyMatrix4( subCamera.matrixWorld );
				const referenceRoot = new ClippingContext();
				referenceRoot.updateGlobal( scene, subCamera );
				const reference = referenceRoot.getGroupContext( parent ).getGroupContext( child );

				for ( const name of [ 'unionPlanes', 'intersectionPlanes' ] ) {

					const plane = context[ name ][ 0 ];
					const viewPlane = reference[ name ][ 0 ];
					const worldDistance = reconstructed.dot( new Vector3( plane.x, plane.y, plane.z ) ) - plane.w;
					const viewDistance = positionView.dot( new Vector3( viewPlane.x, viewPlane.y, viewPlane.z ) ) - viewPlane.w;
					assert.ok( Math.abs( worldDistance - viewDistance ) < 1e-12, name + ' agrees in view ' + i );

				}

			}

			const ordinaryCamera = new PerspectiveCamera();
			ordinaryCamera.position.set( - 3, 7, 5 );
			ordinaryCamera.rotation.set( 0.5, - 0.6, 0.2 );
			ordinaryCamera.updateMatrixWorld();
			root.updateGlobal( scene, ordinaryCamera );
			const refreshed = root.getGroupContext( parent ).getGroupContext( child );
			assert.strictEqual( refreshed, context, 'nested contexts are reusable when switching cameras' );
			const projected = worldPoint.clone().applyMatrix4( ordinaryCamera.matrixWorldInverse );
			for ( const [ name, source ] of [[ 'unionPlanes', parent ], [ 'intersectionPlanes', child ]] ) {

				const plane = refreshed[ name ][ 0 ];
				const distance = projected.dot( new Vector3( plane.x, plane.y, plane.z ) ) - plane.w;
				assert.ok( Math.abs( distance + source.clippingPlanes[ 0 ].distanceToPoint( worldPoint ) ) < 1e-12, name + ' updates after camera switch' );

			}

		} );

	} );

} );
