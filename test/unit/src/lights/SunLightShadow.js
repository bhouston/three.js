import { describe, test, expect } from 'vitest';
import { SunLightShadow } from '@src/lights/SunLightShadow.js';
import { LightShadow } from '@src/lights/LightShadow.js';
import { ObjectLoader } from '@src/loaders/ObjectLoader.js';
import { SunLight } from '@src/lights/SunLight.js';
import { PerspectiveCamera } from '@src/cameras/PerspectiveCamera.js';
import { OrthographicCamera } from '@src/cameras/OrthographicCamera.js';
import { Vector3 } from '@src/math/Vector3.js';

describe( 'Lights', () => {

	describe( 'SunLightShadow', () => {

		test( 'Extending', () => {

			const object = new SunLightShadow();
			expect( object instanceof LightShadow ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new SunLightShadow();
			expect( object ).toBeTruthy();

		} );

		test( 'isSunLightShadow', () => {

			const object = new SunLightShadow();
			expect( object.isSunLightShadow ).toBeTruthy();

		} );

		test( 'cascades', () => {

			const light = new SunLight();
			const camera = new PerspectiveCamera( 60, 1, 0.1, 100 );
			const shadow = light.shadow;
			light.position.setFromSphericalCoords( 1, Math.PI / 2 - 0.9, 0.7 );
			light.updateMatrixWorld();
			camera.updateMatrixWorld();

			shadow.camera.far = 80;
			shadow.updateMatrices( light, camera );

			expect( shadow.getViewportCount() ).toBe( 4 );
			expect( shadow.getFrameExtents().toArray() ).toEqual( [ 2, 2 ] );
			expect( shadow._cascadeSplits.length ).toBe( 5 );
			expect( shadow._cascadeSplits[ 0 ] ).toBe( camera.near );
			expect( shadow._cascadeSplits[ 4 ] ).toBe( shadow.camera.far );

			const orthographicCamera = new OrthographicCamera( - 1, 1, 1, - 1, 0, 100 );
			orthographicCamera.updateMatrixWorld();
			shadow.updateMatrices( light, orthographicCamera );
			expect( shadow._cascadeSplits.every( Number.isFinite ) ).toBeTruthy();

			const infiniteCamera = new PerspectiveCamera( 60, 1, 0.1, 100 );
			infiniteCamera.projectionMatrix.elements[ 10 ] = - 1;
			infiniteCamera.projectionMatrix.elements[ 14 ] = - 2 * infiniteCamera.near;
			infiniteCamera.projectionMatrixInverse.copy( infiniteCamera.projectionMatrix ).invert();
			infiniteCamera.far = Infinity;
			infiniteCamera.updateMatrixWorld();
			shadow.camera.far = 80;
			shadow.updateMatrices( light, infiniteCamera );
			expect( shadow._matrices.every( matrix => matrix.elements.every( Number.isFinite ) ) ).toBeTruthy();
			expect( shadow._cascadeSplits[ 4 ] ).toBe( shadow.camera.far );

		} );

		test( 'cascade fade', () => {

			const light = new SunLight();
			const camera = new PerspectiveCamera( 60, 1, 0.1, 100 );
			const shadow = light.shadow;
			light.position.setFromSphericalCoords( 1, Math.PI / 2 - 0.9, 0.7 );
			light.updateMatrixWorld();
			camera.updateMatrixWorld();

			shadow.camera.far = 80;
			shadow.updateMatrices( light, camera );

			for ( let i = 0; i < 4; i ++ ) {

				const cascade = shadow._cascadeData[ i ];

				expect( cascade.y ).toBe( shadow._cascadeSplits[ i + 1 ] );
				expect( cascade.z < cascade.y && cascade.z >= shadow._cascadeSplits[ i ] ).toBeTruthy();

				if ( i === 0 ) expect( cascade.x < 0 ).toBeTruthy();
				else expect( cascade.x ).toBe( shadow._cascadeData[ i - 1 ].z );

			}

		} );

		test( 'cascade caster margin', () => {

			const light = new SunLight();
			const camera = new PerspectiveCamera( 60, 1, 0.1, 100 );
			const shadow = light.shadow;
			light.position.setFromSphericalCoords( 1, Math.PI / 2 - 0.9, 0.7 );
			light.updateMatrixWorld();
			camera.updateMatrixWorld();
			shadow.camera.far = 100;
			shadow.updateMatrices( light, camera );

			const lightDirection = light.position.clone().normalize().negate();

			for ( let i = 0; i < 4; i ++ ) {

				const depth = ( shadow._cascadeSplits[ i ] + shadow._cascadeSplits[ i + 1 ] ) * 0.5;
				const caster = new Vector3( 0, 0, - depth ).addScaledVector( lightDirection, - 90 );
				expect( shadow.getFrustum( i ).containsPoint( caster ) ).toBeTruthy();

			}

		} );

		test( 'cascade containment', () => {

			const light = new SunLight();
			const camera = new PerspectiveCamera( 60, 1, 0.01, 0.1 );
			const shadow = light.shadow;
			light.position.setFromSphericalCoords( 1, Math.PI / 2 - 0.9, 0.7 );
			light.updateMatrixWorld();
			camera.updateMatrixWorld();
			shadow.camera.far = 1;
			shadow.updateMatrices( light, camera );

			for ( let i = 0; i < 4; i ++ ) {

				let containsSlice = true;

				for ( const depth of shadow._cascadeSplits.slice( i, i + 2 ) ) {

					const halfHeight = Math.tan( camera.fov * Math.PI / 360 ) * depth;

					for ( const x of [ - 1, 1 ] ) {

						for ( const y of [ - 1, 1 ] ) {

							containsSlice = containsSlice && shadow.getFrustum( i ).containsPoint( new Vector3( x * halfHeight, y * halfHeight, - depth ) );

						}

					}

				}

				expect( containsSlice ).toBeTruthy();

			}

		} );

		test( 'cascade stabilization containment', () => {

			const light = new SunLight();
			const camera = new PerspectiveCamera( 60, 1, 1, 10 );
			const shadow = light.shadow;
			camera.updateMatrixWorld();
			light.position.setFromSphericalCoords( 1, Math.PI / 2 - 0.6, Math.PI / 4 );
			light.updateMatrixWorld();

			shadow.mapSize.set( 16, 16 );
			shadow.updateMatrices( light, camera );

			for ( const depth of shadow._cascadeSplits.slice( 0, 2 ) ) {

				const halfHeight = Math.tan( camera.fov * Math.PI / 360 ) * depth;

				for ( const x of [ - 1, 1 ] ) {

					for ( const y of [ - 1, 1 ] ) {

						const corner = new Vector3( x * halfHeight * camera.aspect, y * halfHeight, - depth ).applyMatrix4( camera.matrixWorld );
						expect( shadow.getFrustum( 0 ).containsPoint( corner ) ).toBeTruthy();

					}

				}

			}

		} );

		test( 'reversed depth cascade containment', () => {

			const light = new SunLight();
			const camera = new PerspectiveCamera( 60, 1, 0.1, 100 );
			const shadow = light.shadow;
			light.position.setFromSphericalCoords( 1, Math.PI / 2 - 0.9, 0.7 );
			light.updateMatrixWorld();
			camera.updateMatrixWorld();

			for ( const [ viewReversedDepth, shadowReversedDepth ] of [[ false, false ], [ false, true ], [ true, true ]] ) {

				camera._reversedDepth = viewReversedDepth;
				camera.updateProjectionMatrix();
				shadow.camera._reversedDepth = shadowReversedDepth;
				shadow.updateMatrices( light, camera );

				for ( let i = 0; i < 4; i ++ ) {

					const cascadeCamera = shadow.getCamera( i );
					const depth = ( shadow._cascadeSplits[ i ] + shadow._cascadeSplits[ i + 1 ] ) * 0.5;
					expect( cascadeCamera.reversedDepth ).toBe( shadowReversedDepth );
					expect( shadow.getMatrix( i ).elements.every( Number.isFinite ) ).toBeTruthy();
					expect( shadow.getFrustum( i ).containsPoint( new Vector3( 0, 0, - depth ) ) ).toBeTruthy();

				}

			}

		} );

		test( 'clone/copy', () => {

			const a = new SunLightShadow();
			const b = new SunLightShadow();

			expect( a ).not.toEqual( b );

			const c = a.clone();
			expect( a ).toSmartEqual( c );

			c.mapSize.set( 1024, 1024 );
			expect( a ).not.toEqual( c );

			b.copy( a );
			expect( a ).toSmartEqual( b );

			b.mapSize.set( 512, 512 );
			expect( a ).not.toEqual( b );

		} );

		test( 'toJSON', () => {

			const light = new SunLight();
			const shadow = light.shadow;

			shadow.bias = 10;
			shadow.radius = 5;
			shadow.mapSize.set( 1024, 1024 );

			const json = light.toJSON();
			const newLight = new ObjectLoader().parse( json );

			expect( newLight.shadow.isSunLightShadow ).toBeTruthy();
			expect( newLight.shadow ).toSmartEqual( light.shadow );

		} );

	} );

} );
