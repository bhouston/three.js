import { Vector3, Matrix3, Matrix4, PlaneGeometry, ObjectSpaceBumpMap, WorldSpaceBumpMap, MeshStandardMaterial, MaterialLoader, Texture } from '../../../../src/Three.js';
import { createBumpMapRenderer, heightTexture, assertNormal } from '../../utils/bump-map-renderer.js';

QUnit.module( 'Materials > bump height spaces', () => {

	QUnit.test( 'defaults, copying and serialization', assert => {

		const material = new MeshStandardMaterial( { bumpMap: new Texture(), bumpMapType: WorldSpaceBumpMap, bumpScale: 0.005 } );
		assert.strictEqual( new MeshStandardMaterial().bumpMapType, ObjectSpaceBumpMap, 'Object space is the default' );
		assert.strictEqual( material.clone().bumpMapType, WorldSpaceBumpMap, 'Copy preserves the space' );
		const loader = new MaterialLoader().setTextures( { [ material.bumpMap.uuid ]: material.bumpMap } );
		assert.strictEqual( loader.parse( material.toJSON() ).bumpMapType, WorldSpaceBumpMap, 'Serialization preserves the space' );

	} );

	for ( const backend of [ 'legacy', 'webgl', 'webgpu' ] ) {

		QUnit.test( backend, async assert => {

			const test = await createBumpMapRenderer( backend );
			const ramp = heightTexture();
			const normal = new Vector3( 1, 0, 1 ).normalize();
			const tangent = new Vector3( 1, 0, - 1 ).normalize();
			const matrices = [
				new Matrix4(),
				new Matrix4().makeScale( 3, 3, 3 ),
				new Matrix4().makeScale( 2, 1, 1 ),
				new Matrix4().set( 1, 0, 0.4, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 ),
				new Matrix4().makeScale( - 2, 1, 1 )
			];

			try {

				for ( const kind of [ 'mesh', 'instance', 'batch' ] ) {

					for ( let i = 0; i < matrices.length; i ++ ) {

						// Existing batch/instance normal transforms do not support shear.
						if ( i === 3 && kind !== 'mesh' ) continue;
						// Mirrored instance transforms are not supported by InstancedMesh.
						if ( i === 4 && kind !== 'mesh' ) continue;

						const matrix = matrices[ i ];
						const normalMatrix = new Matrix3().getNormalMatrix( matrix );
						const worldNormal = normal.clone().applyMatrix3( normalMatrix ).normalize();
						const worldTangent = tangent.clone().applyMatrix3( new Matrix3().setFromMatrix4( matrix ) );

						for ( const bumpMapType of [ ObjectSpaceBumpMap, WorldSpaceBumpMap ] ) {

							const expected = bumpMapType === ObjectSpaceBumpMap ? normal.clone().addScaledVector( tangent, - 0.5 ).applyMatrix3( normalMatrix ).normalize() : worldNormal.clone().addScaledVector( worldTangent, - 0.5 / worldTangent.lengthSq() ).normalize();
							const geometry = new PlaneGeometry( 2, 2 ).rotateY( Math.PI / 4 );
							const actual = await test.render( { bumpMap: ramp, bumpScale: 1, bumpMapType }, { geometry, matrix, kind } );
							assertNormal( assert, actual, expected.toArray(), `${ kind }, matrix ${ i }, space ${ bumpMapType }` );

						}

					}

				}

				for ( const resolution of [ 32, 64 ] ) {

					for ( const distance of [ 3, 9 ] ) {

						const geometry = new PlaneGeometry( 2, 2 ).rotateY( Math.PI / 4 );
						const expected = normal.clone().addScaledVector( tangent, - 0.5 ).normalize();
						assertNormal( assert, await test.render( { bumpMap: ramp, bumpScale: 1 }, { geometry, resolution, distance, perspective: true } ), expected.toArray(), `Perspective distance ${ distance }, resolution ${ resolution }` );

					}

				}

				// Retiling changes slope without changing the authored height amplitude.
				ramp.repeat.x = 2;
				ramp.offset.x = - 0.5;
				const retiled = normal.clone().addScaledVector( tangent, - 1 ).normalize();
				assertNormal( assert, await test.render( { bumpMap: ramp, bumpScale: 1 }, { geometry: new PlaneGeometry( 2, 2 ).rotateY( Math.PI / 4 ) } ), retiled.toArray(), 'Doubling UV frequency doubles slope' );
				ramp.repeat.x = 1;
				ramp.offset.x = 0;

				const matrix = matrices[ 2 ];
				const expected = normal.clone().addScaledVector( tangent, - 0.5 ).applyMatrix3( new Matrix3().getNormalMatrix( matrix ) ).normalize();
				assertNormal( assert, await test.render( { bumpMap: ramp, bumpScale: 1, flatShading: true }, { geometry: new PlaneGeometry( 2, 2 ).rotateY( Math.PI / 4 ), matrix } ), expected.toArray(), 'Flat shading uses the same object-space metric' );

			} finally {

				ramp.dispose();
				test.dispose();

			}

		} );

	}

} );
