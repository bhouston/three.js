import { describe, test, expect } from 'vitest';
import { UniformsUtils } from '@src/renderers/shaders/UniformsUtils.js';
import { Color } from '@src/math/Color.js';
import { Vector2 } from '@src/math/Vector2.js';
import { Vector3 } from '@src/math/Vector3.js';
import { Vector4 } from '@src/math/Vector4.js';
import { Matrix3 } from '@src/math/Matrix3.js';
import { Matrix4 } from '@src/math/Matrix4.js';
import { Quaternion } from '@src/math/Quaternion.js';
import { Texture } from '@src/textures/Texture.js';
import { CubeReflectionMapping, UVMapping } from '@src/constants.js';
import { CONSOLE_LEVEL } from '@test-utils/console-wrapper.js';

describe( 'Renderers', () => {

	describe( 'Shaders', () => {

		describe( 'UniformsUtils', () => {

			test( 'Instancing', () => {

				expect( UniformsUtils ).toBeTruthy();

			} );

			test( 'cloneUniforms copies values', () => {

				const uniforms = {
					floatValue: { value: 1.23 },
					intValue: { value: 1 },
					boolValue: { value: true },
					colorValue: { value: new Color( 0xFF00FF ) },
					vector2Value: { value: new Vector2( 1, 2 ) },
					vector3Value: { value: new Vector3( 1, 2, 3 ) },
					vector4Value: { value: new Vector4( 1, 2, 3, 4 ) },
					matrix3Value: { value: new Matrix3() },
					matrix4Value: { value: new Matrix4() },
					quatValue: { value: new Quaternion( 1, 2, 3, 4 ) },
					arrayValue: { value: [ 1, 2, 3, 4 ] },
					textureValue: { value: new Texture( null, CubeReflectionMapping ) },
				};

				const uniformClones = UniformsUtils.clone( uniforms );

				expect( uniforms.floatValue.value === uniformClones.floatValue.value ).toBeTruthy();
				expect( uniforms.intValue.value === uniformClones.intValue.value ).toBeTruthy();
				expect( uniforms.boolValue.value === uniformClones.boolValue.value ).toBeTruthy();
				expect( uniforms.colorValue.value.equals( uniformClones.colorValue.value ) ).toBeTruthy();
				expect( uniforms.vector2Value.value.equals( uniformClones.vector2Value.value ) ).toBeTruthy();
				expect( uniforms.vector3Value.value.equals( uniformClones.vector3Value.value ) ).toBeTruthy();
				expect( uniforms.vector4Value.value.equals( uniformClones.vector4Value.value ) ).toBeTruthy();
				expect( uniforms.matrix3Value.value.equals( uniformClones.matrix3Value.value ) ).toBeTruthy();
				expect( uniforms.matrix4Value.value.equals( uniformClones.matrix4Value.value ) ).toBeTruthy();
				expect( uniforms.quatValue.value.equals( uniformClones.quatValue.value ) ).toBeTruthy();
				expect( uniforms.textureValue.value.source.uuid === uniformClones.textureValue.value.source.uuid ).toBeTruthy();
				expect( uniforms.textureValue.value.mapping === uniformClones.textureValue.value.mapping ).toBeTruthy();
				for ( let i = 0; i < uniforms.arrayValue.value.length; ++ i ) {

					expect( uniforms.arrayValue.value[ i ] === uniformClones.arrayValue.value[ i ] ).toBeTruthy();

				}

			} );

			test( 'cloneUniforms clones properties', () => {

				const uniforms = {
					floatValue: { value: 1.23 },
					intValue: { value: 1 },
					boolValue: { value: true },
					colorValue: { value: new Color( 0xFF00FF ) },
					vector2Value: { value: new Vector2( 1, 2 ) },
					vector3Value: { value: new Vector3( 1, 2, 3 ) },
					vector4Value: { value: new Vector4( 1, 2, 3, 4 ) },
					matrix3Value: { value: new Matrix3() },
					matrix4Value: { value: new Matrix4() },
					quatValue: { value: new Quaternion( 1, 2, 3, 4 ) },
					arrayValue: { value: [ 1, 2, 3, 4 ] },
					textureValue: { value: new Texture( null, CubeReflectionMapping ) },
				};

				const uniformClones = UniformsUtils.clone( uniforms );

				// Modify the originals
				uniforms.floatValue.value = 123.0;
				uniforms.intValue.value = 123;
				uniforms.boolValue.value = false;
				uniforms.colorValue.value.r = 123.0;
				uniforms.vector2Value.value.x = 123.0;
				uniforms.vector3Value.value.x = 123.0;
				uniforms.vector4Value.value.x = 123.0;
				uniforms.matrix3Value.value.elements[ 0 ] = 123.0;
				uniforms.matrix4Value.value.elements[ 0 ] = 123.0;
				uniforms.quatValue.value.x = 123.0;
				uniforms.arrayValue.value[ 0 ] = 123.0;
				uniforms.textureValue.value.mapping = UVMapping;

				expect( uniforms.floatValue.value !== uniformClones.floatValue.value ).toBeTruthy();
				expect( uniforms.intValue.value !== uniformClones.intValue.value ).toBeTruthy();
				expect( uniforms.boolValue.value !== uniformClones.boolValue.value ).toBeTruthy();
				expect( ! uniforms.colorValue.value.equals( uniformClones.colorValue.value ) ).toBeTruthy();
				expect( ! uniforms.vector2Value.value.equals( uniformClones.vector2Value.value ) ).toBeTruthy();
				expect( ! uniforms.vector3Value.value.equals( uniformClones.vector3Value.value ) ).toBeTruthy();
				expect( ! uniforms.vector4Value.value.equals( uniformClones.vector4Value.value ) ).toBeTruthy();
				expect( ! uniforms.matrix3Value.value.equals( uniformClones.matrix3Value.value ) ).toBeTruthy();
				expect( ! uniforms.matrix4Value.value.equals( uniformClones.matrix4Value.value ) ).toBeTruthy();
				expect( ! uniforms.quatValue.value.equals( uniformClones.quatValue.value ) ).toBeTruthy();
				expect( uniforms.textureValue.value.mapping !== uniformClones.textureValue.value.mapping ).toBeTruthy();
				expect( uniforms.arrayValue.value[ 0 ] !== uniformClones.arrayValue.value[ 0 ] ).toBeTruthy();

				// Texture source remains same
				expect( uniforms.textureValue.value.source.uuid === uniformClones.textureValue.value.source.uuid ).toBeTruthy();

			} );

			test( 'cloneUniforms clones arrays of objects', () => {

				const uniforms = {
					vector3Array: { value: [ new Vector3( 1, 2, 3 ), new Vector3( 4, 5, 6 ) ] },
				};

				const uniformClones = UniformsUtils.clone( uniforms );

				// Cloned array is a different reference and contains different object references
				expect( uniforms.vector3Array.value !== uniformClones.vector3Array.value ).toBeTruthy();
				expect( uniforms.vector3Array.value[ 0 ] !== uniformClones.vector3Array.value[ 0 ] ).toBeTruthy();
				expect( uniforms.vector3Array.value[ 1 ] !== uniformClones.vector3Array.value[ 1 ] ).toBeTruthy();

				// Values are equal after cloning
				expect( uniforms.vector3Array.value[ 0 ].equals( uniformClones.vector3Array.value[ 0 ] ) ).toBeTruthy();
				expect( uniforms.vector3Array.value[ 1 ].equals( uniformClones.vector3Array.value[ 1 ] ) ).toBeTruthy();

				// Mutating the original does not affect the clone
				uniforms.vector3Array.value[ 0 ].x = 123.0;
				expect( ! uniforms.vector3Array.value[ 0 ].equals( uniformClones.vector3Array.value[ 0 ] ) ).toBeTruthy();

			} );

			test( 'cloneUniforms skips render target textures', () => {

				const uniforms = {
					textureValue: { value: new Texture( null, CubeReflectionMapping ) },
				};

				uniforms.textureValue.value.isRenderTargetTexture = true;

				console.level = CONSOLE_LEVEL.OFF;
				const uniformClones = UniformsUtils.clone( uniforms );
				console.level = CONSOLE_LEVEL.DEFAULT;

				expect( uniformClones.textureValue.value === null ).toBeTruthy();

			} );

		} );

	} );

} );
