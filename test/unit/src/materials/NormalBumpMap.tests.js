import { DoubleSide, ObjectSpaceNormalMap, PlaneGeometry, RGFormat, DataTexture, FloatType } from '../../../../src/Three.js';
import { createBumpMapRenderer, heightTexture, normalTexture, assertNormal } from '../../utils/bump-map-renderer.js';

QUnit.module( 'Materials > normal and bump maps', () => {

	for ( const backend of [ 'legacy', 'webgl', 'webgpu' ] ) {

		QUnit.test( backend, async assert => {

			const test = await createBumpMapRenderer( backend );
			const ramp = heightTexture( 'y' );
			const flat = heightTexture( 'constant' );
			const neutral = normalTexture();
			const tilted = normalTexture( 0.3 );

			try {

				for ( const normalMapType of [ 0, ObjectSpaceNormalMap ] ) {

					const parameters = { normalMapType, bumpScale: 4, side: DoubleSide };
					const normalOnly = await test.render( { ...parameters, normalMap: tilted } );
					const bumpOnly = await test.render( { ...parameters, bumpMap: ramp } );
					const combined = await test.render( { ...parameters, normalMap: tilted, bumpMap: ramp } );
					assert.ok( combined[ 0 ] > 0.2, 'Normal-map tilt contributes to the combined normal.' );
					assert.ok( combined[ 1 ] < - 0.05, 'Bump-map slope contributes to the combined normal.' );
					assertNormal( assert, await test.render( { ...parameters, normalMap: tilted, bumpMap: flat } ), normalOnly, 'Constant height preserves the normal map' );
					assertNormal( assert, await test.render( { ...parameters, normalMap: tilted, bumpMap: ramp, bumpScale: 0 } ), normalOnly, 'Zero bump scale preserves the normal map' );
					assertNormal( assert, await test.render( { ...parameters, normalMap: neutral, bumpMap: ramp } ), bumpOnly, 'Neutral normal map preserves the bump map' );
					assert.ok( ( await test.render( { ...parameters, normalMap: tilted, bumpMap: ramp, bumpScale: - 4 } ) )[ 1 ] > 0.05, 'Negative bump scale inverts the detail' );
					const back = await test.render( { ...parameters, normalMap: tilted, bumpMap: ramp }, { back: true } );
					assert.ok( back.every( Number.isFinite ) && Math.abs( back[ 1 ] ) > 0.05, 'Double-sided back faces include the bump map' );

				}

				const packed = new DataTexture( new Float32Array( [ 0.65, 0.5 ] ), 1, 1, RGFormat, FloatType );
				packed.needsUpdate = true;
				const packedResult = await test.render( { normalMap: packed, bumpMap: ramp, bumpScale: 4 } );
				const unpackedResult = await test.render( { normalMap: tilted, bumpMap: ramp, bumpScale: 4 } );
				assertNormal( assert, packedResult, unpackedResult, 'Packed RG normals combine with bump maps' );
				packed.dispose();

				const geometry = new PlaneGeometry( 2, 2 );
				geometry.setAttribute( 'uv1', geometry.attributes.uv.clone() );
				const uv1 = geometry.attributes.uv1;
				for ( let i = 0; i < uv1.count; i ++ ) uv1.setY( i, 1 - uv1.getX( i ) );
				ramp.channel = 1;
				const separateUV = await test.render( { normalMap: tilted, bumpMap: ramp, bumpScale: 4 }, { geometry } );
				assert.ok( separateUV[ 0 ] > 0.35 && Math.abs( separateUV[ 1 ] ) < 0.002, `Bump and normal maps use their own UV channels: ${ separateUV }` );

			} finally {

				for ( const texture of [ ramp, flat, neutral, tilted ] ) texture.dispose();
				test.dispose();

			}

		} );

	}

} );
