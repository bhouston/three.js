import * as THREE from 'three';
import * as TSL from 'three/tsl';
import { normalize } from '../neural/NeuralVectorMath.js';

function computeAtlasDimensions( capacity, tileSize, alignment = 16 ) {

	const atlasColumns = alignTo( Math.ceil( Math.sqrt( Math.max( 1, capacity ) ) ), alignment );
	const atlasRows = Math.ceil( Math.max( 1, capacity ) / atlasColumns );
	const atlasWidth = atlasColumns * tileSize;
	const atlasHeight = atlasRows * tileSize;

	return {
		atlasColumns,
		atlasRows,
		atlasWidth,
		atlasHeight
	};

}

function createTeacherRenderTarget( width, height ) {

	return new THREE.RenderTarget( width, height, {
		type: THREE.HalfFloatType,
		format: THREE.RGBAFormat,
		colorSpace: THREE.NoColorSpace,
		minFilter: THREE.NearestFilter,
		magFilter: THREE.NearestFilter,
		depthBuffer: false,
		stencilBuffer: false
	} );

}

function createSampleTexture( width, height ) {

	const texture = new THREE.DataTexture( new Float32Array( width * height * 4 ), width, height, THREE.RGBAFormat, THREE.FloatType );
	texture.colorSpace = THREE.NoColorSpace;
	texture.minFilter = THREE.NearestFilter;
	texture.magFilter = THREE.NearestFilter;
	texture.wrapS = THREE.ClampToEdgeWrapping;
	texture.wrapT = THREE.ClampToEdgeWrapping;
	texture.generateMipmaps = false;
	texture.needsUpdate = true;
	return texture;

}

function createAtlasTextures( atlasColumns, atlasRows ) {

	return {
		uv: createSampleTexture( atlasColumns, atlasRows ),
		normal: createSampleTexture( atlasColumns, atlasRows ),
		tangent: createSampleTexture( atlasColumns, atlasRows ),
		bitangent: createSampleTexture( atlasColumns, atlasRows ),
		wi: createSampleTexture( atlasColumns, atlasRows ),
		wo: createSampleTexture( atlasColumns, atlasRows )
	};

}

function createSampleCoordNode( atlasColumns, atlasRows, tileSize ) {

	const tileCoord = TSL.floor( TSL.viewportCoordinate.div( tileSize ) );
	return tileCoord.add( 0.5 ).div( TSL.vec2( atlasColumns, atlasRows ) );

}

function createMaterialUvNode( sampleUv, sampleTangent, sampleBitangent, tileSize ) {

	const tilePixel = TSL.fract( TSL.viewportCoordinate.div( tileSize ) ).sub( 0.5 );
	const duvDx = sampleUv.zw;
	const duvDy = TSL.vec2( sampleTangent.w, sampleBitangent.w );

	return sampleUv.xy.add( duvDx.mul( tilePixel.x ) ).add( duvDy.mul( tilePixel.y ) );

}

function sampleTextureNode( texture, sampleCoord ) {

	return TSL.texture( texture, sampleCoord );

}

function createAtlasShaderNodes( sampleTextures, atlasColumns, atlasRows, tileSize ) {

	const sampleCoord = createSampleCoordNode( atlasColumns, atlasRows, tileSize );
	const sampleUv = sampleTextureNode( sampleTextures.uv, sampleCoord );
	const sampleNormal = sampleTextureNode( sampleTextures.normal, sampleCoord );
	const sampleTangent = sampleTextureNode( sampleTextures.tangent, sampleCoord );
	const sampleBitangent = sampleTextureNode( sampleTextures.bitangent, sampleCoord );
	const sampleWi = sampleTextureNode( sampleTextures.wi, sampleCoord );
	const sampleWo = sampleTextureNode( sampleTextures.wo, sampleCoord );

	return {
		normal: sampleNormal.xyz.normalize(),
		tangent: sampleTangent.xyz.normalize(),
		bitangent: sampleBitangent.xyz.normalize(),
		wi: sampleWi.xyz.normalize(),
		wo: sampleWo.xyz.normalize(),
		materialUv: createMaterialUvNode( sampleUv, sampleTangent, sampleBitangent, tileSize )
	};

}

function uploadAtlasSamples( sampleTextures, samples, atlasColumns, atlasHeight, tileSize, uvGradientScale = 1 / 1024 ) {

	const capacity = atlasColumns * ( atlasHeight / tileSize );
	const textureData = {};

	for ( const [ name, texture ] of Object.entries( sampleTextures ) ) {

		texture.image.data.fill( 0 );
		textureData[ name ] = texture.image.data;

	}

	for ( let i = 0; i < Math.min( samples.length, capacity ); i ++ ) {

		const sample = samples[ i ];
		const offset = i * 4;
		const uv = sample.uv || [ 0.5, 0.5 ];
		const normal = normalize( sample.normal || [ 0, 0, 1 ] );
		const tangent = normalize( sample.tangent || [ 1, 0, 0 ] );
		const bitangent = normalize( sample.bitangent || [ 0, 1, 0 ] );
		const wi = normalize( sample.wi || [ 0, 0, 1 ] );
		const wo = normalize( sample.wo || [ 0, 0, 1 ] );
		const duvDx = sample.duvDx || [ uvGradientScale, 0 ];
		const duvDy = sample.duvDy || [ 0, uvGradientScale ];

		write4( textureData.uv, offset, uv[ 0 ], uv[ 1 ], duvDx[ 0 ], duvDx[ 1 ] );
		write4( textureData.normal, offset, normal[ 0 ], normal[ 1 ], normal[ 2 ], 1 );
		write4( textureData.tangent, offset, tangent[ 0 ], tangent[ 1 ], tangent[ 2 ], duvDy[ 0 ] );
		write4( textureData.bitangent, offset, bitangent[ 0 ], bitangent[ 1 ], bitangent[ 2 ], duvDy[ 1 ] );
		write4( textureData.wi, offset, wi[ 0 ], wi[ 1 ], wi[ 2 ], 0 );
		write4( textureData.wo, offset, wo[ 0 ], wo[ 1 ], wo[ 2 ], 0 );

	}

	for ( const texture of Object.values( sampleTextures ) ) {

		texture.needsUpdate = true;

	}

}

function write4( data, offset, x, y, z, w ) {

	data[ offset ] = x;
	data[ offset + 1 ] = y;
	data[ offset + 2 ] = z;
	data[ offset + 3 ] = w;

}

function alignTo( value, alignment ) {

	return Math.ceil( value / alignment ) * alignment;

}

export {
	computeAtlasDimensions,
	createTeacherRenderTarget,
	createSampleTexture,
	createAtlasTextures,
	createSampleCoordNode,
	createMaterialUvNode,
	sampleTextureNode,
	createAtlasShaderNodes,
	uploadAtlasSamples,
	write4,
	normalize,
	alignTo
};
