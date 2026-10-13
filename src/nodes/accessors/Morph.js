
import { float, Fn, ivec2, int, If, uniform } from '../tsl/TSLBase.js';
import { Loop } from '../utils/LoopNode.js';
import { OnObjectUpdate } from '../utils/EventNode.js';
import { uniformTexture, textureLoad } from './TextureNode.js';
import { positionLocal, positionPrevious } from './Position.js';
import { normalLocal } from './Normal.js';
import { instanceIndex, vertexIndex } from '../core/IndexNode.js';

import { DataArrayTexture } from '../../textures/DataArrayTexture.js';
import { DataTexture } from '../../textures/DataTexture.js';
import { Vector2 } from '../../math/Vector2.js';
import { Vector4 } from '../../math/Vector4.js';
import { FloatType } from '../../constants.js';
import { uniformArray } from './UniformArrayNode.js';

const _morphTextures = /*@__PURE__*/ new WeakMap();
const _morphVec4 = /*@__PURE__*/ new Vector4();
const _morphInfluencesData = /*@__PURE__*/ new WeakMap();
const _previousMorphData = /*@__PURE__*/ new WeakMap();

function getPreviousMorphData( object, frameId ) {

	let data = _previousMorphData.get( object );
	const morphTexture = object.morphTexture;
	const hasMorphTexture = morphTexture !== undefined && morphTexture !== null;
	const influences = object.morphTargetInfluences || [];

	if ( data === undefined || data.currentInfluences.length !== influences.length || data.sourceTexture !== morphTexture ) {

		if ( data && data.texture ) data.texture.dispose();

		data = {
			frameId,
			currentInfluences: influences.slice(),
			influences: influences.slice(),
			sourceTexture: morphTexture,
			texture: null
		};

		if ( hasMorphTexture ) {

			const { data: array, width, height } = morphTexture.image;
			data.currentTextureData = array.slice();
			data.texture = new DataTexture( array.slice(), width, height, morphTexture.format, morphTexture.type );
			data.texture.needsUpdate = true;
			const previousTexture = data.texture;
			morphTexture.addEventListener( 'dispose', () => previousTexture.dispose() );

		}

		_previousMorphData.set( object, data );

	} else if ( data.frameId !== frameId ) {

		data.frameId = frameId;
		data.influences = data.currentInfluences.slice();

		if ( hasMorphTexture ) {

			data.texture.image.data.set( data.currentTextureData );
			data.texture.needsUpdate = true;

		}

	}

	data.currentInfluences = influences.slice();
	if ( hasMorphTexture ) data.currentTextureData.set( morphTexture.image.data );

	return data;

}

/**
 * TSL function that retrieves and scales the morphed attribute (position or normal) texel value.
 *
 * @param {Object} params - The parameter object.
 * @param {Node<texture>} params.bufferMap - The morph target data array texture.
 * @param {Node<float>} params.influence - The target's animation influence weight.
 * @param {number} params.stride - The vertex data stride (e.g. 1 or 2).
 * @param {Node<int>} params.width - The texture width limit.
 * @param {Node<int>} params.depth - The target layer index (morph target index).
 * @param {Node<int>} params.offset - The texture offset (e.g. 0 for position, 1 for normal).
 * @returns {Node<vec3>} The scaled morph target translation value.
 */
const getMorph = /*@__PURE__*/ Fn( ( { bufferMap, influence, stride, width, depth, offset } ) => {

	const texelIndex = int( vertexIndex ).mul( stride ).add( offset );

	const y = texelIndex.div( width );
	const x = texelIndex.sub( y.mul( width ) );

	const bufferAttrib = textureLoad( bufferMap, ivec2( x, y ) ).depth( depth ).xyz;

	return bufferAttrib.mul( influence );

} );

/**
 * Resolves or creates a compiled DataArrayTexture containing encoded vertex morph targets data for WebGL2/WebGPU.
 *
 * @param {BufferGeometry} geometry - The geometry to parse.
 * @returns {Object} The resolved morph targets texture data mapping entry.
 */
function getEntry( geometry ) {

	const hasMorphPosition = geometry.morphAttributes.position !== undefined;
	const hasMorphNormals = geometry.morphAttributes.normal !== undefined;
	const hasMorphColors = geometry.morphAttributes.color !== undefined;

	// instead of using attributes, the WebGL 2 code path encodes morph targets
	// into an array of data textures. Each layer represents a single morph target.

	const morphAttribute = geometry.morphAttributes.position || geometry.morphAttributes.normal || geometry.morphAttributes.color;
	const morphTargetsCount = ( morphAttribute !== undefined ) ? morphAttribute.length : 0;

	let entry = _morphTextures.get( geometry );

	if ( entry === undefined || entry.count !== morphTargetsCount ) {

		if ( entry !== undefined ) entry.texture.dispose();

		const morphTargets = geometry.morphAttributes.position || [];
		const morphNormals = geometry.morphAttributes.normal || [];
		const morphColors = geometry.morphAttributes.color || [];

		let vertexDataCount = 0;

		if ( hasMorphPosition === true ) vertexDataCount = 1;
		if ( hasMorphNormals === true ) vertexDataCount = 2;
		if ( hasMorphColors === true ) vertexDataCount = 3;

		let width = geometry.attributes.position.count * vertexDataCount;
		let height = 1;

		const maxTextureSize = 4096; // @TODO: Use 'capabilities.maxTextureSize'

		if ( width > maxTextureSize ) {

			height = Math.ceil( width / maxTextureSize );
			width = maxTextureSize;

		}

		const buffer = new Float32Array( width * height * 4 * morphTargetsCount );

		const bufferTexture = new DataArrayTexture( buffer, width, height, morphTargetsCount );
		bufferTexture.type = FloatType;
		bufferTexture.needsUpdate = true;

		// fill buffer

		const vertexDataStride = vertexDataCount * 4;

		for ( let i = 0; i < morphTargetsCount; i ++ ) {

			const morphTarget = morphTargets[ i ];
			const morphNormal = morphNormals[ i ];
			const morphColor = morphColors[ i ];

			const offset = width * height * 4 * i;

			for ( let j = 0; j < morphTarget.count; j ++ ) {

				const stride = j * vertexDataStride;

				if ( hasMorphPosition === true ) {

					_morphVec4.fromBufferAttribute( morphTarget, j );

					buffer[ offset + stride + 0 ] = _morphVec4.x;
					buffer[ offset + stride + 1 ] = _morphVec4.y;
					buffer[ offset + stride + 2 ] = _morphVec4.z;
					buffer[ offset + stride + 3 ] = 0;

				}

				if ( hasMorphNormals === true ) {

					_morphVec4.fromBufferAttribute( morphNormal, j );

					buffer[ offset + stride + 4 ] = _morphVec4.x;
					buffer[ offset + stride + 5 ] = _morphVec4.y;
					buffer[ offset + stride + 6 ] = _morphVec4.z;
					buffer[ offset + stride + 7 ] = 0;

				}

				if ( hasMorphColors === true ) {

					_morphVec4.fromBufferAttribute( morphColor, j );

					buffer[ offset + stride + 8 ] = _morphVec4.x;
					buffer[ offset + stride + 9 ] = _morphVec4.y;
					buffer[ offset + stride + 10 ] = _morphVec4.z;
					buffer[ offset + stride + 11 ] = ( morphColor.itemSize === 4 ) ? _morphVec4.w : 1;

				}

			}

		}

		entry = {
			count: morphTargetsCount,
			texture: bufferTexture,
			stride: vertexDataCount,
			size: new Vector2( width, height )
		};

		_morphTextures.set( geometry, entry );

		function disposeTexture() {

			bufferTexture.dispose();

			_morphTextures.delete( geometry );

			geometry.removeEventListener( 'dispose', disposeTexture );

		}

		geometry.addEventListener( 'dispose', disposeTexture );

	}

	return entry;

}

/**
 * TSL function representing the vertex shader morph targets blend setup.
 * Dynamically computes morph targets weights and updates positionLocal and normalLocal in-place.
 *
 * @tsl
 * @function
 * @param {Mesh} mesh - The mesh.
 */
export const morphReference = /*@__PURE__*/ Fn( ( [ mesh ], builder ) => {

	const { geometry } = mesh;

	const hasMorphPosition = geometry.morphAttributes.position !== undefined;
	const hasMorphNormals = geometry.hasAttribute( 'normal' ) && geometry.morphAttributes.normal !== undefined;

	const morphAttribute = geometry.morphAttributes.position || geometry.morphAttributes.normal || geometry.morphAttributes.color;
	const morphTargetsCount = ( morphAttribute !== undefined ) ? morphAttribute.length : 0;

	if ( morphTargetsCount === 0 ) return;

	// Init

	let morphInfluenceData = _morphInfluencesData.get( mesh );

	if ( morphInfluenceData === undefined || morphInfluenceData.count !== morphTargetsCount ) {

		morphInfluenceData = {
			base: uniform( 1 ),
			influences: mesh.morphTargetInfluences ? uniformArray( mesh.morphTargetInfluences, 'float' ) : null,
			count: morphTargetsCount
		};

		_morphInfluencesData.set( mesh, morphInfluenceData );

	}

	const { base, influences } = morphInfluenceData;
	const needsPreviousData = hasMorphPosition && builder.needsPreviousData();
	const hasMorphTexture = mesh.morphTexture !== null && mesh.morphTexture !== undefined;
	const previousBase = needsPreviousData && ! hasMorphTexture ? uniform( 1 ) : null;
	const previousInfluences = needsPreviousData && ! hasMorphTexture ? uniformArray( mesh.morphTargetInfluences.slice(), 'float' ) : null;
	const previousTexture = needsPreviousData && hasMorphTexture ? uniformTexture( mesh.morphTexture ) : null;

	// Shader

	const { texture: bufferMap, stride, size } = getEntry( geometry );

	const baseInfluence = hasMorphTexture ? textureLoad( mesh.morphTexture, ivec2( 0, int( instanceIndex ) ) ).r : base;
	if ( hasMorphPosition === true ) positionLocal.mulAssign( baseInfluence );
	if ( hasMorphNormals === true ) normalLocal.mulAssign( baseInfluence );

	if ( needsPreviousData ) {

		const previousBaseInfluence = hasMorphTexture ? textureLoad( previousTexture, ivec2( 0, int( instanceIndex ) ) ).r : previousBase;
		positionPrevious.mulAssign( previousBaseInfluence );

	}

	const width = int( size.width );

	Loop( morphTargetsCount, ( { i } ) => {

		const influence = float( 0 ).toVar();

		if ( hasMorphTexture ) {

			influence.assign( textureLoad( mesh.morphTexture, ivec2( int( i ).add( 1 ), int( instanceIndex ) ) ).r );

		} else {

			influence.assign( influences.element( i ).toVar() );

		}

		If( influence.notEqual( 0 ), () => {

			if ( hasMorphPosition === true ) {

				positionLocal.addAssign( getMorph( {
					bufferMap,
					influence,
					stride,
					width,
					depth: i,
					offset: int( 0 )
				} ) );

			}

			if ( hasMorphNormals === true ) {

				normalLocal.addAssign( getMorph( {
					bufferMap,
					influence,
					stride,
					width,
					depth: i,
					offset: int( 1 )
				} ) );

			}

		} );

		if ( needsPreviousData ) {

			const previousInfluence = hasMorphTexture ? textureLoad( previousTexture, ivec2( int( i ).add( 1 ), int( instanceIndex ) ) ).r : previousInfluences.element( i );
			positionPrevious.addAssign( getMorph( {
				bufferMap,
				influence: previousInfluence,
				stride,
				width,
				depth: i,
				offset: int( 0 )
			} ) );

		}

	} );

	// Update

	OnObjectUpdate( ( { object, frameId } ) => {

		const { base, influences } = morphInfluenceData;

		if ( object.geometry.morphTargetsRelative || hasMorphTexture ) {

			base.value = 1;

		} else {

			base.value = 1 - object.morphTargetInfluences.reduce( ( a, b ) => a + b, 0 );

		}

		if ( influences ) {

			influences.array = object.morphTargetInfluences;
			influences.update();

		}

		if ( needsPreviousData ) {

			const previous = getPreviousMorphData( object, frameId );
			if ( previousInfluences ) {

				previousBase.value = object.geometry.morphTargetsRelative ? 1 : 1 - previous.influences.reduce( ( a, b ) => a + b, 0 );
				previousInfluences.array = previous.influences;
				previousInfluences.update();

			}

			if ( previousTexture ) previousTexture.value = previous.texture;

		}

	} );

}, 'void' );



