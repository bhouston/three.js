/**
 * Minimal SafeTensors loader for browser examples.
 *
 * Supports the dense tensor dtypes used by small Hugging Face GPT-2 models.
 *
 * @three_import import { SafeTensorsLoader } from 'three/addons/gpgpu/llm/SafeTensorsLoader.js';
 */
class SafeTensorsLoader {

	async load( url ) {

		const response = await fetch( url );

		if ( response.ok === false ) {

			throw new Error( `SafeTensorsLoader: Failed to load "${ url }" (${ response.status } ${ response.statusText })` );

		}

		return this.parse( await response.arrayBuffer() );

	}

	parse( buffer ) {

		return parseSafeTensors( buffer );

	}

}

const DTYPE_BYTES = {
	F32: 4,
	F16: 2,
	BF16: 2,
	I32: 4,
	I64: 8,
	U8: 1,
	BOOL: 1
};

function readHeaderLength( view ) {

	const low = view.getUint32( 0, true );
	const high = view.getUint32( 4, true );

	if ( high !== 0 ) {

		throw new Error( 'SafeTensorsLoader: Header is too large for this JavaScript implementation.' );

	}

	return low;

}

function createTensorArray( buffer, byteOffset, byteLength, dtype ) {

	const alignedBuffer = buffer.slice( byteOffset, byteOffset + byteLength );

	switch ( dtype ) {

		case 'F32':
			return new Float32Array( alignedBuffer );
		case 'F16':
		case 'BF16':
			return new Uint16Array( alignedBuffer );
		case 'I32':
			return new Int32Array( alignedBuffer );
		case 'I64':
			return new BigInt64Array( alignedBuffer );
		case 'U8':
		case 'BOOL':
			return new Uint8Array( alignedBuffer );
		default:
			throw new Error( `SafeTensorsLoader: Unsupported dtype "${ dtype }".` );

	}

}

function elementCount( shape ) {

	return shape.reduce( ( product, value ) => product * value, 1 );

}

function parseSafeTensors( buffer ) {

	const view = new DataView( buffer );
	const headerLength = readHeaderLength( view );
	const headerStart = 8;
	const headerEnd = headerStart + headerLength;
	const headerBytes = new Uint8Array( buffer, headerStart, headerLength );
	const header = JSON.parse( new TextDecoder().decode( headerBytes ) );
	const dataStart = headerEnd;
	const tensors = {};

	for ( const name in header ) {

		if ( name === '__metadata__' ) continue;

		const descriptor = header[ name ];
		const { dtype, shape, data_offsets: dataOffsets } = descriptor;
		const bytesPerElement = DTYPE_BYTES[ dtype ];

		if ( bytesPerElement === undefined ) {

			throw new Error( `SafeTensorsLoader: Unsupported dtype "${ dtype }" for tensor "${ name }".` );

		}

		const [ begin, end ] = dataOffsets;
		const byteLength = end - begin;
		const expectedByteLength = elementCount( shape ) * bytesPerElement;

		if ( byteLength !== expectedByteLength ) {

			throw new Error( `SafeTensorsLoader: Tensor "${ name }" has ${ byteLength } bytes, expected ${ expectedByteLength }.` );

		}

		tensors[ name ] = {
			name,
			dtype,
			shape,
			data: createTensorArray( buffer, dataStart + begin, byteLength, dtype )
		};

	}

	return {
		metadata: header.__metadata__ || {},
		tensors
	};

}

export { SafeTensorsLoader, parseSafeTensors };
