import { describe, test, expect } from 'vitest';
import { HDRLoader } from '../../../../examples/jsm/loaders/HDRLoader.js';

describe( 'Addons', () => {

	describe( 'Loaders', () => {

		describe( 'HDRLoader', () => {

			test( 'Instancing', () => {

				const loader = new HDRLoader();

				expect( loader instanceof HDRLoader ).toBeTruthy();

			} );

			test( 'parses valid HDR with large header (> chunk size)', () => {

				// Regression: fgets uses chunkSize=128. When a line (e.g. pcomb) exceeds
				// 128 bytes, chunking can skip the FORMAT line, causing "missing format
				// specifier". This minimal synthetic HDR reproduces the bug.
				const header = [
					'#?RADIANCE',
					'some large header line' + 'x'.repeat( 128 ),
					'FORMAT=32-bit_rle_rgbe',
					'-Y 0 +X 0',
					''
				].join( '\n' );
				const encoder = new TextEncoder();
				const headerBytes = encoder.encode( header );

				const buffer = new Uint8Array( headerBytes.length );
				buffer.set( headerBytes );

				const loader = new HDRLoader();
				const result = loader.parse( buffer.buffer );

				expect( result ).toBeTruthy();
				expect( result.width ).toBe( 0 );
				expect( result.height ).toBe( 0 );
				expect( result.data ).toBeTruthy();

			} );

		} );

	} );

} );
