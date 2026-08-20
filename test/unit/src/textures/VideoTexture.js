import { describe, test, expect } from 'vitest';
import { VideoTexture } from '@src/textures/VideoTexture.js';
import { Texture } from '@src/textures/Texture.js';

describe( 'Textures', () => {

	describe( 'VideoTexture', () => {

		test( 'Extending', () => {

			const videoDocumentElement = {};
			const object = new VideoTexture( videoDocumentElement );
			expect( object instanceof Texture ).toBe( true );

		} );

		test( 'Instancing', () => {

			const videoDocumentElement = {};
			const object = new VideoTexture( videoDocumentElement );
			expect( object ).toBeTruthy();

		} );

		test( 'isVideoTexture', () => {

			const videoDocumentElement = {};
			const object = new VideoTexture( videoDocumentElement );
			expect( object.isVideoTexture ).toBeTruthy();

		} );

	} );

} );
