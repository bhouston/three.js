import { describe, test, expect } from 'vitest';
import { WebGLExtensions } from '@src/renderers/webgl/WebGLExtensions.js';

import { CONSOLE_LEVEL } from '@test-utils/console-wrapper.js';

const WebglContextMock = function ( supportedExtensions ) {

	this.supportedExtensions = supportedExtensions || [];
	this.getExtension = function ( name ) {

		if ( this.supportedExtensions.indexOf( name ) > - 1 ) {

			return { 'name': name };

		} else {

			return null;

		}

	};

};

describe( 'Renderers', () => {

	describe( 'WebGL', () => {

		describe( 'WebGLExtensions', () => {

			test( 'Instancing', () => {

				const gl = new WebglContextMock();
				const extensions = new WebGLExtensions( gl );
				expect( typeof extensions === 'object' ).toBeTruthy();

			} );

			test( 'has', () => {

				const gl = new WebglContextMock( [ 'Extension1', 'Extension2' ] );
				const extensions = new WebGLExtensions( gl );
				expect( extensions.has( 'Extension1' ) ).toBeTruthy();
				expect( extensions.has( 'Extension2' ) ).toBeTruthy();
				expect( extensions.has( 'Extension1' ) ).toBeTruthy();
				expect( extensions.has( 'NonExistingExtension' ) ).toBeFalsy();

			} );

			test( 'get', () => {

				const gl = new WebglContextMock( [ 'Extension1', 'Extension2' ] );
				const extensions = new WebGLExtensions( gl );
				expect( extensions.get( 'Extension1' ) ).toBeTruthy();
				expect( extensions.get( 'Extension2' ) ).toBeTruthy();
				expect( extensions.get( 'Extension1' ) ).toBeTruthy();

				// suppress the following console message when testing
				// THREE.WebGLRenderer: NonExistingExtension extension not supported.

				console.level = CONSOLE_LEVEL.OFF;
				expect( extensions.get( 'NonExistingExtension' ) ).toBeFalsy();
				console.level = CONSOLE_LEVEL.DEFAULT;

			} );

		} );

	} );

} );
