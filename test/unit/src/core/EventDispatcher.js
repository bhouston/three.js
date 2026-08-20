import { describe, test, expect } from 'vitest';

import { EventDispatcher } from '@src/core/EventDispatcher.js';

describe( 'Core', () => {

	describe( 'EventDispatcher', () => {

		test( 'Instancing', () => {

			const object = new EventDispatcher();
			expect( object ).toBeTruthy();

		} );

		test( 'addEventListener', () => {

			const eventDispatcher = new EventDispatcher();

			const listener = {};
			eventDispatcher.addEventListener( 'anyType', listener );

			expect( eventDispatcher._listeners.anyType.length === 1 ).toBeTruthy();
			expect( eventDispatcher._listeners.anyType[ 0 ] === listener ).toBeTruthy();

			eventDispatcher.addEventListener( 'anyType', listener );

			expect( eventDispatcher._listeners.anyType.length === 1 ).toBeTruthy();
			expect( eventDispatcher._listeners.anyType[ 0 ] === listener ).toBeTruthy();

		} );

		test( 'hasEventListener', () => {

			const eventDispatcher = new EventDispatcher();

			const listener = {};
			eventDispatcher.addEventListener( 'anyType', listener );

			expect( eventDispatcher.hasEventListener( 'anyType', listener ) ).toBeTruthy();
			expect( ! eventDispatcher.hasEventListener( 'anotherType', listener ) ).toBeTruthy();

		} );

		test( 'removeEventListener', () => {

			const eventDispatcher = new EventDispatcher();

			const listener = {};

			expect( eventDispatcher._listeners === undefined ).toBeTruthy();

			eventDispatcher.addEventListener( 'anyType', listener );
			expect( Object.keys( eventDispatcher._listeners ).length === 1 &&
				eventDispatcher._listeners.anyType.length === 1 ).toBeTruthy();

			eventDispatcher.removeEventListener( 'anyType', listener );
			expect( eventDispatcher._listeners.anyType.length === 0 ).toBeTruthy();

			eventDispatcher.removeEventListener( 'unknownType', listener );
			expect( eventDispatcher._listeners.unknownType === undefined ).toBeTruthy();

			eventDispatcher.removeEventListener( 'anyType', undefined );
			expect( eventDispatcher._listeners.anyType.length === 0 ).toBeTruthy();

		} );

		test( 'dispatchEvent', () => {

			const eventDispatcher = new EventDispatcher();

			let callCount = 0;
			const listener = function () {

				callCount ++;

			};

			eventDispatcher.addEventListener( 'anyType', listener );
			expect( callCount === 0 ).toBeTruthy();

			eventDispatcher.dispatchEvent( { type: 'anyType' } );
			expect( callCount === 1 ).toBeTruthy();

			eventDispatcher.dispatchEvent( { type: 'anyType' } );
			expect( callCount === 2 ).toBeTruthy();

		} );

	} );

} );
