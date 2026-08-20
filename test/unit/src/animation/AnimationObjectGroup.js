import { describe, test, expect } from 'vitest';
import { AnimationObjectGroup } from '@src/animation/AnimationObjectGroup.js';

import { Object3D } from '@src/core/Object3D.js';
import { PropertyBinding } from '@src/animation/PropertyBinding.js';

describe( 'Animation', () => {

	describe( 'AnimationObjectGroup', () => {

		const ObjectA = new Object3D(),
			ObjectB = new Object3D(),
			ObjectC = new Object3D(),

			PathA = 'object.position',
			PathB = 'object.rotation',
			PathC = 'object.scale',

			ParsedPathA = PropertyBinding.parseTrackName( PathA ),
			ParsedPathB = PropertyBinding.parseTrackName( PathB ),
			ParsedPathC = PropertyBinding.parseTrackName( PathC );

		test( 'Instancing', () => {

			const groupA = new AnimationObjectGroup();
			expect( groupA instanceof AnimationObjectGroup ).toBeTruthy();

		} );

		test( 'isAnimationObjectGroup', () => {

			const object = new AnimationObjectGroup();
			expect( object.isAnimationObjectGroup ).toBeTruthy();

		} );

		test( 'smoke test', () => {

			const expectGroup = function ( testIndex, group, bindings, path, cached, roots ) {

				let pathsOk = true, nodesOk = true;
				const rootNodes = [];

				for ( let i = group.nCachedObjects_, n = bindings.length; i !== n; ++ i ) {

					if ( bindings[ i ].path !== path ) pathsOk = false;
					rootNodes.push( bindings[ i ].rootNode );

				}

				for ( let i = 0, n = roots.length; i !== n; ++ i ) {

					if ( rootNodes.indexOf( roots[ i ] ) === - 1 ) nodesOk = false;

				}

				expect( pathsOk ).toBeTruthy();
				expect( nodesOk ).toBeTruthy();
				expect( group.nCachedObjects_ === cached ).toBeTruthy();
				expect( bindings.length - group.nCachedObjects_ === roots.length ).toBeTruthy();

			};

			// initial state

			const groupA = new AnimationObjectGroup();
			expect( groupA instanceof AnimationObjectGroup ).toBeTruthy();

			const bindingsAA = groupA.subscribe_( PathA, ParsedPathA );
			expectGroup( 0, groupA, bindingsAA, PathA, 0, [] );

			const groupB = new AnimationObjectGroup( ObjectA, ObjectB );
			expect( groupB instanceof AnimationObjectGroup ).toBeTruthy();

			const bindingsBB = groupB.subscribe_( PathB, ParsedPathB );
			expectGroup( 1, groupB, bindingsBB, PathB, 0, [ ObjectA, ObjectB ] );

			// add

			groupA.add( ObjectA, ObjectB );
			expectGroup( 2, groupA, bindingsAA, PathA, 0, [ ObjectA, ObjectB ] );

			groupB.add( ObjectC );
			expectGroup( 3, groupB, bindingsBB, PathB, 0, [ ObjectA, ObjectB, ObjectC ] );

			// remove

			groupA.remove( ObjectA, ObjectC );
			expectGroup( 4, groupA, bindingsAA, PathA, 1, [ ObjectB ] );

			groupB.remove( ObjectA, ObjectB, ObjectC );
			expectGroup( 5, groupB, bindingsBB, PathB, 3, [] );

			// subscribe after re-add

			groupA.add( ObjectC );
			expectGroup( 6, groupA, bindingsAA, PathA, 1, [ ObjectB, ObjectC ] );
			const bindingsAC = groupA.subscribe_( PathC, ParsedPathC );
			expectGroup( 7, groupA, bindingsAC, PathC, 1, [ ObjectB, ObjectC ] );

			// re-add after subscribe

			const bindingsBC = groupB.subscribe_( PathC, ParsedPathC );
			groupB.add( ObjectA, ObjectB );
			expectGroup( 8, groupB, bindingsBB, PathB, 1, [ ObjectA, ObjectB ] );

			// unsubscribe

			const copyOfBindingsBC = bindingsBC.slice();
			groupB.unsubscribe_( PathC );
			groupB.add( ObjectC );
			expect( bindingsBC ).toEqual( copyOfBindingsBC );

			// uncache active

			groupB.uncache( ObjectA );
			expectGroup( 9, groupB, bindingsBB, PathB, 0, [ ObjectB, ObjectC ] );

			// uncache cached

			groupA.uncache( ObjectA );
			expectGroup( 10, groupA, bindingsAC, PathC, 0, [ ObjectB, ObjectC ] );

		} );

	} );

} );
