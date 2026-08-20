import { describe, test, expect } from 'vitest';
import { Object3D } from '@src/core/Object3D.js';
import { Raycaster } from '@src/core/Raycaster.js';
import { LOD } from '@src/objects/LOD.js';

describe( 'Objects', () => {

	describe( 'LOD', () => {

		test( 'Extending', () => {

			const lod = new LOD();

			expect( lod instanceof Object3D ).toBe( true );

		} );

		test( 'type', () => {

			const object = new LOD();
			expect( object.type === 'LOD' ).toBeTruthy();

		} );

		test( 'levels', () => {

			const lod = new LOD();
			const levels = lod.levels;

			expect( Array.isArray( levels ) ).toBe( true );
			expect( levels.length ).toBe( 0 );

		} );

		test( 'autoUpdate', () => {

			const lod = new LOD();

			expect( lod.autoUpdate ).toBe( true );

		} );

		test( 'isLOD', () => {

			const lod = new LOD();

			expect( lod.isLOD ).toBe( true );

		} );

		test( 'copy', () => {

			const lod1 = new LOD();
			const lod2 = new LOD();

			const high = new Object3D();
			const mid = new Object3D();
			const low = new Object3D();

			lod1.addLevel( high, 5 );
			lod1.addLevel( mid, 25 );
			lod1.addLevel( low, 50 );

			lod1.autoUpdate = false;

			lod2.copy( lod1 );

			expect( lod2.autoUpdate ).toBe( false );
			expect( lod2.levels.length ).toBe( 3 );

		} );

		test( 'addLevel', () => {

			const lod = new LOD();

			const high = new Object3D();
			const mid = new Object3D();
			const low = new Object3D();

			lod.addLevel( high, 5, 0.00 );
			lod.addLevel( mid, 25, 0.05 );
			lod.addLevel( low, 50, 0.10 );

			expect( lod.levels.length ).toBe( 3 );
			expect( lod.levels[ 0 ] ).toEqual( { distance: 5, object: high, hysteresis: 0.00 } );
			expect( lod.levels[ 1 ] ).toEqual( { distance: 25, object: mid, hysteresis: 0.05 } );
			expect( lod.levels[ 2 ] ).toEqual( { distance: 50, object: low, hysteresis: 0.10 } );

		} );

		test( 'getObjectForDistance', () => {

			const lod = new LOD();

			const high = new Object3D();
			const mid = new Object3D();
			const low = new Object3D();

			expect( lod.getObjectForDistance( 5 ) ).toBe( null );

			lod.addLevel( high, 5 );

			expect( lod.getObjectForDistance( 0 ) ).toBe( high );
			expect( lod.getObjectForDistance( 10 ) ).toBe( high );

			lod.addLevel( mid, 25 );
			lod.addLevel( low, 50 );

			expect( lod.getObjectForDistance( 0 ) ).toBe( high );
			expect( lod.getObjectForDistance( 10 ) ).toBe( high );
			expect( lod.getObjectForDistance( 25 ) ).toBe( mid );
			expect( lod.getObjectForDistance( 50 ) ).toBe( low );
			expect( lod.getObjectForDistance( 60 ) ).toBe( low );

		} );

		test( 'raycast', () => {

			const lod = new LOD();
			const raycaster = new Raycaster();
			const intersections = [];

			lod.raycast( raycaster, intersections );

			expect( intersections.length ).toBe( 0 );

		} );

	} );

} );
