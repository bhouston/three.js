import { DataUtils, FloatType, HalfFloatType, UnsignedByteType } from 'three';
import { IESLoader } from '../../../../examples/jsm/loaders/IESLoader.js';

function document( options = {} ) {

	const vertical = options.vertical || [ 0, 90, 180 ];
	const horizontal = options.horizontal || [ 0 ];
	const values = options.values || horizontal.flatMap( () => [ 100, 50, 0 ] );
	const header = options.header === undefined ? 'IESNA:LM-63-2002' : options.header;
	return [
		header,
		'[TEST] Synthetic IESLoader regression fixture',
		options.tilt || 'TILT=NONE',
		`1 -1 ${ options.multiplier ?? 1 } ${ vertical.length } ${ horizontal.length } ${ options.type ?? 1 } 2 0 0 0`,
		`1 ${ options.secondFactor ?? 1 } 10`,
		vertical.join( ' ' ),
		horizontal.join( ' ' ),
		values.join( ' ' )
	].join( '\n' );

}

function parse( text, type = FloatType ) {

	const loader = new IESLoader();
	loader.type = type;
	return loader.parse( text );

}

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'Loaders', () => {

		QUnit.module( 'IESLoader', () => {

			QUnit.test( 'parses tables independently of physical line boundaries', ( assert ) => {

				const source = document( { horizontal: [ 0, 90 ], values: [ 100, 50, 0, 20, 10, 0 ] } );
				const start = source.indexOf( 'TILT=NONE' ) + 'TILT=NONE'.length;
				const body = source.slice( start ).trim().split( /\s+/ );
				// Both candela rows share one line, and other records split at arbitrary tokens.
				const wrapped = source.slice( 0, start ) + '\r\n' + body.map( ( token, i ) => token + ( i % 5 === 0 ? '\r\n\t' : ', ' ) ).join( '' );
				assert.deepEqual( parse( wrapped ).image.data, parse( source ).image.data );
				assert.deepEqual( parse( source.replaceAll( '100', '1e2' ).replaceAll( '50', '5E+1' ) ).image.data, parse( source ).image.data );
				assert.deepEqual( parse( '\uFEFF' + source.replaceAll( '\n', '\r' ) ).image.data, parse( source ).image.data );

			} );

			QUnit.test( 'zeroes unmeasured lower and upper hemisphere ranges', ( assert ) => {

				for ( const vertical of [[ 0, 90 ], [ 90, 180 ]] ) {

					const texture = parse( document( { vertical, values: [ 100, 50 ] } ) );
					for ( let i = 0; i < 180; i ++ ) {

						const angle = i * 180 / 179;
						const value = texture.image.data[ i ];
						assert.ok( Number.isFinite( value ) && value >= 0 && value <= 1 );
						if ( angle < vertical[ 0 ] || angle > vertical[ 1 ] ) assert.strictEqual( value, 0 );

					}

				}

			} );

			QUnit.test( 'preserves linear relative intensity and full-sphere endpoints', ( assert ) => {

				const texture = parse( document() );
				assert.strictEqual( texture.image.width, 180 );
				assert.strictEqual( texture.image.height, 1 );
				assert.strictEqual( texture.image.data[ 0 ], 1 );
				assert.strictEqual( texture.image.data[ 179 ], 0 );
				assert.ok( Math.abs( texture.image.data[ 90 ] - ( 1 - 90 / 179 ) ) < 1e-7 );

			} );

			for ( const header of [ '', 'IESNA91', 'IESNA:LM-63-1991', 'IESNA:LM-63-1995', 'IESNA:LM-63-2002', 'IES:LM-63-2019' ] ) {

				QUnit.test( `supports ${ header || 'headerless 1986' }`, ( assert ) => {

					assert.strictEqual( parse( document( { header } ) ).image.data[ 0 ], 1 );

				} );

			}

			QUnit.test( 'does not apply 2019 file generation metadata to candela values', ( assert ) => {

				for ( const secondFactor of [ 1.00001, 1.00010, 1.00100, 1.01000, 1.10000, 1.11100 ] ) {

					const loader = new IESLoader();
					const original = loader._getIESValues;
					loader._getIESValues = function ( lamp, type ) {

						assert.strictEqual( lamp.blpFactor, 1 );
						return original.call( this, lamp, type );

					};

					assert.deepEqual( loader.parse( document( { header: 'IES:LM-63-2019', secondFactor } ) ).image.data, new IESLoader().parse( document() ).image.data );

				}

			} );

			QUnit.test( 'does not use the 1995/2002 reserved field as a multiplier', ( assert ) => {

				// Tolerate a non-unity reserved field without changing the light distribution.
				for ( const header of [ 'IESNA:LM-63-1995', 'IESNA:LM-63-2002' ] ) {

					assert.deepEqual( parse( document( { header, secondFactor: 0 } ) ).image.data, parse( document() ).image.data );

				}

			} );

			QUnit.test( 'retains the legacy ballast-lamp factor for 1986/1991 files', ( assert ) => {

				for ( const header of [ '', 'IESNA91' ] ) {

					const texture = parse( document( { header, secondFactor: 0 } ) );
					assert.ok( Array.from( texture.image.data ).every( value => value === 0 ) );

				}

			} );

			QUnit.test( 'supports all output types and black distributions', ( assert ) => {

				for ( const type of [ FloatType, HalfFloatType, UnsignedByteType ] ) {

					const texture = parse( document( { vertical: [ 0, 90 ], values: [ 100, 50 ] } ), type );
					assert.strictEqual( texture.image.data[ 179 ], 0 );
					assert.strictEqual( texture.image.data[ 0 ], type === FloatType ? 1 : type === HalfFloatType ? DataUtils.toHalfFloat( 1 ) : 255 );
					assert.ok( Array.from( parse( document( { values: [ 0, 0, 0 ] } ), type ).image.data ).every( value => value === 0 ) );

				}

			} );

			QUnit.test( 'preserves quadrant, bilateral and wraparound angular maps', ( assert ) => {

				for ( const horizontal of [[ 0, 90 ], [ 90, 270 ], [ 0, 120, 240, 345 ], [ 0, 180, 360 ]] ) {

					const values = horizontal.flatMap( ( _, i ) => [ 100 / ( i + 1 ), 50 / ( i + 1 ), 0 ] );
					const texture = parse( document( { horizontal, values } ) );
					assert.strictEqual( texture.image.height, 360 );
					assert.strictEqual( texture.image.data.length, texture.image.width * texture.image.height );
					assert.ok( Array.from( texture.image.data ).every( value => Number.isFinite( value ) && value >= 0 && value <= 1 ) );

				}

			} );

			QUnit.test( 'consumes embedded TILT records without depending on their line breaks', ( assert ) => {

				const tilted = document( { tilt: 'TILT=INCLUDE\n1 2 0 90 1 0.5' } );
				assert.deepEqual( parse( tilted ).image.data, parse( document() ).image.data );

			} );

		} );

	} );

} );
