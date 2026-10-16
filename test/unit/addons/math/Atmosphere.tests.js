import { atmosphere, luminance, sunDirection, discVisibility, transmittance, sunLightState, skyRadiance, exposure } from '../../../../examples/jsm/math/Atmosphere.js';

export default QUnit.module( 'Maths', () => {

	QUnit.module( 'Atmosphere', () => {

		QUnit.test( 'solar disc crosses the horizon continuously', assert => {

			assert.strictEqual( discVisibility( atmosphere.sunRadius ), 1 );
			assert.strictEqual( discVisibility( 0 ), 0.5 );
			assert.strictEqual( discVisibility( - atmosphere.sunRadius ), 0 );
			let previous = 0;
			for ( let h = - 1; h <= 1; h += 0.001 ) {

				const value = discVisibility( h );
				assert.ok( value >= previous && value - previous < 0.003, 'continuous and monotonic' );
				previous = value;

			}

		} );

		QUnit.test( 'compass directions', assert => {

			const close = ( actual, expected ) => actual.every( ( value, i ) => Math.abs( value - expected[ i ] ) < 1e-12 );
			assert.ok( close( sunDirection( 0, 0 ), [ 0, 0, - 1 ] ), 'north' );
			assert.ok( close( sunDirection( 0, 90 ), [ 1, 0, 0 ] ), 'east' );
			assert.ok( close( sunDirection( 0, 180 ), [ 0, 0, 1 ] ), 'south' );
			assert.ok( close( sunDirection( 90, 0 ), [ 0, 1, 0 ] ), 'zenith' );
			assert.ok( close( sunDirection( 0, 0, 90 ), [ 1, 0, 0 ] ), 'north rotation' );

		} );

		QUnit.test( 'physical sunlight and reddening', assert => {

			const zenith = sunLightState( 90 );
			assert.ok( zenith.illuminance > 95000 && zenith.illuminance < 120000, 'zenith normal illuminance in lux' );
			let previous = 0, previousRatio = Infinity;
			for ( let h = 0; h <= 90; h ++ ) {

				const state = sunLightState( h );
				assert.ok( state.illuminance >= previous, 'brighter toward zenith' );
				const ratio = state.transmission[ 0 ] / state.transmission[ 2 ];
				assert.ok( ratio <= previousRatio, 'redder toward horizon' );
				assert.ok( Math.abs( luminance( state.color ) - 1 ) < 1e-12, 'luminance-normalized colour' );
				previous = state.illuminance;
				previousRatio = ratio;

			}

			assert.strictEqual( sunLightState( - atmosphere.sunRadius ).illuminance, 0, 'no direct sunlight below the disc' );

		} );

		QUnit.test( 'transmittance converges and planet occludes sunlight', assert => {

			const radius = atmosphere.planetRadius + atmosphere.observerAltitude;
			assert.deepEqual( transmittance( radius, - 1 ), [ 0, 0, 0 ] );
			for ( const altitude of [ 0, 1, 5, 10, 30, 60, 90 ] ) {

				const cosine = Math.sin( altitude * Math.PI / 180 );
				const actual = transmittance( radius, cosine );
				const reference = transmittance( radius, cosine, atmosphere, 512 );
				assert.ok( actual.every( ( value, c ) => Math.abs( value / reference[ c ] - 1 ) < 0.01 ), '64-step transmittance within 1% at ' + altitude );

			}

		} );

		QUnit.test( 'finite sky and exposure through twilight', assert => {

			for ( let altitude = - 20; altitude <= 90; altitude += 0.5 ) {

				for ( const azimuth of [ 0, 90, 180, 270 ] ) {

					const rgb = skyRadiance( sunDirection( 5, azimuth ), sunDirection( altitude, 0 ) );
					assert.ok( rgb.every( value => Number.isFinite( value ) && value >= 0 ), 'finite nonnegative radiance' );

				}

			}

			assert.ok( Math.abs( exposure( 100000 ).ev - 15.287712 ) < 1e-6, 'incident light meter' );
			assert.strictEqual( exposure( 0 ).ev, - 3, 'night clamp' );
			assert.strictEqual( exposure( 1e10 ).ev, 17, 'day clamp' );
			assert.strictEqual( exposure( 100000, 10 ).scale, 1 / ( 1.2 * 2 ** 10 ), 'manual pre-exposure' );

		} );

	} );

} );
