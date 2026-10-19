import { moveSunTransmission } from '../../../../examples/jsm/math/SunSmoothing.js';

export default QUnit.module( 'Maths', () => {

	QUnit.module( 'SunSmoothing', () => {

		QUnit.test( 'rate limit, zero target, and retargeting', assert => {

			const value = moveSunTransmission( 1, 0, 0.25, 1 );
			assert.strictEqual( value, 0.75, 'fade toward complete occlusion' );
			assert.strictEqual( moveSunTransmission( value, 1, 0.25, 0.5 ), 0.875, 'reverse immediately toward a new target' );
			assert.strictEqual( moveSunTransmission( 0.9, 1, 0.25, 1 ), 1, 'no overshoot' );
			assert.strictEqual( moveSunTransmission( 0.1, 0, 0.25, 1 ), 0, 'reach exact zero' );
			assert.strictEqual( moveSunTransmission( 0.5, 1, 0, 1 ), 0.5, 'zero rate' );
			assert.strictEqual( moveSunTransmission( 0.5, 1, 1, 0 ), 0.5, 'zero elapsed time' );

		} );

		QUnit.test( 'frame rate independence and changing targets', assert => {

			for ( const fps of [ 30, 60, 144 ] ) {

				let value = 1;
				for ( let frame = 0; frame < fps; frame ++ ) value = moveSunTransmission( value, 0, 0.25, 1 / fps );
				assert.ok( Math.abs( value - 0.75 ) < 1e-12, `${fps} fps` );

			}

			let value = 0.5;
			for ( let frame = 0; frame < 1000; frame ++ ) {

				const previous = value;
				value = moveSunTransmission( value, frame % 2, 0.25, 1 / 60 );
				assert.ok( Math.abs( value - previous ) <= 0.25 / 60 + 1e-12, 'changing targets respect the bound' );
				assert.ok( value >= 0 && value <= 1, 'bounded transmission' );

			}

		} );

	} );

} );
