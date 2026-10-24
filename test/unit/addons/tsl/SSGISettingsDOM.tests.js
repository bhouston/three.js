import { ValueSlider } from '../../../../examples/jsm/inspector/ui/Values.js';

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'TSL', () => {

		QUnit.module( 'SSGISettingsDOM', () => {

			QUnit.test( 'explicit slider steps preserve scene settings during listen updates', assert => {

				// Inspector listen() calls setValue(), whose dispatched change writes
				// the HTML range value back to the settings object. The range's default
				// step 0.01 would quantize small distances around its minimum.
				for ( const [ property, min, max, step, values ] of [
					[ 'thickness', 0.0001, 0.05, 0.0001, [ 0.0025, 0.005, 0.01 ]],
					[ 'radius', 0.01, 1, 0.001, [ 0.1, 0.15, 0.25 ]]
				] ) {

					const settings = { [ property ]: values[ 0 ] };
					const slider = new ValueSlider( { value: settings[ property ], min, max, step } );
					slider.addEventListener( 'change', event => {

						settings[ property ] = event.value;

					} );
					for ( const value of values ) {

						settings[ property ] = value;
						slider.setValue( settings[ property ] );
						assert.ok( Math.abs( settings[ property ] - value ) < 1e-12, `${ property } ${ value } survives the real range input and change event` );

					}

				}

				const legacy = new ValueSlider( { value: 0.005, min: 0.0001, max: 0.05 } );
				assert.notStrictEqual( legacy.getValue(), 0.005, 'Negative control: the implicit step does not preserve the thinner candidate' );

			} );

		} );

	} );

} );
