import { Color } from 'three';
import { Parameters } from '../../../../examples/jsm/inspector/tabs/Parameters.js';

export default QUnit.module( 'Inspector', () => {

	QUnit.module( 'Parameters', () => {

		QUnit.test( 'URL overrides update controls and call callbacks after all values are applied', async ( assert ) => {

			const parameters = new Parameters( { urlParameters: new URLSearchParams( {
				'inspector.Settings.amount': '0.75',
				'inspector.Settings.enabled': 'false',
				'inspector.Settings.text': 'a & b = ✓',
				'inspector.Settings.empty': ''
			} ) } );
			const group = parameters.createGroup( 'Settings' );
			const object = { amount: 0, enabled: true, text: 'default', empty: 'default' };
			const calls = [];
			const amount = group.add( object, 'amount', 0, 1 ).onChange( value => calls.push( { value, ...object } ) );
			const enabled = group.add( object, 'enabled' );
			const text = group.add( object, 'text' );
			group.add( object, 'empty' );

			await new Promise( resolve => setTimeout( resolve, 20 ) );

			assert.deepEqual( object, { amount: 0.75, enabled: false, text: 'a & b = ✓', empty: '' } );
			assert.strictEqual( amount.getValue(), object.amount );
			assert.strictEqual( enabled.getValue(), object.enabled );
			assert.strictEqual( text.getValue(), object.text );
			assert.deepEqual( calls, [ { value: 0.75, ...object } ], 'Chained callback sees all overrides' );

		} );

		QUnit.test( 'group, folder and display names disambiguate uniform values', async ( assert ) => {

			const parameters = new Parameters( { urlParameters: new URLSearchParams( {
				'inspector.Settings.Surface.Roughness': '0.8',
				'inspector.Settings.Surface.value': '0.2',
				'inspector.Settings.Other.value': '0.4',
				'inspector.Lighting.value': '0.6'
			} ) } );
			const group = parameters.createGroup( 'Initial' ).name( 'Settings' );
			const surface = group.addFolder( 'Initial' ).name( 'Surface' );
			const roughness = { value: 0 };
			const metalness = { value: 0 };
			const other = { value: 0 };
			const lighting = { value: 0 };
			surface.add( roughness, 'value', 0, 1 ).name( 'Roughness' );
			surface.add( metalness, 'value', 0, 1 ).name( 'Metalness' );
			group.addFolder( 'Other' ).add( other, 'value' );
			parameters.createGroup( 'Lighting' ).add( lighting, 'value' );

			await Promise.resolve();

			assert.strictEqual( roughness.value, 0.8, 'Display name takes precedence' );
			assert.strictEqual( metalness.value, 0.2, 'Property name is a fallback' );
			assert.strictEqual( other.value, 0.4, 'Folder is part of the path' );
			assert.strictEqual( lighting.value, 0.6, 'Group is part of the path' );

		} );

		QUnit.test( 'numbers respect bounds and slider steps', async ( assert ) => {

			const parameters = new Parameters( { urlParameters: new URLSearchParams( {
				'inspector.Test.high': '100',
				'inspector.Test.low': '-100',
				'inspector.Test.slider': '0.66',
				'inspector.Test.unbounded': '-2.5e3'
			} ) } );
			const group = parameters.createGroup( 'Test' );
			const object = { high: 0, low: 0, slider: 0, unbounded: 0 };
			group.addNumber( object, 'high', - 1, 1 );
			group.addNumber( object, 'low', - 1, 1 );
			const slider = group.add( object, 'slider', 0, 1 ).step( 0.25 );
			group.add( object, 'unbounded' );

			await Promise.resolve();

			assert.deepEqual( object, { high: 1, low: - 1, slider: 0.75, unbounded: - 2500 } );
			assert.strictEqual( Number( slider.numberInput.value ), slider.getValue(), 'Slider inputs stay in sync' );

		} );

		QUnit.test( 'selects preserve option types and colors preserve Color identity', async ( assert ) => {

			const parameters = new Parameters( { urlParameters: new URLSearchParams( {
				'inspector.Test.array': '2',
				'inspector.Test.mapping': '20',
				'inspector.Test.boolean': 'false',
				'inspector.Test.color': '#ff8800',
				'inspector.Test.hex': '0x112233'
			} ) } );
			const group = parameters.createGroup( 'Test' );
			const color = new Color();
			const object = { array: 1, mapping: 10, boolean: true, color, hex: 0 };
			group.add( object, 'array', [ 1, 2, 3 ] );
			group.add( object, 'mapping', { Low: 10, High: 20 } );
			group.addSelect( object, 'boolean', [ true, false ] );
			const editor = group.addColor( object, 'color' );
			group.addColor( object, 'hex' );

			await Promise.resolve();

			assert.strictEqual( object.array, 2 );
			assert.strictEqual( object.mapping, 20 );
			assert.strictEqual( object.boolean, false );
			assert.strictEqual( object.color, color );
			assert.strictEqual( color.getHex(), 0xff8800 );
			assert.strictEqual( editor.colorInput.value, '#ff8800' );
			assert.strictEqual( object.hex, 0x112233 );

		} );

		QUnit.test( 'invalid values, unknown parameters and buttons are ignored', async ( assert ) => {

			for ( const invalid of [ '', ' ', 'NaN', 'Infinity', '-Infinity', '1bad' ] ) {

				const parameters = new Parameters( { urlParameters: new URLSearchParams( {
					'inspector.Test.number': invalid,
					'inspector.Test.boolean': 'yes',
					'inspector.Test.select': 'missing',
					'inspector.Test.color': '#bad',
					'inspector.Test.action': 'true',
					'inspector.Test.unknown': '10',
					'number': '99'
				} ) } );
				const group = parameters.createGroup( 'Test' );
				let calls = 0;
				const object = { number: 1, boolean: true, select: 'a', color: 0xffffff, action: () => calls ++ };
				const expected = { ...object };
				group.add( object, 'number' ).onChange( () => calls ++ );
				group.add( object, 'boolean' );
				group.add( object, 'select', [ 'a', 'b' ] );
				group.addColor( object, 'color' );
				group.add( object, 'action' );

				await new Promise( resolve => setTimeout( resolve, 20 ) );

				assert.deepEqual( object, expected, `Invalid number: ${invalid}` );
				assert.strictEqual( calls, 0, 'No callbacks or actions triggered' );

			}

		} );

		QUnit.test( 'controls added asynchronously also apply overrides once', async ( assert ) => {

			const parameters = new Parameters( { urlParameters: new URLSearchParams( { 'inspector.Test.value': '2' } ) } );
			const group = parameters.createGroup( 'Test' );
			await Promise.resolve();

			const object = { value: 1 };
			const editor = group.add( object, 'value' );
			await Promise.resolve();

			assert.strictEqual( object.value, 2 );
			editor.setValue( 3 );
			await Promise.resolve();
			assert.strictEqual( object.value, 3, 'Later user edits are preserved' );

		} );

		QUnit.test( 'URL overrides are disabled for other parameter tabs by default', async ( assert ) => {

			const parameters = new Parameters();
			const object = { value: 1 };
			parameters.createGroup( 'Test' ).add( object, 'value' );
			await Promise.resolve();

			assert.strictEqual( parameters.urlParameters, null );
			assert.strictEqual( object.value, 1 );

		} );

	} );

} );
