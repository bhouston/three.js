import { Color } from 'three';
import { uniform } from 'three/tsl';
import { Parameters } from '../../../../examples/jsm/inspector/tabs/Parameters.js';

export default QUnit.module( 'Inspector', () => {

	QUnit.module( 'Parameters', () => {

		QUnit.test( 'URL overrides update controls and call callbacks after all values are applied', async ( assert ) => {

			const parameters = new Parameters( { urlParameters: new URLSearchParams( {
				'amount': '0.75',
				'enabled': 'false',
				'text': 'a & b = ✓',
				'empty': ''
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

		QUnit.test( 'group paths use original names and properties, ignoring display names', async ( assert ) => {

			const parameters = new Parameters( { urlParameters: new URLSearchParams( {
				roughness: '0.2',
				'material.roughness': '0.4',
				'material.surface.roughness': '0.8',
				'Material.Surface.roughness': '0.9',
				Roughness: '0.1',
				Metalness: '0.9'
			} ) } );
			const group = parameters.createGroup( 'material' ).name( 'Material' );
			const surface = group.addFolder( 'surface' ).name( 'Surface' );
			const object = { roughness: 0, metalness: 0 };
			const nested = { roughness: 0 };
			const other = { roughness: 0 };
			group.add( object, 'roughness', 0, 1 ).name( 'Roughness' );
			group.add( object, 'metalness', 0, 1 ).name( 'Metalness' );
			surface.add( nested, 'roughness', 0, 1 ).name( 'Roughness' );
			parameters.createGroup( 'other' ).add( other, 'roughness', 0, 1 ).name( 'Roughness' );

			await Promise.resolve();

			assert.strictEqual( object.roughness, 0.4, 'Group path takes precedence' );
			assert.strictEqual( nested.roughness, 0.8, 'Folder path uses original names' );
			assert.strictEqual( other.roughness, 0, 'Ambiguous unqualified property is ignored' );
			assert.strictEqual( object.metalness, 0, 'Display names are ignored' );

		} );

		QUnit.test( 'uniform values use explicit uniform names', async ( assert ) => {

			const parameters = new Parameters( { urlParameters: new URLSearchParams( {
				'water.waveHeight': '0.5',
				waveHeight: '0.2',
				waveSpeed: '0.8',
				value: '0.3',
				'Wave Height': '0.9'
			} ) } );
			const group = parameters.createGroup( 'water' );
			const height = uniform( 0 );
			const speed = uniform( 0 ).setName( 'waveSpeed' );
			const unnamed = uniform( 0 );
			group.add( height, 'value', 0, 1 ).name( 'Wave Height' );
			group.add( speed, 'value', 0, 1 ).name( 'Wave Speed' );
			group.add( unnamed, 'value', 0, 1 );
			height.setName( 'waveHeight' );

			await Promise.resolve();

			assert.strictEqual( height.value, 0.5, 'Group path takes precedence' );
			assert.strictEqual( speed.value, 0.8, 'Named uniforms are independent' );
			assert.strictEqual( unnamed.value, 0.3, 'Unnamed uniforms use the property' );

		} );

		QUnit.test( 'ambiguous qualified names do not override multiple controls', async ( assert ) => {

			const parameters = new Parameters( { urlParameters: new URLSearchParams( { 'water.value': '0.5' } ) } );
			const group = parameters.createGroup( 'water' );
			const height = uniform( 0.1 );
			const speed = uniform( 0.2 );
			group.add( height, 'value', 0, 1 );
			group.add( speed, 'value', 0, 1 );

			await Promise.resolve();

			assert.strictEqual( height.value, 0.1 );
			assert.strictEqual( speed.value, 0.2 );

		} );

		QUnit.test( 'a URL key cannot be reused by a control registered later', async ( assert ) => {

			const parameters = new Parameters( { urlParameters: new URLSearchParams( { amount: '0.5' } ) } );
			const first = { amount: 0.1 };
			const second = { amount: 0.2 };
			parameters.createGroup( 'first' ).add( first, 'amount' );
			await Promise.resolve();
			parameters.createGroup( 'second' ).add( second, 'amount' );
			await Promise.resolve();

			assert.strictEqual( first.amount, 0.5 );
			assert.strictEqual( second.amount, 0.2 );

		} );

		QUnit.test( 'numbers respect bounds and slider steps', async ( assert ) => {

			const parameters = new Parameters( { urlParameters: new URLSearchParams( {
				'high': '100',
				'low': '-100',
				'slider': '0.66',
				'unbounded': '-2.5e3'
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
				'array': '2',
				'mapping': '20',
				'boolean': 'false',
				'color': '#ff8800',
				'hex': '0x112233'
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

		QUnit.test( 'unknown parameters and buttons are ignored', async ( assert ) => {

			const parameters = new Parameters( { urlParameters: new URLSearchParams( {
				'action': 'true',
				'unknown': '10'
			} ) } );
			const group = parameters.createGroup( 'Test' );
			let calls = 0;
			const object = { number: 1, action: () => calls ++ };
			group.add( object, 'number' ).onChange( () => calls ++ );
			group.add( object, 'action' );

			await new Promise( resolve => setTimeout( resolve, 20 ) );

			assert.strictEqual( object.number, 1 );
			assert.notOk( Object.hasOwn( object, 'unknown' ) );
			assert.strictEqual( calls, 0 );

		} );

		QUnit.test( 'controls added asynchronously also apply overrides once', async ( assert ) => {

			const parameters = new Parameters( { urlParameters: new URLSearchParams( { 'value': '2' } ) } );
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

		QUnit.test( 'share URLs use qualified keys and omit defaults', async ( assert ) => {

			const parameters = new Parameters( { urlParameters: new URLSearchParams( { amount: '0.75' } ) } );
			const object = { amount: 0.25, enabled: true, text: '', untouched: 10 };
			const group = parameters.createGroup( 'settings' ).name( 'Settings' );
			const amount = group.add( object, 'amount', 0, 1 ).name( 'Amount' );
			group.add( object, 'enabled' );
			group.add( object, 'text' );
			group.add( object, 'untouched' );
			await Promise.resolve();

			object.enabled = false;
			object.text = 'a & b = ✓';
			const base = 'https://example.com/demo?amount=0.75&settings.untouched=99&mode=webgpu#view';
			const url = new URL( parameters._getShareURL( new URL( base ) ) );

			assert.strictEqual( url.searchParams.get( 'settings.amount' ), '0.75', 'URL override differs from the original default' );
			assert.strictEqual( url.searchParams.get( 'settings.enabled' ), 'false' );
			assert.strictEqual( url.searchParams.get( 'settings.text' ), object.text );
			assert.notOk( url.searchParams.has( 'settings.untouched' ), 'Default removes a stale override' );
			assert.notOk( url.searchParams.has( 'amount' ), 'Unqualified aliases are replaced' );
			assert.strictEqual( url.searchParams.get( 'mode' ), 'webgpu' );
			assert.strictEqual( url.hash, '#view' );

			const restored = new Parameters( { urlParameters: url.searchParams } );
			const restoredObject = { amount: 0.25, enabled: true, text: '', untouched: 10 };
			const restoredGroup = restored.createGroup( 'settings' );
			for ( const property of Object.keys( restoredObject ) ) restoredGroup.add( restoredObject, property );
			await Promise.resolve();
			assert.deepEqual( restoredObject, object, 'Shared URL restores the same parameter values' );

			amount.setValue( 0.25 );
			assert.notOk( new URL( parameters._getShareURL( new URL( base ) ) ).searchParams.has( 'settings.amount' ), 'Reset values are omitted' );

		} );

		QUnit.test( 'share URLs snapshot colors and use uniform names in nested groups', async ( assert ) => {

			const parameters = new Parameters();
			const group = parameters.createGroup( 'water' ).addFolder( 'waves' );
			const height = uniform( 0.1 ).setName( 'waveHeight' );
			const object = { color: new Color( 0xffffff ), hex: '#FFFFFF' };
			group.add( height, 'value', 0, 1 );
			group.addColor( object, 'color' );
			group.addColor( object, 'hex' );
			height.value = 0.5;
			object.color.setHex( 0xff8800 );
			object.hex = 0xffffff;

			const url = new URL( parameters._getShareURL( new URL( 'https://example.com/demo' ) ) );

			assert.strictEqual( url.searchParams.get( 'water.waves.waveHeight' ), '0.5' );
			assert.strictEqual( url.searchParams.get( 'water.waves.color' ), '#ff8800' );
			assert.notOk( url.searchParams.has( 'water.waves.hex' ), 'Equivalent color representations use the same default' );

			object.color.setHex( 0xffffff );
			assert.notOk( new URL( parameters._getShareURL( new URL( 'https://example.com/demo' ) ) ).searchParams.has( 'water.waves.color' ) );

		} );

		QUnit.test( 'share URLs index repeated paths and restore controls independently', async ( assert ) => {

			const parameters = new Parameters();
			const group = parameters.createGroup( 'water' );
			const height = uniform( 0.1 );
			const speed = uniform( 0.2 );
			group.add( height, 'value', 0, 1 );
			group.add( speed, 'value', 0, 1 );
			height.value = 0.5;
			speed.value = 0.8;

			const base = 'https://example.com/demo?water.value=99&water.value[0]=99&mode=test';
			const url = new URL( parameters._getShareURL( new URL( base ) ) );
			assert.strictEqual( url.searchParams.get( 'water.value[0]' ), '0.5' );
			assert.strictEqual( url.searchParams.get( 'water.value[1]' ), '0.8' );
			assert.notOk( url.searchParams.has( 'water.value' ), 'Unindexed aliases are removed' );
			assert.strictEqual( url.searchParams.get( 'mode' ), 'test' );

			const restored = new Parameters( { urlParameters: url.searchParams } );
			const restoredGroup = restored.createGroup( 'water' );
			const restoredHeight = uniform( 0.1 );
			const restoredSpeed = uniform( 0.2 );
			restoredGroup.add( restoredHeight, 'value', 0, 1 );
			restoredGroup.add( restoredSpeed, 'value', 0, 1 );
			await Promise.resolve();

			assert.strictEqual( restoredHeight.value, 0.5 );
			assert.strictEqual( restoredSpeed.value, 0.8 );

			height.value = 0.1;
			const reset = new URL( parameters._getShareURL( new URL( base ) ) );
			assert.notOk( reset.searchParams.has( 'water.value[0]' ), 'Default controls are omitted without renumbering' );
			assert.strictEqual( reset.searchParams.get( 'water.value[1]' ), '0.8' );

			const sparse = new Parameters( { urlParameters: reset.searchParams } );
			const sparseGroup = sparse.createGroup( 'water' );
			const sparseHeight = uniform( 0.1 );
			const sparseSpeed = uniform( 0.2 );
			sparseGroup.add( sparseHeight, 'value', 0, 1 );
			sparseGroup.add( sparseSpeed, 'value', 0, 1 );
			await Promise.resolve();
			assert.strictEqual( sparseHeight.value, 0.1 );
			assert.strictEqual( sparseSpeed.value, 0.8, 'Sparse indices restore the intended control' );

		} );

		QUnit.test( 'indexed keys work across asynchronous registration', async ( assert ) => {

			const parameters = new Parameters( { urlParameters: new URLSearchParams( {
				'water.value[0]': '0.5',
				'water.value[1]': '0.8',
				'other.value[0]': '0.9'
			} ) } );
			const group = parameters.createGroup( 'water' );
			const height = uniform( 0.1 );
			const speed = uniform( 0.2 );
			const other = uniform( 0.3 );
			group.add( height, 'value', 0, 1 );
			await Promise.resolve();
			group.add( speed, 'value', 0, 1 );
			parameters.createGroup( 'other' ).add( other, 'value', 0, 1 );
			await Promise.resolve();

			assert.strictEqual( height.value, 0.5 );
			assert.strictEqual( speed.value, 0.8 );
			assert.strictEqual( other.value, 0.9, 'Indices are local to the matching path' );

		} );

		QUnit.test( 'URL overrides are disabled for other parameter tabs by default', async ( assert ) => {

			const parameters = new Parameters();
			const object = { value: 1 };
			parameters.createGroup( 'Test' ).add( object, 'value' );
			await Promise.resolve();

			assert.strictEqual( parameters._urlParameters, null );
			assert.strictEqual( object.value, 1 );

		} );

	} );

} );
