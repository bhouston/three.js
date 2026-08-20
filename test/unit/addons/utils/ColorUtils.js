import { describe, test, expect, afterEach } from 'vitest';
import { Color, ColorManagement } from 'three';
import * as ColorUtils from '../../../../examples/jsm/utils/ColorUtils.js';

describe( 'Addons', () => {

	describe( 'Utils', () => {

		describe( 'ColorUtils', () => {

			const colorManagementEnabled = ColorManagement.enabled;

			afterEach( () => {

				ColorManagement.enabled = colorManagementEnabled;

			} );

			test( 'setKelvin', () => {

				ColorManagement.enabled = false; // TODO: Update and enable.

				const c = new Color();

				// ~1900K candle flame — warm reddish-orange, no blue
				ColorUtils.setKelvin( c, 1900 );
				expect( c.r > c.g && c.g > c.b && c.b === 0 ).toBeTruthy();

				// ~6500K daylight — roughly white, all channels near 1
				ColorUtils.setKelvin( c, 6500 );
				expect( c.r > 0.9 && c.g > 0.9 && c.b > 0.9 ).toBeTruthy();

				// clamping: below 1000K should equal 1000K
				const atMin = ColorUtils.setKelvin( new Color(), 1000 );
				ColorUtils.setKelvin( c, 500 );
				expect( c.equals( atMin ) ).toBeTruthy();

				// clamping: above 40000K should equal 40000K
				const atMax = ColorUtils.setKelvin( new Color(), 40000 );
				ColorUtils.setKelvin( c, 50000 );
				expect( c.equals( atMax ) ).toBeTruthy();

				// ~10000K cool blue sky — blue channel above red
				ColorUtils.setKelvin( c, 10000 );
				expect( c.b > c.r ).toBeTruthy();

			} );

		} );

	} );

} );
