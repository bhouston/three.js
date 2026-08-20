import { describe, expect, it } from 'vitest';
import {
	BASE_RESOLUTION,
	CHANNELS_PER_LEVEL,
	DECODER_INPUT_SIZE,
	DEFAULT_WRAP,
	FORMAT,
	IBL_INPUT_SIZE,
	IBL_OUTPUT_SIZE,
	IBL_TARGET_SIZE,
	INDIRECT_INPUT_SIZE,
	INDIRECT_OUTPUT_SIZE,
	LATENT_CHANNELS,
	LEVELS,
	ROTATION_OUTPUT_SIZE,
	TARGET_RESOLUTION,
	VERSION
} from '../../../../examples/jsm/neural-appearance/NeuralAppearanceFormat.js';

describe( 'Addons > Neural > Neural-Appearance > NeuralAppearanceFormat', () => {

	it( 'FORMAT is the exact manifest format identifier string', () => {

		expect( FORMAT ).toBe( 'three-neural-appearance' );

	} );

	it( 'VERSION is a positive integer', () => {

		expect( Number.isInteger( VERSION ) ).toBe( true );
		expect( VERSION ).toBeGreaterThan( 0 );

	} );

	it( 'grid geometry constants are positive integers', () => {

		for ( const value of [ LEVELS, BASE_RESOLUTION, TARGET_RESOLUTION, CHANNELS_PER_LEVEL ] ) {

			expect( Number.isInteger( value ) ).toBe( true );
			expect( value ).toBeGreaterThan( 0 );

		}

	} );

	it( 'LATENT_CHANNELS === LEVELS * CHANNELS_PER_LEVEL', () => {

		expect( LATENT_CHANNELS ).toBe( LEVELS * CHANNELS_PER_LEVEL );

	} );

	it( 'DECODER_INPUT_SIZE === LATENT_CHANNELS + 12', () => {

		expect( DECODER_INPUT_SIZE ).toBe( LATENT_CHANNELS + 12 );

	} );

	it( 'IBL_INPUT_SIZE === LATENT_CHANNELS + 6', () => {

		expect( IBL_INPUT_SIZE ).toBe( LATENT_CHANNELS + 6 );

	} );

	it( 'INDIRECT_INPUT_SIZE === LATENT_CHANNELS + 3 + 3', () => {

		expect( INDIRECT_INPUT_SIZE ).toBe( LATENT_CHANNELS + 3 + 3 );

	} );

	it( 'IBL_OUTPUT_SIZE, INDIRECT_OUTPUT_SIZE, IBL_TARGET_SIZE and ROTATION_OUTPUT_SIZE hold their documented fixed values', () => {

		expect( IBL_OUTPUT_SIZE ).toBe( 4 );
		expect( INDIRECT_OUTPUT_SIZE ).toBe( 3 );
		expect( IBL_TARGET_SIZE ).toBe( 17 );
		expect( ROTATION_OUTPUT_SIZE ).toBe( 12 );

	} );

	it( 'DEFAULT_WRAP is the exact default texture wrap mode string', () => {

		expect( DEFAULT_WRAP ).toBe( 'repeat' );

	} );

	it( 'all *_SIZE constants are positive integers, keeping flat-array offset math well-defined', () => {

		const sizes = [
			DECODER_INPUT_SIZE,
			IBL_INPUT_SIZE,
			IBL_OUTPUT_SIZE,
			INDIRECT_INPUT_SIZE,
			INDIRECT_OUTPUT_SIZE,
			IBL_TARGET_SIZE,
			ROTATION_OUTPUT_SIZE
		];

		for ( const size of sizes ) {

			expect( Number.isInteger( size ) ).toBe( true );
			expect( size ).toBeGreaterThan( 0 );

		}

	} );

} );
