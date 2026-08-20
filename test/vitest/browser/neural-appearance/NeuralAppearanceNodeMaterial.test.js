import { describe, expect, it } from 'vitest';
import { NeuralAppearanceNodeMaterial } from '../../../../examples/jsm/neural-appearance/NeuralAppearanceNodeMaterial.js';

// Documents a real, separate limitation surfaced while fixing the "grid
// levels" training bug (see NeuralAppearanceModel.levels-bug.test.js):
// NeuralAppearanceTSL.js's shader graph is written directly against exactly
// 4 named latent-texel inputs (LEVEL_NAMES) for TSL graph-shape reasons, not
// a variable-length loop over however many grid levels a model actually has.
// Training/export now correctly supports any `levels` count, but *rendering*
// still doesn't - NeuralAppearanceLoader.parse() already throws a clear
// error for a manifest with `latents.levels.length !== 4` (see
// NeuralAppearanceLoader.js's own validateManifest check), so this is the
// second layer of the same guard for anyone constructing
// `neuralAppearanceData` directly (bypassing the loader) instead of failing
// silently deep inside the shader graph.
describe( 'Addons > Neural > NeuralAppearance > NeuralAppearanceNodeMaterial (grid-level-count guard)', () => {

	function buildFakeData( latentTextureCount ) {

		return {
			isNeuralAppearanceData: true,
			name: 'test',
			latentTextures: new Array( latentTextureCount ).fill( null ),
			levels: latentTextureCount,
			wrap: 'repeat',
			outputs: {},
			referenceEvaluations: []
		};

	}

	it( 'throws a clear error when latentTextures.length is not exactly 4', () => {

		expect( () => new NeuralAppearanceNodeMaterial( buildFakeData( 2 ) ) ).toThrow( /latent grid level/i );
		expect( () => new NeuralAppearanceNodeMaterial( buildFakeData( 8 ) ) ).toThrow( /latent grid level/i );
		expect( () => new NeuralAppearanceNodeMaterial( buildFakeData( 0 ) ) ).toThrow( /latent grid level/i );

	} );

	it( 'still throws the pre-existing "Expected data from NeuralAppearanceLoader" error for non-loader input, before the new guard even runs', () => {

		expect( () => new NeuralAppearanceNodeMaterial( { latentTextures: [ 1, 2, 3, 4 ] } ) ).toThrow( /Expected data from NeuralAppearanceLoader/ );

	} );

} );
