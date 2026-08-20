import { describe, expect, it } from 'vitest';
import { NeuralAppearanceNodeMaterial } from '../../../../examples/jsm/neural-appearance/NeuralAppearanceNodeMaterial.js';

// NeuralAppearanceTSL.js's shader graph used to be written directly against
// exactly 4 named latent-texel inputs (LEVEL_NAMES), so this constructor
// rejected any other level count. It now generates that many named
// `latent0..latentN-1` inputs from the model's own `levels` at construction
// time (see NeuralAppearanceTSL.js's levelNames/createEvaluateNeuralBRDFFn),
// so training/export/NeuralAppearanceLoader/rendering all correctly support
// any `levels` count - see NeuralAppearanceTSL.test.js's variable-level-count
// coverage for the shader-graph side of that. All that's left here is a
// basic non-empty-array shape check for anyone constructing
// `neuralAppearanceData` directly (bypassing the loader).
describe( 'Addons > Neural > NeuralAppearance > NeuralAppearanceNodeMaterial (latentTextures shape guard)', () => {

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

	it( 'throws a clear error when latentTextures is empty', () => {

		expect( () => new NeuralAppearanceNodeMaterial( buildFakeData( 0 ) ) ).toThrow( /latentTextures must be a non-empty array/ );

	} );

	it( 'still throws the pre-existing "Expected data from NeuralAppearanceLoader" error for non-loader input, before the new guard even runs', () => {

		expect( () => new NeuralAppearanceNodeMaterial( { latentTextures: [ 1, 2, 3, 4 ] } ) ).toThrow( /Expected data from NeuralAppearanceLoader/ );

	} );

} );
