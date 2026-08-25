import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { uv } from 'three/tsl';
import { NTCNodeMaterial } from '../../../../examples/jsm/ntc/NTCNodeMaterial.js';
import { classifyMaterialChannels } from '../../../../examples/jsm/ntc/NTCSource.js';
import { createTestRenderer } from '../helpers/webgpuEval.js';

// End-to-end coverage that the trainer actually fits the channels added on
// top of the original CHANNELS vocabulary (iridescence, iridescenceIOR,
// iridescenceThickness, ior, thickness, attenuationColor,
// attenuationDistance, dispersion, retroreflectivity, specularColor) - not
// just that they're *classified*/*applied* correctly (see
// NeuralMaterialCustomChannels.test.js / NeuralMaterialConstantChannels.
// render.test.js), but that a real `NTCTrainer` run against a
// material driving every one of them with a spatially-varying node actually
// converges: finite, decreasing loss, exactly like the pre-existing channels
// this trainer was already known to handle.
//
// This is the only test in the suite that drives `NTCNodeMaterial
// .fit()` - the full classify -> bake -> train -> reconstruct pipeline - for
// these specific channels, so it's what actually answers "is the trainer set
// up to train these new channels", as opposed to inferring it from the
// generic (channel-agnostic) shape of NTCSource.js/
// NTCTrainer.js.
describe( 'Addons > Neural > NeuralMaterial > new channels train end-to-end (real WebGPU)', () => {

	let renderer;

	beforeAll( async () => {

		renderer = await createTestRenderer();

	} );

	afterAll( () => {

		renderer?.dispose();
		renderer = undefined;

	} );

	// Every added channel driven by a distinct, simple UV-based procedural
	// node, so each one is genuinely spatially-varying (and therefore
	// "active"/trained, not folded into constantValues) - deliberately not
	// using the exact same expression for every channel, so a bug that only
	// shows up for e.g. a 'softplus'-activated channel bleeding into a
	// 'sigmoid' one's packed slot wouldn't be masked by every channel having
	// an identical (and therefore accidentally-consistent) target.
	function buildSourceMaterial() {

		const material = new THREE.MeshPhysicalNodeMaterial();

		material.iridescenceNode = uv().x;
		material.iridescenceIORNode = uv().y.mul( 0.5 ).add( 1.3 ); // ~[1.3, 1.8]
		material.iridescenceThicknessNode = uv().x.mul( 300 ).add( 100 ); // ~[100, 400] nm
		material.iorNode = uv().y.mul( 0.4 ).add( 1.4 ); // ~[1.4, 1.8]
		material.thicknessNode = uv().x.mul( 2.0 ); // ~[0, 2] scene units
		material.attenuationColorNode = uv().x.mul( 0.5 ).add( 0.5 ); // ~[0.5, 1]
		material.attenuationDistanceNode = uv().y.mul( 5.0 ).add( 1.0 ); // ~[1, 6] - a real, finite, user-authored value, not the Infinity-fallback sentinel
		material.dispersionNode = uv().x.mul( 0.3 );
		material.retroreflectivityNode = uv().y.mul( 0.2 );
		material.specularColorNode = uv().y.mul( 0.5 ).add( 0.5 );

		return material;

	}

	it( 'fits every new channel jointly with finite, sharply-decreasing loss', async () => {

		const material = buildSourceMaterial();
		const channelClassification = classifyMaterialChannels( material );

		const activeKeys = channelClassification.activeChannels.map( ( c ) => c.key );
		for ( const key of [
			'iridescence', 'iridescenceIOR', 'iridescenceThickness', 'ior', 'thickness',
			'attenuationColor', 'attenuationDistance', 'dispersion', 'retroreflectivity', 'specularColor'
		] ) {

			expect( activeKeys, `${key} should classify as active (node-driven)` ).toContain( key );

		}

		const lossHistory = [];

		const { material: neuralMaterial, loss } = await NTCNodeMaterial.fit( renderer, material, {
			resolution: 64,
			levels: 2,
			baseResolution: 8,
			growthFactor: 2,
			hiddenSizes: [ 16, 16 ],
			batchSize: 2048,
			iterations: 300,
			learningRate: 0.02,
			seed: 1,
			onProgress: ( progress ) => lossHistory.push( progress.loss )
		} );

		expect( Number.isFinite( loss ) ).toBe( true );
		expect( lossHistory.length ).toBeGreaterThan( 0 );
		expect( lossHistory.every( Number.isFinite ) ).toBe( true );

		// Sharp early-training drop, matching the shape every other channel
		// combination in this codebase's convergence tests exhibits (see
		// NTCTrainer.convergence.test.js) - not a strict "monotonic
		// decrease" (stochastic minibatches don't guarantee that), just that
		// it actually learned something rather than staying flat/diverging.
		expect( loss ).toBeLessThan( lossHistory[ 0 ] * 0.5 );

		neuralMaterial.dispose();

	}, 60000 );

} );
