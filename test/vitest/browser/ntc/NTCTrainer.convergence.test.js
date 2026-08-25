import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { NTCTrainer } from '../../../../examples/jsm/ntc/training/NTCTrainer.js';
import { createTestRenderer } from '../helpers/webgpuEval.js';

// Convergence-rate regression coverage: trains the same synthetic target
// under a few different network/optimizer configurations and checks the
// *relative* ordering of the results (more capacity converges to a lower
// final loss; a too-low learning rate converges slower early on) rather than
// exact loss numbers, which would be too brittle across engine/driver
// changes. This exists to catch a real regression class none of the other
// tests would: a change that still trains "successfully" (finite, decreasing
// loss) but silently makes convergence *worse* - e.g. a broken learning-rate
// schedule, a grid-encoding regression that reduces effective capacity, or
// an Adam/gradient-norm change that makes optimization less efficient.
describe( 'Addons > Neural > NeuralTexture > NTCTrainer convergence rate (real WebGPU)', () => {

	let renderer;

	beforeAll( async () => {

		renderer = await createTestRenderer();

	} );

	afterAll( () => {

		renderer?.dispose();
		renderer = undefined;

	} );

	// A smooth radial gradient, not a hard-edged pattern - deliberately easy
	// to fit well so differences in final loss reflect network capacity /
	// optimizer quality rather than irreducible high-frequency detail the
	// grid can't represent at any of these small resolutions.
	function buildGradientTexture( resolution ) {

		const data = new Uint16Array( resolution * resolution * 4 );

		for ( let y = 0; y < resolution; y ++ ) {

			for ( let x = 0; x < resolution; x ++ ) {

				const u = ( x + 0.5 ) / resolution;
				const v = ( y + 0.5 ) / resolution;
				const r = u;
				const g = v;
				const b = 0.5 + 0.5 * Math.sin( 2 * Math.PI * u ) * Math.cos( 2 * Math.PI * v );
				const p = ( y * resolution + x ) * 4;

				data[ p + 0 ] = THREE.DataUtils.toHalfFloat( r );
				data[ p + 1 ] = THREE.DataUtils.toHalfFloat( g );
				data[ p + 2 ] = THREE.DataUtils.toHalfFloat( b );
				data[ p + 3 ] = THREE.DataUtils.toHalfFloat( 1 );

			}

		}

		const texture = new THREE.DataTexture( data, resolution, resolution, THREE.RGBAFormat, THREE.HalfFloatType );
		texture.wrapS = THREE.RepeatWrapping;
		texture.wrapT = THREE.RepeatWrapping;
		texture.magFilter = THREE.LinearFilter;
		texture.minFilter = THREE.LinearFilter;
		texture.generateMipmaps = false;
		texture.needsUpdate = true;

		return texture;

	}

	async function trainAndCollectLossHistory( overrides ) {

		const texture = buildGradientTexture( 32 );

		const trainer = new NTCTrainer( {
			channels: 4,
			levels: 3,
			baseResolution: 8,
			hiddenSizes: [ 16, 16 ],
			outputChannels: 3,
			batchSize: 2048,
			iterations: 200,
			learningRate: 0.02,
			seed: 1,
			// Forces maxLod to 0 (see NTCGridPyramidModel.js), so every sample's
			// LOD is always 0 - this file tests basic gradient-descent
			// convergence against a texture with no real mip chain
			// (`generateMipmaps: false` above); a genuinely sampled LOD > 0
			// would be reading an undefined mip level.
			textureResolution: 1,
			...overrides
		} );

		const lossHistory = [];

		const result = await trainer.train( {
			renderer,
			sourceTextures: [ texture ],
			onProgress: ( progress ) => lossHistory.push( { iteration: progress.iteration, loss: progress.loss } )
		} );

		texture.dispose();

		return { result, lossHistory };

	}

	it( 'a wider hidden layer converges to a final loss at least as good as a narrower one, on the same target/iterations/seed', async () => {

		const narrow = await trainAndCollectLossHistory( { hiddenSizes: [ 4, 4 ] } );
		const wide = await trainAndCollectLossHistory( { hiddenSizes: [ 32, 32 ] } );

		console.log( `[convergence] hidden [4,4] final loss=${narrow.result.loss.toExponential( 3 )}, hidden [32,32] final loss=${wide.result.loss.toExponential( 3 )}` );

		expect( Number.isFinite( narrow.result.loss ) ).toBe( true );
		expect( Number.isFinite( wide.result.loss ) ).toBe( true );

		// Not a strict inequality - both configurations may reach the same
		// near-zero floor on an easy target - but the wider network must not
		// end up *worse* by more than a small numerical-noise margin.
		expect( wide.result.loss ).toBeLessThanOrEqual( narrow.result.loss * 1.2 + 1e-6 );

	}, 60000 );

	it( 'a near-zero learning rate converges markedly slower (higher loss at the same iteration) than a well-tuned one', async () => {

		const slow = await trainAndCollectLossHistory( { learningRate: 0.0002, iterations: 80 } );
		const normal = await trainAndCollectLossHistory( { learningRate: 0.02, iterations: 80 } );

		console.log( `[convergence] lr=0.0002 final loss=${slow.result.loss.toExponential( 3 )}, lr=0.02 final loss=${normal.result.loss.toExponential( 3 )}` );

		expect( Number.isFinite( slow.result.loss ) ).toBe( true );
		expect( Number.isFinite( normal.result.loss ) ).toBe( true );

		// A 100x smaller learning rate over the same (short) iteration budget
		// must not have already converged as far as the well-tuned one - if
		// it has, either the LR schedule or the loss computation has a bug
		// that makes the learning rate not actually matter.
		expect( slow.result.loss ).toBeGreaterThan( normal.result.loss );

	}, 60000 );

	it( 'loss is (noisily) non-increasing over the course of training - no runaway divergence', async () => {

		const { lossHistory } = await trainAndCollectLossHistory( {} );

		expect( lossHistory.length ).toBeGreaterThan( 2 );

		const first = lossHistory[ 0 ].loss;
		const last = lossHistory[ lossHistory.length - 1 ].loss;

		console.log( `[convergence] loss trajectory: first=${first.toExponential( 3 )} -> last=${last.toExponential( 3 )} over ${lossHistory.length} logged points` );

		expect( last ).toBeLessThan( first );
		// No individual logged point should blow up to more than a few times
		// the starting loss - catches a diverging (too-high effective
		// learning rate / exploding gradient) run rather than just checking
		// the endpoints.
		for ( const point of lossHistory ) {

			expect( point.loss ).toBeLessThan( first * 5 + 1e-6 );

		}

	}, 60000 );

} );
