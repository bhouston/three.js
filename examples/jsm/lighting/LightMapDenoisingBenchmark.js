import * as THREE from 'three/webgpu';
import { texture, uv } from 'three/tsl';

function metrics( value, reference, mask ) {

	let squared = 0, absolute = 0, energy = 0, count = 0;
	for ( let i = 0; i < mask.length; i += 4 ) {

		if ( mask[ i + 3 ] <= 0 ) continue;
		count ++;
		for ( let c = 0; c < 3; c ++ ) {

			const delta = value[ i + c ] - reference[ i + c ];
			if ( ! Number.isFinite( delta ) ) throw new Error( 'Non-finite irradiance in covered texel.' );
			squared += delta * delta;
			absolute += Math.abs( delta );
			energy += reference[ i + c ] * reference[ i + c ];

		}

	}

	if ( count === 0 ) throw new Error( 'No covered texels.' );
	return { coveredTexels: count, rmse: Math.sqrt( squared / ( 3 * count ) ), relativeRMSE: Math.sqrt( squared / Math.max( energy, 1e-20 ) ), mae: absolute / ( 3 * count ) };

}

function aggregate( trials ) {

	const mean = key => trials.reduce( ( sum, trial ) => sum + trial[ key ], 0 ) / trials.length;
	const relativeMean = mean( 'relativeRMSE' );
	return {
		coveredTexels: trials[ 0 ].coveredTexels,
		rmse: Math.sqrt( trials.reduce( ( sum, trial ) => sum + trial.rmse ** 2, 0 ) / trials.length ),
		relativeRMSE: Math.sqrt( trials.reduce( ( sum, trial ) => sum + trial.relativeRMSE ** 2, 0 ) / trials.length ),
		mae: mean( 'mae' ),
		relativeRMSEStd: Math.sqrt( trials.reduce( ( sum, trial ) => sum + ( trial.relativeRMSE - relativeMean ) ** 2, 0 ) / trials.length ),
		relativeRMSEMin: Math.min( ...trials.map( trial => trial.relativeRMSE ) ),
		relativeRMSEMax: Math.max( ...trials.map( trial => trial.relativeRMSE ) )
	};

}


/**
 * Compare denoisers on an already prepared LightBaker. Mutates its samples and denoiser.
 * Timings include complete, GPU-fenced bakes; quality readbacks are outside timing.
 * @param {Object} options - Renderer, prepared baker, configuration and optional callbacks.
 * @return {Promise<Object>} Throughput and covered-texel quality results for every atlas page.
 */
async function runLightMapDenoisingBenchmark( { renderer, baker, config, onStatus = () => {}, onPreview = () => {}, sceneName = 'fixture' } ) {

	config = { ...config, seconds: Math.max( 5, config.seconds || 5 ), qualityTrials: 5 };
	const mapper = baker.lightMapper;
	if ( ! mapper || mapper.pages.length === 0 ) throw new Error( 'Prepare the baker before benchmarking.' );
	const size = mapper.size;
	const drain = () => renderer.backend.device.queue.onSubmittedWorkDone();
	const readTarget = new THREE.RenderTarget( size, size, { type: THREE.FloatType, depthBuffer: false } );
	const readMaterial = new THREE.NodeMaterial();
	const quad = new THREE.QuadMesh( readMaterial );
	const previousTarget = renderer.getRenderTarget();
	function update( samples ) {

		const target = baker.samples + samples;
		mapper.update( samples );
		if ( baker.samples !== target ) throw new Error( 'Requested sample count exceeds the adapter accumulation limit: ' + target );

	}

	async function readPages( sources ) {

		const result = new Float32Array( sources.length * size * size * 4 );
		for ( let page = 0; page < sources.length; page ++ ) {

			readMaterial.fragmentNode = texture( sources[ page ], uv() );
			readMaterial.needsUpdate = true;
			renderer.setRenderTarget( readTarget );
			quad.render( renderer );
			renderer.setRenderTarget( previousTarget );
			result.set( new Float32Array( await renderer.readRenderTargetPixelsAsync( readTarget, 0, 0, size, size ) ), page * size * size * 4 );

		}

		return result;

	}

	function previewPages( label, pixels ) {

		for ( let page = 0; page < mapper.pages.length; page ++ ) onPreview( label, pixels.subarray( page * size * size * 4, ( page + 1 ) * size * size * 4 ), size, page );

	}

	async function process( mode ) {

		baker.denoiser = mode;
		return await baker.processAsync();

	}

	try {

		const modes = config.oidn ? [ 'none', 'variance', 'oidn' ] : [ 'none', 'variance' ];
		const rows = [];
		for ( const mode of modes ) {

			onStatus( 'Warming ' + mode );
			baker.reset();
			update( config.samples );
			await process( mode );
			await drain();
			onStatus( 'Measuring completed ' + mode + ' bakes for ' + config.seconds + ' seconds' );
			let completed = 0;
			const start = performance.now();
			do {

				baker.reset();
				update( config.samples );
				await process( mode );
				await drain();
				completed ++;

			} while ( performance.now() - start < config.seconds * 1000 );

			const seconds = ( performance.now() - start ) / 1000;
			const output = await readPages( await process( mode ) );
			rows.push( { mode, samples: config.samples, completedBakes: completed, measurementSeconds: seconds, bakesPerSecond: completed / seconds, meanCompletedBakeSeconds: seconds / completed, output } );

		}

		onStatus( 'Building high sample reference' );
		baker.reset();
		// Fixed batches keep the browser responsive; only whole completed runs are timed.
		while ( baker.samples < config.reference ) {

			update( Math.min( 32, config.reference - baker.samples ) );
			await drain();

		}

		const reference = await readPages( mapper.pages.map( page => page.texture ) );
		const mask = await readPages( mapper.pages.map( page => page.sampleTexture ) );
		previewPages( 'Reference: ' + config.reference + ' samples', reference );
		const baseline = rows[ 0 ];
		for ( const row of rows ) {

			onStatus( 'Quality trials: ' + row.mode );
			row.qualityTrials = [];
			for ( let trial = 0; trial < config.qualityTrials; trial ++ ) {

				baker.reset();
				update( config.samples );
				const output = await readPages( await process( row.mode ) );
				row.qualityTrials.push( metrics( output, reference, mask ) );
				if ( trial === 0 ) previewPages( row.mode + ': ' + config.samples + ' samples (first quality trial)', output );

			}

			row.error = aggregate( row.qualityTrials );
			delete row.output;
			// More raw rays at the denoiser's measured end-to-end time budget.
			const samples = Math.max( config.samples, Math.min( config.reference - 1, Math.round( config.samples * row.meanCompletedBakeSeconds / baseline.meanCompletedBakeSeconds ) ) );
			// Warm the candidate, then measure complete batches for several seconds too.
			baker.reset();
			update( samples );
			await process( 'none' );
			await drain();
			let completedBakes = 0;
			const start = performance.now();
			do {

				baker.reset();
				update( samples );
				await process( 'none' );
				await drain();
				completedBakes ++;

			} while ( performance.now() - start < config.seconds * 1000 );

			const measurementSeconds = ( performance.now() - start ) / 1000;
			const actualSeconds = measurementSeconds / completedBakes;
			const qualityTrials = [];
			for ( let trial = 0; trial < config.qualityTrials; trial ++ ) {

				baker.reset();
				update( samples );
				await process( 'none' );
				const output = await readPages( mapper.pages.map( page => page.texture ) );
				qualityTrials.push( metrics( output, reference, mask ) );
				if ( trial === 0 ) previewPages( 'Raw time-matched to ' + row.mode + ': ' + samples + ' samples (first quality trial)', output );

			}

			row.moreSamples = { samples, targetSeconds: row.meanCompletedBakeSeconds, actualSeconds, completedBakes, measurementSeconds, budgetRatio: actualSeconds / row.meanCompletedBakeSeconds, error: aggregate( qualityTrials ), qualityTrials };

		}

		const result = { config, sceneName, atlasSize: size, atlasPages: mapper.pages.length, rows, notes: [ 'Linear RGB irradiance over all atlas pages; caller supplies the prepared scene and baker.', 'Reset repeats UV jitter but the ray source continues its render-pass sequence. Quality trials are sequential phases, not independently seeded trials. Reference uses the same bounce cache and is not ground truth.', 'Quality uses five sequential phase trials per mode and raw comparator outside throughput timing; RMSE is sqrt(mean trial MSE), with relative RMSE standard deviation and range. Readback and preview excluded from timings; GPU completion included. Setup, shader/model warmup excluded.', 'Equal-time sample count estimated from warmed throughput, rounded and capped below reference. Actual elapsed time reported; CPU/browser variance can cause budget mismatch.', 'Each mode measured sequentially for at least five seconds, counting complete GPU-fenced bakes. No requestAnimationFrame or per-frame timers.' ] };
		return result;

	} finally {

		await drain();
		renderer.setRenderTarget( previousTarget );
		readTarget.dispose();
		readMaterial.dispose();

	}

}

export { runLightMapDenoisingBenchmark };
