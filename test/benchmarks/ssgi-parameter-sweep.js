/** Fifteen isolated comparisons; every case starts from the same 2 x 8 budget. */
export const SSGI_PARAMETER_CONFIGS = Object.freeze( [
	...[ 0.1, 0.15, 0.25 ].flatMap( radius => [ 0.0025, 0.005, 0.01 ].map( thickness => ( {
		label: `radius-${ radius }-thickness-${ thickness }`, settings: Object.freeze( { radius, thickness } )
	} ) ) ),
	{ label: 'slices-3-steps-8', settings: Object.freeze( { sliceCount: 3, stepCount: 8 } ) },
	{ label: 'slices-2-steps-12', settings: Object.freeze( { sliceCount: 2, stepCount: 12 } ) },
	{ label: 'full-resolution', settings: Object.freeze( { resolutionScale: 1 } ) },
	{ label: 'history-8', settings: Object.freeze( { maxFrames: 8 } ) },
	{ label: 'history-32', settings: Object.freeze( { maxFrames: 32 } ) },
	{ label: 'denoise-radius-1', settings: Object.freeze( { denoiseRadius: 1 } ) }
].map( Object.freeze ) );

/**
 * Run only in the centrally controlled hardware-GPU page. The caller disables
 * vsync/frame limiting, stops automatic animation, verifies a real adapter, and
 * supplies whole-frame step(), GPU-completion drain(), and float RGBA read().
 * GPU throughput is measured before any readback or CPU reference work.
 *
 * The reference module must be served by the common harness. Its radiance
 * comparison reuses screen lighting; it is not full path-traced illumination.
 * @param {Object} options - State, frame helpers, dimensions and optional reference injection for CPU harness tests.
 * @return {Promise<Array<Object>>} Compact comparison records, with no image arrays.
 */
export async function runSSGIParameterSweep( { state, step, drain, read, width = 960, height = 720, geometryReference = null } ) {

	const reference = geometryReference || ( await import( '/test/benchmarks/ssgi-geometry-reference.js' ) ).ssgiGeometryReference;
	const original = { ...state.ssgiParams };
	const originalOutput = state.params.output;
	const baseline = {
		...original,
		enabled: true, denoise: true,
		resolutionScale: 0.5, sliceCount: 2, stepCount: 8,
		radius: 0.25, thickness: 0.01,
		maxFrames: 16, denoiseRadius: 1.5
	};
	const results = [];
	try {

		for ( const config of SSGI_PARAMETER_CONFIGS ) {

			Object.assign( state.ssgiParams, baseline, config.settings );
			state.updateSSGIParameters();
			state.params.output = 'Beauty';
			state.updateOutput();
			for ( let frame = 0; frame < 64; frame ++ ) step();

			await drain();
			const initialGI = state.ssgiDenoise._frames;
			if ( ! Number.isFinite( initialGI ) ) throw new Error( 'Missing GI frame counter; cannot verify completed full frames' );

			let completed = 0;
			const started = performance.now();
			while ( performance.now() - started < 5000 ) {

				for ( let frame = 0; frame < 4; frame ++ ) step();

				await drain();
				completed += 4;

			}

			const seconds = ( performance.now() - started ) / 1000;
			const giFrames = state.ssgiDenoise._frames - initialGI;
			if ( giFrames !== completed ) throw new Error( `GI updated ${ giFrames } times for ${ completed } completed frames in ${ config.label }` );

			const rawTarget = state.ssgiPass.getRenderTarget();
			const filteredTarget = state.ssgiDenoise.getRenderTarget();
			// These operations are outside the measured interval. Keep them serial so
			// CPU geometry work and GPU readback cannot contaminate throughput.
			const raw = await read( rawTarget, 1 );
			const filtered = await read( filteredTarget, 0 );
			const source = await read( state.lightingHistory, 0 );
			const worldRadius = state.ssgiPass.radius.value;
			const worldThickness = state.ssgiPass.thickness.value;
			const geometry = await reference( {
				model: state.model || state.scene,
				camera: state.camera, width, height, radius: worldRadius,
				raw: { data: raw, width: rawTarget.width, height: rawTarget.height },
				filtered: { data: filtered, width: filteredTarget.width, height: filteredTarget.height },
				radiance: { data: source, width: state.lightingHistory.width, height: state.lightingHistory.height }
			} );
			const validCount = geometry.valid.filter( Boolean ).length;
			const coverageSum = geometry.coverage.reduce( ( sum, value, index ) => sum + ( geometry.valid[ index ] ? value : 0 ), 0 );
			results.push( {
				label: config.label,
				settings: { ...state.ssgiParams }, worldRadius, worldThickness,
				throughput: { completed, giFrames, seconds, fps: completed / seconds },
				geometry: {
					method: geometry.method,
					gridWidth: geometry.gridWidth, gridHeight: geometry.gridHeight, rays: geometry.rays,
					validCount, sampleCount: geometry.valid.length,
					meanTrueCoverage: validCount > 0 ? coverageSum / validCount : null,
					comparisons: geometry.comparisons, milliseconds: geometry.milliseconds
				}
			} );

		}

	} finally {

		// Every measured case has drained before CPU work. Also drain on errors
		// before restoring targets/history for the caller's next implementation.
		try {

			await drain();

		} finally {

			for ( const key of Object.keys( state.ssgiParams ) ) {

				if ( ! Object.hasOwn( original, key ) ) delete state.ssgiParams[ key ];

			}

			Object.assign( state.ssgiParams, original );
			state.params.output = originalOutput;
			state.updateSSGIParameters();
			state.updateOutput();

		}

	}

	return results;

}
