import { Matrix4 } from 'three/webgpu';
import { vec4 } from 'three/tsl';
import { temporalReproject } from './TemporalReprojectNode.js';
import { recurrentDenoise } from './RecurrentDenoiseNode.js';

/** Shared controls for the experimental SSR comparison examples. */
export function readSSRComparison( defaultMode, defaultDenoise, defaultSteps = 16, defaultClamp = 10 ) {

	const query = new URLSearchParams( window.location.search );
	const number = ( name, fallback, min, max ) => {

		const value = query.has( name ) ? Number( query.get( name ) ) : fallback;
		return Number.isFinite( value ) ? Math.max( min, Math.min( max, value ) ) : fallback;

	};

	const requestedMode = query.get( 'mode' );
	return {
		mode: [ 'mirror', 'ggx', 'power' ].includes( requestedMode ) ? requestedMode : defaultMode,
		denoise: query.has( 'denoise' ) ? query.get( 'denoise' ) !== '0' : defaultDenoise,
		resolutionScale: number( 'resolution', 1, 0.25, 1 ),
		steps: Math.round( number( 'steps', defaultSteps, 1, 64 ) ),
		rayCount: requestedMode === 'power' ? Math.round( number( 'rays', 1, 1, 4 ) ) : 1,
		seed: Math.round( number( 'seed', 1, 0, 65535 ) ),
		maxDistance: query.has( 'reach' ) ? number( 'reach', 1, 0.001, 20 ) : null,
		thickness: query.has( 'thickness' ) ? number( 'thickness', 0.03, 0.0001, 1 ) : null,
		maxLuminance: number( 'clamp', query.has( 'mode' ) ? 0 : defaultClamp, 0, 10000 ),
		explicit: query.has( 'mode' ),
		query
	};

}

/** Mode and denoise changes reload so disabled paths do not execute hidden passes. */
export function addSSRComparisonGUI( gui, comparison, apply ) {

	const reload = ( key, value ) => {

		const url = new URL( window.location.href );
		for ( const [ field, query ] of Object.entries( { resolutionScale: 'resolution', steps: 'steps', rayCount: 'rays', seed: 'seed', maxDistance: 'reach', thickness: 'thickness', maxLuminance: 'clamp' } ) ) {

			if ( comparison[ field ] !== null ) url.searchParams.set( query, comparison[ field ] );

		}

		url.searchParams.set( 'denoise', comparison.denoise ? '1' : '0' );
		url.searchParams.set( key, value );
		window.location.href = url.href;

	};

	gui.add( comparison, 'mode', { 'Mirror / blur': 'mirror', 'Stochastic GGX': 'ggx', 'Biased power lobe': 'power' } ).onChange( value => reload( 'mode', value ) );
	gui.add( comparison, 'denoise' ).name( 'recurrent denoise' ).onChange( value => reload( 'denoise', value ? '1' : '0' ) );
	gui.add( comparison, 'resolutionScale', 0.25, 1, 0.25 ).name( 'trace resolution' ).onChange( apply );
	gui.add( comparison, 'steps', 1, 64, 1 ).name( comparison.mode === 'mirror' ? 'steps (mirror: adaptive)' : 'trace steps' ).onChange( apply );
	if ( comparison.mode === 'power' ) {

		gui.add( comparison, 'rayCount', 1, 4, 1 ).name( 'rays (stock=1)' ).onChange( apply );
		gui.add( comparison, 'seed', 0, 65535, 1 ).name( 'seed' ).onChange( apply );

	}

	gui.add( comparison, 'maxLuminance', 0, 1000, 1 ).name( 'radiance clamp (0=off)' ).onChange( value => {

		comparison.query.set( 'clamp', value );
		apply();

	} );
	const note = document.createElement( 'div' );
	note.style.cssText = 'position:fixed;bottom:8px;left:8px;max-width:580px;padding:6px;background:#000b;color:white;font:12px sans-serif;pointer-events:none;z-index:1000';
	note.textContent = `${ comparison.mode } · ${ comparison.denoise ? 'recurrent denoise' : 'raw' } · ${ comparison.mode === 'power' ? 'biased estimator; projected steps; previous opaque HDR' : 'stock estimator; current beauty; one ray' }. Explicit modes share an absolute reach cap. Mirror retains adaptive steps, plane attenuation, blur and no environment miss fallback; GGX retains its step distribution and BRDF weights. Scene/filter controls reset on mode reload; shared trace controls persist.`;
	document.body.appendChild( note );

}

export function applySSRComparison( node, comparison ) {

	node.resolutionScale = comparison.resolutionScale;
	if ( comparison.explicit || comparison.query.has( 'clamp' ) ) {

		node.maxLuminance.value = comparison.maxLuminance === 0 && comparison.mode !== 'power' ? 1e20 : comparison.maxLuminance;

	}

	if ( comparison.mode === 'power' ) {

		node.steps.value = comparison.steps;
		node.rayCount.value = comparison.rayCount;
		node.seed.value = comparison.seed;

	} else {

		node.quality.value = comparison.steps / 64;

	}

	if ( comparison.maxDistance !== null ) node.maxDistance.value = comparison.maxDistance;
	if ( comparison.thickness !== null ) node.thickness.value = comparison.thickness;
	if ( node.maxRayLength ) node.maxRayLength.value = node.maxDistance.value;

}

/**
 * The raw estimator contract is weighted radiance RGB + ray distance alpha.
 * Temporal/recurrent outputs instead carry inverse history age in alpha.
 * Coverage remains on PowerLobeSSRNode.getMetadataNode(), outside this adapter.
 */
export function createSSRComparisonDenoiser( node, depth, packedNormal, velocity, metalRoughness, camera ) {

	if ( ! camera.isPerspectiveCamera ) throw new Error( 'SSR comparison: recurrent specular filtering requires a perspective camera. Use denoise=0 for orthographic cameras.' );
	const rawDistance = node.getTextureNode(); // Weighted RGB/distance, with explicit pass scheduling.
	const temporal = temporalReproject( rawDistance, depth, packedNormal, velocity, camera, { mode: 'specular', accumulate: false } );
	const denoise = recurrentDenoise( temporal, camera, {
		depth, normal: packedNormal, raw: rawDistance, metalRoughness, mode: 'specular', accumulate: true
	} );
	denoise.alphaSource = 'raylength';
	temporal.setHistoryTexture( denoise );
	return { temporal, denoise, color: vec4( denoise.rgb, 1 ) };

}

/** Resets all estimator/filter history together without changing their algorithms. */
export function createSSRHistoryReset( node, camera, temporal = null, denoise = null ) {

	const previousWorld = new Matrix4();
	const reset = () => {

		node.resetHistory?.();
		temporal?.resetHistory();
		denoise?.resetHistory();

	};

	const update = () => {

		camera.updateMatrixWorld();
		const delta = camera.matrixWorld.elements.reduce( ( sum, v, i ) => sum + Math.abs( v - previousWorld.elements[ i ] ), 0 );
		if ( delta > 1 ) reset();
		previousWorld.copy( camera.matrixWorld );

	};

	return { reset, update };

}
