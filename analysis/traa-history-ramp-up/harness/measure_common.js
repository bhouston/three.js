// Measurement harness shared by measure_vxgi.html / measure_ssr.html / measure_bake.html.
import * as THREE from 'three/webgpu';

const query = new URLSearchParams( location.search );
globalThis.__NOISE = query.get( 'noise' ) ?? 'default';
globalThis.__WHITE_SEED = Number( query.get( 'seed' ) ?? 0 );

// variant H: load the FAST atlas with the PR's own loader before any node is built
if ( globalThis.__NOISE === 'h' ) {

	const t0 = performance.now();
	const { loadFASTNoise } = await import( './jsm/tsl/utils/FASTNoise.js' );
	globalThis.__FAST_ATLAS = await loadFASTNoise( 'textures/noises/fast/gauss1-ema01-spatio-temporal.png' );
	globalThis.__FAST_LOAD_MS = performance.now() - t0; // module import + fetch from localhost + PNG decode (not the GPU upload)

}

export const WIDTH = Number( query.get( 'w' ) ?? 960 ), HEIGHT = Number( query.get( 'h' ) ?? 720 );

export async function adapterInfo() {

	const adapter = await navigator.gpu.requestAdapter();
	const i = adapter.info;
	return { vendor: i.vendor, architecture: i.architecture, device: i.device, description: i.description, isFallbackAdapter: i.isFallbackAdapter ?? null, userAgent: navigator.userAgent };

}

function toBase64( typed ) {

	const bytes = new Uint8Array( typed.buffer, typed.byteOffset, typed.byteLength );
	let s = '';
	for ( let i = 0; i < bytes.length; i += 0x8000 ) s += String.fromCharCode.apply( null, bytes.subarray( i, i + 0x8000 ) );
	return btoa( s );

}

export const save = ( name, typed ) => window.__save( name, toBase64( typed ) );

// Advance the renderer's node frame exactly like Animation.js does for one requestAnimationFrame tick.
export function tick( renderer ) {

	if ( renderer._animation._requestId !== null ) renderer._animation.stop(); // the harness, not requestAnimationFrame, advances frames
	renderer._nodes.nodeFrame.update();
	renderer.info.frame = renderer._nodes.nodeFrame.frameId;

}

export const gpuDone = ( renderer ) => renderer.backend.device.queue.onSubmittedWorkDone();

/**
 * renderFrame(): renders one frame into `target` (FloatType RGBA, linear).
 * Returns per-pixel temporal mean (saved as <name>_mean.f32, RGB float32), the last frame (<name>_last.f32),
 * the mean per-pixel temporal variance and the mean squared consecutive-frame difference (both over RGB).
 */
export async function capture( renderer, target, renderFrame, { name, warmup, frames } ) {

	const w = target.width, h = target.height, n = w * h;
	const t0 = performance.now();

	for ( let i = 0; i < warmup; i ++ ) {

		tick( renderer );
		renderFrame();
		if ( i % 4 === 3 ) await gpuDone( renderer );

	}

	const sum = new Float64Array( n * 3 ), sumSq = new Float64Array( n * 3 );
	const previous = new Float32Array( n * 3 ), current = new Float32Array( n * 3 );
	let flicker = 0, nonFinite = 0;

	for ( let f = 0; f < frames; f ++ ) {

		tick( renderer );
		renderFrame();
		const rgba = await renderer.readRenderTargetPixelsAsync( target, 0, 0, w, h );

		for ( let i = 0, j = 0; i < n * 4; i += 4, j += 3 ) {

			for ( let c = 0; c < 3; c ++ ) {

				let v = rgba[ i + c ];
				if ( ! Number.isFinite( v ) ) { nonFinite ++; v = 0; }
				current[ j + c ] = v;
				sum[ j + c ] += v;
				sumSq[ j + c ] += v * v;
				if ( f > 0 ) { const d = v - previous[ j + c ]; flicker += d * d; }

			}

		}

		previous.set( current );

	}

	const mean = new Float32Array( n * 3 );
	let variance = 0;

	for ( let j = 0; j < n * 3; j ++ ) {

		const m = sum[ j ] / frames;
		mean[ j ] = m;
		variance += Math.max( 0, sumSq[ j ] / frames - m * m );

	}

	await save( name + '_mean.f32', mean );
	await save( name + '_last.f32', current );

	return { name, width: w, height: h, warmup, frames, temporalVariance: variance / ( n * 3 ), flickerMSE: flicker / ( n * 3 * Math.max( 1, frames - 1 ) ), nonFinite, ms: performance.now() - t0, frameId: renderer._nodes.nodeFrame.frameId, noise: globalThis.__NOISE, fastLoadMs: globalThis.__FAST_LOAD_MS ?? null, vcGenerateMs: globalThis.__VC_MS ?? null, vcTextureBytes: globalThis.__VC_BYTES ?? null };

}

// Sustained throughput: no vsync (never presents; renders offscreen), warm up, then count GPU-completed frames for `seconds`.
export async function throughput( renderer, renderFrame, seconds = 5 ) {

	for ( let i = 0; i < 60; i ++ ) { tick( renderer ); renderFrame(); if ( i % 4 === 3 ) await gpuDone( renderer ); }
	await gpuDone( renderer );

	const t0 = performance.now();
	let frames = 0;

	while ( performance.now() - t0 < seconds * 1000 ) {

		for ( let i = 0; i < 4; i ++ ) { tick( renderer ); renderFrame(); frames ++; }
		await gpuDone( renderer );

	}

	await gpuDone( renderer );
	return { frames, seconds: ( performance.now() - t0 ) / 1000, fps: frames / ( ( performance.now() - t0 ) / 1000 ), vcGenerateMs: globalThis.__VC_MS ?? null };

}

export { THREE };
