import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { cos, sin, uv, vec2 } from 'three/tsl';
import { CHANNELS } from '../../../../examples/jsm/ntc/NTCFormat.js';
import { classifyMaterialChannels } from '../../../../examples/jsm/ntc/training/NTCSource.js';
import { reconstructFinalNormal } from '../../../../examples/jsm/ntc/NTCNodeMaterial.js';
import { createTestRenderer } from '../helpers/webgpuEval.js';

// Regression coverage for a real bug: `NTCFormat.js`'s anisotropy
// channel's `applyConstant` used to unconditionally assign a literal
// `anisotropyNode`, including for the overwhelmingly common zero-strength
// case - which still satisfies `MeshPhysicalNodeMaterial.useAnisotropy`'s
// `anisotropyNode !== null` check (see src/materials/nodes/
// MeshPhysicalNodeMaterial.js), so it force-enabled the whole anisotropic-GGX
// shading branch (including a `TBNViewMatrix` tangent/bitangent frame built
// from screen-space derivatives) on every neural material, whether or not the
// source ever had real anisotropy. That branch is numerically fragile exactly
// where the surface normal varies sharply or is viewed at a grazing angle -
// producing black/NaN pixels that looked like "the trained normal
// reconstruction is broken for some directions" even though
// `reconstructFinalNormal` itself was untouched and correct the whole time.
//
// Unlike NTCNodeMaterial.test.js (which evaluates
// `reconstructFinalNormal` in isolation, unlit, on a flat quad - so it can
// never see a bug like this one, which only exists because of *other*
// channels' constant-application side effects interacting with a curved,
// lit, tangent-varying surface), this file renders a real lit
// `MeshPhysicalNodeMaterial`, built the same way `NTCNodeMaterial`
// itself builds one (every channel's own `applyConstant`, plus a spatially-
// varying reconstructed normal standing in for a trained one), on real
// curved geometry, and checks the actual rendered pixels for the general
// symptom this bug produced - NaN or unexpectedly black lit pixels - rather
// than re-asserting the specific `anisotropyNode` fix. That's deliberate: the
// next regression in this family won't be an anisotropy bug specifically,
// but it will very likely still show up as "some pixels went black or NaN
// under normal lighting" - which is what this test actually watches for.
describe( 'Addons > Neural > NeuralMaterial > constant-channel rendering (real WebGPU, lit)', () => {

	let renderer;

	beforeAll( async () => {

		renderer = await createTestRenderer();

	} );

	afterAll( () => {

		renderer?.dispose();
		renderer = undefined;

	} );

	// Builds a material the same way `NTCNodeMaterial`'s
	// constructor does for a source material whose only spatially-varying
	// (trained) channel is `normal` - every other channel goes through its
	// own `applyConstant` with `classifyMaterialChannels({})`'s all-default
	// constant values, exactly like reconstructing a material with no
	// texture-driven channels at all except bump detail.
	function buildTestMaterial() {

		const material = new THREE.MeshPhysicalNodeMaterial();
		const { constantValues } = classifyMaterialChannels( {} );

		for ( const channel of CHANNELS ) {

			if ( channel.key === 'normal' || channel.key === 'clearcoatNormal' ) continue;
			channel.applyConstant( material, constantValues[ channel.key ] );

		}

		// Stands in for a trained normal channel's decoded (dx, dy): a high-
		// frequency UV-driven oscillation sweeps through flat (~0,0),
		// moderate, and near/over-the-unit-circle-boundary values, in all
		// four sign combinations, many times across the sphere's surface -
		// the same representative/edge-case coverage
		// NTCNodeMaterial.test.js uses, but applied per-fragment
		// on a real curved mesh instead of one flat quad per case.
		const dx = sin( uv().x.mul( 41.0 ) ).mul( 0.95 );
		const dy = cos( uv().y.mul( 37.0 ) ).mul( 0.95 );
		material.normalNode = reconstructFinalNormal( vec2( dx, dy ) );

		return material;

	}

	async function renderLitSphere( material, size = 48 ) {

		const geometry = new THREE.SphereGeometry( 1, 32, 24 );
		geometry.computeTangents();

		const scene = new THREE.Scene();
		scene.add( new THREE.Mesh( geometry, material ) );
		scene.add( new THREE.AmbientLight( 0xffffff, 0.2 ) );

		const light = new THREE.DirectionalLight( 0xffffff, 3 );
		light.position.set( 2, 3, 4 );
		scene.add( light );

		const camera = new THREE.PerspectiveCamera( 45, 1, 0.1, 10 );
		camera.position.set( 0, 0, 3 );
		camera.lookAt( 0, 0, 0 );

		const target = new THREE.RenderTarget( size, size, {
			type: THREE.HalfFloatType,
			format: THREE.RGBAFormat,
			colorSpace: THREE.NoColorSpace,
			minFilter: THREE.NearestFilter,
			magFilter: THREE.NearestFilter,
			depthBuffer: true,
			stencilBuffer: false
		} );

		const previousTarget = renderer.getRenderTarget();
		renderer.setRenderTarget( target );
		renderer.setClearColor( 0x000000, 0 );
		renderer.clear( true, true, true );
		renderer.render( scene, camera );
		const rawPixels = await renderer.readRenderTargetPixelsAsync( target, 0, 0, size, size, 0 );
		renderer.setRenderTarget( previousTarget );
		target.dispose();
		geometry.dispose();

		const pixels = new Float32Array( rawPixels.length );
		for ( let i = 0; i < rawPixels.length; i ++ ) pixels[ i ] = THREE.DataUtils.fromHalfFloat( rawPixels[ i ] );

		return { pixels, size };

	}

	it( 'every lit surface pixel is finite and non-black under a directional + ambient light', async () => {

		const material = buildTestMaterial();
		const { pixels, size } = await renderLitSphere( material );

		let surfacePixelCount = 0;
		let nonFiniteCount = 0;
		let blackCount = 0;
		const firstOffenders = [];

		for ( let i = 0; i < size * size; i ++ ) {

			const r = pixels[ i * 4 ];
			const g = pixels[ i * 4 + 1 ];
			const b = pixels[ i * 4 + 2 ];
			const a = pixels[ i * 4 + 3 ];

			// Only the sphere itself (alpha written by an opaque
			// MeshPhysicalNodeMaterial fragment), not the cleared background.
			if ( a < 0.5 ) continue;

			surfacePixelCount ++;

			const finite = Number.isFinite( r ) && Number.isFinite( g ) && Number.isFinite( b );
			if ( ! finite ) {

				nonFiniteCount ++;
				if ( firstOffenders.length < 5 ) firstOffenders.push( { i, r, g, b } );
				continue;

			}

			// The ambient light alone guarantees every non-degenerate shaded
			// pixel a nonzero minimum radiance (diffuse albedo is opaque
			// white here) - an exact (or near-exact) black among lit surface
			// pixels means some term in the BRDF evaluated to zero/NaN and
			// got clamped, not "this pixel is just dim".
			const luminance = r + g + b;
			if ( luminance < 1e-4 ) {

				blackCount ++;
				if ( firstOffenders.length < 5 ) firstOffenders.push( { i, r, g, b } );

			}

		}

		// Sanity check on the test itself: the sphere must actually cover a
		// meaningful part of the frame, or "zero bad pixels" would be
		// vacuously true.
		expect( surfacePixelCount ).toBeGreaterThan( size * size * 0.2 );

		expect( nonFiniteCount, `${nonFiniteCount}/${surfacePixelCount} surface pixels were NaN/Infinity - first offenders: ${JSON.stringify( firstOffenders )}` ).toBe( 0 );
		expect( blackCount, `${blackCount}/${surfacePixelCount} surface pixels were unexpectedly black despite ambient light - first offenders: ${JSON.stringify( firstOffenders )}` ).toBe( 0 );

	} );

	it( 'a material with genuine nonzero constant anisotropy also renders cleanly (the non-default branch is still exercised)', async () => {

		const material = buildTestMaterial();
		// Directly exercises the *other* branch of the anisotropy channel's
		// applyConstant - a real, nonzero constant should still get a true
		// TSL constant node and render without NaN, not just silently no-op.
		material.anisotropyNode = vec2( 0.6, 0.3 );

		const { pixels, size } = await renderLitSphere( material );

		let sawSurfacePixel = false;

		for ( let i = 0; i < size * size; i ++ ) {

			const a = pixels[ i * 4 + 3 ];
			if ( a < 0.5 ) continue;
			sawSurfacePixel = true;

			const r = pixels[ i * 4 ];
			const g = pixels[ i * 4 + 1 ];
			const b = pixels[ i * 4 + 2 ];
			expect( Number.isFinite( r ) && Number.isFinite( g ) && Number.isFinite( b ), `pixel ${i}: (${r}, ${g}, ${b})` ).toBe( true );

		}

		expect( sawSurfacePixel ).toBe( true );

	} );

} );
