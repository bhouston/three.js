import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { vec2, vec3 } from 'three/tsl';
import { previewColor, buildUvGridColorNode } from '../../../../examples/jsm/ntc/NTCFormat.js';
import { withTestRenderer, evalFloats } from '../helpers/webgpuEval.js';

// Regression coverage for a real bug found while debugging the neural-material
// example's "brick" preset: the normal/clearcoatNormal debug view showed a
// non-standard yellow/red/green palette (should be the familiar blue-dominant
// tangent-space normal map look) because both call sites that build that
// preview (NTCNodeMaterial.setDebugView and NTCSource.
// buildChannelPreviewMaterials) pass a fully TBN-reconstructed 3-component
// view-space normal into previewColor, but tag it with the *trained payload's*
// channel descriptor - size 2 (correct for the raw (dx, dy) the network
// outputs, wrong for the already-blended vec3 both call sites actually hand
// in). previewColor's size===2 branch hardcodes blue to 0, silently dropping
// the normal's z everywhere this happens. Fix: both call sites now pass a
// `{ ...channel, size: 3 }` override for normal/clearcoatNormal specifically -
// this file locks down exactly why that override is required.
describe( 'Addons > Neural > NeuralMaterial > NTCFormat.previewColor (real WebGPU)', () => {

	const getRenderer = withTestRenderer( { beforeAll, afterAll } );

	it( 'size 2 (the network\'s raw (dx, dy) payload) hardcodes blue to 0 - this is intentional for a genuine 2-component value', async () => {

		const channel = { activation: 'tanh', size: 2 };
		const value = vec2( 0.4, - 0.6 ); // raw signed offset, in [-1, 1]

		const [ r, g, b ] = await evalFloats( getRenderer(), 3, ( out ) => {

			const color = previewColor( value, channel, false );
			out.element( 0 ).assign( color.x );
			out.element( 1 ).assign( color.y );
			out.element( 2 ).assign( color.z );

		} );

		expect( r ).toBeCloseTo( 0.4 * 0.5 + 0.5, 5 );
		expect( g ).toBeCloseTo( - 0.6 * 0.5 + 0.5, 5 );
		expect( b ).toBe( 0 ); // documented, deliberate for a *true* 2-component value

	} );

	it( 'size 3 override on a reconstructed vec3 normal preserves z (the actual fix)', async () => {

		const channel = { activation: 'tanh', size: 3 };
		// Stands in for reconstructFinalNormal(...)'s output: a full TBN-blended
		// view-space normal, same shape NTCNodeMaterial.setDebugView
		// and NTCSource.buildChannelPreviewMaterials actually pass.
		const value = vec3( 0.4, - 0.6, 0.9 );

		const [ r, g, b ] = await evalFloats( getRenderer(), 3, ( out ) => {

			const color = previewColor( value, channel, false );
			out.element( 0 ).assign( color.x );
			out.element( 1 ).assign( color.y );
			out.element( 2 ).assign( color.z );

		} );

		expect( r ).toBeCloseTo( 0.4 * 0.5 + 0.5, 5 );
		expect( g ).toBeCloseTo( - 0.6 * 0.5 + 0.5, 5 );
		expect( b ).toBeCloseTo( 0.9 * 0.5 + 0.5, 5 ); // z survives - this is what was broken

	} );

	it( 'a flat (no-bump) normal previews as pale blue with the size-3 override, not olive/yellow', async () => {

		const channel = { activation: 'tanh', size: 3 };
		const flatNormal = vec3( 0, 0, 1 ); // no bump: dx = dy = 0, z = 1

		const [ r, g, b ] = await evalFloats( getRenderer(), 3, ( out ) => {

			const color = previewColor( flatNormal, channel, false );
			out.element( 0 ).assign( color.x );
			out.element( 1 ).assign( color.y );
			out.element( 2 ).assign( color.z );

		} );

		expect( r ).toBeCloseTo( 0.5, 5 );
		expect( g ).toBeCloseTo( 0.5, 5 );
		expect( b ).toBeCloseTo( 1.0, 5 ); // was 0 before the fix - the "yellow instead of blue" symptom

	} );

	it( 'clamps out-of-range (undertrained) values for display instead of NaN/negative colors', async () => {

		const channel = { activation: 'tanh', size: 3 };
		// tanh is bounded to (-1, 1) so this can't happen from a real trained
		// channel, but previewColor's doc comment explicitly calls out that it
		// must clamp defensively for display - lock that down.
		const value = vec3( 5, - 5, 2 );

		const [ r, g, b ] = await evalFloats( getRenderer(), 3, ( out ) => {

			const color = previewColor( value, channel, false );
			out.element( 0 ).assign( color.x );
			out.element( 1 ).assign( color.y );
			out.element( 2 ).assign( color.z );

		} );

		expect( r ).toBe( 1 );
		expect( g ).toBe( 0 );
		expect( b ).toBe( 1 );

	} );

} );

describe( 'Addons > NTC > NTCFormat.buildUvGridColorNode (real WebGPU)', () => {

	const getRenderer = withTestRenderer( { beforeAll, afterAll } );

	async function sampleAt( u, v, cellCount ) {

		const [ r, g, b ] = await evalFloats( getRenderer(), 3, ( out ) => {

			const color = buildUvGridColorNode( vec2( u, v ), cellCount );
			out.element( 0 ).assign( color.x );
			out.element( 1 ).assign( color.y );
			out.element( 2 ).assign( color.z );

		} );

		return { r, g, b };

	}

	it( 'the red/green channels track the wrapped UV coordinate (ratio-preserving under the shared checker shade)', async () => {

		// Kept off exact cell boundaries (0.23/0.61, not 0.2/0.6 at
		// cellCount=10) to avoid a floor() edge case at an exact integer -
		// r/g == u/v regardless of which cell's shade multiplies both,
		// so this doesn't need to know the checker's actual shade value.
		const { r, g } = await sampleAt( 0.23, 0.61, 10 );

		expect( r / g ).toBeCloseTo( 0.23 / 0.61, 4 );

	} );

	it( 'wraps coordinates outside [0, 1) via fract(), matching the periodic grid it visualizes', async () => {

		const inside = await sampleAt( 0.23, 0.61, 10 );
		const wrapped = await sampleAt( 1.23, - 0.39, 10 ); // 1.23 -> 0.23, -0.39 -> 0.61

		expect( wrapped.r ).toBeCloseTo( inside.r, 4 );
		expect( wrapped.g ).toBeCloseTo( inside.g, 4 );
		expect( wrapped.b ).toBeCloseTo( inside.b, 4 );

	} );

	it( 'shades adjacent cells differently (a real checkerboard, not a flat gradient)', async () => {

		// cellCount=2: (0.2, 0.2) and (0.7, 0.2) land in cell (0,0) and (1,0) -
		// adjacent cells, opposite checker parity.
		const cellA = await sampleAt( 0.2, 0.2, 2 );
		const cellB = await sampleAt( 0.7, 0.2, 2 );

		// Isolate the checker shading from the red-channel gradient (which
		// differs between these two points anyway) by comparing blue - the
		// constant-0.5-before-shading channel, so any difference is purely the
		// checkerboard's doing.
		expect( Math.abs( cellA.b - cellB.b ) ).toBeGreaterThan( 0.05 );

	} );

} );
