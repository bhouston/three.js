import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { vec3 } from 'three/tsl';
import { buildPackedColorNodes } from '../../../../examples/jsm/ntc/training/NTCSource.js';
import { getChannel, layoutChannels } from '../../../../examples/jsm/ntc/NTCFormat.js';
import { withTestRenderer, evalFloats } from '../helpers/webgpuEval.js';

// buildPackedColorNodes (-> flattenChannelComponents, not itself exported) is
// where the normal/clearcoatNormal channel's y-component gets negated to
// compensate for the baking quad's V-flip (see the long comment at
// NTCSource.js's flattenChannelComponents) - a wrong sign here
// would train the network against upside-down ground truth for exactly the
// axis the reported "brick" bug (missing bump on some sides) narrowed down
// to. Pin the exact sign contract down with a real material.normalNode whose
// x/y/z are all distinguishable, so a sign flip, an axis swap, or an
// accidental double-negation all fail loudly.
describe( 'Addons > Neural > NeuralMaterial > NTCSource.buildPackedColorNodes (real WebGPU)', () => {

	const getRenderer = withTestRenderer( { beforeAll, afterAll } );

	it( 'normal channel: x is passed through, y is negated, z is dropped (2-component payload)', async () => {

		const activeChannels = layoutChannels( [ getChannel( 'normal' ) ] ).channels;
		const fakeMaterial = { normalNode: vec3( 0.3, 0.7, 0.9 ) };

		const [ packs ] = buildPackedColorNodes( activeChannels, fakeMaterial );

		const [ dx, dy, packZ, packW ] = await evalFloats( getRenderer(), 4, ( out ) => {

			out.element( 0 ).assign( packs.x );
			out.element( 1 ).assign( packs.y );
			out.element( 2 ).assign( packs.z );
			out.element( 3 ).assign( packs.w );

		} );

		expect( dx ).toBeCloseTo( 0.3, 5 ); // x: passed through unchanged
		expect( dy ).toBeCloseTo( - 0.7, 5 ); // y: negated (V-flip compensation)
		expect( packZ ).toBe( 0 ); // z of normalNode is dropped - only (dx, dy) is trained
		expect( packW ).toBe( 0 ); // padding, unused by a lone 2-wide channel

	} );

	it( 'clearcoatNormal channel gets the same x-through/y-negated treatment as normal', async () => {

		const activeChannels = layoutChannels( [ getChannel( 'clearcoatNormal' ) ] ).channels;
		const fakeMaterial = { clearcoatNormalNode: vec3( - 0.2, - 0.5, 1 ) };

		const [ packs ] = buildPackedColorNodes( activeChannels, fakeMaterial );

		const [ dx, dy ] = await evalFloats( getRenderer(), 2, ( out ) => {

			out.element( 0 ).assign( packs.x );
			out.element( 1 ).assign( packs.y );

		} );

		expect( dx ).toBeCloseTo( - 0.2, 5 );
		expect( dy ).toBeCloseTo( 0.5, 5 ); // -(-0.5)

	} );

	it( 'a channel packed alone as the 5th total component (its own render target) keeps the same sign convention as when packed in the first four', async () => {

		// Mirrors the exact layout the "brick" preset hits: albedo (3) + normal
		// (2) = 5 total components, so normal's y (flat index 4) lands alone as
		// the R channel of a *second* packed render target, right next to three
		// always-zero padding slots (see NTCFormat.js/
		// packComponentsIntoVec4). Confirms that isolation doesn't itself
		// perturb the value or its sign.
		const activeChannels = layoutChannels( [ getChannel( 'albedo' ), getChannel( 'normal' ) ] ).channels;
		const fakeMaterial = {
			colorNode: vec3( 0.1, 0.2, 0.3 ),
			normalNode: vec3( 0.3, 0.7, 0.9 )
		};

		const packs = buildPackedColorNodes( activeChannels, fakeMaterial );
		expect( packs.length ).toBe( 2 ); // 5 components -> ceil(5/4) = 2 render targets

		const [ pack0x, pack0y, pack0z, pack0w, pack1x, pack1y, pack1z, pack1w ] = await evalFloats( getRenderer(), 8, ( out ) => {

			out.element( 0 ).assign( packs[ 0 ].x );
			out.element( 1 ).assign( packs[ 0 ].y );
			out.element( 2 ).assign( packs[ 0 ].z );
			out.element( 3 ).assign( packs[ 0 ].w );
			out.element( 4 ).assign( packs[ 1 ].x );
			out.element( 5 ).assign( packs[ 1 ].y );
			out.element( 6 ).assign( packs[ 1 ].z );
			out.element( 7 ).assign( packs[ 1 ].w );

		} );

		expect( pack0x ).toBeCloseTo( 0.1, 5 ); // albedo.r
		expect( pack0y ).toBeCloseTo( 0.2, 5 ); // albedo.g
		expect( pack0z ).toBeCloseTo( 0.3, 5 ); // albedo.b
		expect( pack0w ).toBeCloseTo( 0.3, 5 ); // normal dx, same value/sign as the solo-channel case above

		expect( pack1x ).toBeCloseTo( - 0.7, 5 ); // normal dy - negated, unaffected by landing alone in pack 1
		expect( pack1y ).toBe( 0 ); // zero padding
		expect( pack1z ).toBe( 0 );
		expect( pack1w ).toBe( 0 );

	} );

} );
