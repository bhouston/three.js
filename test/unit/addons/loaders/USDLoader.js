import { describe, test, expect } from 'vitest';
import { USDLoader } from '../../../../examples/jsm/loaders/USDLoader.js';

describe( 'Addons', () => {

	describe( 'Loaders', () => {

		describe( 'USDLoader', () => {

			test( 'uses timeCodesPerSecond for USDA animation timing', () => {

				const usda = `#usda 1.0
(
	defaultPrim = "Root"
	framesPerSecond = 24
	timeCodesPerSecond = 60
)

def Xform "Root"
{
	def Xform "Animated"
	{
		float3 xformOp:translate = (0, 0, 0)
		float3 xformOp:translate.timeSamples = {
			0: (0, 0, 0),
			60: (1, 0, 0),
		}
		uniform token[] xformOpOrder = ["xformOp:translate"]
	}
}`;

				const loader = new USDLoader();
				const scene = loader.parse( usda );
				const clip = scene.animations[ 0 ];
				const track = clip.tracks[ 0 ];

				expect( scene.animations.length ).toBe( 1 );
				expect( clip.name ).toBe( 'TransformAnimation' );
				expect( Math.abs( clip.duration - 1 ) ).toBeLessThanOrEqual( 0.000001 );
				expect( track.name ).toBe( 'Animated.position' );
				expect( Array.from( track.times ) ).toEqual( [ 0, 1 ] );

			} );

		} );

	} );

} );
