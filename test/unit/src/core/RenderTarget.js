import { describe, test, expect } from 'vitest';

import { NearestFilter } from '@src/constants.js';
import { RenderTarget } from '@src/core/RenderTarget.js';

describe( 'Core', () => {

	describe( 'RenderTarget', () => {

		test( 'Constructor', () => {

			const empty = new RenderTarget();
			expect( empty.width != null && empty.height != null && empty.textures.length === 1 ).toBeTruthy();

			const sized = new RenderTarget( 1, 1 );
			expect( sized.width === 1 && sized.height === 1 && sized.textures.length === 1 ).toBeTruthy();

			const mrt = new RenderTarget( 1, 1, { count: 2 } );
			expect( mrt.width === 1 && mrt.height === 1 && mrt.textures.length === 2 ).toBeTruthy();

			const options = new RenderTarget( 1, 1, { magFilter: NearestFilter } );
			expect( options.width === 1 && options.height === 1 && options.texture.magFilter === NearestFilter ).toBeTruthy();

		} );

		test( 'setSize', () => {

			const renderTarget = new RenderTarget();
			renderTarget.setSize( 128, 128 );
			expect( renderTarget.width === 128 && renderTarget.height === 128 ).toBeTruthy();
			expect( renderTarget.texture.image.width === 128 && renderTarget.texture.image.height === 128 ).toBeTruthy();
			expect( renderTarget.viewport.width === 128 && renderTarget.viewport.height === 128 ).toBeTruthy();
			expect( renderTarget.scissor.width === 128 && renderTarget.scissor.height === 128 ).toBeTruthy();

			const mrt = new RenderTarget( 0, 0, { count: 2 } );
			mrt.setSize( 128, 128 );
			expect( mrt.width === 128 && mrt.height === 128 ).toBeTruthy();
			expect( mrt.textures[ 0 ].image.width === 128 && mrt.textures[ 0 ].image.height === 128 && mrt.textures[ 1 ].image.width === 128 && mrt.textures[ 1 ].image.height === 128 ).toBeTruthy();
			expect( mrt.viewport.width === 128 && mrt.viewport.height === 128 ).toBeTruthy();
			expect( mrt.scissor.width === 128 && mrt.scissor.height === 128 ).toBeTruthy();

			const renderTarget3D = new RenderTarget();
			renderTarget3D.setSize( 128, 128, 16 );
			expect( renderTarget3D.width === 128 && renderTarget3D.height === 128 && renderTarget3D.depth === 16 ).toBeTruthy();
			expect( renderTarget3D.texture.image.width === 128 && renderTarget3D.texture.image.height === 128 && renderTarget3D.texture.image.depth === 16 ).toBeTruthy();
			expect( renderTarget3D.viewport.width === 128 && renderTarget3D.viewport.height === 128 ).toBeTruthy();
			expect( renderTarget3D.scissor.width === 128 && renderTarget3D.scissor.height === 128 ).toBeTruthy();

		} );

	} );

} );
