import Textures from '../../../../../src/renderers/common/Textures.js';
import { CompressedCubeTexture } from '../../../../../src/textures/CompressedCubeTexture.js';
import { CubeTexture } from '../../../../../src/textures/CubeTexture.js';

export default QUnit.module( 'Renderers', () => {

	QUnit.module( 'Textures', () => {

		QUnit.test( 'updateTexture cube mip levels', ( assert ) => {

			const allocations = [];
			const uploads = [];
			const backend = {
				createTexture: ( texture, options ) => allocations.push( options.levels ),
				updateTexture: ( texture ) => uploads.push( texture )
			};
			const textures = new Textures( {}, backend, { createTexture() {} } );
			const faces = Array.from( { length: 6 }, () => ( {
				width: 4,
				height: 4,
				mipmaps: [ { width: 4, height: 4 }, { width: 2, height: 2 }, { width: 1, height: 1 } ]
			} ) );
			const compressed = new CompressedCubeTexture( faces );
			compressed.needsUpdate = true;

			assert.deepEqual( compressed.mipmaps, [], 'Compressed cubes store mipmaps per face' );
			textures.updateTexture( compressed );
			assert.strictEqual( allocations[ 0 ], 3, 'Face mipmaps include the base level' );
			assert.strictEqual( uploads[ 0 ], compressed, 'Compressed cube reaches the backend upload' );

			const cube = new CubeTexture( faces );
			cube.mipmaps = [ { image: faces }, { image: faces } ];
			cube.generateMipmaps = false;
			cube.needsUpdate = true;
			textures.updateTexture( cube );

			assert.strictEqual( allocations[ 1 ], 3, 'Ordinary cube mipmaps exclude the base level' );
			assert.strictEqual( uploads[ 1 ], cube, 'Ordinary cube reaches the backend upload' );

		} );

	} );

} );
