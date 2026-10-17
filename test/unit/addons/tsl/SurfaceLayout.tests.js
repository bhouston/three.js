import { surfaceLayout } from '../../../../../examples/jsm/tsl/utils/SurfaceLayout.js';
import { Scene, PerspectiveCamera, UnsignedByteType } from '../../../../../src/Three.WebGPU.js';
import { pass, vec4, uv } from '../../../../../src/Three.TSL.js';

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'TSL', () => {

		QUnit.module( 'SurfaceLayout', () => {

			const makePass = () => pass( new Scene(), new PerspectiveCamera() );
			const makeLayout = () => ( {
				attachments: {
					output: { node: vec4( 0 ) },
					packed: { node: vec4( 0.25, 0.5, 0.75, 1 ), texture: { type: UnsignedByteType } },
					unused: { node: vec4( 1 ) }
				},
				fields: {
					normal: { attachment: 'packed', channels: 'rgb', decode: value => value.mul( 2 ).sub( 1 ) },
					roughness: { attachment: 'packed', channels: 'a' },
					custom: { attachment: 'unused', channels: 'rg' }
				}
			} );

			QUnit.test( 'A requested field activates its complete shared attachment', assert => {

				const producer = makePass();
				const declaration = makeLayout();
				const views = surfaceLayout( producer, declaration, [ 'roughness', 'roughness' ] );
				assert.deepEqual( views.attachmentNames, [ 'output', 'packed' ] );
				assert.strictEqual( producer.renderTarget.textures.length, 2 );
				assert.strictEqual( producer.getMRT().get( 'packed' ), declaration.attachments.packed.node, 'full value writes the other channels too' );
				assert.deepEqual( Object.keys( views.nodes ), [ 'roughness' ] );
				assert.ok( views.nodes.roughness.isSampleNode, 'ordinary sampleable node, no extra RTT' );
				assert.strictEqual( producer.getTexture( 'packed' ).type, UnsignedByteType );
				producer.dispose();

			} );

			QUnit.test( 'Invalid declarations fail before allocating; compiled layouts are not extended', assert => {

				const producer = makePass();
				assert.throws( () => surfaceLayout( producer, makeLayout(), [ 'missing' ] ), /unknown field/ );
				const unresolved = makeLayout();
				unresolved.fields.custom.attachment = 'missing';
				assert.throws( () => surfaceLayout( producer, unresolved, [] ), /unresolved attachment/ );
				const missingProducer = makeLayout();
				delete missingProducer.attachments.packed.node;
				assert.throws( () => surfaceLayout( producer, missingProducer, [ 'roughness' ] ), /missing producer/ );
				assert.strictEqual( producer.renderTarget.textures.length, 1 );
				surfaceLayout( producer, makeLayout(), [] );
				assert.throws( () => surfaceLayout( producer, makeLayout(), [ 'custom' ] ), /fresh PassNode/ );
				producer.dispose();

			} );

			QUnit.test( 'Earlier consumers retain producer identity across new configurations', assert => {

				const first = makePass();
				const second = makePass();
				const declaration = makeLayout();
				const raw = surfaceLayout( first, declaration, [ 'custom' ] );
				const consumer = raw.nodes.custom.sample( uv() );
				declaration.fields.custom.attachment = 'packed';
				const later = surfaceLayout( second, declaration, [ 'custom' ] );
				assert.strictEqual( consumer.node.referenceNode.passNode, first );
				assert.strictEqual( raw.nodes.custom.sample( uv() ).node.referenceNode, raw.textures.unused, 'declaration edits cannot retarget captured storage' );
				assert.notStrictEqual( raw.nodes.custom, later.nodes.custom );
				assert.ok( Object.isFrozen( raw.nodes ) );
				first.dispose();
				second.dispose();

			} );

			QUnit.test( 'Rebuild removes demand; targets resize and dispose under caller ownership', assert => {

				const first = makePass();
				surfaceLayout( first, makeLayout(), [ 'normal', 'custom' ] );
				first.setSize( 31, 19 );
				assert.strictEqual( first.getTexture( 'packed' ).image.width, 31 );
				assert.strictEqual( first.getTexture( 'unused' ).image.height, 19 );
				let disposed = 0;
				first.renderTarget.addEventListener( 'dispose', () => disposed ++ );
				first.dispose();
				assert.strictEqual( disposed, 1 );
				const replacement = makePass();
				assert.deepEqual( surfaceLayout( replacement, makeLayout(), [] ).attachmentNames, [ 'output' ] );
				assert.strictEqual( replacement.renderTarget.textures.length, 1 );
				replacement.dispose();

			} );

		} );

	} );

} );
