import { BoxGeometry, CubeArrayTexture, Mesh, MeshStandardNodeMaterial, PMREMGenerator, Scene, Vector3, WebGPUCoordinateSystem } from 'three/webgpu';
import { vec3 } from 'three/tsl';
import { ReflectionProbe, ReflectionProbes } from '../../../../../examples/jsm/lighting/ReflectionProbes.js';
import { ReflectionProbesNode } from '../../../../../examples/jsm/tsl/lighting/ReflectionProbesNode.js';
import WGSLNodeBuilder from '../../../../../src/renderers/webgpu/nodes/WGSLNodeBuilder.js';

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'TSL', () => {

		QUnit.module( 'ReflectionProbesNode', () => {

			QUnit.test( 'bounces sample complete previous passes and restore state on failure', ( assert ) => {

				const probes = new ReflectionProbes();
				probes.add( new ReflectionProbe(), new ReflectionProbe() );
				const scene = new Scene();
				const environmentNode = vec3( 0 );
				scene.environmentNode = environmentNode;
				const fromScene = PMREMGenerator.prototype.fromScene;
				let captures = [];
				let targets = [];
				let disposed = [];
				let fail = false;
				const renderer = {
					backend: { isWebGPUBackend: true },
					initRenderTarget: target => {

						targets.push( target );
						target.addEventListener( 'dispose', () => disposed.push( target ) );

					},
					copyTextureToTexture: ( source, destination ) => {

						if ( probes._captureBounce > 0 ) assert.notStrictEqual( destination, probes.texture, 'never writes to the sampled texture' );

					}
				};

				PMREMGenerator.prototype.fromScene = function () {

					if ( fail && probes._captureBounce === 1 ) throw new Error( 'capture failed' );
					captures.push( { bounce: probes._captureBounce, texture: probes.texture } );
					return { texture: {}, dispose() {} };

				};

				try {

					probes.bake( renderer, scene, { bounces: 3 } );
					assert.deepEqual( captures.map( capture => capture.bounce ), [ 0, 0, 1, 1, 2, 2, 3, 3 ], 'captures all probes in each pass' );
					assert.strictEqual( captures[ 2 ].texture, targets[ 0 ].texture, 'first bounce reads the base pass' );
					assert.strictEqual( captures[ 4 ].texture, targets[ 1 ].texture, 'second bounce reads the first bounce' );
					assert.strictEqual( probes.texture, targets[ 1 ].texture, 'publishes the last completed pass' );
					assert.deepEqual( disposed, [ targets[ 0 ] ], 'disposes the spare target' );
					assert.strictEqual( scene.environmentNode, environmentNode, 'restores the scene environment node' );

					const previousTexture = probes.texture;
					const previousProbes = probes.bakedProbes;
					captures = [];
					targets = [];
					disposed = [];
					fail = true;
					assert.throws( () => probes.bake( renderer, scene, { bounces: 3 } ), /capture failed/ );
					assert.strictEqual( probes.texture, previousTexture, 'retains the old bake after failure' );
					assert.strictEqual( probes.bakedProbes, previousProbes, 'retains the old probe order after failure' );
					assert.strictEqual( disposed.length, 2, 'disposes both failed bake targets' );
					assert.strictEqual( scene.environmentNode, environmentNode, 'restores the scene after failure' );
					assert.false( probes.capturing, 'clears the capture flag after failure' );

					fail = false;
					captures = [];
					probes.bake( renderer, scene );
					assert.deepEqual( captures.map( capture => capture.bounce ), [ 0, 0 ], 'default bake uses only one pass' );

				} finally {

					PMREMGenerator.prototype.fromScene = fromScene;
					probes.dispose();

				}

			} );

			QUnit.test( 'clone copies the probe settings', ( assert ) => {

				const probes = new ReflectionProbes();
				probes.boxProjection = false;
				const probe = new ReflectionProbe( 8, 5, 8 );
				probe.projectionSize = new Vector3( 16, 5, 8 );
				probe.projectionOffset.set( 0, 0.6, 0 );
				probe.blendDistance = 0.25;
				probes.add( probe );

				const clone = probes.clone();
				const cloneProbe = clone.children[ 0 ];

				assert.strictEqual( clone.boxProjection, false, 'copies boxProjection' );
				assert.strictEqual( clone.texture, null, 'does not share the baked texture' );
				assert.true( cloneProbe.isReflectionProbe, 'clones the probes' );
				assert.deepEqual( cloneProbe.influenceSize.toArray(), [ 8, 5, 8 ], 'copies influenceSize' );
				assert.deepEqual( cloneProbe.projectionSize.toArray(), [ 16, 5, 8 ], 'copies projectionSize' );
				assert.notStrictEqual( cloneProbe.projectionSize, probe.projectionSize, 'does not share projectionSize' );
				assert.deepEqual( cloneProbe.projectionOffset.toArray(), [ 0, 0.6, 0 ], 'copies projectionOffset' );
				assert.strictEqual( cloneProbe.blendDistance, 0.25, 'copies blendDistance' );

			} );

			QUnit.test( 'updates uniforms in every material program', ( assert ) => {

				const probe = new ReflectionProbe( 8, 5, 8 );
				const probes = new ReflectionProbes();
				probes.texture = new CubeArrayTexture();
				probes.texture.mipmaps = [ {}, {} ];
				probes.bakedProbes = [ probe ];
				const node = new ReflectionProbesNode( probes );
				const geometry = new BoxGeometry();
				const material = new MeshStandardNodeMaterial();
				const mesh = new Mesh( geometry, material );
				const renderer = {
					backend: { utils: { getTextureSampleData: () => ( { primarySamples: 1 } ) } },
					coordinateSystem: WebGPUCoordinateSystem,
					debug: { diagnostics: { keywords: false } },
					hasFeature: () => true
				};

				const compile = () => {

					const builder = new WGSLNodeBuilder( mesh, renderer );
					builder.setShaderStage( 'fragment' );
					builder.context.getUV = () => vec3( 1, 0, 0 );
					builder.context.getTextureLevel = () => 0;
					builder.flowStagesNode( node );
					return builder.uniforms.fragment;

				};

				const firstUniforms = compile();
				const firstSlot = node._slots[ 0 ];
				const secondUniforms = compile();

				for ( const name of Object.keys( firstSlot ) ) {

					const uniform = firstSlot[ name ];
					assert.true( firstUniforms.some( binding => binding.node === uniform ), `${ name } is bound in the first program` );
					assert.true( secondUniforms.some( binding => binding.node === uniform ), `${ name } is shared with the second program` );

				}

				probes.boxProjection = false;
				probe.position.set( 1, 2, 3 );
				probe.scale.setScalar( 2 );
				probe.blendDistance = 0.75;
				probe.projectionSize = new Vector3( 10, 6, 10 );
				probe.projectionOffset.set( 0, 0.5, 0 );
				node.updateBefore();

				assert.strictEqual( node._boxProjection.value, 0, 'disables projection' );
				assert.deepEqual( firstSlot.position.value.toArray(), [ 1, 2, 3 ], 'updates the world position in both programs' );
				assert.deepEqual( firstSlot.extent.value.toArray(), [ 8, 5, 8 ], 'updates the influence extent in both programs' );
				assert.strictEqual( firstSlot.fade.value, 0.75, 'updates the fade in both programs' );
				assert.deepEqual( firstSlot.projectionSize.value.toArray(), [ 20, 12, 20 ], 'updates the projection size in both programs' );
				assert.deepEqual( firstSlot.projectionOffset.value.toArray(), [ 0, 1, 0 ], 'updates the projection offset in both programs' );

				probes.boxProjection = true;
				node.updateBefore();
				assert.strictEqual( node._boxProjection.value, 1, 're-enables projection' );

				probes.capturing = true;
				node.updateBefore();
				assert.strictEqual( node._local.value, 0, 'base pass excludes local reflections' );
				probes._captureBounce = 1;
				node.updateBefore();
				assert.strictEqual( node._local.value, 1, 'later passes include previous reflections' );

				geometry.dispose();
				material.dispose();
				probes.texture.dispose();

			} );

		} );

	} );

} );
