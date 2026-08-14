import {
	createModel,
	createLatentGrid,
	sampleLatents,
	scatterLatentGradients,
	forwardDecoderInput,
	backwardDecoderInput,
	trainBatch,
	clipModelGradients,
	assertModelFinite
} from '../../../../examples/jsm/neural/NeuralAppearanceModel.js';

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'Neural', () => {

		QUnit.module( 'NeuralAppearanceModel', () => {

			QUnit.test( 'creates multi-level latent mip grids', ( assert ) => {

				const random = () => 0.5;
				const model = createModel( {
					resolution: 4,
					hiddenSize: 8,
					outputFeatures: { emission: true, opacity: true }
				}, random );

				assert.strictEqual( model.latentGrids.length, 3, 'creates 4x4, 2x2, 1x1 grids' );
				assert.strictEqual( model.latentGrids[ 0 ].width, 4, 'base grid width is 4' );
				assert.strictEqual( model.latentGrids[ 2 ].width, 1, 'finest grid width is 1' );
				assert.ok( model.emissionHead !== null, 'creates emission head' );
				assert.ok( model.opacityHead !== null, 'creates opacity head' );
				assert.strictEqual( model.rotationWeights.length, 96, 'allocates 8 * 12 rotation weights' );

			} );

			QUnit.test( 'samples and scatters latents bilinearly with adjoint consistency', ( assert ) => {

				const grid = createLatentGrid( 2, 2, () => 1 );
				const uv = [ 0.25, 0.25 ];
				const run = sampleLatents( grid, uv );

				assert.strictEqual( run.output.length, 8, 'samples 8 latent channels' );
				assert.strictEqual( run.taps.length, 4, 'samples 4 taps' );

				const tapWeightSum = run.taps.reduce( ( sum, tap ) => sum + tap.weight, 0 );
				assert.ok( Math.abs( tapWeightSum - 1.0 ) < 1e-12, 'bilinear weights sum to 1' );

				const gradLatents = [ 1, 2, 3, 4, 5, 6, 7, 8 ];
				scatterLatentGradients( grid, run, gradLatents );

				let totalGridGrad = 0;
				for ( let c = 0; c < 8; c ++ ) {

					let channelGradSum = 0;
					for ( let i = 0; i < 4; i ++ ) {

						channelGradSum += grid.grad[ i * 8 + c ];

					}

					totalGridGrad += channelGradSum;
					assert.ok( Math.abs( channelGradSum - gradLatents[ c ] ) < 1e-12, `adjoint property holds for channel ${c}` );

				}

				assert.ok( Math.abs( totalGridGrad - 36 ) < 1e-12, 'total scattered gradient matches input sum' );

			} );

			QUnit.test( 'evaluates and backpropagates learned-frame decoder input', ( assert ) => {

				const latents = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 ];
				const rotationWeights = new Array( 96 ).fill( 0 );
				const wi = [ 0, 0, 1 ];
				const wo = [ 0, 0, 1 ];

				const inputRun = forwardDecoderInput( latents, rotationWeights, wi, wo );
				assert.strictEqual( inputRun.output.length, 20, 'produces 20-channel decoder input (8 latents + 2 * 6 frame projections)' );

				const gradOutput = new Array( 20 ).fill( 1 );
				const gradients = backwardDecoderInput( inputRun, gradOutput, rotationWeights );

				assert.strictEqual( gradients.latents.length, 8, 'computes 8 latent gradients' );
				assert.strictEqual( gradients.rotationWeights.length, 96, 'computes 96 rotation weight gradients' );

			} );

			QUnit.test( 'matches the non-negative output clamp in backpropagation', ( assert ) => {

				const outputLayer = {
					inputSize: 20,
					outputSize: 3,
					activation: 'linear',
					weights: new Array( 60 ).fill( 0 ),
					biases: [ - 1, 1, 0 ],
					mWeights: new Array( 60 ).fill( 0 ),
					vWeights: new Array( 60 ).fill( 0 ),
					mBiases: new Array( 3 ).fill( 0 ),
					vBiases: new Array( 3 ).fill( 0 )
				};
				const model = {
					decoder: { layers: [ outputLayer ] },
					rotationWeights: new Array( 96 ).fill( 0 ),
					rotationGrad: new Array( 96 ).fill( 0 ),
					rotationM: new Array( 96 ).fill( 0 ),
					rotationV: new Array( 96 ).fill( 0 ),
					latentGrids: [ {
						width: 1,
						height: 1,
						data: new Array( 8 ).fill( 0 ),
						grad: new Array( 8 ).fill( 0 ),
						m: new Array( 8 ).fill( 0 ),
						v: new Array( 8 ).fill( 0 )
					} ]
				};

				trainBatch( model, [ {
					uv: [ 0.5, 0.5 ],
					wi: [ 0, 0, 1 ],
					wo: [ 0, 0, 1 ],
					mip: 0,
					target: [ 1, 0, 1 ],
					weight: 1
				} ], null, 0.1, 1, 100 );

				assert.strictEqual( outputLayer.biases[ 0 ], - 1, 'does not send gradients through a clamped negative output' );
				assert.ok( outputLayer.biases[ 1 ] < 1, 'still updates an active positive output' );
				assert.strictEqual( outputLayer.biases[ 2 ], 0, 'uses the zero derivative at the clamp boundary' );

			} );

			QUnit.test( 'clips model gradients and asserts finite parameters', ( assert ) => {

				const model = {
					decoder: {
						layers: [ {
							weights: [ 0, 0 ],
							biases: [ 0 ],
							gradWeights: [ 100, 100 ],
							gradBiases: [ 100 ]
						} ]
					},
					rotationWeights: [ 0, 0 ],
					rotationGrad: [ 100, 100 ],
					latentGrids: [ {
						width: 1,
						height: 1,
						data: [ 0, 0 ],
						grad: [ 100, 100 ]
					} ]
				};

				clipModelGradients( model, 1.0 );
				assertModelFinite( model );

				const totalSquared = model.rotationGrad.reduce( ( s, v ) => s + v * v, 0 ) +
					model.latentGrids[ 0 ].grad.reduce( ( s, v ) => s + v * v, 0 ) +
					model.decoder.layers[ 0 ].gradWeights.reduce( ( s, v ) => s + v * v, 0 ) +
					model.decoder.layers[ 0 ].gradBiases.reduce( ( s, v ) => s + v * v, 0 );

				assert.ok( Math.abs( Math.sqrt( totalSquared ) - 1.0 ) < 1e-6, 'scales gradients to specified max norm' );

			} );

		} );

	} );

} );
