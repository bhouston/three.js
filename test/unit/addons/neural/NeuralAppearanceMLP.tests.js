import {
	createMLP,
	forwardMLP,
	backwardMLP,
	zeroGradients,
	applyAdam,
	activate,
	activateDerivative,
	sigmoid,
	binaryCrossEntropy,
	powerLog
} from '../../../../examples/jsm/neural/NeuralAppearanceMLP.js';

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'Neural', () => {

		QUnit.module( 'NeuralAppearanceMLP', () => {

			QUnit.test( 'evaluates activation functions and derivatives', ( assert ) => {

				assert.strictEqual( activate( 2.5, 'relu' ), 2.5, 'relu positive' );
				assert.strictEqual( activate( - 1.5, 'relu' ), 0, 'relu negative' );
				assert.strictEqual( activateDerivative( 2.5, 'relu' ), 1, 'relu derivative positive' );
				assert.strictEqual( activateDerivative( - 1.5, 'relu' ), 0, 'relu derivative negative' );

				assert.strictEqual( activate( 2.0, 'linear' ), 2.0, 'linear activation' );
				assert.strictEqual( activateDerivative( 2.0, 'linear' ), 1, 'linear derivative' );

				assert.ok( Math.abs( sigmoid( 0 ) - 0.5 ) < 1e-6, 'sigmoid at zero' );
				assert.ok( Math.abs( binaryCrossEntropy( 0.5, 1 ) - ( - Math.log( 0.5 ) ) ) < 1e-6, 'binary cross entropy' );
				assert.strictEqual( powerLog( 1, 3 ), 0, 'power log at 1' );

			} );

			QUnit.test( 'performs forward and backward propagation', ( assert ) => {

				const random = () => 0.75;
				const mlp = createMLP( 2, [ 3 ], 1, random, 'relu', 'linear' );

				const input = [ 1, 0.5 ];
				const run = forwardMLP( mlp, input );

				assert.strictEqual( run.activations.length, 3, 'tracks activations for input, hidden, output' );
				assert.strictEqual( run.output.length, 1, 'produces output of specified size' );

				const gradOutput = [ 1.0 ];
				const gradInput = backwardMLP( mlp, run, gradOutput );

				assert.strictEqual( gradInput.length, 2, 'computes input gradients' );
				assert.ok( mlp.layers[ 0 ].gradWeights.some( ( v ) => v !== 0 ), 'accumulates layer weight gradients' );
				assert.ok( mlp.layers[ 0 ].gradBiases.some( ( v ) => v !== 0 ), 'accumulates layer bias gradients' );

				zeroGradients( mlp );
				assert.ok( mlp.layers[ 0 ].gradWeights.every( ( v ) => v === 0 ), 'zeros layer weight gradients' );
				assert.ok( mlp.layers[ 0 ].gradBiases.every( ( v ) => v === 0 ), 'zeros layer bias gradients' );

			} );

			QUnit.test( 'applies Adam optimizer updates to layer weights and biases', ( assert ) => {

				const random = () => 0.5;
				const mlp = createMLP( 2, [], 1, random, 'linear', 'linear' );
				const initialWeight = mlp.layers[ 0 ].weights[ 0 ];

				mlp.layers[ 0 ].gradWeights[ 0 ] = 0.5;
				mlp.layers[ 0 ].gradBiases[ 0 ] = 0.5;

				applyAdam( mlp, 0.01, 1 );

				assert.notStrictEqual( mlp.layers[ 0 ].weights[ 0 ], initialWeight, 'updates weights with Adam step' );
				assert.notStrictEqual( mlp.layers[ 0 ].biases[ 0 ], 0, 'updates biases with Adam step' );

			} );

			QUnit.test( 'initializes a linear RGB head with a shared gray bias', ( assert ) => {

				const random = () => 0.75;
				const mlp = createMLP( 32, [ 32 ], 3, random, 'relu', 'linear' );
				const hiddenLayer = mlp.layers[ 0 ];
				const outputLayer = mlp.layers[ 1 ];
				const heScale = Math.sqrt( 2 / 32 );
				const rgbScale = heScale * 0.45;

				assert.deepEqual( outputLayer.biases, [ 0.3, 0.3, 0.3 ], 'starts RGB biases at a shared positive gray' );
				assert.ok( hiddenLayer.weights.every( ( weight ) => Math.abs( weight ) <= heScale + 1e-12 ), 'keeps He scale on hidden ReLU weights' );
				assert.ok( outputLayer.weights.every( ( weight ) => Math.abs( weight ) <= rgbScale + 1e-12 ), 'uses a moderate RGB head scale so coarse color can move without a loud random field' );
				assert.ok( outputLayer.weights.some( ( weight ) => Math.abs( weight ) > rgbScale * 0.4 ), 'still uses nonzero RGB weights for a little hue variation' );

			} );

		} );

	} );

} );
