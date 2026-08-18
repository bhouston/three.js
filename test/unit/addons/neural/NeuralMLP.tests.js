import {
	createMLP,
	forwardMLP,
	activate,
	sigmoid,
	powerLog
} from '../../../../examples/jsm/neural/NeuralMLP.js';

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'Neural', () => {

		QUnit.module( 'NeuralMLP', () => {

			QUnit.test( 'evaluates activation functions', ( assert ) => {

				assert.strictEqual( activate( 2.5, 'relu' ), 2.5, 'relu positive' );
				assert.strictEqual( activate( - 1.5, 'relu' ), 0, 'relu negative' );

				assert.strictEqual( activate( 2.0, 'linear' ), 2.0, 'linear activation' );

				assert.ok( Math.abs( sigmoid( 0 ) - 0.5 ) < 1e-6, 'sigmoid at zero' );
				assert.strictEqual( powerLog( 1, 3 ), 0, 'power log at 1' );

			} );

			QUnit.test( 'performs forward propagation', ( assert ) => {

				const random = () => 0.75;
				const mlp = createMLP( 2, [ 3 ], 1, random, 'relu', 'linear' );

				const input = [ 1, 0.5 ];
				const run = forwardMLP( mlp, input );

				assert.strictEqual( run.activations.length, 3, 'tracks activations for input, hidden, output' );
				assert.strictEqual( run.output.length, 1, 'produces output of specified size' );

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
