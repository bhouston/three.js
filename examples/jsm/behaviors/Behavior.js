
const BEHAVIOR_TYPES = {
	EXECUTION: 'execute',
	NUMBER: 'number',
	VEC3: 'vec3',
	BOOLEAN: 'boolean',
	STRING: 'string'
};

// structure for defining BehaviorNodes
class BehaviorNodeDefinition {

	constructor( type, name, inputDefinitions, outputDefinitions, func ) {

		this.type = type;
		this.name = name;
		this.inputDefinitions = inputDefinitions;
		this.outputDefinitions = outputDefinitions;
		this.func = func; // optional promise to support async operation.

	}

}

const BEHAVIOR_NODE_DEFINITIONS = [

	// TRIGGERS

	new BehaviorNodeDefinition(
		'trigger/sceneStart',
		[],
		[ { name: 'execution', type: BEHAVIOR_TYPES.EXECUTION } ],
		( context, inputs ) => {
			return [ { name: 'execution' } ];
		}
	),
	new BehaviorNodeDefinition(
		'trigger/tick',
		[],
		[ { name: 'execution', type: BEHAVIOR_TYPES.EXECUTION } ]
	),
	new BehaviorNodeDefinition(
		'trigger/nodeClick',
		[],
		[ { name: 'execution', type: BEHAVIOR_TYPES.EXECUTION }, { name: 'node', type: BEHAVIOR_TYPES.NUMBER } ]
	),

	// LOGIC

	new BehaviorNodeDefinition(
		'logic/if',
		[ { name: 'execution', type: BEHAVIOR_TYPES.EXECUTION }, { name: 'condition', type: BEHAVIOR_TYPES.BOOLEAN } ],
		[ { name: 'true', type: BEHAVIOR_TYPES.EXECUTION },  { name: 'false', type: BEHAVIOR_TYPES.EXECUTION } ],
		( context, inputs ) => {

			return inputs[ 'condition' ].value ? [ { name: 'true' } ] : [ { name: 'false' } ];

		}
	),


	// ASYNC

	new BehaviorNodeDefinition(
		'logic/sleep',
		[ { name: 'execution', type: BEHAVIOR_TYPES.EXECUTION }, { name: 'milliseconds', type: BEHAVIOR_TYPES.NUMBER } ],
		[ { name: 'execution', type: BEHAVIOR_TYPES.EXECUTION } ],
		( context, inputs ) => {

			return null; // TODO: return a promise that results with an async delay

		}
	),

	// MATH

	new BehaviorNodeDefinition(
		'math/random',
		[ { name: 'execution', type: BEHAVIOR_TYPES.EXECUTION } ],
		[ { name: 'value', type: BEHAVIOR_TYPES.NUMBER } ],
		( context, inputs ) => {

			return [ { value: Math.random() } ];

		}
	),

	// ACTIONS

	new BehaviorNodeDefinition(
		'action/debugOutput',
		[ { name: 'execution', type: BEHAVIOR_TYPES.EXECUTION }, { name: 'text', type: BEHAVIOR_TYPES.STRING } ],
		[ { name: 'execution', type: BEHAVIOR_TYPES.EXECUTION } ],
		( context, inputs ) => {

			console.log( 'Debug Output: ' + inputs[ 'text' ] );

		}
	),
	new BehaviorNodeDefinition(
		'action/show',
		[ { name: 'execution', type: BEHAVIOR_TYPES.EXECUTION }, { name: 'node', type: BEHAVIOR_TYPES.NUMBER } ],
		[ { name: 'execution', type: BEHAVIOR_TYPES.EXECUTION } ],
		( context, inputs ) => {

			const node = context.getSceneNodeByIndex( inputs[ 'node' ] );
			node.visible = false;

		}
	),
	new BehaviorNodeDefinition(
		'action/hide',
		[ { name: 'execution', type: BEHAVIOR_TYPES.EXECUTION }, { name: 'node', type: BEHAVIOR_TYPES.NUMBER } ],
		[ { name: 'execution', type: BEHAVIOR_TYPES.EXECUTION } ],
		( context, inputs ) => {

			const node = context.getSceneNodeByIndex( inputs[ 'node' ] );
			node
			.visible = true;

		}
	),
	new BehaviorNodeDefinition(
		'action/translate',
		[ { name: 'execution', type: BEHAVIOR_TYPES.EXECUTION }, { name: 'node', type: BEHAVIOR_TYPES.NUMBER }, { name: 'offset', type: BEHAVIOR_TYPES.VEC3 } ],
		[ { name: 'execution', type: BEHAVIOR_TYPES.EXECUTION } ],
		( context, inputs ) => {

			const node = context.getSceneNodeByIndex( inputs[ 'node' ] );
			node.translation.add( inputs[ 'offset' ] );

		}
	),
	new BehaviorNodeDefinition(
		'action/rotation',
		[ { name: 'execution', type: BEHAVIOR_TYPES.EXECUTION }, { name: 'node', type: BEHAVIOR_TYPES.NUMBER }, { name: 'eulerDelta', type: BEHAVIOR_TYPES.VEC3 } ],
		[ { name: 'execution', type: BEHAVIOR_TYPES.EXECUTION } ],
		( context, inputs ) => {

			const node = context.getSceneNodeByIndex( inputs[ 'node' ] );
			node.rotation.add( inputs[ 'eulerDelta' ] );

		}
	),
	new BehaviorNodeDefinition(
		'action/scale',
	 	[ { name: 'execution', type: BEHAVIOR_TYPES.EXECUTION }, { name: 'node', type: BEHAVIOR_TYPES.NUMBER }, { name: 'factor', type: BEHAVIOR_TYPES.VEC3 } ],
		[ { name: 'execution', type: BEHAVIOR_TYPES.EXECUTION } ],
		( context, inputs ) => {

			const node = context.getSceneNodeByIndex( inputs[ 'node' ] );
			node.scale.multiplyByVector( inputs[ 'factor' ] );

		}
	),
];

// sort in alphabetical order
BEHAVIOR_NODE_DEFINITIONS.sort( ( a, b ) => ( a.type.localCompare( b.type) ) );

class BehaviorNode {

	constructor( index, definition, inputs, func ) {

		this.index = index;
		this.definition = definition;
		this.inputs = inputs;
		this.func = func;
		this.outputs = {};
		this.definition.outputs.forEach( ( value ) => {
			this.outputs[value] = {
				name: value,
				downlinks: []
			};
		})

	}



}
class BehaviorContext {

	constructor( scene, indexToNodeMap ) {

		this.scene = scene;
		this.indexToNodeMap = indexToNodeMap;

	}
}

class Behavior {

	constructor() {

		this.name = '';
		this.nodes = []; // contains BehaviorNodes
		this.workQueue = []; // contains the next BehaviorNodes to process

	}

	trigger( triggerName ) {

		// look up any nodes with this trigger name and add them to the executionQueue
		const triggerNodes = this.nodes.filter( ( item ) => ( item.definition.type === triggerName ) );

		if( triggerNodes.length > 0 ) {
			// add to the back of the queue
			this.workQueue.push( triggerNodes );
		}

		// inform how many trigger nodes were triggered
		return triggerNodes.length;

	}

	prioritizeNode( node ) {

		// remove from the queue if it is exists
		this.workQueue = this.workQueue.filter( ( item ) => ( item !== node ) );

		// add to front of queue
		this.workQueue.unshift( node );

	}

	// resolve non-execution inputs so that each has a value stored in them.  Then and only then we can execute the node's function.
	resolveInputs( node ) {

		let unresolvedInputs = 0;

		node.inputs.forEach( ( inputName, index ) => {

			const inputDefinition = node.definition.inputs.find( ( item ) => { item.name === inputName } );
			const input = node.inputs[ inputName ];

			// no need to resolve execution inputs.
			if( inputDefinition.type === BEHAVIOR_TYPES.EXECUTION ) {

				continue;

			}

			// if the input has a value, it is resolved
			if( input.value === undefined ) {

				var sourceNode = this.nodes[ node.definition.inputs.node ];
				this.scheduleNode( sourceNode );
				unresolvedInputs ++;

				return;

			}

		} );

		return unresolvedInputs;

	}

	// returns the number of new execution steps created as a result of this one step
	executeStep() {

		// no work waiting!
		if( this.workQeueue.length === 0 ) {
			return 0;
		}

		// look at the next item in the queue
		const peekNextItem = this.workQueue[0];

		// resolve inputs if they are not.  If all are resolved, function returns 0, and we can execute it.
		if( this.resolveInputs( peekNextItem ) > 0 ) {

			return this.executeStep();

		}

		// pop off item
		const nextItem = this.workQueue.shift();
		if( peekNextItem !== nextItem ) {
			throw new Error( 'should not happen' );
		}

		// collect all inputs, while clearing their values.
		let inputValues = {};
		node.inputs.forEach( ( inputName, index ) => {

			const input = node.inputs[ inputName ];

			if( input.value !== undefined ) {

				inputValues[ input.name ] = input.value;
				delete input.value;

			}

		});

		// this is where the promise would be;
		const result = nextItem.func( inputValues );

		// push results to the inputs of downstream nodes.
		result.outputs.forEach( ( output, outputIndex ) => {

			if( output.downlinks !== undefined ) {

				output.downlinks.forEach( ( downLinks ) => {

					var downlinkNode = this.nodes[ downlinks.nodeIndex ];

					if( output.value !== undefined ) {
						downlinkNode.inputs[ downlinkNode.inputName ].value = output.value;
					}
					else {
						if( nextItem.definition.outputs[ output.name ] !== BEHAVIOR_TYPES.EXECUTION ) {
							throw new Error( "outputs without values must be execution" );
						}

						this.workQueue.push( downlinkNode );
					}

				});

			}

		});

		return 1;
	}

	executeSteps( maximumSteps ) {
		let numberOfStepsExecuted = 0;
		while( maximumSteps > 0 ) {

			const workDone = this.internal_executeStep();
			if( ! workDone ) {

				if( this.triggerQueue.length ) {

				}

			}

			numberOfStepsExecuted ++;
		}

		return numberOfStepsExecuted;
	}

}

class BehaviorParser {

	constructor() {

		this.behavior = new Behavior();

	}

	parse( json ) {

		const nodesJson = json;

		// create new BehaviorNode instances for each node in the json.
		for ( let i = 0; i < nodesJson.length; i ++ ) {

			const nodeJson = nodesJson[ i ];
			const nodeType = nodeJson[ 'type' ];
			const definitions = BEHAVIOR_NODE_DEFINITIONS.filter( ( item ) => ( item.type === nodeType ) );

			if ( definitions.length <= 0 ) {

				throw new Error( `Can not find Behavior Node Definition for ${nodeType}` );

			}
			if ( definitions.length > 1 ) {

				throw new Error( `Too many matching Behavior Node Definition for ${nodeType}` );

			}

			this.behavior.nodes.push( new BehaviorNode( i, definitions[ 0 ], nodeJson[ 'inputs' ] );

		}

		// connect up the graph edges from BehaviorNode inputs to outputs.  This is required to follow execution
		this.behavior.nodes.forEach( ( node ) => {
			// initialize the inputs by resolving to the reference nodes.
			node.inputs.forEach( ( inputName, index ) => {
				const input = node.inputs[ inputName ];

				if( input['type'] === 'link' ) {
					const uplinkNode = this.behavior.nodes[ input[ 'node' ] ];
					const uplinkOutput = uplink.outputs[ input[ 'output' ] ];
					if( ! uplinkOutput.downlinks ) {
						uplinkOutput.downlinks = [];
					}
					uplinkOutput.downlinks.push( { node: value[ 'node' ], input: input.name })
				}
			});

		})
	}

}

const behaviorExample = [
	{
		'type': 'trigger/sceneStart'
	},
	{
		'type': 'action/debugOutput',
		'inputs': {
			'execute': { 'type': 'uplink', 'nodeIndex': 0, 'outputName': 'execute' },
			'text': { 'type': 'constant', 'value': 'Hello World!' }
		}
	}
];

class BehaviorTest {

	constructor() {
		this.parser = new BehaviorParser();
		this.behavior = parser.behavior;
	}

	load( json ) {
		parser.parse( json );
	}

	init() {
		behavior.trigger( "trigger/sceneStart" );
	}

	tick() {
		behavior.trigger( "trigger/tick" );
	}

	executeSteps( maximumSteps ) {
		this.behavior.executeSteps( maximumSteps );
	}

}

export { CinematicCamera };

