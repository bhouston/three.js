import {
	Bone,
	BoxGeometry,
	Color,
	CylinderGeometry,
	Euler,
	Matrix4,
	Mesh,
	MeshBasicMaterial,
	Object3D,
	Quaternion,
	SphereGeometry,
	Vector3
} from 'three';

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

class BehaviorNode {

	constructor( index, definition, inputs ) {

		this.index = index;
		this.definition = definition;
		this.inputs = inputs;

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
		this.behaviorNodes = [];

	}

}

const BEHAVIOR_TYPES = {
	EXECUTION: 'execute',
	NUMBER: 'number',
	VEC3: "vec3",
	BOOLEAN: 'boolean',
	STRING: 'string'
};

const BEHAVIOR_NODE_DEFINITIONS = [

	new BehaviorNodeDefinition( 'trigger', 'sceneStart', [], [ { name: 'execution', type: BEHAVIOR_TYPES.EXECUTION } ] ),
	new BehaviorNodeDefinition( 'trigger', 'tick', [], [ { name: 'execution', type: BEHAVIOR_TYPES.EXECUTION } ] ),
	new BehaviorNodeDefinition( 'trigger', 'nodeClick', [], [ { name: 'execution', type: BEHAVIOR_TYPES.EXECUTION }, { name: 'node', type: BEHAVIOR_TYPES.NUMBER } ] ),

	new BehaviorNodeDefinition( 'action', 'debugOutput', [ { name: 'execution', type: BEHAVIOR_TYPES.EXECUTION }, { name: 'text', type: BEHAVIOR_TYPES.STRING } ], [],
		( context, inputs ) => {

			console.log( 'Debug Output: ' + inputs[ 'text' ] );

		}
	),

	new BehaviorNodeDefinition( 'action', 'show', [ { name: 'execution', type: BEHAVIOR_TYPES.EXECUTION }, { name: 'node', type: BEHAVIOR_TYPES.NUMBER } ], [],
		( context, inputs ) => {

			const node = context.getSceneNodeByIndex( inputs[ 'node' ] );
			node.visible = false;

		}
	),
	new BehaviorNodeDefinition( 'action', 'hide', [ { name: 'execution', type: BEHAVIOR_TYPES.EXECUTION }, { name: 'node', type: BEHAVIOR_TYPES.NUMBER } ], [],
		( context, inputs ) => {

			const node = context.getSceneNodeByIndex( inputs[ 'node' ] );
			node.visible = true;

		}
	),
	new BehaviorNodeDefinition( 'action', 'translate', [ { name: 'execution', type: BEHAVIOR_TYPES.EXECUTION }, { name: 'node', type: BEHAVIOR_TYPES.NUMBER }, { name: 'offset', type: BEHAVIOR_TYPES.VEC3 } ], [],
		( context, inputs ) => {

			const node = context.getSceneNodeByIndex( inputs[ 'node' ] );
			node.translation.add( inputs[ 'offset' ] );

		}
	),
	new BehaviorNodeDefinition( 'action', 'rotation', [ { name: 'execution', type: BEHAVIOR_TYPES.EXECUTION }, { name: 'node', type: BEHAVIOR_TYPES.NUMBER }, { name: 'eulerDelta', type: BEHAVIOR_TYPES.VEC3 } ], [],
		( context, inputs ) => {

			const node = context.getSceneNodeByIndex( inputs[ 'node' ] );
			node.rotation.add( inputs[ 'eulerDelta' ] );

		}
	),
	new BehaviorNodeDefinition( 'action', 'scale', [ { name: 'execution', type: BEHAVIOR_TYPES.EXECUTION }, { name: 'node', type: BEHAVIOR_TYPES.NUMBER }, { name: 'factor', type: BEHAVIOR_TYPES.VEC3 } ], [],
		( context, inputs ) => {

			const node = context.getSceneNodeByIndex( inputs[ 'node' ] );
			node.scale.multiplyByVector( inputs[ 'factor' ] );

		}
	),
];

class BehaviorParser {

	constructor() {

		this.behavior = new Behavior();

	}

	parse( json ) {

		const nodesJson = json;

		for ( let i = 0; i < nodesJson.length; i ++ ) {

			const nodeJson = nodesJson[ i ];
			const nodeType = nodeJson[ 'type' ];
			const nodeName = nodeJson[ 'name' ];
			const definition = BEHAVIOR_NODE_DEFINITIONS.findIndex( ( item ) => ( item.type === nodeType && item.name === nodeName ) );
			this.behavior.behaviorNodes.push( new BehaviorNode( i, definition, nodeJson[ 'inputs ' ] ) );

		}

	}

}
