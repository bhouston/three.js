import { describe, test, expect } from 'vitest';
import {
	AmbientLight,
	PerspectiveCamera,
	PointLight,
	Scene,
} from 'three';
import { WebGLNodesHandler } from '../../../../examples/jsm/tsl/WebGLNodesHandler.js';

describe( 'Addons', () => {

	describe( 'TSL', () => {

		describe( 'WebGLNodesHandler', () => {

			test( 'nested render lifecycle', () => {

				const handler = new WebGLNodesHandler();
				handler.setRenderer( {
					extensions: {},
					getContext: () => ( {} ),
				} );
				expect( handler.nodeFrame.renderer ).toBe( handler.renderer );

				const scene = new Scene();
				const outerCamera = new PerspectiveCamera();
				const nestedCamera = new PerspectiveCamera();
				const outerLight = new AmbientLight();
				const nestedLight = new PointLight();

				handler.renderStart( scene, outerCamera );
				handler.updateLights( [ outerLight ] );
				handler.renderStart( scene, nestedCamera );
				handler.updateLights( [ nestedLight ] );
				handler.renderEnd();

				const sceneContext = handler.renderStack[ 0 ].sceneContext;
				expect( sceneContext.lightsNode.getLights() ).toEqual( [ outerLight ] );
				expect( handler.nodeFrame.camera ).toBe( outerCamera );

				handler.renderEnd();
				expect( handler.renderStack.length ).toBe( 0 );

			} );

		} );

	} );

} );
