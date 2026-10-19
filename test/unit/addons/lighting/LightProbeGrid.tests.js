import { Scene } from 'three';
import { cameraNear, cameraFar } from 'three/tsl';
import { LightProbeGrid } from '../../../../examples/jsm/lighting/LightProbeGrid.js';

export default QUnit.module( 'Lighting', () => {

	QUnit.module( 'LightProbeGrid', () => {

		QUnit.test( 'bake updates capture near/far uniforms when reusing the camera', ( assert ) => {

			const grid = new LightProbeGrid( 1, 1, 1, 1, 1, 1 );
			const scene = new Scene();
			let near, far;
			let captures = 0;

			// Observe the camera passed to the renderer without submitting GPU work.
			const renderer = {
				isWebGPURenderer: true,
				initialized: true,
				inspector: { enabled: false },
				xr: { enabled: false },
				autoClear: true,
				getRenderTarget: () => null,
				getActiveCubeFace: () => 0,
				getActiveMipmapLevel: () => 0,
				setRenderTarget: () => {},
				render: ( object, camera ) => {

					if ( camera.isArrayCamera !== true ) return;

					captures ++;
					cameraNear.update( { camera } );
					cameraFar.update( { camera } );
					assert.strictEqual( cameraNear.value, near, 'cameraNear uses the bake near plane' );
					assert.strictEqual( cameraFar.value, far, 'cameraFar uses the bake far plane' );
					assert.ok( camera.cameras.every( ( face ) => face.near === near && face.far === far ), 'Face cameras match the uniforms' );

				}
			};

			try {

				near = 0.2; far = 50;
				grid.bake( renderer, scene, { near, far } );
				near = 0.4; far = 75;
				grid.bake( renderer, scene, { near, far } );
				near = 0.1; far = 100;
				grid.bake( renderer, scene );
				assert.strictEqual( captures, 3, 'Checked initial, changed and default bake settings' );

			} finally {

				grid.dispose();

			}

		} );

	} );

} );
