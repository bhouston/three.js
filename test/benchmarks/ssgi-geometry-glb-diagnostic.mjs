// CPU-only reproduction of the interior benchmark cameras and geometry coverage.
import { readFile } from 'node:fs/promises';
import { URL } from 'node:url';
import console from 'node:console';
import { Box3, Vector3, WebGPUCoordinateSystem } from '../../src/Three.WebGPU.js';
import { GLTFLoader } from '../../examples/jsm/loaders/GLTFLoader.js';
import { ssgiGeometryReference } from './ssgi-geometry-reference.js';

const threeModule = new URL( '../../src/Three.WebGPU.js', import.meta.url ).href;
for ( const name of [ 'cornell-box', 'dual-rooms' ] ) {

	const data = await readFile( new URL( `../../examples/models/gltf/${ name }.glb`, import.meta.url ) );
	const gltf = await new GLTFLoader().parseAsync( data.buffer.slice( data.byteOffset, data.byteOffset + data.byteLength ), '' );
	const model = gltf.scene;
	model.updateMatrixWorld( true );
	const camera = gltf.cameras[ 0 ];
	camera.aspect = 4 / 3;
	camera.coordinateSystem = WebGPUCoordinateSystem;
	camera.updateProjectionMatrix();
	camera.updateMatrixWorld( true );
	const radius = new Box3().setFromObject( model ).getSize( new Vector3() ).length() * 0.25;
	const result = await ssgiGeometryReference( { model, camera, width: 960, height: 720, radius, threeModule } );
	const valid = result.valid.filter( Boolean ).length;
	console.log( JSON.stringify( {
		model: name, radius, valid,
		meanTrueCoverage: result.coverage.reduce( ( sum, value ) => sum + value, 0 ) / Math.max( 1, valid ),
		maxTrueCoverage: Math.max( ...result.coverage ),
		milliseconds: result.milliseconds,
		roomSide: model.getObjectByName( 'room' )?.material.side,
		method: result.method
	} ) );

}
