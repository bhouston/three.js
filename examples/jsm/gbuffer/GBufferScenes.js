import * as THREE from 'three/webgpu';
import { GLTFLoader } from '../loaders/GLTFLoader.js';
import { DRACOLoader } from '../loaders/DRACOLoader.js';
import { RoundedBoxGeometry } from '../geometries/RoundedBoxGeometry.js';

// Keep the diagnostic geometry and materials aligned with the standalone examples.
function createCornell( scene ) {

	// Cornell box

	const wallMaterial = new THREE.MeshStandardMaterial( { color: 0xcccccc } );
	const redMaterial = new THREE.MeshStandardMaterial( { color: 0xff0000 } );
	const greenMaterial = new THREE.MeshStandardMaterial( { color: 0x00ff00 } );

	// Floor
	const floor = new THREE.Mesh( new THREE.PlaneGeometry( 6, 6 ), wallMaterial );
	floor.rotation.x = - Math.PI / 2;
	floor.receiveShadow = true;
	scene.add( floor );

	// Ceiling
	const ceiling = new THREE.Mesh( new THREE.PlaneGeometry( 6, 6 ), wallMaterial );
	ceiling.rotation.x = Math.PI / 2;
	ceiling.position.y = 5;
	ceiling.receiveShadow = true;
	scene.add( ceiling );

	// Back wall
	const backWall = new THREE.Mesh( new THREE.PlaneGeometry( 6, 5 ), wallMaterial );
	backWall.position.set( 0, 2.5, - 3 );
	backWall.receiveShadow = true;
	scene.add( backWall );

	// Front wall
	const frontWall = new THREE.Mesh( new THREE.PlaneGeometry( 6, 5 ), wallMaterial );
	frontWall.rotation.y = Math.PI;
	frontWall.position.set( 0, 2.5, 3 );
	frontWall.receiveShadow = true;
	scene.add( frontWall );

	// Left wall (red)
	const leftWall = new THREE.Mesh( new THREE.PlaneGeometry( 6, 5 ), redMaterial );
	leftWall.rotation.y = Math.PI / 2;
	leftWall.position.set( - 3, 2.5, 0 );
	leftWall.receiveShadow = true;
	scene.add( leftWall );

	// Right wall (green)
	const rightWall = new THREE.Mesh( new THREE.PlaneGeometry( 6, 5 ), greenMaterial );
	rightWall.rotation.y = - Math.PI / 2;
	rightWall.position.set( 3, 2.5, 0 );
	rightWall.receiveShadow = true;
	scene.add( rightWall );

	// Objects inside the box

	const objectMaterial = new THREE.MeshStandardMaterial( { color: 0xeeeeee } );

	// Tall box
	const tallBox = new THREE.Mesh( new THREE.BoxGeometry( 1.2, 2.5, 1.2 ), objectMaterial );
	tallBox.position.set( - 0.8, 1.25, - 0.8 );
	tallBox.rotation.y = Math.PI / 8;
	tallBox.castShadow = true;
	tallBox.receiveShadow = true;
	scene.add( tallBox );

	// Short box
	const shortBox = new THREE.Mesh( new THREE.BoxGeometry( 1.2, 1.2, 1.2 ), objectMaterial );
	shortBox.position.set( 1, 0.6, 0.5 );
	shortBox.rotation.y = - Math.PI / 6;
	shortBox.castShadow = true;
	shortBox.receiveShadow = true;
	scene.add( shortBox );

	// Sphere (to show smooth GI variation)
	const sphere = new THREE.Mesh( new THREE.SphereGeometry( 0.5, 32, 32 ), objectMaterial.clone() );
	sphere.position.set( 1, 1.9, 0.5 );
	sphere.castShadow = true;
	sphere.receiveShadow = true;
	scene.add( sphere );

	// Light
	const light = new THREE.PointLight( 0xffffff, 40 );
	light.position.set( 0, 4.5, 0 );
	light.castShadow = true;
	light.shadow.mapSize.setScalar( 256 );
	light.shadow.radius = 10;
	light.shadow.normalBias = - 0.02;
	scene.add( light );


	// Two roughness levels make SSR and probe reflections easy to compare.
	sphere.material.metalness = 1;
	sphere.material.roughness = 0.18;
	const roughSphere = new THREE.Mesh( new THREE.SphereGeometry( 0.45, 32, 32 ), new THREE.MeshStandardMaterial( { color: 0xe9b96e, metalness: 1, roughness: 0.45 } ) );
	roughSphere.position.set( - 1.5, 0.45, 1.1 );
	scene.add( roughSphere );

}

function createTwoRooms( scene ) {

	// Materials

	const wallMaterial = new THREE.MeshStandardMaterial( { color: 0xcccccc, side: THREE.BackSide } );
	const whiteMat = new THREE.MeshStandardMaterial( { color: 0xeeeeee } );

	// Room shell: x=-8 to 8, z=-4 to 4, y=0 to 5

	const room = new THREE.Mesh( new THREE.BoxGeometry( 16, 5, 8 ), wallMaterial );
	room.position.set( 0, 2.5, 0 );
	room.receiveShadow = true;
	scene.add( room );

	const redWall = new THREE.Mesh(
		new THREE.PlaneGeometry( 8, 5 ),
		new THREE.MeshStandardMaterial( { color: 0xdd2200 } )
	);
	redWall.rotation.y = Math.PI / 2;
	redWall.position.set( - 7.99, 2.5, 0 );
	scene.add( redWall );

	const blueWall = new THREE.Mesh(
		new THREE.PlaneGeometry( 8, 5 ),
		new THREE.MeshStandardMaterial( { color: 0x0044ff } )
	);
	blueWall.rotation.y = - Math.PI / 2;
	blueWall.position.set( 7.99, 2.5, 0 );
	scene.add( blueWall );

	// Dividing wall at x=0 with doorway

	const dividerMat = new THREE.MeshStandardMaterial( { color: 0xcccccc } );
	const doorwayHalfGap = 1.25;

	// Left section of divider (z = -4 to -1.25)
	const dividerLeft = new THREE.Mesh(
		new THREE.BoxGeometry( 0.3, 5, 4 - doorwayHalfGap ),
		dividerMat
	);
	dividerLeft.position.set( 0, 2.5, - ( doorwayHalfGap + ( 4 - doorwayHalfGap ) / 2 ) );
	dividerLeft.castShadow = true;
	dividerLeft.receiveShadow = true;
	scene.add( dividerLeft );

	// Right section of divider (z = 1.25 to 4)
	const dividerRight = new THREE.Mesh(
		new THREE.BoxGeometry( 0.3, 5, 4 - doorwayHalfGap ),
		dividerMat
	);
	dividerRight.position.set( 0, 2.5, doorwayHalfGap + ( 4 - doorwayHalfGap ) / 2 );
	dividerRight.castShadow = true;
	dividerRight.receiveShadow = true;
	scene.add( dividerRight );

	// Lintel above doorway
	const lintel = new THREE.Mesh(
		new THREE.BoxGeometry( 0.3, 1.2, doorwayHalfGap * 2 ),
		dividerMat
	);
	lintel.position.set( 0, 4.4, 0 );
	lintel.castShadow = true;
	lintel.receiveShadow = true;
	scene.add( lintel );

	// Left room objects

	// Columns
	const columnGeom = new THREE.CylinderGeometry( 0.3, 0.3, 4, 16 );

	for ( const x of [ - 6, - 2 ] ) {

		for ( const z of [ - 2.5, 2.5 ] ) {

			const col = new THREE.Mesh( columnGeom, whiteMat );
			col.position.set( x, 2, z );
			col.castShadow = true;
			col.receiveShadow = true;
			scene.add( col );

		}

	}

	// Table with golden sphere
	const tableTop = new THREE.Mesh(
		new THREE.BoxGeometry( 2.4, 0.12, 1.4 ),
		new THREE.MeshStandardMaterial( { color: 0x886644 } )
	);
	tableTop.position.set( - 4, 1.0, 0 );
	tableTop.castShadow = true;
	tableTop.receiveShadow = true;
	scene.add( tableTop );

	const legGeom = new THREE.BoxGeometry( 0.1, 1, 0.1 );

	for ( const x of [ - 5.05, - 2.95 ] ) {

		for ( const z of [ - 0.55, 0.55 ] ) {

			const leg = new THREE.Mesh( legGeom, tableTop.material );
			leg.position.set( x, 0.5, z );
			leg.castShadow = true;
			scene.add( leg );

		}

	}

	const sphere = new THREE.Mesh(
		new THREE.SphereGeometry( 0.35, 32, 32 ),
		new THREE.MeshStandardMaterial( { color: 0xffd700, metalness: 0.3, roughness: 0.4 } )
	);
	sphere.position.set( - 4, 1.41, 0 );
	sphere.castShadow = true;
	sphere.receiveShadow = true;
	scene.add( sphere );

	// Stepped blocks near red wall
	const step1 = new THREE.Mesh( new THREE.BoxGeometry( 1.5, 0.5, 1 ), whiteMat );
	step1.position.set( - 7, 0.25, 0 );
	step1.castShadow = true;
	step1.receiveShadow = true;
	scene.add( step1 );

	const step2 = new THREE.Mesh( new THREE.BoxGeometry( 1.0, 1.0, 1 ), whiteMat );
	step2.position.set( - 7, 0.5, 0 );
	step2.castShadow = true;
	step2.receiveShadow = true;
	scene.add( step2 );

	const step3 = new THREE.Mesh( new THREE.BoxGeometry( 0.5, 1.5, 1 ), whiteMat );
	step3.position.set( - 7, 0.75, 0 );
	step3.castShadow = true;
	step3.receiveShadow = true;
	scene.add( step3 );

	// Right room objects

	// Columns
	for ( const x of [ 2, 6 ] ) {

		for ( const z of [ - 2.5, 2.5 ] ) {

			const col = new THREE.Mesh( columnGeom, whiteMat );
			col.position.set( x, 2, z );
			col.castShadow = true;
			col.receiveShadow = true;
			scene.add( col );

		}

	}

	// Pedestal with torus knot
	const pedestal = new THREE.Mesh(
		new THREE.BoxGeometry( 0.8, 1.2, 0.8 ),
		whiteMat
	);
	pedestal.position.set( 4, 0.6, 0 );
	pedestal.castShadow = true;
	pedestal.receiveShadow = true;
	scene.add( pedestal );

	const torusKnot = new THREE.Mesh(
		new THREE.TorusKnotGeometry( 0.3, 0.1, 64, 16 ),
		new THREE.MeshStandardMaterial( { color: 0xff44aa, metalness: 0.2, roughness: 0.5 } )
	);
	torusKnot.position.set( 4, 1.65, 0 );
	torusKnot.castShadow = true;
	torusKnot.receiveShadow = true;
	scene.add( torusKnot );

	// Tall cone sculpture
	const sculpture = new THREE.Mesh(
		new THREE.ConeGeometry( 0.4, 2.5, 5 ),
		whiteMat
	);
	sculpture.position.set( 6.5, 1.25, - 1.5 );
	sculpture.castShadow = true;
	sculpture.receiveShadow = true;
	scene.add( sculpture );

	// Lights

	// Warm point light in left room
	const warmLight = new THREE.PointLight( 0xffaa44, 30 );
	warmLight.position.set( - 4, 4.5, 0 );
	warmLight.castShadow = true;
	warmLight.shadow.mapSize.setScalar( 256 );
	warmLight.shadow.radius = 12;
	warmLight.shadow.bias = - 0.002;
	scene.add( warmLight );

	// Cool point light in right room
	const coolLight = new THREE.PointLight( 0x88bbff, 30 );
	coolLight.position.set( 4, 4.5, 0 );
	coolLight.castShadow = true;
	coolLight.shadow.mapSize.setScalar( 256 );
	coolLight.shadow.radius = 12;
	coolLight.shadow.bias = - 0.002;
	scene.add( coolLight );


}

async function createGallery( scene ) {

	// wall-mounted spotlights

	function addSpotLight( position, targetPosition ) {

		const light = new THREE.SpotLight( 0xffe09e, 40 );
		light.position.copy( position );
		light.angle = 0.6;
		light.penumbra = 1;
		light.distance = 6.5;
		light.decay = 2;
		light.target.position.copy( targetPosition );
		scene.add( light );
		scene.add( light.target );

	}

	addSpotLight( new THREE.Vector3( - 2.5, 5, - 4.8 ), new THREE.Vector3( - 2.5, - 2, - 4.8 ) );
	addSpotLight( new THREE.Vector3( 2.5, 5, - 4.8 ), new THREE.Vector3( 2.5, - 2, - 4.8 ) );
	addSpotLight( new THREE.Vector3( - 5.3, 5, 0 ), new THREE.Vector3( - 5.3, - 2, 0 ) );

	// checkerboard floor

	const tilesX = 16;
	const tilesZ = 16;
	const floorCanvas = document.createElement( 'canvas' );
	floorCanvas.width = tilesX * 32;
	floorCanvas.height = tilesZ * 32;
	const floorCtx = floorCanvas.getContext( '2d' );

	for ( let x = 0; x < tilesX; x ++ ) {

		for ( let z = 0; z < tilesZ; z ++ ) {

			floorCtx.fillStyle = ( x + z ) % 2 === 0 ? '#d8d0c8' : '#b8b0a8';
			floorCtx.fillRect( x * 32, z * 32, 32, 32 );

		}

	}

	const floorTexture = new THREE.CanvasTexture( floorCanvas );
	floorTexture.colorSpace = THREE.SRGBColorSpace;
	floorTexture.wrapS = THREE.RepeatWrapping;
	floorTexture.wrapT = THREE.RepeatWrapping;

	const floorMat = new THREE.MeshStandardMaterial( { map: floorTexture, roughness: 0.7, metalness: 0.05 } );
	const floor = new THREE.Mesh( new THREE.PlaneGeometry( tilesX, tilesZ ), floorMat );
	floor.rotation.x = - Math.PI / 2;
	floor.position.y = - 2;
	scene.add( floor );

	// walls

	const wallMat = new THREE.MeshStandardMaterial( { color: '#e0d8d0', roughness: 0.9, metalness: 0 } );

	const backWall = new THREE.Mesh( new THREE.PlaneGeometry( 16, 10 ), wallMat );
	backWall.position.set( 0, 3, - 5 );
	scene.add( backWall );

	const leftWall = new THREE.Mesh( new THREE.PlaneGeometry( 16, 10 ), wallMat );
	leftWall.rotation.y = Math.PI / 2;
	leftWall.position.set( - 5.5, 3, 0 );
	scene.add( leftWall );

	// central pedestal

	const pedestalMat = new THREE.MeshStandardMaterial( { color: '#f0ece8', roughness: 0.32, metalness: 1 } );

	const pedestalBase = new THREE.Mesh( new THREE.CylinderGeometry( 0.9, 1.0, 0.2, 32 ), pedestalMat );
	pedestalBase.position.set( 0, - 1.9, 0 );
	scene.add( pedestalBase );

	const pedestalShaft = new THREE.Mesh( new THREE.CylinderGeometry( 0.55, 0.65, 1.5, 32 ), pedestalMat );
	pedestalShaft.position.set( 0, - 1.05, 0 );
	scene.add( pedestalShaft );

	const pedestalTop = new THREE.Mesh( new THREE.CylinderGeometry( 0.8, 0.7, 0.25, 32 ), pedestalMat );
	pedestalTop.position.set( 0, - 0.18, 0 );
	scene.add( pedestalTop );

	// torus knot

	const knotMat = new THREE.MeshStandardMaterial( { color: '#c0a060', roughness: 0.28, metalness: 1 } );
	const torusKnot = new THREE.Mesh( new THREE.TorusKnotGeometry( 0.5, 0.17, 128, 32 ), knotMat );
	torusKnot.position.set( 0, 0.82, 0 );
	scene.add( torusKnot );

	// columns

	const columnMat = new THREE.MeshStandardMaterial( { color: '#e8e4de', roughness: 0.5, metalness: 0 } );

	function addColumn( x, z ) {

		const columnGroup = new THREE.Group();

		const base = new THREE.Mesh( new THREE.BoxGeometry( 0.6, 0.2, 0.6 ), columnMat );
		base.position.y = - 1.9;
		columnGroup.add( base );

		const shaft = new THREE.Mesh( new THREE.CylinderGeometry( 0.18, 0.22, 5, 16 ), columnMat );
		shaft.position.y = 0.7;
		columnGroup.add( shaft );

		const capital = new THREE.Mesh( new THREE.BoxGeometry( 0.55, 0.25, 0.55 ), columnMat );
		capital.position.y = 3.35;
		columnGroup.add( capital );

		columnGroup.scale.setScalar( 1.5 );
		columnGroup.position.set( x, 1.0, z );
		scene.add( columnGroup );

	}

	addColumn( - 5.05, - 4.55 );
	addColumn( 4.5, - 4.55 );
	addColumn( - 5.05, 3 );

	// vase on its own pedestal

	const vaseProfile = [
		new THREE.Vector2( 0, 0.5 ),
		new THREE.Vector2( 0.12, 0.5 ),
		new THREE.Vector2( 0.18, 0.7 ),
		new THREE.Vector2( 0.28, 0.9 ),
		new THREE.Vector2( 0.32, 1.0 ),
		new THREE.Vector2( 0.3, 1.1 ),
		new THREE.Vector2( 0.22, 1.15 ),
		new THREE.Vector2( 0.2, 1.2 ),
		new THREE.Vector2( 0.22, 1.25 ),
		new THREE.Vector2( 0, 1.25 )
	];

	const vaseMat = new THREE.MeshStandardMaterial( { color: '#d4806a', roughness: 0.6, metalness: 0.02 } );

	const vasePedestalBase = new THREE.Mesh( new THREE.CylinderGeometry( 0.6, 0.7, 0.15, 32 ), pedestalMat );
	vasePedestalBase.position.set( - 5, - 1.925, - 2 );
	scene.add( vasePedestalBase );

	const vasePedestalShaft = new THREE.Mesh( new THREE.CylinderGeometry( 0.35, 0.42, 1, 32 ), pedestalMat );
	vasePedestalShaft.position.set( - 5, - 1.35, - 2 );
	scene.add( vasePedestalShaft );

	const vasePedestalTop = new THREE.Mesh( new THREE.CylinderGeometry( 0.55, 0.48, 0.15, 32 ), pedestalMat );
	vasePedestalTop.position.set( - 5, - 0.775, - 2 );
	scene.add( vasePedestalTop );

	const vase = new THREE.Mesh( new THREE.LatheGeometry( vaseProfile, 24 ), vaseMat );
	vase.position.set( - 5, - 1.6, - 2 );
	vase.scale.setScalar( 1.8 );
	scene.add( vase );

	// armchair

	const woodMat = new THREE.MeshStandardMaterial( { color: '#8b6840', roughness: 0.8, metalness: 0 } );
	const fabricMat = new THREE.MeshStandardMaterial( { color: '#8b3a3a', roughness: 0.9, metalness: 0 } );

	const chairGroup = new THREE.Group();

	const seat = new THREE.Mesh( new RoundedBoxGeometry( 0.9, 0.25, 0.8, 4, 0.06 ), fabricMat );
	seat.position.set( 0, - 1.35, 0 );
	chairGroup.add( seat );

	const backrest = new THREE.Mesh( new RoundedBoxGeometry( 0.9, 0.6, 0.12, 4, 0.04 ), fabricMat );
	backrest.position.set( 0, - 0.95, - 0.4 );
	backrest.rotation.x = - 0.3;
	chairGroup.add( backrest );

	for ( const side of [ - 1, 1 ] ) {

		const armrest = new THREE.Mesh( new THREE.BoxGeometry( 0.1, 0.25, 0.7 ), woodMat );
		armrest.position.set( side * 0.5, - 1.2, 0 );
		chairGroup.add( armrest );

		const armTop = new THREE.Mesh( new THREE.BoxGeometry( 0.14, 0.06, 0.8 ), woodMat );
		armTop.position.set( side * 0.5, - 1.07, 0 );
		chairGroup.add( armTop );

	}

	const legPositions = [[ - 0.38, - 0.32 ], [ - 0.38, 0.32 ], [ 0.38, - 0.32 ], [ 0.38, 0.32 ]];
	for ( const [ lx, lz ] of legPositions ) {

		const leg = new THREE.Mesh( new THREE.CylinderGeometry( 0.03, 0.035, 0.2, 8 ), woodMat );
		leg.position.set( lx, - 1.575, lz );
		chairGroup.add( leg );

	}

	chairGroup.scale.setScalar( 1.76 );
	chairGroup.position.set( 4, 0.948, - 2.5 );
	chairGroup.rotation.y = - 0.6;
	scene.add( chairGroup );

	// side table with cup

	const tableGroup = new THREE.Group();

	const tableTop = new THREE.Mesh( new THREE.CylinderGeometry( 0.4, 0.4, 0.05, 24 ), woodMat );
	tableTop.position.y = - 1.05;
	tableGroup.add( tableTop );

	const tableLeg = new THREE.Mesh( new THREE.CylinderGeometry( 0.04, 0.06, 0.9, 8 ), woodMat );
	tableLeg.position.y = - 1.5;
	tableGroup.add( tableLeg );

	const tableBase = new THREE.Mesh( new THREE.CylinderGeometry( 0.25, 0.28, 0.06, 24 ), woodMat );
	tableBase.position.y = - 1.92;
	tableGroup.add( tableBase );

	const cupMat = new THREE.MeshStandardMaterial( { color: '#f0ece0', roughness: 0.4, metalness: 0.05 } );
	const cupBody = new THREE.Mesh( new THREE.CylinderGeometry( 0.1, 0.08, 0.2, 16 ), cupMat );
	cupBody.scale.setScalar( 1 / 2.2 );
	cupBody.position.set( 0.15, - 0.98, 0 );
	tableGroup.add( cupBody );

	tableGroup.scale.setScalar( 2.2 );
	tableGroup.position.set( 2, 2.29, - 4 );
	scene.add( tableGroup );

	// rug

	const rugMat = new THREE.MeshStandardMaterial( { color: '#c8a0a8', roughness: 0.95, metalness: 0 } );
	const rug = new THREE.Mesh( new THREE.BoxGeometry( 6, 0.02, 5 ), rugMat );
	rug.position.set( 0, - 1.99, 0.5 );
	scene.add( rug );

	const rugBorderMat = new THREE.MeshStandardMaterial( { color: '#d4b880', roughness: 0.95, metalness: 0 } );
	const rugBorder = new THREE.Mesh( new THREE.BoxGeometry( 6.3, 0.015, 5.3 ), rugBorderMat );
	rugBorder.position.set( 0, - 1.9925, 0.5 );
	scene.add( rugBorder );

	// picture frames

	function addFrame( x, y, z, width, height, paintColor, rotY = 0 ) {

		const frameGroup = new THREE.Group();
		const t = 0.1;

		const outerW = width + t * 2;
		const outerH = height + t * 2;
		const frameShape = new THREE.Shape();
		frameShape.moveTo( - outerW / 2, - outerH / 2 );
		frameShape.lineTo( outerW / 2, - outerH / 2 );
		frameShape.lineTo( outerW / 2, outerH / 2 );
		frameShape.lineTo( - outerW / 2, outerH / 2 );
		frameShape.closePath();

		const hole = new THREE.Path();
		hole.moveTo( - width / 2, - height / 2 );
		hole.lineTo( width / 2, - height / 2 );
		hole.lineTo( width / 2, height / 2 );
		hole.lineTo( - width / 2, height / 2 );
		hole.closePath();
		frameShape.holes.push( hole );

		const frameMat = new THREE.MeshStandardMaterial( { color: '#8b6840', roughness: 0.7, metalness: 0 } );
		const geo = new THREE.ExtrudeGeometry( frameShape, {
			depth: 0.12,
			bevelEnabled: true,
			bevelThickness: 0.02,
			bevelSize: 0.02,
			bevelSegments: 2
		} );
		const frame = new THREE.Mesh( geo, frameMat );
		frameGroup.add( frame );

		const paintMat = new THREE.MeshStandardMaterial( { color: paintColor, roughness: 0.95, metalness: 0 } );
		const canvas = new THREE.Mesh( new THREE.PlaneGeometry( width, height ), paintMat );
		canvas.position.z = 0.001;
		frameGroup.add( canvas );

		frameGroup.position.set( x, y, z );
		if ( rotY ) frameGroup.rotation.y = rotY;
		scene.add( frameGroup );

	}

	addFrame( - 3.2, 2.0, - 4.9, 1.8, 1.2, '#e8a8a0' );
	addFrame( - 0.5, 2.6, - 4.9, 1.1, 1.6, '#a0c0e0' );
	addFrame( 2, 1.8, - 4.9, 2.0, 1.4, '#a0d0a8' );
	addFrame( - 5.4, 2.2, - 3, 1.5, 1.1, '#d0b0d8', Math.PI / 2 );
	addFrame( - 5.4, 1.8, 1, 1.8, 1.3, '#e0c8a0', Math.PI / 2 );

	// bust pedestal

	const bustX = - 3;
	const bustZ = - 3.2;

	const bustBase = new THREE.Mesh( new THREE.CylinderGeometry( 0.6, 0.7, 0.15, 32 ), pedestalMat );
	bustBase.position.set( bustX, - 1.925, bustZ );
	scene.add( bustBase );

	const bustShaft = new THREE.Mesh( new THREE.CylinderGeometry( 0.35, 0.42, 1, 32 ), pedestalMat );
	bustShaft.position.set( bustX, - 1.35, bustZ );
	scene.add( bustShaft );

	const bustTop = new THREE.Mesh( new THREE.CylinderGeometry( 0.55, 0.48, 0.15, 32 ), pedestalMat );
	bustTop.position.set( bustX, - 0.775, bustZ );
	scene.add( bustTop );



	const dracoLoader = new DRACOLoader();
	dracoLoader.setDecoderPath( new URL( '../libs/draco/gltf/', import.meta.url ).href );
	const loader = new GLTFLoader().setDRACOLoader( dracoLoader );
	try {

		const gltf = await loader.loadAsync( new URL( '../../models/gltf/tennyson-bust.glb', import.meta.url ).href );
		const bust = gltf.scene;
		bust.rotation.y = Math.PI;

		const targetHeight = 2.64;
		const sizeBox = new THREE.Box3().setFromObject( bust );
		const size = new THREE.Vector3();
		sizeBox.getSize( size );
		bust.scale.setScalar( targetHeight / size.y );

		const fitBox = new THREE.Box3().setFromObject( bust );
		const center = new THREE.Vector3();
		fitBox.getCenter( center );
		bust.position.set( bustX - center.x, - 0.7 - fitBox.min.y, bustZ - center.z + 0.1 );
		scene.add( bust );


	} finally {

		dracoLoader.dispose();

	}

}

function volume( min, max ) {

	const box = new THREE.Box3( new THREE.Vector3( ...min ), new THREE.Vector3( ...max ) );
	return { position: box.getCenter( new THREE.Vector3() ), box };

}

/** Build an independent scene; the caller owns its lighting environment and disposal. */
export async function createGBufferScene( name ) {

	const scene = new THREE.Scene();
	scene.name = name;
	let cameraPosition, target, probeVolumes;

	switch ( name ) {

		case 'Cornell':
			createCornell( scene );
			cameraPosition = new THREE.Vector3( 0, 2.8, 9 );
			target = new THREE.Vector3( 0, 2, - 0.5 );
			probeVolumes = [ volume( [ - 2.8, 0.1, - 2.8 ], [ 2.8, 4.8, 2.8 ] ) ];
			break;

		case 'Two rooms':
			createTwoRooms( scene );
			cameraPosition = new THREE.Vector3( 0, 5, 16 );
			target = new THREE.Vector3( 1, 1.8, - 0.5 );
			probeVolumes = [ volume( [ - 7.8, 0.1, - 3.8 ], [ 0, 4.8, 3.8 ] ), volume( [ 0, 0.1, - 3.8 ], [ 7.8, 4.8, 3.8 ] ) ];
			break;

		case 'GTAO':
			await createGallery( scene );
			cameraPosition = new THREE.Vector3( 1, 3, 7 );
			target = new THREE.Vector3( 0, 1.2, 0 );
			probeVolumes = [ volume( [ - 5.2, - 1.8, - 4.7 ], [ 5.2, 5.5, 5 ] ) ];
			break;

		default:
			throw new Error( 'Unknown G-buffer scene: ' + name );

	}

	scene.traverse( object => {

		if ( object.isMesh ) {

			object.castShadow = true;
			object.receiveShadow = true;

		} else if ( object.isLight && object.shadow ) {

			object.castShadow = true;

		}

	} );
	scene.updateMatrixWorld( true );
	const bounds = new THREE.Box3().setFromObject( scene );
	return { scene, cameraPosition, target, bounds, probeVolumes };

}

