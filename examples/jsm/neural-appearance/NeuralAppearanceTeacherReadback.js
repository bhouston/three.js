import * as THREE from 'three';

function readPixelValue( pixels, index ) {

	if ( pixels instanceof Uint16Array ) return THREE.DataUtils.fromHalfFloat( pixels[ index ] );
	return pixels[ index ];

}

function readSamplePixel( pixels, sampleIndex, atlasColumns, atlasWidth, tileSize ) {

	const tileX = sampleIndex % atlasColumns;
	const tileY = Math.floor( sampleIndex / atlasColumns );
	const x = tileX * tileSize + Math.floor( tileSize / 2 );
	const y = tileY * tileSize + Math.floor( tileSize / 2 );
	const index = ( y * atlasWidth + x ) * 4;

	return [
		readPixelValue( pixels, index ),
		readPixelValue( pixels, index + 1 ),
		readPixelValue( pixels, index + 2 ),
		readPixelValue( pixels, index + 3 )
	];

}

async function renderAndReadTeacher( renderer, scene, camera, target, atlasWidth, atlasHeight ) {

	const previousTarget = renderer.getRenderTarget();
	const previousToneMapping = renderer.toneMapping;
	const previousClearColor = new THREE.Color();
	const previousClearAlpha = renderer.getClearAlpha();

	renderer.getClearColor( previousClearColor );

	try {

		renderer.toneMapping = THREE.NoToneMapping;
		renderer.setClearColor( 0x000000, 1 );
		renderer.setRenderTarget( target );
		renderer.render( scene, camera );

		const pixels = await renderer.readRenderTargetPixelsAsync( target, 0, 0, atlasWidth, atlasHeight );

		if ( pixels instanceof Uint16Array === false ) {

			const type = pixels && pixels.constructor ? pixels.constructor.name : typeof pixels;
			throw new Error( `THREE.NeuralAppearanceTeacherEvaluator: Half-float teacher readback is required; received ${ type }.` );

		}

		return pixels;

	} finally {

		renderer.setRenderTarget( previousTarget );
		renderer.setClearColor( previousClearColor, previousClearAlpha );
		renderer.toneMapping = previousToneMapping;

	}

}

export {
	readPixelValue,
	readSamplePixel,
	renderAndReadTeacher
};
