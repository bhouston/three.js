import * as THREE from 'three';
import {
	computeFootprintArea,
	createGaussianSampleKernel,
	getGaussianSampleGridSize
} from './NeuralAppearanceFilterUtils.js';

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

function readFilteredSample( pixels, sampleIndex, sample, {
	atlasColumns,
	atlasWidth,
	tileSize,
	sourceResolution = 1,
	uvGradientScale = 1 / 1024,
	filterMinSamples = 1,
	filterMaxSamples = 64,
	filterSigma = 0.25,
	filterKernels = null
} = {} ) {

	const footprintArea = computeFootprintArea(
		sample.duvDx || [ uvGradientScale, 0 ],
		sample.duvDy || [ 0, uvGradientScale ],
		sourceResolution,
		sourceResolution
	);
	const maxSamples = Math.min( filterMaxSamples, tileSize * tileSize );
	const gridSize = getGaussianSampleGridSize( footprintArea, filterMinSamples, maxSamples );
	let kernel = filterKernels ? filterKernels.get( gridSize ) : undefined;

	if ( kernel === undefined ) {

		kernel = createGaussianSampleKernel( gridSize, tileSize, filterSigma );
		if ( filterKernels ) filterKernels.set( gridSize, kernel );

	}

	const tileX = sampleIndex % atlasColumns;
	const tileY = Math.floor( sampleIndex / atlasColumns );
	const target = [ 0, 0, 0, 0 ];

	for ( const tap of kernel ) {

		const x = tileX * tileSize + tap.x;
		const y = tileY * tileSize + tap.y;
		const index = ( y * atlasWidth + x ) * 4;

		target[ 0 ] += readPixelValue( pixels, index ) * tap.weight;
		target[ 1 ] += readPixelValue( pixels, index + 1 ) * tap.weight;
		target[ 2 ] += readPixelValue( pixels, index + 2 ) * tap.weight;
		target[ 3 ] += readPixelValue( pixels, index + 3 ) * tap.weight;

	}

	return target;

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
	readFilteredSample,
	renderAndReadTeacher
};
