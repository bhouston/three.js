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

function validateHalfFloatPixels( pixels ) {

	if ( pixels instanceof Uint16Array === false ) {

		const type = pixels && pixels.constructor ? pixels.constructor.name : typeof pixels;
		throw new Error( `THREE.NeuralAppearanceTeacherEvaluator: Half-float teacher readback is required; received ${ type }.` );

	}

	return pixels;

}

async function renderAndReadTeacher( renderer, scene, camera, target, atlasWidth, atlasHeight ) {

	const [ pixels ] = await renderAndReadTeacherAttachments( renderer, scene, camera, target, atlasWidth, atlasHeight, [ 0 ] );
	return pixels;

}

// Renders `scene` once and reads back one or more MRT color attachments from
// the same draw -- lets callers merge several teacher outputs (that used to
// require separate render+readback round trips) into a single render pass.
// `textureIndices` defaults to just attachment 0 for non-MRT targets.
async function renderAndReadTeacherAttachments( renderer, scene, camera, target, atlasWidth, atlasHeight, textureIndices = [ 0 ] ) {

	const previousTarget = renderer.getRenderTarget();
	const previousToneMapping = renderer.toneMapping;
	const previousClearColor = new THREE.Color();
	const previousClearAlpha = renderer.getClearAlpha();

	renderer.getClearColor( previousClearColor );

	renderer.toneMapping = THREE.NoToneMapping;
	renderer.setClearColor( 0x000000, 1 );
	renderer.setRenderTarget( target );
	renderer.render( scene, camera );

	// Restore the renderer's render target (and other overridden state)
	// *before* the async readback below, not in a `finally` after it. The
	// example's rAF-driven animation loop calls `renderer.render()` without
	// first calling `setRenderTarget( null )`, so it inherits whatever
	// render target is currently active; `readRenderTargetPixelsAsync()`
	// involves a real GPU round trip (`GPUBuffer.mapAsync()`) that can span
	// one or more animation frames, and every training iteration goes
	// through this await. Restoring only in `finally` left a window, hit on
	// essentially every frame while training was active, during which a
	// concurrent animation-loop render would draw the whole viewport scene
	// into this offscreen teacher target instead of the canvas -- surfacing
	// as a WebGPU GPUValidationError ("color and depth targets from pass do
	// not match pipeline") from pipelines cached against mismatched
	// attachment layouts. `readRenderTargetPixelsAsync()` takes `target`
	// explicitly and doesn't depend on the renderer's "current" render
	// target, so restoring first is safe.
	renderer.setRenderTarget( previousTarget );
	renderer.setClearColor( previousClearColor, previousClearAlpha );
	renderer.toneMapping = previousToneMapping;

	const attachments = [];

	for ( const textureIndex of textureIndices ) {

		const pixels = await renderer.readRenderTargetPixelsAsync( target, 0, 0, atlasWidth, atlasHeight, textureIndex );
		attachments.push( validateHalfFloatPixels( pixels ) );

	}

	return attachments;

}

export {
	readPixelValue,
	readSamplePixel,
	renderAndReadTeacher,
	renderAndReadTeacherAttachments
};
