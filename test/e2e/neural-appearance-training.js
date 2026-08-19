import puppeteer from 'puppeteer';
import * as fs from 'fs/promises';
import { Image } from './image.js';
import { createServer } from '../../utils/server.js';

const port = 1234;
const width = 400;
const height = 250;
const viewScale = 2;
const networkTimeout = 5; // minutes
const trainingTimeout = 3; // minutes
const outputDir = 'test/e2e/output-screenshots';
const pixelThreshold = 0.15;
// `targetResolution`/`baseResolution` size the shared multiresolution latent
// grid (see NeuralGridModel.js / NeuralAppearanceFormat.js) - cases with
// spatially detailed teacher patterns (UV grids, checkerboards, normal maps)
// use a finer target resolution than the flat-color cases, which are fine
// with the html defaults (baseResolution 16, targetResolution 256).
const allTestCases = [
	{ name: 'lambertBlue', label: 'diffuse blue', iterations: 1200, maxMeanRgbError: 8, maxDifferentPixels: 5, baseResolution: 4, targetResolution: 64 },
	{ name: 'lambertRed', label: 'diffuse red', iterations: 1200, maxMeanRgbError: 8, maxDifferentPixels: 5 },
	{ name: 'lambertGreen', label: 'diffuse green', iterations: 1200, maxMeanRgbError: 8, maxDifferentPixels: 5 },
	{ name: 'lambert', label: 'diffuse rust', iterations: 400, maxMeanRgbError: 18, maxDifferentPixels: 22 },
	{ name: 'glossy', label: 'glossy blue', iterations: 800, maxMeanRgbError: 30, maxDifferentPixels: 36 },
	{ name: 'glossyRed', label: 'glossy red', iterations: 800, maxMeanRgbError: 30, maxDifferentPixels: 36 },
	{ name: 'glossyGold', label: 'glossy gold', iterations: 800, maxMeanRgbError: 22, maxDifferentPixels: 36 },
	{ name: 'uvGrid', label: 'uv grid', iterations: 1000, maxMeanRgbError: 28, maxDifferentPixels: 32, baseResolution: 4, targetResolution: 64 },
	{ name: 'normalMap', label: 'normal map', iterations: 1000, maxMeanRgbError: 35, maxDifferentPixels: 40, baseResolution: 4, targetResolution: 64, hiddenSize: 64 },
	{ name: 'emissiveGrid', label: 'emissive grid', iterations: 1200, maxMeanRgbError: 35, maxDifferentPixels: 40, baseResolution: 4, targetResolution: 64, expectedOutputs: [ 'emission' ] },
	{ name: 'alphaCutoff', label: 'alpha cutoff grid', iterations: 1200, maxMeanRgbError: 35, maxDifferentPixels: 45, baseResolution: 4, targetResolution: 64, expectedOutputs: [ 'opacity' ], expectedOpacityMode: 'mask' },
	{ name: 'checkerboardTransparency', label: 'checkerboard transparency', iterations: 1200, maxMeanRgbError: 35, maxDifferentPixels: 45, baseResolution: 4, targetResolution: 64, expectedOutputs: [ 'opacity' ], expectedOpacityMode: 'blend' }
];
const testCases = process.env.TEST_CASE ? allTestCases.filter( ( testCase ) => testCase.name === process.env.TEST_CASE ) : allTestCases;
const background = [ 0x15, 0x17, 0x1c ];
const rotations = [ 0, Math.PI * 0.35, Math.PI * 0.7 ];

const server = createServer();
let browser;

console.red = msg => console.log( `\x1b[31m${msg}\x1b[39m` );
console.green = msg => console.log( `\x1b[32m${msg}\x1b[39m` );

server.listen( port, main );
process.on( 'SIGINT', () => close( 1 ) );

async function main() {

	try {

		await fs.rm( outputDir, { recursive: true, force: true } );
		await fs.mkdir( outputDir, { recursive: true } );

		const page = await launchPage();
		const signatures = new Set();

		for ( const testCase of testCases ) {

			const result = await runTrainingCase( page, testCase );
			const signature = getReferenceSignature( result.json );

			if ( signatures.has( signature ) ) {

				throw new Error( `${testCase.label}: teacher inputs match a previous fixture, so the material is not unique.` );

			}

			signatures.add( signature );

		}

		console.green( `TEST PASSED! Neural appearance training matched ${testCases.length} teacher materials.` );
		await close( 0 );

	} catch ( error ) {

		console.red( error );
		await close( 1 );

	}

}

function getReferenceSignature( json ) {

	return json.referenceEvaluations
		.flatMap( ( reference ) => reference.targetRgb || reference.rgb )
		.map( ( value ) => value.toFixed( 4 ) )
		.join( ',' );

}

async function runTrainingCase( page, testCase ) {

	page.error = undefined;
	const targetResolutionParam = testCase.targetResolution ? `&targetResolution=${testCase.targetResolution}` : '';
	const baseResolutionParam = testCase.baseResolution ? `&baseResolution=${testCase.baseResolution}` : '';
	await page.goto( `http://localhost:${port}/examples/webgpu_materials_neural_appearance.html?test=${testCase.name}&autoTrain=1&noRotate=1&iterations=${testCase.iterations}&batchSize=256&hiddenSize=${testCase.hiddenSize || 32}${baseResolutionParam}${targetResolutionParam}&seed=7`, {
		waitUntil: 'networkidle0',
		timeout: networkTimeout * 60000
	} );

	await page.waitForFunction( () => window.__neuralAppearanceTrainingReady === true, { timeout: 30000 } );
	await page.waitForFunction( () => window.__neuralAppearanceTrainingDone === true, { timeout: trainingTimeout * 60000 } );

	if ( page.error !== undefined ) throw new Error( `${testCase.label}: ${page.error}` );

	const result = await page.evaluate( () => ( {
		loss: window.__neuralAppearanceLastLoss,
		firstLoss: window.__neuralAppearanceFirstLoss,
		validationLoss: window.__neuralAppearanceValidationLoss,
		validation: window.__neuralAppearanceValidation,
		json: window.__neuralAppearanceExportJson,
		teacherInputs: window.__neuralAppearanceTeacherInputs
	} ) );

	if ( Number.isFinite( result.loss ) === false ) {

		throw new Error( `${testCase.label}: training finished with non-finite loss: ${result.loss}` );

	}

	validateExportJson( result.json, testCase );
	validateTeacherInputs( result.teacherInputs, testCase );
	validateLosses( result, testCase );

	let worstComparison = null;
	let worstWhiteComparison = null;
	let worstRotation = 0;

	for ( const rotation of rotations ) {

		const teacher = await captureCanvasView( page, 'teacher', rotation );
		const neural = await captureCanvasView( page, 'neural', rotation );
		const white = await captureCanvasView( page, 'white', rotation );
		const comparison = compareImages( teacher.image, neural.image );
		const whiteComparison = compareImages( teacher.image, white.image );

		if ( neural.state.activeMaterialType !== 'NeuralAppearanceNodeMaterial' ) {

			throw new Error( `${testCase.label}: neural view rendered ${neural.state.activeMaterialType} instead of NeuralAppearanceNodeMaterial.` );

		}

		if ( worstComparison === null || comparison.differentPixels > worstComparison.differentPixels ) {

			worstComparison = comparison;
			worstWhiteComparison = whiteComparison;
			worstRotation = rotation;

			await teacher.image.write( `${outputDir}/webgpu_materials_neural_appearance-${testCase.name}-teacher.jpg` );
			await neural.image.write( `${outputDir}/webgpu_materials_neural_appearance-${testCase.name}-neural.jpg` );
			await white.image.write( `${outputDir}/webgpu_materials_neural_appearance-${testCase.name}-white-control.jpg` );
			await comparison.mask.write( `${outputDir}/webgpu_materials_neural_appearance-${testCase.name}-mask.jpg` );
			await comparison.diff.write( `${outputDir}/webgpu_materials_neural_appearance-${testCase.name}-diff.jpg` );

		}

	}

	if ( worstWhiteComparison.meanRgbError <= worstComparison.meanRgbError * 1.1 ) {

		throw new Error( `${testCase.label}: forced-white control was not rejected: white mean RGB error ${worstWhiteComparison.meanRgbError.toFixed( 2 )}, neural mean RGB error ${worstComparison.meanRgbError.toFixed( 2 )}.` );

	}

	if ( worstComparison.meanRgbError > testCase.maxMeanRgbError || worstComparison.differentPixels > testCase.maxDifferentPixels ) {

		throw new Error( `${testCase.label}: teacher/neural foreground mismatch at rotation ${worstRotation.toFixed( 2 )}: mean RGB error ${worstComparison.meanRgbError.toFixed( 2 )} (max ${testCase.maxMeanRgbError}), RMSE ${worstComparison.rmse.toFixed( 2 )}, different pixels ${worstComparison.differentPixels.toFixed( 2 )}% (max ${testCase.maxDifferentPixels}%), foreground ${worstComparison.foregroundPixels} pixels` );

	}

	console.green( `${testCase.label}: loss ${result.loss.toExponential( 3 )}, validation ${result.validationLoss.toExponential( 3 )}, worst foreground mean RGB error ${worstComparison.meanRgbError.toFixed( 2 )}, different pixels ${worstComparison.differentPixels.toFixed( 2 )}%` );

	return result;

}

async function launchPage() {

	const flags = [
		'--hide-scrollbars',
		'--enable-unsafe-webgpu',
		'--enable-features=Vulkan',
		'--disable-vulkan-surface',
		'--ignore-gpu-blocklist',
		'--disable-gpu-driver-bug-workarounds',
		'--disable-gpu-watchdog',
		'--no-sandbox'
	];
	const viewport = { width: width * viewScale, height: height * viewScale };

	browser = await puppeteer.launch( {
		headless: ( 'CI' in process.env || process.env.VISIBLE ) ? false : 'new',
		env: { ...process.env, VK_DRIVER_FILES: '/usr/share/vulkan/icd.d/lvp_icd.x86_64.json' },
		args: flags,
		defaultViewport: viewport,
		handleSIGINT: false,
		protocolTimeout: 0,
		userDataDir: './.puppeteer_profile'
	} );

	const page = await browser.newPage();
	page.on( 'console', async msg => {

		const text = msg.text().trim();

		if ( text === '' || text.includes( 'Timestamp tracking is disabled' ) ) return;
		if ( text === 'Failed to load resource: the server responded with a status of 404 (Not Found)' ) return;
		if ( msg.type() === 'error' ) page.error = text;

	} );
	page.on( 'response', response => {

		if ( response.status() < 400 || response.url().endsWith( '/favicon.ico' ) ) return;

		page.error = `Failed to load ${response.url()}: ${response.status()}`;

	} );
	page.on( 'pageerror', error => {

		page.error = error.message;

	} );

	return page;

}

async function captureCanvasView( page, view, rotation ) {

	const state = await page.evaluate( async ( view, rotation ) => {

		const startToken = window.__neuralAppearanceFrameToken;
		window.__neuralAppearanceSetRotation( rotation );
		window.__neuralAppearanceSetView( view );
		for ( let i = 0; i < 8; i ++ ) {

			await new Promise( resolve => requestAnimationFrame( resolve ) );
			if ( window.__neuralAppearanceFrameToken > startToken && window.__neuralAppearanceRenderView === ( view === 'neural' ? 'trained neural' : ( view === 'white' ? 'forced white' : 'MaterialX teacher' ) ) ) break;

		}

		return {
			activeMaterialType: window.__neuralAppearanceActiveMaterialType,
			frameToken: window.__neuralAppearanceFrameToken,
			renderView: window.__neuralAppearanceRenderView
		};

	}, view, rotation );

	if ( page.error !== undefined ) throw new Error( page.error );

	const canvas = await page.$( '#neuralAppearanceCanvas' );
	if ( canvas === null ) throw new Error( 'Could not find renderer canvas.' );

	return {
		image: await Image.read( await canvas.screenshot( { type: 'png' } ) ),
		state
	};

}

function compareImages( teacher, neural ) {

	if ( teacher.width !== neural.width || teacher.height !== neural.height ) {

		throw new Error( 'Teacher and neural screenshots have different sizes.' );

	}

	const diff = teacher.clone();
	const mask = new Image( teacher.width, teacher.height, Buffer.alloc( teacher.width * teacher.height * 4 ) );
	let error = 0;
	let squaredError = 0;
	let differentPixels = 0;
	let foregroundPixels = 0;
	const maxPixelDistance = 255 * Math.sqrt( 3 );
	const threshold = pixelThreshold * maxPixelDistance;

	for ( let i = 0; i < teacher.data.length; i += 4 ) {

		const backgroundDistance = Math.sqrt(
			Math.pow( teacher.data[ i ] - background[ 0 ], 2 ) +
			Math.pow( teacher.data[ i + 1 ] - background[ 1 ], 2 ) +
			Math.pow( teacher.data[ i + 2 ] - background[ 2 ], 2 )
		);

		if ( backgroundDistance < 24 ) {

			diff.data[ i ] = teacher.data[ i ] * 0.2;
			diff.data[ i + 1 ] = teacher.data[ i + 1 ] * 0.2;
			diff.data[ i + 2 ] = teacher.data[ i + 2 ] * 0.2;
			diff.data[ i + 3 ] = 255;
			mask.data[ i ] = 0;
			mask.data[ i + 1 ] = 0;
			mask.data[ i + 2 ] = 0;
			mask.data[ i + 3 ] = 255;
			continue;

		}

		const dr = teacher.data[ i ] - neural.data[ i ];
		const dg = teacher.data[ i + 1 ] - neural.data[ i + 1 ];
		const db = teacher.data[ i + 2 ] - neural.data[ i + 2 ];
		const distance = Math.sqrt( dr * dr + dg * dg + db * db );
		const pixelError = ( Math.abs( dr ) + Math.abs( dg ) + Math.abs( db ) ) / 3;

		error += pixelError;
		squaredError += pixelError * pixelError;
		foregroundPixels ++;
		mask.data[ i ] = 255;
		mask.data[ i + 1 ] = 255;
		mask.data[ i + 2 ] = 255;
		mask.data[ i + 3 ] = 255;

		if ( distance > threshold ) {

			differentPixels ++;
			diff.data[ i ] = 255;
			diff.data[ i + 1 ] = 0;
			diff.data[ i + 2 ] = 0;
			diff.data[ i + 3 ] = 255;

		} else {

			diff.data[ i ] = teacher.data[ i ] * 0.2;
			diff.data[ i + 1 ] = teacher.data[ i + 1 ] * 0.2;
			diff.data[ i + 2 ] = teacher.data[ i + 2 ] * 0.2;
			diff.data[ i + 3 ] = 255;

		}

	}

	if ( foregroundPixels < teacher.width * teacher.height * 0.03 ) {

		throw new Error( `Foreground mask only covered ${foregroundPixels} pixels.` );

	}

	return {
		diff,
		mask,
		foregroundPixels,
		meanRgbError: error / foregroundPixels,
		rmse: Math.sqrt( squaredError / foregroundPixels ),
		differentPixels: differentPixels / foregroundPixels * 100
	};

}

function validateExportJson( json, testCase ) {

	if ( json === null || json.format !== 'three-neural-appearance' ) {

		throw new Error( `${testCase.label}: training did not produce a neural appearance export.` );

	}

	if ( json.version !== 3 || json.latents.textures.length !== 2 || json.outputs.brdf.inputSize !== 20 || json.outputs.ibl.inputSize !== 14 ) {

		throw new Error( `${testCase.label}: training produced an invalid neural appearance export shape.` );

	}

	for ( const outputName of testCase.expectedOutputs || [] ) {

		if ( json.outputs[ outputName ] === undefined ) {

			throw new Error( `${testCase.label}: training did not export outputs.${outputName}.` );

		}

	}

	if ( testCase.expectedOpacityMode && ( ! json.outputs.opacity || json.outputs.opacity.mode !== testCase.expectedOpacityMode ) ) {

		throw new Error( `${testCase.label}: training exported opacity mode "${json.outputs.opacity ? json.outputs.opacity.mode : undefined}", expected "${testCase.expectedOpacityMode}".` );

	}

}

function validateLosses( result, testCase ) {

	if ( Number.isFinite( result.firstLoss ) === false ) {

		throw new Error( `${testCase.label}: training did not expose a finite first loss.` );

	}

	if ( result.loss >= result.firstLoss * 0.75 ) {

		throw new Error( `${testCase.label}: training loss did not improve enough: first ${result.firstLoss}, final ${result.loss}.` );

	}

	if ( Number.isFinite( result.validationLoss ) === false ) {

		throw new Error( `${testCase.label}: training did not expose a finite validation loss.` );

	}

	if ( result.validationLoss > Math.max( result.loss * 2.5, 0.15 ) ) {

		throw new Error( `${testCase.label}: validation loss is too high: ${result.validationLoss}.` );

	}

	// The multiresolution latent grid always has 4 levels (see
	// NeuralAppearanceFormat.js) regardless of baseResolution/targetResolution,
	// so there's no per-case mip-count to check anymore - just confirm the
	// runtime validation exported the expected level count.
	const expectedLevels = testCase.levels || 4;

	if ( result.validation === null || result.validation.levels !== expectedLevels ) {

		throw new Error( `${testCase.label}: runtime validation did not cover all ${expectedLevels} exported latent grid levels.` );

	}

	for ( const direction of [ 'wi', 'wo' ] ) {

		const bins = result.validation.angularBins[ direction ];

		if ( Array.isArray( bins ) === false || bins.length !== 3 || bins.reduce( ( count, bin ) => count + bin.count, 0 ) === 0 ) {

			throw new Error( `${testCase.label}: runtime validation did not report ${direction} angular-bin errors.` );

		}

	}

}

function validateTeacherInputs( teacherInputs, testCase ) {

	if ( Array.isArray( teacherInputs ) === false || teacherInputs.length !== 14 ) {

		throw new Error( `${testCase.label}: training did not expose 14 teacher inputs.` );

	}

	for ( const value of teacherInputs ) {

		if ( Number.isFinite( value ) === false ) {

			throw new Error( `${testCase.label}: teacher inputs contain non-finite values.` );

		}

	}

}

async function close( exitCode = 1 ) {

	if ( browser ) {

		try {

			await browser.close();

		} catch {

			// ignore

		}

	}

	if ( server.closeAllConnections ) {

		server.closeAllConnections();

	}

	server.close();
	process.exit( exitCode );

}
