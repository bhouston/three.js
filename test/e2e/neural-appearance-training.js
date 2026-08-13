import puppeteer from 'puppeteer';
import * as fs from 'fs/promises';
import { Image } from './image.js';
import { createServer } from '../../utils/server.js';

const port = 1234;
const width = 400;
const height = 250;
const viewScale = 2;
const networkTimeout = 5; // minutes
const trainingTimeout = 2; // minutes
const outputDir = 'test/e2e/output-screenshots';
const pixelThreshold = 0.2;
const testCases = [
	{ name: 'lambert', label: 'diffuse rust', iterations: 400, maxMeanRgbError: 25, maxDifferentPixels: 30 },
	{ name: 'lambertGreen', label: 'diffuse green', iterations: 400, maxMeanRgbError: 25, maxDifferentPixels: 30 },
	{ name: 'lambertBlue', label: 'diffuse blue', iterations: 400, maxMeanRgbError: 25, maxDifferentPixels: 30 },
	{ name: 'glossy', label: 'glossy blue', iterations: 800, maxMeanRgbError: 35, maxDifferentPixels: 45 },
	{ name: 'glossyRed', label: 'glossy red', iterations: 800, maxMeanRgbError: 35, maxDifferentPixels: 45 },
	{ name: 'glossyGold', label: 'glossy gold', iterations: 800, maxMeanRgbError: 35, maxDifferentPixels: 45 }
];

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
			const signature = result.teacherInputs.map( ( value ) => value.toFixed( 4 ) ).join( ',' );

			if ( signatures.has( signature ) ) {

				throw new Error( `${testCase.label}: teacher inputs match a previous fixture, so the material is not unique.` );

			}

			signatures.add( signature );

		}

		console.green( `TEST PASSED! Neural appearance training matched ${testCases.length} teacher materials.` );
		close( 0 );

	} catch ( error ) {

		console.red( error );
		close( 1 );

	}

}

async function runTrainingCase( page, testCase ) {

	page.error = undefined;
	await page.goto( `http://localhost:${port}/examples/webgpu_materials_neural_appearance_train.html?test=${testCase.name}&autoTrain=1&noRotate=1&iterations=${testCase.iterations}&batchSize=256&resolution=1&seed=7`, {
		waitUntil: 'networkidle0',
		timeout: networkTimeout * 60000
	} );

	await page.waitForFunction( () => window.__neuralAppearanceTrainingReady === true, { timeout: 30000 } );
	await page.waitForFunction( () => window.__neuralAppearanceTrainingDone === true, { timeout: trainingTimeout * 60000 } );

	if ( page.error !== undefined ) throw new Error( `${testCase.label}: ${page.error}` );

	const result = await page.evaluate( () => ( {
		loss: window.__neuralAppearanceLastLoss,
		json: window.__neuralAppearanceExportJson,
		teacherInputs: window.__neuralAppearanceTeacherInputs
	} ) );

	if ( Number.isFinite( result.loss ) === false ) {

		throw new Error( `${testCase.label}: training finished with non-finite loss: ${result.loss}` );

	}

	validateExportJson( result.json, testCase );
	validateTeacherInputs( result.teacherInputs, testCase );

	const teacher = await captureCanvasView( page, 'teacher' );
	const neural = await captureCanvasView( page, 'neural' );
	const comparison = compareImages( teacher, neural );

	await teacher.write( `${outputDir}/webgpu_materials_neural_appearance_train-${testCase.name}-teacher.jpg` );
	await neural.write( `${outputDir}/webgpu_materials_neural_appearance_train-${testCase.name}-neural.jpg` );
	await comparison.diff.write( `${outputDir}/webgpu_materials_neural_appearance_train-${testCase.name}-diff.jpg` );

	if ( comparison.meanRgbError > testCase.maxMeanRgbError || comparison.differentPixels > testCase.maxDifferentPixels ) {

		throw new Error( `${testCase.label}: teacher/neural image mismatch: mean RGB error ${comparison.meanRgbError.toFixed( 2 )} (max ${testCase.maxMeanRgbError}), different pixels ${comparison.differentPixels.toFixed( 2 )}% (max ${testCase.maxDifferentPixels}%)` );

	}

	console.green( `${testCase.label}: loss ${result.loss.toExponential( 3 )}, mean RGB error ${comparison.meanRgbError.toFixed( 2 )}, different pixels ${comparison.differentPixels.toFixed( 2 )}%` );

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

async function captureCanvasView( page, view ) {

	await page.evaluate( async view => {

		window.__neuralAppearanceSetView( view );
		await new Promise( resolve => requestAnimationFrame( resolve ) );
		await new Promise( resolve => requestAnimationFrame( resolve ) );

	}, view );

	if ( page.error !== undefined ) throw new Error( page.error );

	const canvas = await page.$( 'canvas' );
	if ( canvas === null ) throw new Error( 'Could not find renderer canvas.' );

	return Image.read( await canvas.screenshot( { type: 'png' } ) );

}

function compareImages( teacher, neural ) {

	if ( teacher.width !== neural.width || teacher.height !== neural.height ) {

		throw new Error( 'Teacher and neural screenshots have different sizes.' );

	}

	const diff = teacher.clone();
	let error = 0;
	let differentPixels = 0;
	const maxPixelDistance = 255 * Math.sqrt( 3 );
	const threshold = pixelThreshold * maxPixelDistance;

	for ( let i = 0; i < teacher.data.length; i += 4 ) {

		const dr = teacher.data[ i ] - neural.data[ i ];
		const dg = teacher.data[ i + 1 ] - neural.data[ i + 1 ];
		const db = teacher.data[ i + 2 ] - neural.data[ i + 2 ];
		const distance = Math.sqrt( dr * dr + dg * dg + db * db );

		error += ( Math.abs( dr ) + Math.abs( dg ) + Math.abs( db ) ) / 3;

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

	const pixelCount = teacher.width * teacher.height;

	return {
		diff,
		meanRgbError: error / pixelCount,
		differentPixels: differentPixels / pixelCount * 100
	};

}

function validateExportJson( json, testCase ) {

	if ( json === null || json.format !== 'three-neural-appearance' ) {

		throw new Error( `${testCase.label}: training did not produce a neural appearance export.` );

	}

	if ( json.latents.textures.length !== 2 || json.decoder.inputSize !== 20 ) {

		throw new Error( `${testCase.label}: training produced an invalid neural appearance export shape.` );

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

function close( exitCode = 1 ) {

	if ( browser ) browser.close();
	server.close();
	process.exit( exitCode );

}
