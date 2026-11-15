// usage: node compare.mjs bench|error|shots [factor]   (PORT env, default 8310; serves examples/_cmp.html, see mkpage.sh)
// bench: mean frame time in ms at 1920x1080, vsync off, 5 s window bracketed by onSubmittedWorkDone, 3 interleaved rounds
// error: mean of the example's Error output (x4) over the frame, 0-255, static and rotating
// shots: beauty and error screenshots per upsampler
import { createRequire } from 'module';
import fs from 'fs';
const require = createRequire( new URL( '../../../package.json', import.meta.url ) );
const puppeteer = require( 'puppeteer' );
const { PNG } = require( 'pngjs' );
const wait = ms => new Promise( r => setTimeout( r, ms ) );
const port = process.env.PORT || 8312;
const [ mode, factorArg ] = process.argv.slice( 2 );
const factor = Number( factorArg || 3 );
const upsamplers = ( process.env.UPSAMPLERS || 'none bilinear joint-bilateral joint-bilateral-traa fsr1 taau' ).split( ' ' );

const open = async ( params, benchmark ) => {

	const args = [ '--enable-unsafe-webgpu', '--use-angle=metal', '--enable-gpu', '--window-size=1920,1080' ];
	if ( benchmark ) args.push( '--disable-gpu-vsync', '--disable-frame-rate-limit' );
	const browser = await puppeteer.launch( { executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: 'new', args, defaultViewport: { width: 1920, height: 1080, deviceScaleFactor: 1 } } );
	const page = await browser.newPage();
	const errors = [];
	page.on( 'pageerror', e => errors.push( e.message ) );
	page.on( 'console', m => { if ( ( m.type() === 'error' || m.type() === 'warn' ) && ! /404|Failed to load resource/.test( m.text() ) ) errors.push( m.text().split( '\n' )[ 0 ].slice( 0, 300 ) ); } );
	await page.evaluateOnNewDocument( () => { const request = GPUAdapter.prototype.requestDevice; GPUAdapter.prototype.requestDevice = async function ( ...a ) { const device = await request.apply( this, a ); window.__device = device; return device; }; } );
	await page.goto( `http://localhost:${port}/examples/_cmp.html#${encodeURIComponent( JSON.stringify( params ) )}` );
	await page.waitForFunction( () => window.__test?.frames > 60, { timeout: 60000 } );
	const info = await page.evaluate( async () => { const a = await navigator.gpu.requestAdapter(); return {vendor:a.info.vendor,architecture:a.info.architecture,device:a.info.device,description:a.info.description}; } );
	if ( ! /apple|nvidia|amd|intel/i.test( JSON.stringify(info) ) || /swiftshader|llvmpipe|lavapipe/i.test(JSON.stringify(info)) ) throw new Error('Non-hardware adapter '+JSON.stringify(info));
	await page.addStyleTag( { content: 'body > *:not(canvas) { display: none !important; }' } );
	return { browser, page, errors, close: () => Promise.race( [ browser.close(), wait( 3000 ) ] ) };

};

const meanOf = ( buffer ) => {

	const png = PNG.sync.read( Buffer.from( buffer ) );
	let sum = 0;
	for ( let i = 0; i < png.data.length; i += 4 ) sum += png.data[ i ] + png.data[ i + 1 ] + png.data[ i + 2 ];
	return sum / ( png.data.length / 4 * 3 );

};

if ( mode === 'psnr' ) {

	// references: the native render, and a supersampled one (device pixel ratio 3, box filtered in linear light)
	const shot = async ( params, scale = 1 ) => {

		const s = await open( { rotate: false, ...params }, false );
		if ( scale !== 1 ) { await s.page.setViewport( { width: 1920, height: 1080, deviceScaleFactor: scale } ); await s.page.reload(); await s.page.waitForFunction( () => window.__device !== undefined ); await s.page.addStyleTag( { content: 'body > *:not(canvas) { display: none !important; }' } ); }
		await wait( Number(process.env.WARM_MS || 4000) );
		await s.page.waitForFunction(()=>window.__test?.frames>=320,{timeout:60000});
		await s.page.evaluate(()=>window.__device.queue.onSubmittedWorkDone());
		const png = PNG.sync.read( Buffer.from( await s.page.screenshot() ) );
		if ( s.errors.length ) console.log( JSON.stringify( params ), 'ERRORS:', s.errors[ 0 ] );
		await s.close();
		return png;

	};

	const toLinear = v => { v /= 255; return v <= 0.04045 ? v / 12.92 : ( ( v + 0.055 ) / 1.055 ) ** 2.4; };
	const toSRGB = v => 255 * ( v <= 0.0031308 ? v * 12.92 : 1.055 * v ** ( 1 / 2.4 ) - 0.055 );
	const W = 1920, H = 1080, K = 3;
	const native = process.env.REF_CACHE ? PNG.sync.read(fs.readFileSync('native-beauty.png')) : await shot( { upsampler: 'bilinear', upscaleFactor: 1 } );
	fs.writeFileSync( 'native-beauty.png', PNG.sync.write( native ) );
	const big = process.env.REF_CACHE ? PNG.sync.read(fs.readFileSync('supersampled-beauty.png')) : await shot( { upsampler: 'bilinear', upscaleFactor: 1 }, K );
	if ( big.width !== W * K ) throw new Error( 'supersampled size ' + big.width );
	fs.writeFileSync('supersampled-beauty.png', PNG.sync.write(big));
	const ssaa = new Float32Array( W * H * 3 );
	for ( let y = 0; y < H; y ++ ) for ( let x = 0; x < W; x ++ ) for ( let c = 0; c < 3; c ++ ) {

		let sum = 0;
		for ( let j = 0; j < K; j ++ ) for ( let i = 0; i < K; i ++ ) sum += toLinear( big.data[ ( ( y * K + j ) * W * K + x * K + i ) * 4 + c ] );
		ssaa[ ( y * W + x ) * 3 + c ] = toSRGB( sum / ( K * K ) );

	}

	const nativeF = new Float32Array( W * H * 3 );
	for ( let i = 0; i < W * H; i ++ ) for ( let c = 0; c < 3; c ++ ) nativeF[ i * 3 + c ] = native.data[ i * 4 + c ];

	// helmet crop excludes the blurry background, which every upsampler reproduces well
	const crop = { x0: 500, x1: 1230, y0: 170, y1: 1000 };
	const psnr = ( png, ref, box ) => {

		let se = 0, n = 0;
		for ( let y = box.y0; y < box.y1; y ++ ) for ( let x = box.x0; x < box.x1; x ++ ) for ( let c = 0; c < 3; c ++ ) { const d = png.data[ ( y * W + x ) * 4 + c ] - ref[ ( y * W + x ) * 3 + c ]; se += d * d; n ++; }
		return ( 10 * Math.log10( 255 * 255 / ( se / n ) ) ).toFixed( 2 );

	};

	const full = { x0: 0, x1: W, y0: 0, y1: H };
	console.log( 'PSNR in dB on displayed sRGB: vs native full/helmet | vs supersampled full/helmet' );
	console.log( `native 1x: - / - | ${psnr( native, ssaa, full )} / ${psnr( native, ssaa, crop )}` );
	for ( const f of ( process.env.FACTORS || '2 3' ).split( ' ' ).map( Number ) ) for ( const upsampler of upsamplers ) {

		const png = await shot( { upsampler, upscaleFactor: f } );
		fs.writeFileSync(`${upsampler}-beauty-${f}x.png`, PNG.sync.write(png));
		console.log( `${upsampler} @${f}x: ${psnr( png, nativeF, full )} / ${psnr( png, nativeF, crop )} | ${psnr( png, ssaa, full )} / ${psnr( png, ssaa, crop )}` );

	}

 } else if ( mode === 'motion' ) {
	for ( const upsampler of upsamplers ) {
		const s = await open({upsampler, upscaleFactor:factor, output:'Error', rotate:false, motionTest:true}, false);
		const values=[];
		for (let i=0;i<8;i++) {
			const target=120+i*30;
			if (i) await s.page.evaluate(target=>{window.__test.stopAt=target;window.__test.renderer.setAnimationLoop(window.__test.render);},target);
			await s.page.waitForFunction(target=>window.__test.frames>=target,{timeout:30000},target);
			await s.page.evaluate(()=>window.__device.queue.onSubmittedWorkDone());
			values.push(meanOf(await s.page.screenshot()));
		}
		console.log(`${upsampler} @${factor}x deterministic motion: ${(values.reduce((a,b)=>a+b)/8).toFixed(3)} [${values.map(v=>v.toFixed(3))}]${s.errors.length?' ERRORS '+s.errors[0]:''}`);
		await s.close();
	}

} else if ( mode === 'bench' ) {

	const variants = [ 'native', ...upsamplers ];
	const results = Object.fromEntries( variants.map( v => [ v, [] ] ) );
	for ( let round = 0; round < Number(process.env.ROUNDS || 3); round ++ ) {

		for ( const upsampler of variants ) {

			const params = upsampler === 'native' ? { upsampler: 'bilinear', upscaleFactor: 1 } : { upsampler, upscaleFactor: factor };
			const s = await open( params, true );
			await wait( Number(process.env.WARM_MS || 4000) );
			const r = await s.page.evaluate( async () => {

				const adapter = await navigator.gpu.requestAdapter();
				window.__test.renderer.setAnimationLoop( null );
				await window.__device.queue.onSubmittedWorkDone();
				window.__test.frames = 0;
				window.__test.completed = 0;
				window.__test.counting = true;
				const start = performance.now();
				window.__test.renderer.setAnimationLoop( window.__test.render );
				await new Promise( r => setTimeout( r, 5000 ) );
				window.__test.renderer.setAnimationLoop( null );
				await window.__device.queue.onSubmittedWorkDone();
				window.__test.completed = window.__test.frames;
				return { ms: ( performance.now() - start ) / window.__test.completed, frames: window.__test.completed, adapter: adapter.info.vendor + ' ' + adapter.info.architecture };

			} );
			results[ upsampler ].push( r.ms );
			console.log( `round ${round} ${upsampler}: ${r.ms.toFixed( 3 )} ms (${r.frames} GPU-completed frames; ${r.adapter})${s.errors.length ? ' ERRORS: ' + s.errors[ 0 ] : ''}` );
			await s.close();
			await wait( Number(process.env.COOL_MS || 15000) );

		}

	}

	for ( const v of variants ) console.log( `${v}: mean ${( results[ v ].reduce( ( a, b ) => a + b ) / results[ v ].length ).toFixed( 2 )} ms  [${results[ v ].map( x => x.toFixed( 2 ) ).join( ', ' )}]` );

} else {

	for ( const upsampler of upsamplers ) {

		if ( mode === 'shots' ) {

			for ( const output of [ 'Beauty', 'Error' ] ) {

				const s = await open( { upsampler, upscaleFactor: factor, output, rotate: false }, false );
				await wait( 7000 );
				await s.page.screenshot( { path: `${upsampler}-${output.toLowerCase()}-${factor}x.png` } );
				if ( s.errors.length ) console.log( upsampler, 'ERRORS:', s.errors.slice( 0, 3 ) );
				await s.close();

			}

			continue;

		}

		const line = [];
		for ( const rotate of [ false, true ] ) {

			const s = await open( { upsampler, upscaleFactor: factor, output: 'Error', rotate }, false );
			await wait( 7000 );
			let sum = 0;
			const n = rotate ? 8 : 1;
			for ( let i = 0; i < n; i ++ ) { sum += meanOf( await s.page.screenshot() ); await wait( 300 ); }
			line.push( `${rotate ? 'rotating' : 'static'} ${( sum / n ).toFixed( 3 )}` );
			if ( s.errors.length ) line.push( 'ERRORS: ' + s.errors[ 0 ] );
			await s.close();

		}

		console.log( `${upsampler} @${factor}x: ${line.join( ', ' )}` );

	}

}

process.exit( 0 );
