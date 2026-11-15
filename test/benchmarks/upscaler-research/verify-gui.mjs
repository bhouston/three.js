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

const s = await open({upsampler:'joint-bilateral',rotate:true},false);
const choices = await s.page.evaluate(()=>{
 const select=[...document.querySelectorAll('select')].find(s=>[...s.options].some(o=>o.textContent==='taau-mip-fused'));
 if(!select) throw new Error('Upsampler GUI missing');
 return [...select.options].map(o=>o.textContent);
});
console.log('GUI modes',choices.length);
for(const key of choices) {
 const before=await s.page.evaluate(()=>window.__test.frames);
 await s.page.evaluate(key=>{
  const select=[...document.querySelectorAll('select')].find(s=>[...s.options].some(o=>o.textContent==='taau-mip-fused'));
  const option=[...select.options].find(o=>o.textContent===key);
  select.value=option.value; select.dispatchEvent(new Event('change',{bubbles:true}));
 },key);
 await s.page.waitForFunction(before=>window.__test.frames>=before+40,{timeout:30000},before);
 await s.page.evaluate(()=>window.__device.queue.onSubmittedWorkDone());
 if(s.errors.length) throw new Error(key+': '+s.errors.join('\n'));
 const selected=await s.page.evaluate(()=>window.__test.params.upsampler);
 if(selected!==key) throw new Error('GUI selection failed '+selected+' != '+key);
 console.log('switch',key,'ok');
}
// Return to previously used temporal nodes, then resize and change the material.
for(const key of ['taau-mip-fused','joint-bilateral-wide-jitter-mip','taau-mip-fast-fused']) {
 await s.page.evaluate(key=>{
 const select=[...document.querySelectorAll('select')].find(s=>[...s.options].some(o=>o.textContent===key));
 select.value=[...select.options].find(o=>o.textContent===key).value;select.dispatchEvent(new Event('change',{bubbles:true}));
 },key);
 await s.page.setViewport({width:1280,height:720,deviceScaleFactor:1});
 await wait(750);
 await s.page.setViewport({width:1920,height:1080,deviceScaleFactor:1});
 await wait(750);
}
for(const factor of [1,2,3,3.5,3]) {
 await s.page.evaluate(factor=>{
  const row=[...document.querySelectorAll('.list-item-row')].find(row=>row.firstElementChild?.textContent==='Upscale factor');
  if(!row) throw new Error('Upscale factor GUI missing');
  const input=row.querySelector('input[type=number]');input.value=factor;input.dispatchEvent(new Event('change',{bubbles:true}));
 },factor);
 await wait(500);
 if(await s.page.evaluate(()=>window.__test.params.upscaleFactor)!==factor) throw new Error('factor change failed');
}
console.log('factor 1/2/3/3.5 changes ok');
await s.page.evaluate(()=>{
 const select=[...document.querySelectorAll('select')].find(s=>[...s.options].some(o=>o.textContent==='Metallic'));
 select.value=[...select.options].find(o=>o.textContent==='Metallic').value;select.dispatchEvent(new Event('change',{bubbles:true}));
});
await wait(1000);
await s.page.evaluate(()=>window.__device.queue.onSubmittedWorkDone());
if(s.errors.length) throw new Error(s.errors.join('\n'));
console.log('reactivation, resize and material changes ok');
await s.close();
process.exit(0);
