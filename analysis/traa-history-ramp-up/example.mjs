import puppeteer from 'puppeteer';
import fs from 'node:fs';
const browser=await puppeteer.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:['--enable-unsafe-webgpu','--use-angle=metal','--enable-gpu','--ignore-gpu-blocklist']});
try{
 const page=await browser.newPage();await page.setViewport({width:1200,height:800});const errors=[];
 page.on('console',m=>{if(m.type()==='error'&&!m.text().includes('404'))errors.push(m.text());});page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://localhost:8141/examples/webgpu_postprocessing_ssr_denoise.html?historyRampUp=1');
 await page.waitForFunction(()=>document.querySelector('canvas')&&document.body.innerText.includes('TRAA settings'),{timeout:120000});
 await page.evaluate(()=>new Promise(resolve=>{let n=0;function next(){if(++n===120)resolve();else requestAnimationFrame(next);}requestAnimationFrame(next);}));
 const controls = await page.evaluate( () => {
  const label = Array.from( document.querySelectorAll( '*' ) ).find( e => e.children.length === 0 && e.textContent.trim() === 'history ramp-up' );
  let row = label.parentElement;
  while ( ! row.querySelector( 'input' ) ) row = row.parentElement;
  const input = row.querySelector( 'input' );
  if ( ! input.checked ) throw Error( 'Enabled URL did not initialize the switch' );
  input.click();
  if ( input.checked ) throw Error( 'Switch did not turn off' );
  const reset = Array.from( document.querySelectorAll( 'button' ) ).find( e => e.textContent.trim() === 'reset history' );
  reset.click();
  const select = document.querySelector( 'select' );
  const option = Array.from( select.options ).find( e => e.textContent === 'Compare (Denoise)' );
  select.value = option.value;
  select.dispatchEvent( new Event( 'change' ) );
  input.click();
  reset.click();
  return { initialEnabled: true, toggledOffAndOn: input.checked, resetButtonClicked: true, compareMode: option.textContent };
 } );
 await page.evaluate(()=>new Promise(resolve=>{let n=0;function next(){if(++n===60)resolve();else requestAnimationFrame(next);}requestAnimationFrame(next);}));
 console.log( controls );
 fs.writeFileSync( new URL( './example-controls.json', import.meta.url ), JSON.stringify( controls, null, 2 ) );
 await page.screenshot({path:new URL('./example.png',import.meta.url).pathname});
 fs.writeFileSync(new URL('./example-errors.json',import.meta.url),JSON.stringify(errors,null,2));console.log('Errors:',errors);
}finally{await browser.close();}
