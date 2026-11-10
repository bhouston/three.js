import puppeteer from 'puppeteer';
import fs from 'node:fs';
const browser=await puppeteer.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,protocolTimeout:180000,args:['--enable-unsafe-webgpu','--use-angle=metal','--enable-gpu','--ignore-gpu-blocklist']});
const names=process.env.EXAMPLES?.split(',')??['postprocessing_ssr_denoise','vxgi','postprocessing_ssgi','postprocessing_ao','vxgi_sponza','volume_lighting_traa','postprocessing_traa'];
const reports=fs.existsSync(new URL('./examples-controls.json',import.meta.url))?JSON.parse(fs.readFileSync(new URL('./examples-controls.json',import.meta.url))):[];
try{
 for(const name of names){
  const page=await browser.newPage(),errors=[],failed=[];await page.setViewport({width:800,height:600});
  page.on('requestfailed',r=>console.log('request failed',r.url(),r.failure()?.errorText));page.on('pageerror',e=>{errors.push(e.message);console.log('page error',e.message);});page.on('console',m=>{if(m.type()==='error'&&!m.text().includes('404'))errors.push(m.text());});page.on('response',r=>{if(r.status()>=400&&!r.url().includes('favicon'))failed.push([r.status(),r.url()]);});
  if(process.env.SPONZA_CACHE){
   await page.setRequestInterception(true);
   page.on('request',async r=>{
    const url=r.url();
    if(url.startsWith('https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/')){
     const file=new URL('./raw/sponza-cache/'+url.split('/').at(-1),import.meta.url);
     if(fs.existsSync(file))return r.respond({status:200,headers:{'access-control-allow-origin':'*'},body:fs.readFileSync(file)});
    }
    r.continue();
   });
  }
  await page.goto(`http://localhost:8141/examples/webgpu_${name}.html?historyRampUp=1`);
  await page.waitForFunction(()=>document.body.innerText.includes('history ramp-up'),{timeout:180000});
  await page.evaluate(async()=>{
   const {default:TRAA}=await import('./jsm/tsl/display/TRAANode.js');
   const update=TRAA.prototype.updateBefore,reset=TRAA.prototype.reset;
   window.__rampObserved=[];window.__resetCalls=0;
   TRAA.prototype.updateBefore=function(frame){
    __rampObserved.push(this.useHistoryRampUp);
    const i=frame.renderer.backend.device.adapterInfo;window.__hardware={vendor:i.vendor,architecture:i.architecture,isFallbackAdapter:i.isFallbackAdapter};
    return update.call(this,frame);
   };
   TRAA.prototype.reset=function(){__resetCalls++;return reset.call(this);};
  });
  await page.waitForFunction('window.__rampObserved.length > 0',{timeout:180000});
  await page.evaluate(()=>{
   if(!__rampObserved.at(-1))throw Error('Enabled query not honored');
   const label=Array.from(document.querySelectorAll('*')).find(e=>e.children.length===0&&e.textContent.trim()==='history ramp-up');
   let row=label.parentElement;while(!row.querySelector('input'))row=row.parentElement;
   window.__checkbox=row.querySelector('input');if(!__checkbox.checked)throw Error('Checkbox initial state');
   __checkbox.click();
  });
  await page.waitForFunction('window.__rampObserved.at(-1) === false');
  const result=await page.evaluate(()=>{
   const before=__resetCalls;Array.from(document.querySelectorAll('button')).find(e=>e.textContent.trim()==='reset history').click();
   if(__resetCalls<=before)throw Error('Reset button did not reset TRAA');
   __checkbox.click();return {hardware:__hardware,offObserved:true,resetCalls:__resetCalls};
  });
  await page.waitForFunction('window.__rampObserved.at(-1) === true');
  if(result.hardware.vendor!=='apple'||result.hardware.isFallbackAdapter!==false)throw Error('Unexpected adapter');
  reports.push({name,result,errors,failed});console.log(name,JSON.stringify(reports.at(-1)));
  if(errors.length||failed.length)throw Error('Example errors');
  await page.screenshot({path:new URL(`./${name}-controls.png`,import.meta.url).pathname});await page.close();
 }
}finally{fs.writeFileSync(new URL('./examples-controls.json',import.meta.url),JSON.stringify(reports,null,2));await browser.close();}
