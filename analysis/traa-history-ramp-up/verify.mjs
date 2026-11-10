import puppeteer from 'puppeteer';
import fs from 'node:fs';
const browser=await puppeteer.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:['--enable-unsafe-webgpu','--use-angle=metal','--enable-gpu','--ignore-gpu-blocklist']});
const reports=[];
try{
 for(const backend of ['webgpu','webgl','webgpu-1x1']){
  const page=await browser.newPage(),errors=[];
  page.on('console',m=>{if(['warn','error'].includes(m.type())&&!m.text().includes('404'))errors.push(m.text());});page.on('pageerror',e=>errors.push(e.message));
  await page.goto(`http://localhost:8141/examples/measure_ssr.html?variant=new&w=${backend==='webgpu-1x1'?1:320}&h=${backend==='webgpu-1x1'?1:240}&${backend==='webgl'?'webgl=1':''}`);
  await page.waitForFunction('__ready===true',{timeout:120000});
  const result=await page.evaluate(async()=>{
   const {renderer,renderFrame,traa,target}=__test,{tick,gpuDone}=await import('./measure_common.js'),{DataUtils}=await import('three/webgpu');
   renderer._animation.stop();renderer._animation.start=()=>{};renderer._nodes.nodeFrame.frameId=0;
   const tiny = target.width === 1;
   let hardware;
   if(renderer.backend.isWebGLBackend){const gl=renderer.backend.gl,ext=gl.getExtension('WEBGL_debug_renderer_info');hardware={renderer:gl.getParameter(ext.UNMASKED_RENDERER_WEBGL),vendor:gl.getParameter(ext.UNMASKED_VENDOR_WEBGL)};}
   else{const i=renderer.backend.device.adapterInfo;hardware={vendor:i.vendor,architecture:i.architecture,isFallbackAdapter:i.isFallbackAdapter};}
   if(/swiftshader|llvmpipe|lavapipe/i.test(JSON.stringify(hardware)))throw Error('Software GPU');
   async function draw(){tick(renderer);renderFrame();if(renderer.backend.isWebGLBackend)renderer.backend.gl.finish();else await gpuDone(renderer);}
   async function stats(){
    const rt=traa._resolveRenderTarget,n=await renderer.readRenderTargetPixelsAsync(rt,0,0,rt.width,rt.height,1),color=await renderer.readRenderTargetPixelsAsync(target,0,0,target.width,target.height);let min=Infinity,max=-Infinity,alphaMin=Infinity,alphaMax=-Infinity;
    for(const bits of n){const v=DataUtils.fromHalfFloat(bits);min=Math.min(min,v);max=Math.max(max,v);}
    for(let i=3;i<color.length;i+=4){alphaMin=Math.min(alphaMin,color[i]);alphaMax=Math.max(alphaMax,color[i]);}
    return {min,max,center:DataUtils.fromHalfFloat(n[Math.floor(rt.height/2)*rt.width+Math.floor(rt.width/2)]),alphaMin,alphaMax};
   }
   await draw();const first=await stats();
   await draw();const second=await stats();
   for(let i=2;i<24;i++)await draw();const capped=await stats();
   traa.useHistoryRampUp=false;await draw();const runtimeSwitch=traa._useHistoryRampUp.value===false;
   traa.reset();await draw();const reset=await stats();
   renderer.setSize(160,120);target.setSize(160,120);await draw();const resize=await stats();
   for(const [name,s]of Object.entries({first,reset,resize}))if(s.min!==1||s.max!==1||s.alphaMin!==1||s.alphaMax!==1)throw Error(name+JSON.stringify(s));
   if((!tiny&&(second.center!==2||capped.center!==20))||!runtimeSwitch)throw Error('Count or switch failure');
   return {hardware,first,second,capped,runtimeSwitch,reset,resize};
  });
  reports.push({backend,result,errors});console.log(backend,JSON.stringify(result),errors);await page.close();
  if(errors.length)throw Error(errors.join('\n'));
 }
}finally{fs.writeFileSync(new URL('./verification.json',import.meta.url),JSON.stringify(reports,null,2));await browser.close();}
