// Hardware GPU measurement driver, adapted from houston-stratified-noise/analysis/gpu/run.mjs.
import puppeteer from 'puppeteer';
import fs from 'node:fs';
const OUT = new URL('./raw/', import.meta.url);
fs.mkdirSync(OUT,{recursive:true});
const browser = await puppeteer.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,protocolTimeout:0,args:['--enable-unsafe-webgpu','--use-angle=metal','--enable-gpu','--ignore-gpu-blocklist','--disable-frame-rate-limit','--disable-gpu-vsync','--disable-background-timer-throttling','--disable-renderer-backgrounding']});
const results=[];
async function open(effect,variant){
 const page=await browser.newPage();
 const errors=[];
 page.on('console',m=>{if(m.type()==='error'&&!m.text().includes('404')) errors.push(m.text());});
 page.on('pageerror',e=>errors.push(e.message));
 await page.exposeFunction('__save',(name,b64)=>fs.writeFileSync(new URL(name,OUT),Buffer.from(b64,'base64')));
 await page.goto(`http://localhost:8141/examples/measure_${effect}.html?variant=${variant}&w=640&h=480`);
 await page.waitForFunction('window.__ready===true',{timeout:120000});
 const adapter=await page.evaluate(()=>__adapter);
 if(adapter.isFallbackAdapter!==false||!/apple|nvidia|amd|intel/i.test(adapter.vendor)||/swiftshader|llvmpipe|lavapipe/i.test(JSON.stringify(adapter))) throw Error(JSON.stringify(adapter));
 await page.evaluate(()=>{const renderer=__test.renderer; renderer._animation.stop(); renderer._animation.start=()=>{}; renderer._nodes.nodeFrame.frameId=0; renderer.info.frame=0; const i=renderer.backend.device.adapterInfo; window.__actualAdapter={vendor:i.vendor,architecture:i.architecture,device:i.device,description:i.description,isFallbackAdapter:i.isFallbackAdapter};});
 console.log(effect,variant,adapter,await page.evaluate(()=>__actualAdapter));
 return {page,errors,adapter};
}
try {
 for(const effect of ['ssr','vxgi']){
  // Long temporal mean of the actual unchanged noise and denoiser, linear post-TRAA.
  let {page,errors,adapter}=await open(effect,'old');
  const reference=await page.evaluate(()=>__capture({name:'reference',warmup:512,frames:512}));
  // __save names must be effect specific before opening the next effect.
  for(const suffix of ['mean','last'])fs.renameSync(new URL(`reference_${suffix}.f32`,OUT),new URL(`${effect}_reference_${suffix}.f32`,OUT));
  results.push({effect,kind:'reference',adapter,reference,errors});
  fs.writeFileSync(new URL('runs.json',OUT),JSON.stringify(results,null,2));
  console.log(effect,'reference complete',reference.ms);
  await page.close();
  for(let repeat=0;repeat<3;repeat++)for(const variant of ['old','new']){
   ({page,errors,adapter}=await open(effect,variant));
   const name=`${effect}_${variant}_${repeat}`;
   const convergence=await page.evaluate(async({name,repeat})=>{
    const {renderer,renderFrame,traa,denoise,target}=__test;
    const {tick,gpuDone,save}=await import('./measure_common.js');
    for(let i=0;i<256+repeat*37;i++){tick(renderer);renderFrame();if(i%4===3)await gpuDone(renderer);}await gpuDone(renderer);
    traa.reset(); if(denoise)denoise.reset();
    // Underlying stochastic sequence and jitter continue after the history reset.
    const checks=[1,2,4,8,16,32,64],stats=[];
    for(let f=1;f<=64;f++){
     tick(renderer);renderFrame();
     if(checks.includes(f)){
      const rgba=await renderer.readRenderTargetPixelsAsync(target,0,0,target.width,target.height);
      const rgb=new Float32Array(target.width*target.height*3);let nonFinite=0;
      for(let i=0,j=0;i<rgba.length;i+=4,j+=3)for(let c=0;c<3;c++){rgb[j+c]=rgba[i+c];if(!Number.isFinite(rgb[j+c]))nonFinite++;}
      await save(`${name}_f${f}.f32`,rgb);stats.push({frame:f,nonFinite});
     }else if(f%4===0)await gpuDone(renderer);
    }
    await gpuDone(renderer);return stats;
   },{name,repeat});
   const steady=await page.evaluate(name=>__capture({name,warmup:256,frames:256}),name+'_steady');
   results.push({effect,variant,repeat,kind:'quality',adapter,convergence,steady,errors});
   fs.writeFileSync(new URL('runs.json',OUT),JSON.stringify(results,null,2));
   console.log(name,'quality complete',steady.ms,'errors',errors.length);
   if(errors.length)throw Error(errors.join('\n'));
   await page.close();
  }
  for(const variant of ['old','new']){
   ({page,errors,adapter}=await open(effect,variant));
   const motion=await page.evaluate(async name=>{
    const {renderer,renderFrame,camera,lookAt,target}=__test;
    const {tick,gpuDone,save}=await import('./measure_common.js');
    for(let i=0;i<256;i++){tick(renderer);renderFrame();if(i%4===3)await gpuDone(renderer);}await gpuDone(renderer);
    const center=new (await import('three/webgpu')).Vector3(...lookAt),start=camera.position.clone().sub(center);
    for(let f=1;f<=128;f++){
     camera.position.copy(start).applyAxisAngle(new (await import('three/webgpu')).Vector3(0,1,0),f*0.001).add(center);camera.lookAt(center);camera.updateMatrixWorld();
     tick(renderer);renderFrame();
     if([32,64,96,128].includes(f)){
      const rgba=await renderer.readRenderTargetPixelsAsync(target,0,0,target.width,target.height),rgb=new Float32Array(target.width*target.height*3);
      for(let i=0,j=0;i<rgba.length;i+=4,j+=3)for(let c=0;c<3;c++)rgb[j+c]=rgba[i+c];
      await save(`${name}_f${f}.f32`,rgb);
     }else if(f%4===0)await gpuDone(renderer);
    }await gpuDone(renderer);return {frames:128,radiansPerFrame:0.001};
   },`${effect}_${variant}_motion`);
   results.push({effect,variant,kind:'motion',adapter,motion,errors});
   fs.writeFileSync(new URL('runs.json',OUT),JSON.stringify(results,null,2));
   console.log(effect,variant,'motion complete');await page.close();
  }
  for(let repeat=0;repeat<3;repeat++)for(const variant of ['original','old','new']){
   ({page,errors,adapter}=await open(effect,variant));
   const throughput=await page.evaluate(()=>__throughput(5));
   results.push({effect,variant,repeat,kind:'throughput',adapter,throughput,errors});
   fs.writeFileSync(new URL('runs.json',OUT),JSON.stringify(results,null,2));
   console.log(effect,variant,repeat,throughput.fps,'fps');if(errors.length)throw Error(errors.join('\n'));
   await page.close();
  }
 }
}finally{await browser.close();}
