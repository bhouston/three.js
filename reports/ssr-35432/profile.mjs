import { execFileSync } from 'node:child_process';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createServer } from 'node:http';
import { resolve, extname } from 'node:path';
import puppeteer from 'puppeteer';

const base = '29c99f4e170d01c2ecc47e44c48b29a6c57379dd';
const head = 'c85daef49196e28a687bf40c48a397defaf304be';
const root = process.cwd();
const out = resolve(process.env.SWARM_ARTIFACTS || '../.swarm/artifacts');
await mkdir(out, { recursive: true });
const source = (rev, file) => process.platform === 'win32'
 ? execFileSync('cmd.exe', ['/d', '/s', '/c', `git-dedup.cmd show ${rev}:examples/${file}`], { encoding: 'utf8', maxBuffer: 4e6 })
 : execFileSync('git-dedup', ['show', `${rev}:examples/${file}`], { encoding: 'utf8', maxBuffer: 4e6 });
const oldMirror = source(base, 'webgpu_postprocessing_ssr.html');
const newMirror = source(head, 'webgpu_postprocessing_ssr_mirror.html');
const oldDungeon = source(base, 'webgpu_postprocessing_ssr_denoise.html');
const newDungeon = source(head, 'webgpu_postprocessing_ssr_stochastic.html');
const refine = s => s.replace('binaryRefine: false', 'binaryRefine: true');
let packed = refine(oldMirror).replace(/import \{ pass, mrt,[^\n]+/, newMirror.match(/import \{ pass, mrt,[^\n]+/)[0]);
let block = newMirror.slice(newMirror.indexOf('\t\t\tpipeline = new THREE.RenderPipeline'), newMirror.indexOf('\t\t\tconst raw ='));
block = block.replaceAll('pipeline', 'renderPipeline').replace('const reflection =', 'const reflection = ssrPass =');
packed = packed.slice(0, packed.indexOf('\t\t\trenderPipeline = new THREE.RenderPipeline')) + block + '\n const outputNode = smaa( beauty.add( reflection.rgb ) );\n renderPipeline.outputNode = outputNode;\n' + packed.slice(packed.indexOf('\t\t\tcontrols = new OrbitControls'));
const noAlbedo = packed.replace('diffuseNode: albedo, ', '');
const composed = packed.replace('const outputNode = smaa( beauty.add( reflection.rgb ) );', `const raw = passTexture( reflection, reflection.getRenderTarget().texture );
 const blurred = reflection.getTextureNode();
 const outputNode = smaa( vec4( beauty.rgb.add( mix( raw.rgb, blurred.rgb, uniform( 1 ) ) ), 1 ) );`);
const variants = [
 ['mirror', '0-before', oldMirror, 'renderPipeline'],
 ['mirror', '1-binary-refine', refine(oldMirror), 'renderPipeline'],
 ['mirror', '2-repack-material-buffers', noAlbedo, 'renderPipeline'],
 ['mirror', '3-albedo-input', packed, 'renderPipeline'],
 ['mirror', '4-output-composition', composed, 'renderPipeline'],
 ['mirror', '5-final', newMirror, 'pipeline'],
 ['dungeon', '0-before', oldDungeon, 'postProcessing'],
 ['dungeon', '1-binary-refine', refine(oldDungeon), 'postProcessing'],
 ['dungeon', '2-final', newDungeon, 'postProcessing']
];
// Debug views changed their actual rendering work; measure separately from Beauty.
for (const [name, mode] of [['raw-SSR', 1], ['denoised-SSR', 4], ['compare-denoise', 8], ['compare-reflections', 9]]) {
 for (const [step, html] of [['0-before', oldDungeon], ['1-binary-refine', refine(oldDungeon)], ['2-final', newDungeon]]) {
  variants.push([`dungeon-${name}`, step, html.replace('output: 0,', `output: ${mode},`), 'postProcessing']);
 }
}
let current;
const mime = { '.js': 'text/javascript', '.html': 'text/html', '.css': 'text/css', '.json': 'application/json', '.wasm': 'application/wasm' };
const server = createServer(async (req, res) => {
 try {
  const path = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
  if (path === '/favicon.ico') { res.writeHead(204); res.end(); return; }
  if (path === '/examples/benchmark.html') {
   res.setHeader('Content-Type', 'text/html'); res.end(current); return;
  }
  const entries = { 'three.webgpu.js': 'Three.WebGPU.js', 'three.tsl.js': 'Three.TSL.js', 'three.core.js': 'Three.Core.js' };
  if (path.startsWith('/build/') && entries[path.slice(7)]) {
   res.setHeader('Content-Type', 'text/javascript'); res.end(`export * from '../src/${entries[path.slice(7)]}';`); return;
  }
  const file = resolve(root, '.' + path);
  if (!file.startsWith(root + '\\') && !file.startsWith(root + '/')) throw new Error('Outside root');
  res.setHeader('Content-Type', mime[extname(file)] || 'application/octet-stream'); res.end(await readFile(file));
 } catch { res.writeHead(404); res.end(); }
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
let browser;
const results = [];
try {
 browser = await puppeteer.launch({ executablePath: process.env.CHROME_PATH || (process.platform === 'win32' ? 'C:/Program Files/Google/Chrome/Application/chrome.exe' : puppeteer.executablePath()), headless: true,
  args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--disable-frame-rate-limit', '--disable-gpu-vsync', '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows'] });
 const probe = await browser.newPage();
 await probe.goto(`http://127.0.0.1:${server.address().port}/examples/benchmark.html`).catch(() => {});
 const cdp = await browser.target().createCDPSession();
 const system = await cdp.send('SystemInfo.getInfo');
 await writeFile(resolve(out, 'gpu-system.json'), JSON.stringify({gpu:system.gpu, commandLine:system.commandLine}, null, 2));
 await probe.close();
 const selected = variants.filter(v => !process.env.ONLY || `${v[0]}/${v[1]}`.includes(process.env.ONLY));
 for (const [example, step, html, pipeline] of selected) {
  current = html.replaceAll('renderer.setAnimationLoop( animate );', '');
  const end = current.lastIndexOf('</script>');
  current = current.slice(0, end) + `\n window.__bench = { ready: () => !!( model && ${pipeline} && controls ), renderer: () => renderer, frame: animate };\n` + current.slice(end);
  const page = await browser.newPage();
  await page.setViewport({width:1280,height:720,deviceScaleFactor:1});
  const errors = [];
  page.on('pageerror', e => errors.push(String(e)));
  page.on('console', m => {if(m.type()==='error') errors.push(m.text());});
  await page.goto(`http://127.0.0.1:${server.address().port}/examples/benchmark.html`, {waitUntil:'networkidle0',timeout:120000});
  await page.waitForFunction(() => window.__bench?.ready(), {timeout:120000});
  const result = await page.evaluate(async () => {
   const b = window.__bench;
   const renderer = b.renderer();
   // Render passes update once per node frame, not once per pipeline.render().
   // Replace RAF scheduling while reproducing Animation.update for every frame.
   renderer._animation.stop();
   renderer._animation.start = () => {};
   function frame() {
    if (renderer._inspector.isRunning) renderer._inspector.finish();
    if (renderer.info.autoReset) renderer.info.reset();
    renderer._nodes.nodeFrame.update();
    renderer.info.frame = renderer._nodes.nodeFrame.frameId;
    renderer._inspector.begin();
    b.frame();
   }
   const device = renderer.backend.device;
   if (!device) throw new Error('Not WebGPU');
   const info = device.adapterInfo;
   const adapter = Object.fromEntries(['vendor','architecture','device','description','isFallbackAdapter'].map(k=>[k,info?.[k]]));
   if (!/nvidia|10de/i.test(JSON.stringify(adapter)) || /swiftshader|llvmpipe|lavapipe|software/i.test(JSON.stringify(adapter))) throw new Error('Physical NVIDIA adapter required: '+JSON.stringify(adapter));
   async function run(ms) {
    await device.queue.onSubmittedWorkDone();
    let frames=0;
    const start=performance.now();
    do {
     for(let i=0;i<8;i++) frame();
     await device.queue.onSubmittedWorkDone();
     frames+=8;
    } while(performance.now()-start<ms);
    const elapsed=performance.now()-start;
    return {frames,seconds:elapsed/1000,fps:frames*1000/elapsed};
   }
   await run(3000);
   const trials=[];
   for(let i=0;i<3;i++) trials.push(await run(5000));
   await device.queue.onSubmittedWorkDone();
   return {adapter,trials,medianFPS:trials.map(t=>t.fps).sort((a,b)=>a-b)[1],render: {calls:renderer.info.render.calls,triangles:renderer.info.render.triangles}};
  });
  if (errors.length) throw new Error(`${example}/${step}: ${errors.join('\n')}`);
  await page.screenshot({path:resolve(out, `${example}-${step}.png`)});
  results.push({example,step,...result});
  await writeFile(resolve(out,'results.json'),JSON.stringify({base,head,browser:await browser.version(),viewport:[1280,720],batchSize:8,warmupSeconds:3,trialSeconds:5,results},null,2));
  console.log(`${example}/${step}: ${result.medianFPS.toFixed(2)} FPS`);
  await page.close();
 }
} finally {
 await browser?.close();
 await new Promise(r=>server.close(r));
}
