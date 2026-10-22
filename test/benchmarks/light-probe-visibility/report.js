import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';

const output = path.resolve( process.env.PROBE_RESULTS || path.join( os.homedir(), 'Coding/.benchmarks/three-light-probe-visibility' ) );
const report = JSON.parse( await readFile( path.join( output, 'results.json' ), 'utf8' ) );
const csv = [ 'variant,scene,wallMean,wallP95,wallMax,litMean,interiorMean,distanceBytes,completedFrames,seconds,fps' ];
for ( const v of report.variants ) for ( const m of v.measurements ) csv.push( [ v.variant, m.name, m.leak.mean, m.leak.p95, m.leak.max, m.lit.mean, m.interior.mean, m.distanceBytes, v.throughput.completed, v.throughput.seconds, v.throughput.fps ].join( ',' ) );
await writeFile( path.join( output, 'summary.csv' ), csv.join( '\n' ) + '\n' );
const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Light probe visibility experiment results</title>
<style>
:root{color-scheme:dark}body{margin:0;background:#111822;color:#e5edf8;font:16px/1.55 system-ui}main{max-width:1350px;padding:32px;margin:auto}h1{font-size:32px;line-height:1.2}h2{font-size:22px;margin-top:36px}p{max-width:1050px;color:#bacbdd}a{color:#83c8ff}select{font:inherit;padding:8px;margin-right:20px;background:#243448;color:white;border:1px solid #718196;border-radius:5px}table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}th,td{padding:10px;border-bottom:1px solid #344152;text-align:right}th:first-child,td:first-child{text-align:left}tr:hover{background:#203145}th{color:#a7caec;background:#1a2635}section{overflow:auto}.images{display:grid;grid-template-columns:1fr 1fr;gap:20px}.images img{width:100%;display:block}.heat{image-rendering:pixelated;max-width:240px}figure{margin:0;background:#1a2635;padding:12px;border-radius:6px}figcaption{margin-bottom:10px;color:#bcd1e7}.tag{display:inline-block;background:#284639;color:#b5f4d3;border-radius:5px;padding:4px 9px}small{color:#acbdd1}pre{overflow:auto;font-size:13px;padding:16px;background:#1a2635}.notes{border-left:4px solid #72b5ff;padding-left:18px}@media(max-width:750px){main{padding:18px}.images{grid-template-columns:1fr}}
</style></head><body><main>
<span class="tag">16 independent variants · 10 geometries · real Apple GPU</span>
<h1>Where probe light leakage comes from</h1>
<p>Every experiment changes one factor from commit <b>966be8df9d0a</b>. No fixes are stacked. The production example and lighting implementation remain unchanged. Measurements below are from synthetic fixtures with a known zero-light reference on the dark-facing wall. The complex-scene screenshots are visual comparisons, not ground-truth error scores.</p>
<p class="notes"><b>Main finding:</b> 32² visibility maps, stronger bias and weak-weight suppression improve thin-wall visibility. An exact wall-segment test removes closed-wall errors but leaves most doorway error. Building irradiance directly from captured radiance, instead of low-order SH, produces zero dark-wall readings in these fixtures while retaining 99.8% of baseline thin-wall lit brightness and 98.1% of baseline opening brightness. This identifies angular approximation as a separate source of error; it does not prove general leak-free lighting.</p>
<h2>Measured comparisons</h2>
<label>Geometry <select id="scene"></select></label>
<p><small>Wall mean, p95 and maximum are linear luminance; the reference is zero. Lower is better. Lit and opening columns help expose settings that simply darken everything. The dark-room surface should be black for closed partitions; through the doorway its illumination is legitimate. Images use the same 8× exposure for every variant.</small></p>
<section><table><thead><tr><th>Independent change</th><th>Wall mean</th><th>p95</th><th>Maximum</th><th>Lit reference</th><th>Interior/opening</th><th>Frames/s*</th><th>Distance MiB</th></tr></thead><tbody id="numbers"></tbody></table></section>
<p><small>* Single five-second sustained offscreen 512×512 microbenchmark per variant. Forty warmup renders, queue drained first, batches counted only after GPU completion. Vsync and frame limits disabled. Simple receiver scene; these are not complex-scene frame rates. Small throughput differences are within run-to-run noise. Direct convolution adds bake work that this runtime benchmark does not measure.</small></p>
<h2>Controlled fixture: selected variant against baseline</h2>
<label>Variant <select id="variant"></select></label><label>Surface <select id="surface"><option value="leak">Dark-facing wall</option><option value="lit">Lit reference</option><option value="interior">Dark-room surface/opening</option></select></label>
<div class="images"><figure><figcaption>Baseline · fixed exposure</figcaption><img id="baseImage" class="heat"></figure><figure><figcaption id="testCaption"></figcaption><img id="testImage" class="heat"></figure></div>
<h2>Actual complex scene</h2>
<label>View <select id="view"><option value="door">Divider close view</option><option value="front">Original front view</option></select></label>
<div class="images"><figure><figcaption>Committed single-grid baseline</figcaption><img id="complexBase"></figure><figure><figcaption id="complexCaption"></figcaption><img id="complexTest"></figure></div>
<h2>Probe inputs at the center of the selected dark-wall patch</h2>
<p>These diagnostic values come from baseline distance-atlas readback. The exact blocker label is an independent CPU segment/box intersection. A probe may have nonzero visibility while its trilinear weight is zero. The sample is a point inspection, not an average over the entire patch.</p>
<section><table><thead><tr><th>Probe position</th><th>Mean distance</th><th>Receiver distance</th><th>Visibility</th><th>Final weight</th><th>Blocked exactly?</th></tr></thead><tbody id="diagnostics"></tbody></table></section>
<h2>What the tests cover</h2>
<p>Closed walls of thickness 0.05, 0.15, 0.3, 0.6 and 1.2; a doorway; an L corner; two parallel walls; a 30° wall; and an oblique camera. All use 8×5×7 probes with fixed positions, 32² cubemap faces, a black absorbing room and partitions, and a white radiance emitter in the right chamber. The white receiver is added after baking. A no-GI control verifies the leakage crop contains no directly visible source or directly lit pixels. Only the isolated radiance source emits; there are no direct lights, shadow-map errors or multiple bounces in the controlled fixtures.</p>
<p><b>Limitations:</b> one fixed probe layout, deterministic capture samples, static geometry and one hardware adapter. These fixtures are not a full path-traced reference for the complex scene. Exact visibility tests only the explicit partition boxes. Region masking uses known sides of the main partition plane, including its rotation; arbitrary rooms and portal transitions need a proper region representation. Large sampling biases may erase details or become view dependent elsewhere. More directional depth texels increase storage by 3.57× here. Direct convolution uses 256 cosine samples per irradiance texel and retains the existing 8² directional irradiance atlas, so sampling and spatial approximation remain.</p>
<h2>Reproduce</h2>
<pre>node test/benchmarks/light-probe-visibility/runner.js
node test/benchmarks/light-probe-visibility/complex.js
node test/benchmarks/light-probe-visibility/report.js</pre>
<p><a href="results.json">Full JSON, including GPU identity and completion counts</a> · <a href="summary.csv">CSV measurements</a> · <a href="http://localhost:8081/test/benchmarks/light-probe-visibility/index.html">Interactive baseline fixtures</a></p>
<script>
const report=${JSON.stringify( report )};
const names={'baseline':'Baseline','visibility-off':'Visibility disabled (control)','slack-0':'Slack 0','slack-0.005':'Slack 0.005','slack-0.01':'Slack 0.01','crush':'Weak-weight suppression','distance-32':'32² distance maps','normal-0':'Normal bias 0','normal-0.02':'Normal bias 0.02','normal-0.16':'Normal bias 0.16','view-0':'View bias 0','view-0.08':'View bias 0.08','view-0.16':'View bias 0.16','exact':'Exact partition visibility','region-mask':'Partition-region mask','direct-convolution':'Direct irradiance convolution'};
const scene=document.querySelector('#scene'),variant=document.querySelector('#variant'),surface=document.querySelector('#surface'),view=document.querySelector('#view');
for(const m of report.variants[0].measurements)scene.add(new Option(m.name,m.name));
for(const v of report.variants)variant.add(new Option(names[v.variant],v.variant));
variant.value='direct-convolution';scene.value='doorway';
const num=n=>n===0?'0':n<0.001?n.toExponential(3):n.toFixed(5);
function update(){
 const base=report.variants[0].measurements.find(m=>m.name===scene.value);
 document.querySelector('#numbers').innerHTML=report.variants.map(v=>{const m=v.measurements.find(m=>m.name===scene.value);return '<tr><td>'+names[v.variant]+'</td><td>'+num(m.leak.mean)+'</td><td>'+num(m.leak.p95)+'</td><td>'+num(m.leak.max)+'</td><td>'+num(m.lit.mean)+'</td><td>'+num(m.interior.mean)+'</td><td>'+Math.round(v.throughput.fps)+'</td><td>'+(m.distanceBytes/1048576).toFixed(3)+'</td></tr>';}).join('');
 const test=report.variants.find(v=>v.variant===variant.value).measurements.find(m=>m.name===scene.value);
 document.querySelector('#baseImage').src=base[surface.value].image;document.querySelector('#testImage').src=test[surface.value].image;
 document.querySelector('#testCaption').textContent=names[variant.value]+' · fixed exposure';
 document.querySelector('#complexCaption').textContent=names[variant.value];
 document.querySelector('#complexBase').src='complex/baseline-'+view.value+'.png';document.querySelector('#complexTest').src='complex/'+variant.value+'-'+view.value+'.png';
 document.querySelector('#diagnostics').innerHTML=(base.diagnostics?.wall.corners||[]).map(c=>'<tr><td>'+c.position.map(x=>x.toFixed(3)).join(', ')+'</td><td>'+num(c.meanDistance)+'</td><td>'+num(c.distance)+'</td><td>'+num(c.visibility)+'</td><td>'+num(c.weight)+'</td><td>'+c.exactBlocked+'</td></tr>').join('');
}
for(const s of [scene,variant,surface,view])s.onchange=update;update();
</script></main></body></html>`;
await writeFile( path.join( output, 'report.html' ), html );
console.log( path.join( output, 'report.html' ) );
