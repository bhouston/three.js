import { readFile, writeFile, copyFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const dir = import.meta.dirname;
const artifacts = resolve(process.env.SWARM_ARTIFACTS || '../.swarm/artifacts');
const data = JSON.parse(await readFile(resolve(artifacts, 'results.json'), 'utf8'));
const groups = Map.groupBy(data.results, r => r.example);
const titles = {
 '0-before': 'Before PR', '1-binary-refine': '+ Binary refine enabled',
 '2-repack-material-buffers': '+ Repack material MRT buffers',
 '3-albedo-input': '+ Albedo input to SSR', '4-output-composition': '+ Raw/blur mix and vec4 composition',
 '5-final': '+ Remaining UI/output refactor (final)', '2-final': '+ Output/UI refactor (final)'
};
const pct = n => `${n >= 0 ? '+' : ''}${n.toFixed(1)}%`;
let md = `# SSR example profiling — PR #35432\n\n[Original PR](https://github.com/mrdoob/three.js/pull/35432). Base: \`${data.base}\`; head: \`${data.head}\`.\n\n![Incremental FPS chart](chart.svg)\n\n`;
for (const name of ['mirror', 'dungeon']) {
 const rows = groups.get(name);
 const first = rows[0].medianFPS;
 const binary = rows[1].medianFPS;
 const last = rows.at(-1).medianFPS;
 md += `**${name}: ${first.toFixed(2)} → ${last.toFixed(2)} FPS (${pct((last/first-1)*100)}).** Binary refinement alone changes it to ${binary.toFixed(2)} FPS (${pct((binary/first-1)*100)}).\n\n`;
}
md += 'Binary refinement accounts for nearly all the default Beauty slowdown. In SSRNode it changes coarse hit tolerance and adds eight bisection iterations for eligible crossings, sampling depth and normals again to refine the intersection. This is additional shader work for improved hit accuracy. The other default changes are small compared with the refinement cost; sub-percent differences should not be assigned a causal performance benefit.\n\n';
let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="${data.results.length * 32 + groups.size * 62 + 80}" role="img" aria-label="Incremental sustained FPS for SSR examples"><rect width="100%" height="100%" fill="#111827"/><style>text{fill:#e5e7eb;font:14px sans-serif}.title{font-size:20px;font-weight:bold}</style><text x="24" y="32" class="title">SSR PR #35432 — sustained GPU-completed FPS (higher is faster)</text><text x="24" y="56">GTX 1050 · 1280×720 · median of 3 × 5 s · uncapped hardware WebGPU</text>`;
let y = 80;
const max = Math.max(...data.results.map(r => r.medianFPS));
for (const [name, rows] of groups) {
 md += `## ${name}${name === 'mirror' || name === 'dungeon' ? ' — default Beauty output' : ''}\n\n| Cumulative change | Median FPS | Trial range | Change vs previous | Change vs before |\n|---|---:|---:|---:|---:|\n`;
 y += 30;
 svg += `<text x="24" y="${y}" class="title">${name}</text>`;
 for (const [i, r] of rows.entries()) {
  const prev = rows[i-1];
  const delta = prev ? pct((r.medianFPS/prev.medianFPS-1)*100) : '—';
  const overall = pct((r.medianFPS/rows[0].medianFPS-1)*100);
  const values = r.trials.map(t => t.fps);
  md += `| ${titles[r.step]} | ${r.medianFPS.toFixed(2)} | ${Math.min(...values).toFixed(2)}–${Math.max(...values).toFixed(2)} | ${delta} | ${overall} |\n`;
  y += 32;
  svg += `<text x="24" y="${y}">${titles[r.step].replaceAll('&', '&amp;')}</text><rect x="430" y="${y-17}" width="${r.medianFPS/max*470}" height="23" fill="${i===0?'#94a3b8':i===rows.length-1?'#34d399':'#60a5fa'}"/><text x="${440+r.medianFPS/max*470}" y="${y}">${r.medianFPS.toFixed(2)} (${delta})</text>`;
 }
 md += '\n';
 y += 32;
}
svg += '</svg>';
md += `## Method and interpretation\n\n- Hardware: physical NVIDIA GeForce GTX 1050 (Pascal, 2 GB), driver 560.94 (32.0.15.6094). Every variant checks the actual WebGPU device adapter: NVIDIA, non-fallback. Browser: ${data.browser}, headless, default WebGPU backend. Raw GPU diagnostics are in gpu-system.json in the swarm artifacts.\n- Fixed 1280×720 canvas, device pixel ratio 1, initial example camera, loaded models/environment, default parameters. Controls settle during warmup. No animation example is compared: it is new and has no before version.\n- Sequential variants; three-second minimum warmup, then three trials of at least five seconds. Each trial counts complete batches of eight full frames, waits for WebGPU queue completion, and divides completed frames by total wall time. GPU queues are drained before each trial and before switching variants. No individual frame timings are taken.\n- Vsync and frame limits are disabled with Chromium flags. The harness drives rendering directly rather than waiting for requestAnimationFrame. It reproduces Three.js Animation frame bookkeeping (node frame update, info reset and Inspector begin/finish), so scene, SSR, blur/denoising and AA passes execute for every counted frame. Merely calling pipeline.render repeatedly without advancing the node frame would incorrectly reuse pass results.\n- This measures sustained rendering throughput including CPU submission and GPU completion, rather than visible monitor presentation rate or a GPU-only shader time. Batching bounds queue depth and amortizes completion waits. Inspector remains attached for both versions. Screenshots are outside timed regions.\n- Both endpoints use the same head library source: this PR changes examples, metadata, screenshots and an e2e exception name, with no core or addon implementation changes. Build imports are served as source re-exports without generating repository-root build files. Original HTML is read from the pinned commits; incremental variants are generated in memory.\n- Steps are cumulative in the listed order, not independent additive costs. Small changes near trial variation should be treated as noise. Shader compilation, loading, initial shadow rendering, PMREM generation and initial temporal convergence are excluded by warmup. Results are specific to this GPU, viewport and stationary initial camera.\n\n## What actually changed\n\nThe only changed default SSR option in either existing example is binary refinement: false → true. Quality stays 0.5 (mirror) and 0.25 (dungeon); max distance stays 1 and 0.4; thickness stays 0.03 and 0.1; mirror blur quality stays 1. Existing material roughness and lighting defaults remain. The dungeon already used TRAA before the PR; the new TRAA toggle defaults to true, so it does not add a pass to default Beauty.\n\nThe mirror also replaces its normal + metal/roughness MRT layout with normal/roughness + albedo/metalness, switches from metalness/roughness to materialMetalness/materialRoughness, supplies albedo to SSR, and changes final composition to an explicit raw/blur mix. Those are measured separately above. Both layouts use two unsigned-byte auxiliary attachments; the repacking does not remove an attachment.\n\nDungeon raw SSR, denoised SSR and denoise comparison outputs now bypass TRAA/sharpen, which changes their rendering cost. The reflection comparison still applies temporal postprocessing to both sides. These views have their own before → binary-refine → final series, so their change is not confused with default Beauty.\n\nRenames, titles, descriptions, slider ranges/labels, folder placement, added pointer-release handlers and the e2e exception rename do not change default GPU rendering work. The final steps include their aggregate CPU/UI effect rather than assigning unsupported FPS gains to individual labels. Blur and denoise toggles select via shader mix; disabling one does not establish that its producer pass stops executing. No off-toggle speedup is claimed.\n\n## Reproduce and validation\n\nFrom the repository root at the profiled head or this report branch:\n\n\`\`\`powershell\nnpm.cmd ci\nnode reports/ssr-35432/profile.mjs\nnode reports/ssr-35432/report.mjs\n\`\`\`\n\nOptional environment variables: CHROME_PATH, SWARM_ARTIFACTS (output directory), ONLY (substring selecting example/step). Use a physical NVIDIA GPU or adapt the explicit hardware assertion to another physical adapter. The server and browser shut down on completion or failure. Do not run other GPU workloads during measurement.\n\nValidation: npm test passed: lint-core, 1,982 core tests (one TODO), 147 addon tests; zero failed tests. The addon runner emitted a resource 404 despite passing all tests. node --check passed for the harness. No typecheck script exists. Hardware WebGPU rendering was exercised for every measured variant; screenshots and per-trial frame counts/durations are preserved in the swarm artifacts.\n`;
await writeFile(resolve(dir, 'README.md'), md);
await writeFile(resolve(dir, 'chart.svg'), svg);
await copyFile(resolve(artifacts, 'results.json'), resolve(dir, 'results.json'));
await writeFile(resolve(artifacts, 'ssr-performance-report.md'), md);
await writeFile(resolve(artifacts, 'chart.svg'), svg);
console.log('Generated README.md, chart.svg and results.json');
