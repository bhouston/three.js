# WebGPU LLM performance log

This file records comparable performance runs for the WebGPU LLM example. New optimization results should be appended rather than replacing earlier measurements.

## Benchmark protocol

- Command: `npm run test-performance-llm -- --models=tinystories,smollm2 --trials=5 --tokens=32`
- Prompt: `Once upon a time, in a quiet village by the sea, a curious child found a small brass key beneath an old oak tree.`
- Prompt length: 27 tokens for both baseline models
- Sampling: greedy (`temperature = 0`, `topK = 1`), with repetition and n-gram penalties disabled
- Trials: one first-use compilation run followed by five measured warm runs
- Cache policy: browser HTTP cache disabled; the runner cache is reset before every generation trial
- Rates: prefill includes the last-prompt logits readback; decode excludes TTFT and uses intervals between streamed tokens
- Date: 2026-09-02
- Source revision: `5a3cd603c82556fe8be15492cfb8248ca0998be9`, with the current uncommitted LLM worktree changes
- Browser: Chrome 152.0.0.0
- GPU backend: Apple, Metal 3

Absolute values are machine-, browser-, thermal-, and driver-dependent. Compare sections produced on the same machine with the same command and protocol.

## Baseline: scalar kernels and one submission per dispatch

### TinyStories GPT-2 3M

- Checkpoint transferred from local server: 14.3 MB
- Renderer initialization: 3.9 ms
- Model load, parse, unpack, and runner construction: 148.0 ms
- First-use TTFT, including pipeline creation and GPU upload: 231.2 ms
- Warm TTFT median: 170.8 ms (170.0–172.0 ms)
- Warm prefill throughput median: 158.080 tokens/s
- Warm decode throughput median: 119.507 tokens/s (115.199–119.599 tokens/s)
- Warm end-to-end generation median: 72.959 tokens/s
- Typical decode-token latency: 8.4 ms; trial p95 values were 8.7–9.3 ms
- Command submissions per model forward: 112
- Compute dispatches per model forward: 112
- Logit readbacks per 27-prompt + 32-generated-token trial: 231

### SmolLM2 135M

- Checkpoint transferred from local server: 256.6 MB BF16
- Renderer initialization: 3.4 ms
- Model load, BF16 conversion, unpack, and runner construction: 2629.1 ms
- First-use TTFT, including pipeline creation and GPU upload: 1168.2 ms
- Warm TTFT median: 759.8 ms (759.4–765.2 ms)
- Warm prefill throughput median: 35.536 tokens/s
- Warm decode throughput median: 33.532 tokens/s (33.448–33.586 tokens/s)
- Warm end-to-end generation median: 18.648 tokens/s
- Typical decode-token latency: 29.9 ms; trial p95 values were 30.3–30.7 ms
- Command submissions per model forward: 427
- Compute dispatches per model forward: 427
- Logit readbacks per 27-prompt + 32-generated-token trial: 198

### Baseline observations

- SmolLM2 submits 25,193 command buffers during each measured trial. Queue submission overhead is therefore a high-confidence optimization target.
- Dispatch count and command submission count are currently identical because every TSL kernel calls `renderer.compute()` independently.
- TinyStories performs seven sequential vocabulary readbacks per sampled token; SmolLM2 performs six.
- SmolLM2 loading spends roughly 1.63 seconds converting BF16 tensors to float32 and about 0.53 seconds unpacking, transposing, and constructing the runner after download.
- The first-use penalty is substantial, particularly for SmolLM2, and must remain separate from warm token throughput.

## Optimization 1: batch all dispatches into one submission per forward

The runners now pass each token's ordered compute-node list to one `renderer.compute()` call. This preserves the exact 112 TinyStories and 427 SmolLM2 dispatches, but records and submits them in one compute pass.

### TinyStories GPT-2 3M

- Model load: 145.3 ms; effectively unchanged
- First-use TTFT: 129.2 ms, 44.1% lower than baseline
- Warm TTFT median: 31.5 ms, 81.6% lower
- Warm prefill throughput median: 857.143 tokens/s, 442.2% higher
- Warm decode throughput median: 225.619 tokens/s, 88.8% higher
- Warm end-to-end generation median: 182.648 tokens/s, 150.3% higher
- Command submissions per model forward: 1, down from 112
- Compute dispatches per model forward: 112, unchanged
- Logit readbacks per trial: 231, unchanged

### SmolLM2 135M

- Model load: 2616.9 ms; effectively unchanged
- First-use TTFT: 785.7 ms, 32.7% lower than baseline
- Warm TTFT median: 294.0 ms, 61.3% lower
- Warm prefill throughput median: 91.837 tokens/s, 158.4% higher
- Warm decode throughput median: 68.162 tokens/s, 103.3% higher
- Warm end-to-end generation median: 41.956 tokens/s, 125.0% higher
- Command submissions per model forward: 1, down from 427
- Compute dispatches per model forward: 427, unchanged
- Logit readbacks per trial: 198, unchanged

### Result

Submission batching more than doubled representative SmolLM2 decode throughput without changing dispatch count or model arithmetic. TinyStories trial variance increased because each generation now completes in roughly 150–200 ms, but every measured trial remained substantially faster than baseline.

## Experiment 2: shared-workgroup normalization reductions (not retained)

RMSNorm and LayerNorm were rewritten so one workgroup cooperatively reduced each vector instead of every output invocation scanning the vector independently.

### Measured result relative to Optimization 1

- TinyStories warm TTFT: 30.1 ms, 4.4% lower
- TinyStories prefill: 897.010 tokens/s, 4.7% higher
- TinyStories decode: 255.354 tokens/s, 13.2% higher, but with substantial short-trial variance
- SmolLM2 warm TTFT: 293.3 ms, effectively unchanged
- SmolLM2 prefill: 92.056 tokens/s, effectively unchanged
- SmolLM2 decode: 67.745 tokens/s, 0.6% lower
- SmolLM2 end-to-end: 41.645 tokens/s, 0.7% lower
- SmolLM2 first-use TTFT: 918.1 ms, 16.8% worse
- Dispatch and submission counts were unchanged

### Decision

This experiment was reverted. On Apple Metal, the old lockstep reads appear to benefit from cache/broadcast behavior, while the replacement adds workgroup storage, barriers, and more shader compilation. TinyStories improved, but the representative SmolLM2 model did not, and first-use latency regressed materially.

## Optimization 2: parallel chunked-logit readbacks

The full vocabulary is still read back to the CPU, but all vocabulary chunks now begin their copy/map operations together with `Promise.all()` instead of serializing six or seven GPU synchronization round trips.

### TinyStories GPT-2 3M

- Model load: 146.3 ms; effectively unchanged
- First-use TTFT: 122.5 ms, 5.2% lower than Optimization 1
- Warm TTFT median: 23.8 ms, 24.4% lower
- Warm prefill throughput median: 1134.454 tokens/s, 32.4% higher
- Warm decode throughput median: 565.693 tokens/s, 150.7% higher
- Warm end-to-end generation median: 398.506 tokens/s, 118.2% higher
- Command submissions per model forward: 1
- Compute dispatches per model forward: 112
- Logit readback count: unchanged at 231 per trial, but the seven reads per sampling step overlap

### SmolLM2 135M

- Model load: 2749.1 ms; within expected load-time variance
- First-use TTFT: 770.7 ms, 1.9% lower than Optimization 1
- Warm TTFT median: 293.1 ms, effectively unchanged
- Warm prefill throughput median: 92.119 tokens/s, effectively unchanged
- Warm decode throughput median: 75.407 tokens/s, 10.6% higher
- Warm end-to-end generation median: 44.017 tokens/s, 4.9% higher
- Command submissions per model forward: 1
- Compute dispatches per model forward: 427
- Logit readback count: unchanged at 198 per trial, but the six reads per sampling step overlap

### Cumulative result versus baseline

- TinyStories decode improved from 119.507 to 565.693 tokens/s: 4.73×
- SmolLM2 decode improved from 33.532 to 75.407 tokens/s: 2.25×
- SmolLM2 warm TTFT improved from 759.8 to 293.1 ms: 61.4% lower
- SmolLM2 still reads the entire 49,152-value vocabulary for every sampled token. GPU top-k remains the clearest next synchronization reduction.

### Verification

- `npm run test-unit-addons`: 436 passed, 0 failed
- ESLint passed for all benchmark and optimized LLM files
- `git diff --check`: passed

## Experiment 3: vec4-packed GEMV

The standalone command `npm run test-performance-llm-vec4 -- --iterations=100 --trials=7` compares:

- The current scalar kernel with `[input, output]` weights
- Four adjacent outputs accumulated together in one `vec4`
- Input-axis `dot(vec4, vec4)` with weights repacked to `[output, input/4, vec4]`

Each result is the median wall time per dispatch after warmup. Numerical maximum error versus the scalar kernel was below `3.2e-8`.

### Standalone Apple Metal 3 results

- SmolLM2 up projection, 576→1536:
  - Scalar: 0.049 ms, 36.1 GFLOP/s
  - Four-output vec4: 0.079 ms, 0.62× baseline
  - Vec4 dot: 0.042 ms, 1.17× baseline
- SmolLM2 down projection, 1536→576:
  - Scalar: 0.081 ms, 21.8 GFLOP/s
  - Four-output vec4: 0.188 ms, 0.43× baseline
  - Vec4 dot: 0.061 ms, 1.33× baseline
- Vocabulary chunk, 576→8192:
  - Scalar: 0.222 ms, 42.5 GFLOP/s
  - Four-output vec4: 0.228 ms, 0.97× baseline
  - Vec4 dot: 0.227 ms, 0.98× baseline

Repacking one matrix for vec4-dot layout took 1.8 ms, 3.3 ms, and 72.4 ms for the three shapes respectively.

### Whole-model experiments

Vectorizing every eligible linear produced:

- TinyStories decode: 586.011 tokens/s, 3.6% above Optimization 2 but within its high short-trial variance
- SmolLM2 decode: 73.986 tokens/s, 1.9% below Optimization 2
- SmolLM2 warm TTFT: 301.6 ms, 2.9% worse
- SmolLM2 first-use TTFT: 911.0 ms, 18.2% worse
- SmolLM2 model load: 2877.0 ms, with substantial additional temporary CPU memory for repacked weights

Restricting vec4-dot to contraction/down projections produced:

- TinyStories decode: 541.958 tokens/s, 4.2% below Optimization 2
- SmolLM2 decode: 75.962 tokens/s, 0.7% above Optimization 2
- SmolLM2 warm TTFT: 294.9 ms, 0.6% worse
- SmolLM2 end-to-end generation: 44.705 tokens/s, 1.6% higher
- SmolLM2 first-use TTFT: 782.8 ms, 1.6% worse

### Decision

The vec4 implementation was not retained in the production LLM path. A vec4 dot is beneficial for isolated contraction matrices, confirming that explicit vector arithmetic can help, but the whole-model gain was within run-to-run noise and required an extra transposed copy of every selected weight matrix. The additional memory and first-use cost do not justify a 0.7% representative decode improvement.

The four-output vec4 approach is actively worse because the existing scalar layout already lets adjacent GPU invocations read adjacent coefficients, while one invocation accumulating four outputs reduces available parallelism and increases register pressure.

A future implementation should avoid runtime repacking by preserving `[output, input]` checkpoint layout during loading, then retest vec4-dot as part of an F16 or quantized kernel where reduced weight bandwidth can compound the vectorization benefit.

## Optimization 3: GPU greedy sampling and small top-k candidates

The TSL runners now add a GPU-side logit candidate reducer after the chunked vocabulary projection. Greedy sampling uses a workgroup reduction per vocabulary chunk, merges the chunk maxima on the GPU, and reads back one token ID per sampled token instead of reading every chunk of full-vocabulary logits. A small exact top-k candidate path is also available for top-k values within the runner's candidate budget; sampling modes that need repetition, presence, frequency, or no-repeat-ngram penalties still fall back to the full-logit CPU path.

The isolated benchmark used the current harness with chunked prefill disabled so this section measures sampling only:

```sh
npm run test-performance-llm -- --models=tinystories,smollm2 --trials=5 --tokens=32 --headless --gpuSampling=false --prefillMode=false --output=test/benchmark/results/tsl-gpu-sampling-baseline.json
npm run test-performance-llm -- --models=tinystories,smollm2 --trials=5 --tokens=32 --headless --prefillMode=false --output=test/benchmark/results/tsl-gpu-sampling.json
```

### Paired TinyStories and SmolLM2 results

| Model | Mode | Warm TTFT median | Warm decode median | Warm end-to-end median | Readbacks per trial | Dispatches per forward |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TinyStories GPT-2 3M | CPU full-logit sampling | 36.7 ms | 232.558 tok/s | 182.753 tok/s | 231 | 108.475 |
| TinyStories GPT-2 3M | GPU greedy sampling | 23.8 ms | 509.868 tok/s | 372.960 tok/s | 32 | 112.949 |
| SmolLM2 135M | CPU full-logit sampling | 262.7 ms | 79.446 tok/s | 48.331 tok/s | 198 | 423.915 |
| SmolLM2 135M | GPU greedy sampling | 263.1 ms | 82.250 tok/s | 49.953 tok/s | 32 | 427.831 |

### Qwen3.5 high-vocabulary check

The Qwen run used three trials and 16 generated tokens to keep the larger checkpoint iteration time practical:

```sh
npm run test-performance-llm -- --models=qwen3.5-0.8b --trials=3 --tokens=16 --headless --prefillMode=false --gpuSampling=false --output=test/benchmark/results/tsl-gpu-sampling-qwen-baseline.json
npm run test-performance-llm -- --models=qwen3.5-0.8b --trials=3 --tokens=16 --headless --prefillMode=false --output=test/benchmark/results/tsl-gpu-sampling-qwen.json
```

| Model | Mode | Warm TTFT median | Warm decode median | Readbacks per trial | Dispatches per forward |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.5 0.8B | CPU full-logit sampling | 850.0 ms | 20.298 tok/s | 527 | 456.651 |
| Qwen3.5 0.8B | GPU greedy sampling | 851.1 ms | 21.904 tok/s | 16 | 469.302 |

### Result

GPU greedy sampling removed the readback bottleneck exactly as intended: SmolLM2 dropped from 198 to 32 readbacks per 27-prompt + 32-generated-token trial, and Qwen dropped from 527 to 16 readbacks per 27-prompt + 16-generated-token trial. The extra candidate-reduction dispatches are worthwhile for Qwen (+7.9% decode) and modestly positive for SmolLM2 in the paired run (+3.5% decode). TinyStories remains noisy across runs because the whole trial is short, but the paired run improved decode by 119.2%.

The result also narrows the next target: once readbacks are reduced to one token ID per sampled token, remaining decode cost is again dominated by model kernels and dispatch count rather than CPU-side logit transfer.

## Optimization 4: chunked sequential prefill mode

The TSL runners now expose a prompt-ingestion path that uploads a chunk of prompt embeddings once, advances a GPU-visible position cursor, and replays the existing logits-free layer graph in a single compute submission per prefill chunk. This keeps the same one-token math as the previous prefill path, but changes prompt ingestion from one CPU embedding upload and one compute submission per prompt token to one upload and one submission per 32-token chunk.

The benchmark harness now supports exact synthetic prompt lengths with `--promptTokens=...` and `--tokens=0`. Pure prefill runs add one explicit synchronization readback after the prompt graph so the measured duration includes GPU completion; that sync read is reported separately and excluded from model readback counts.

```sh
npm run test-performance-llm -- --models=smollm2 --trials=5 --tokens=0 --promptTokens=32 --headless --gpuSampling=false --prefillMode=false --output=test/benchmark/results/tsl-prefill-old-32.json
npm run test-performance-llm -- --models=smollm2 --trials=5 --tokens=0 --promptTokens=32 --headless --gpuSampling=false --output=test/benchmark/results/tsl-prefill-32.json
npm run test-performance-llm -- --models=smollm2 --trials=5 --tokens=0 --promptTokens=128 --headless --gpuSampling=false --prefillMode=false --output=test/benchmark/results/tsl-prefill-old-128.json
npm run test-performance-llm -- --models=smollm2 --trials=5 --tokens=0 --promptTokens=128 --headless --gpuSampling=false --output=test/benchmark/results/tsl-prefill-128.json
npm run test-performance-llm -- --models=smollm2 --trials=5 --tokens=0 --promptTokens=512 --headless --gpuSampling=false --prefillMode=false --output=test/benchmark/results/tsl-prefill-old-512.json
npm run test-performance-llm -- --models=smollm2 --trials=5 --tokens=0 --promptTokens=512 --headless --gpuSampling=false --output=test/benchmark/results/tsl-prefill-512.json
```

### SmolLM2 pure prefill sweep

| Prompt tokens | Mode | Prefill duration median | Prefill throughput median | Prefill submissions | Prefill dispatches | Model readbacks |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 32 | old per-token loop | 25.2 ms | 1269.841 tok/s | 32 | 13,440 | 0 |
| 32 | chunked prefill | 23.5 ms | 1361.702 tok/s | 1 | 13,504 | 0 |
| 128 | old per-token loop | 49.1 ms | 2606.925 tok/s | 128 | 53,760 | 0 |
| 128 | chunked prefill | 62.6 ms | 2044.728 tok/s | 4 | 54,016 | 0 |
| 512 | old per-token loop | 1840.8 ms | 278.140 tok/s | 512 | 215,040 | 0 |
| 512 | chunked prefill | 279.1 ms | 1834.468 tok/s | 16 | 216,064 | 0 |

### Result

Chunked prefill realizes the intended submission reduction: `N` prompt submissions become `ceil(N / 32)`. The extra copy and cursor-advance kernels add two dispatches per prompt token, so short prompts are close to neutral: 32 tokens improved 6.7%, while 128 tokens regressed 21.5% in this run. At 512 tokens the reduced submission overhead dominates and prefill duration improved 84.8%.

This is still sequential prefill, not batched GEMM. It makes the existing one-token graph cheaper to drive for long prompts and gives a benchmark harness that can measure future true batched-prefill kernels at 32, 128, and 512 prompt tokens.
