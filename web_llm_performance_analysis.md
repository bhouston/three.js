# WebLLM performance advantage analysis

Date: 2026-09-02

## Scope

This report compares the three.js TSL LLM implementation with WebLLM. It uses:

- the current uncommitted three.js LLM worktree and measurements in `llm_performance.md`
- WebLLM commit `fa123eba72978920c6c9f1b1e96c0c7c6be41f0f`
- WebLLM 0.2.84 and its WebLLM paper revision from March 2026

No direct WebLLM versus TSL throughput ratio has been measured yet. A valid ratio requires the same model architecture, weight precision, context, sampling policy, browser, and hardware. WebLLM's published numbers compare WebLLM with native MLC-LLM, not with this project.

## Summary

WebLLM has four substantial performance advantages:

1. It compiles a model ahead of time into model-specific WebGPU kernels.
2. It uses F16 and 4-bit weight formats instead of converting every model to F32.
3. It has separate batched prefill and one-token decode programs.
4. It keeps logit processing and sampling on the GPU, then reads back one token ID.
5. It selects compiled libraries for available WebGPU features such as `shader-f16` and subgroups.

The current TSL implementation has already removed one gap. It records every forward pass in one compute pass, reducing 427 command submissions to one for SmolLM2. WebLLM's worker support and browser cache improve responsiveness and startup behavior, but they do not explain decode throughput.

The next comparison should use SmolLM2-135M. Both projects support it, and WebLLM provides `q0f32` and `q0f16` variants. This allows a near-matched test before introducing 4-bit quantization.

## What the current TSL measurements establish

The retained optimizations already produced large gains:

| Metric | Baseline | Current | Gain |
| --- | ---: | ---: | ---: |
| TinyStories decode | 119.507 tok/s | 565.693 tok/s | 4.73x |
| SmolLM2-135M decode | 33.532 tok/s | 75.407 tok/s | 2.25x |
| SmolLM2 warm TTFT | 759.8 ms | 293.1 ms | 61.4% lower |
| SmolLM2 submissions per forward | 427 | 1 | 99.8% lower |

Those results show that JavaScript-to-WebGPU submission and synchronization overhead dominated the original implementation. They also show that isolated kernel improvements do not guarantee a model-level gain. The attempted shared-workgroup normalization and vec4 GEMV kernels did not improve representative SmolLM2 throughput.

The current SmolLM2 decode path still performs:

- 427 compute dispatches per forward
- six full-vocabulary readbacks per sampled token
- one forward pass per prompt token
- F32 storage and arithmetic after converting the BF16 checkpoint

The worktree now removes one prefill-only cost: intermediate prompt tokens advance the layer and KV-cache state without running the final norm or vocabulary projection. The final prompt token still computes logits so sampling behavior does not change.

Sources: `llm_performance.md`, `examples/jsm/gpgpu/llm/DecoderTSLRunner.js`, and `examples/jsm/gpgpu/llm/LLMGenerate.js`.

## Where WebLLM gets its gains

### 1. Ahead-of-time model compilation

WebLLM does not construct a model from general operators in the browser. MLC-LLM and TVM compile each model into converted weights and a model-specific WASM library. The WASM library contains the WGSL kernels and CPU runtime code.

WebLLM selects different libraries for model, quantization, and context configuration. For example, its catalog points Llama-3.2-1B `q4f16`, `q4f32`, `q0f16`, and `q0f32` variants at different `model_lib` WASM files (`web-llm/src/config.ts:359-408`).

This compilation stage can apply:

- graph-level kernel fusion
- operator-level GEMM and GEMV tiling
- different schedules for prefill and decode
- model-specific attention and cache kernels
- layouts chosen together with quantization

The WebLLM TypeScript repository does not contain most generated WGSL. It loads hosted model libraries and invokes their exported functions. Claims about exact tile sizes, workgroup layouts, or dispatch counts require inspecting a specific compiled model artifact or profiling it at runtime.

Why it matters here: the TSL runner composes general kernels such as `TSLLinear`, `TSLRMSNorm`, `TSLAttention`, and `TSLAdd`. SmolLM2 executes 427 dispatches even though one command buffer now contains them. Pipeline switches, bind-group updates, intermediate buffer traffic, and dispatch overhead remain.

### 2. Reduced-precision and quantized weights

The TSL loader converts F16 and BF16 tensors to F32 (`examples/jsm/gpgpu/llm/LLMTensors.js:82-151`). Every dense layer then stores weights in a float storage buffer (`examples/jsm/gpgpu/llm/TSLLinear.js:21-27`).

WebLLM ships several compiled formats:

- `q0f32`: unquantized weights with F32 computation
- `q0f16`: unquantized weights with F16 computation
- `q4f32_1`: 4-bit weights with F32 computation
- `q4f16_1`: 4-bit weights with F16 computation

Its catalog exposes the memory effect. For Llama-3.2-1B at a 4096-token context:

| WebLLM variant | Required VRAM | Relative to q0f32 |
| --- | ---: | ---: |
| q0f32 | 5106.26 MB | 1.00x |
| q0f16 | 2573.13 MB | 0.50x |
| q4f32_1 | 1128.82 MB | 0.22x |
| q4f16_1 | 879.04 MB | 0.17x |

Source: `web-llm/src/config.ts:359-408`.

The figures include more than weights, so they do not represent compression ratios alone. They show the practical capacity advantage: `q4f16_1` needs 5.8 times less total VRAM than `q0f32` in this configuration.

Autoregressive decode usually reads most model weights for each token. Reduced weight traffic therefore improves decode speed as well as capacity. The exact gain depends on dequantization cost, GPU bandwidth, shader F16 support, and dispatch overhead. The current 135M TSL model may see a smaller proportional gain than a 1B model because its 427 dispatches and sampling synchronization consume a larger part of each token.

### 3. Batched prefill

The TSL generator calls `computeToken` once for every uncached prompt token (`examples/jsm/gpgpu/llm/LLMGenerate.js:164-174`). It uses the same one-token kernels for prefill and decode.

WebLLM loads separate `prefill` or `batch_prefill` and `decode` or `batch_decode` functions (`web-llm/src/llm_chat.ts:224-295`). It reads `prefill_chunk_size` from compiled metadata and sends prompt chunks through the prefill program (`web-llm/src/llm_chat.ts:350-358, 872-898`).

Batched prefill converts repeated matrix-vector work into matrix-matrix work. GPUs can reuse weights across prompt tokens and expose more parallel work. A causal attention kernel can also process the prompt as a block instead of replaying one-token decode.

The TSL prefill used to have another avoidable cost. `generateAsync` only reads logits after the final prompt token, but the runner computed final norm and every logit chunk for each intermediate prompt token. The current worktree adds a logits-free prefill compute list for decoder and Qwen runners. For SmolLM2 this skips seven dispatches per intermediate prompt token: one final RMSNorm plus six 8192-token vocabulary chunks. A 27-token prompt should therefore remove 182 prefill dispatches before any batched GEMM work.

This should produce WebLLM's largest TTFT advantage on medium and long prompts. It contributes little to steady-state decode because decode still has one new token at each step.

### 4. GPU-side logit processing and sampling

The current TSL path computes vocabulary chunks on the GPU, maps every chunk to JavaScript, combines a full `Float32Array`, applies penalties, and samples on the CPU:

- chunk creation and readback: `examples/jsm/gpgpu/llm/TSLLogits.js`
- CPU penalties and top-k: `examples/jsm/gpgpu/llm/LLMMath.js:437-596`

For SmolLM2, each token reads 49,152 F32 logits, or 192 KiB, through six copy-and-map operations. Parallel mapping improved decode by 10.6%, which confirms that synchronization matters more than the byte count.

WebLLM exports GPU functions for bitmasks, penalties, logit bias, softmax, sorting, and top-p sampling (`web-llm/src/llm_chat.ts:230-244, 300-334`). Its normal path copies one sampled `int32` token to the CPU (`web-llm/src/llm_chat.ts:1912-1969`). It reads the full probability vector only when a CPU logit processor or requested log probabilities require it.

This is a transferable WebLLM advantage. The best TSL greedy path would use a hierarchical GPU argmax and read four bytes. A top-k path could reduce each vocabulary chunk to a small candidate set, merge candidates on the GPU, and read one token ID or a few candidates. It does not need WebLLM's full argsort when the API only supports top-k.

### 5. Fused graph and fewer intermediate buffers

The SmolLM2 TSL graph runs about 14 dispatches per decoder layer:

- normalization
- QKV projection
- four attention stages
- attention output projection
- residual add
- second normalization
- gate and up projections
- SiLU multiply
- down projection
- residual add

`TSLGatedMLP` alone uses four dispatches and three intermediate arrays (`examples/jsm/gpgpu/llm/TSLGatedMLP.js`). `TSLAttention` materializes query, scores, and output buffers and uses four stages (`examples/jsm/gpgpu/llm/TSLAttention.js:79-96`).

TVM can fuse compatible graph operations and choose schedules that keep intermediate values in registers or workgroup memory. Useful TSL fusion candidates include:

- one packed gate-plus-up projection
- projection plus residual addition
- residual addition plus the next normalization
- Q/K transformation, RoPE, and KV-cache write
- attention score, softmax, and value accumulation where the target GPU benefits

The normalization experiment in `llm_performance.md` warns against assuming that every reduction rewrite helps Apple GPUs. Fusion should target dispatch count and memory traffic first, then use GPU timestamps to confirm the effect.

### 6. Attention and KV-cache specialization

WebLLM creates a paged KV cache with compiled cache functions and supports sliding windows (`web-llm/src/llm_chat.ts:399-439`). The WebLLM paper states that MLC-LLM can compile PagedAttention and FlashAttention implementations.

The current TSL cache uses contiguous F32 key and value arrays per layer. That design is reasonable for one sequence and a fixed context. Paged KV storage offers a larger advantage for multiple sequences, variable-length batching, and cache memory management than for this single-user demo.

Two attention issues matter more for the current design:

1. `TSLAttention` dispatches score work for `headCount * maxTokens` at every position and discards inactive tokens in the shader (`TSLAttention.js:259-306`). Early tokens can launch far more invocations than needed.
2. Each output dimension repeats the softmax max and denominator loops (`TSLAttention.js:310-370`). The redundant work grows with context length.
3. Cache reset fills each CPU mirror with zero and marks the whole storage buffer for upload (`TSLAttention.js:383-391`). For SmolLM2 with 30 layers, 576 cached values per token, and an 8192-token context, the key and value buffers total about 1.05 GiB. Clearing logical cache length would avoid touching and uploading those buffers.

A runtime score dispatch count of `headCount * (position + 1)` is a small, testable improvement. A fused decode-attention kernel becomes more valuable at long contexts. Batched or flash-style attention matters most for prefill.

### 7. Pipeline loading and artifact caching

WebLLM asks its runtime to load all WebGPU pipelines during model initialization (`web-llm/src/llm_chat.ts:716-718`, called from `web-llm/src/engine.ts:399-414`). This moves shader compilation before the first request and makes first-token latency predictable.

It also caches converted model tensors through Cache API, IndexedDB, OPFS, or an experimental cross-origin backend. The TSL loader uses browser HTTP caching for source files but repeats safetensors parsing, BF16/F16 conversion, transposition, packing, and runner construction after a reload.

The WebGPU runtime reads sharded tensor-cache metadata, runs four parallel download loops, decodes each shard, and uploads tensors to the GPU cache. This reduces serial network and preparation time. The implementation lives in `@mlc-ai/web-runtime`; WebLLM enters it through `fetchTensorCache` (`web-llm/src/engine.ts:394-397`).

These techniques reduce cold-start and reload time. They do not increase warm decode throughput.

### 8. Capability-specific runtime variants

WebLLM checks each model's `required_features` before loading it. Models that need `shader-f16` declare that requirement in the catalog, and the subgroup example swaps a base model library for an `sg32` library when the adapter supports subgroups.

This lets the compiler generate separate kernels for a conservative WebGPU baseline and newer GPU features. A TSL implementation can use the same policy: choose F32, F16, and subgroup paths after renderer initialization rather than forcing one shader design across all adapters. The compiled subgroup and F16 kernel details remain inside model WASM files, so the checkout proves variant selection but not their exact schedules.

### 9. Web Workers and WebAssembly

WebLLM can put its engine in a dedicated worker or service worker. The worker prevents tokenizer, runtime, and sampling work from blocking UI updates. WASM handles CPU-heavy sequence management, tensor manipulation, and grammar processing.

Workers improve responsiveness. A worker can reduce incidental main-thread delays, but message passing and GPU queue serialization mean it should not be counted as a GPU throughput optimization. A service worker mainly preserves model lifetime and cache state across page visits.

The current example also renders a full-screen scene continuously while generating (`examples/webgpu_llm.html:330-347`). The benchmark does not render a scene (`test/benchmark/webgpu_llm.html:150-158`). Removing or pausing the decorative render loop would improve application throughput without changing the LLM implementation.

## Advantages that the TSL implementation already shares

The comparison should not attribute these gains only to WebLLM:

- Both systems use WebGPU compute.
- The TSL runner now records a whole forward pass in one compute pass.
- Both runtimes batch dispatch encoding until a synchronization boundary. WebLLM's runtime reuses a command encoder and submits it when `device.sync()` flushes pending work.
- Both systems retain a KV or recurrent cache between tokens.
- Both systems reuse cached conversation prefixes.
- Both systems compile and reuse WebGPU pipelines after first use.
- Both systems stream tokens to the UI.

The current TSL design also accepts ordinary Hugging Face checkpoints at runtime. WebLLM requires converted weights and a compatible precompiled model library. TSL trades peak performance for deployment simplicity, inspectable JavaScript kernels, and rapid model experimentation.

## Ranked sources of a likely WebLLM lead

### Decode throughput

1. F16 or 4-bit weight bandwidth
2. Model-specific GEMV schedules and graph fusion
3. GPU-side sampling with one scalar readback
4. Fewer dispatches and intermediate buffers
5. Specialized long-context attention

### Prompt prefill and TTFT

1. Batched prefill and GEMM schedules
2. Skipping intermediate prompt-token vocabulary projections
3. Batched causal or flash-style attention
4. Ahead-of-time pipeline loading
5. Reduced-precision weights

### Model load and memory

1. Quantized, preconverted artifacts
2. Persistent tensor cache
3. Parallel sharded download and upload
4. No browser-side BF16-to-F32 expansion
5. Ahead-of-time layouts with no runtime transpose or repack

## Phase execution plan

### Phase 1: closest full-model comparison

Use SmolLM2-135M because both projects support it.

- TSL: current Hugging Face SmolLM2-135M path
- WebLLM: `SmolLM2-135M-Instruct-q0f32-MLC`
- Then WebLLM: `SmolLM2-135M-Instruct-q0f16-MLC`

WebLLM lists both models in `web-llm/src/config.ts:989-1013`. The Instruct checkpoint is not byte-identical to the TSL base checkpoint, but it has the same architecture and parameter scale. The `q0f32` run should isolate compiler, fusion, prefill, and sampling advantages. The `q0f16` run then measures reduced precision.

Run both with:

- the same browser build and GPU
- greedy sampling
- matching context limit
- the same tokenized prompt length
- a warm compile run
- at least five warm trials
- separate TTFT, prefill, decode, and end-to-end rates

Record GPU memory, command submissions, dispatches, and readback bytes if the runtime exposes them.

Commands now available in this worktree:

```sh
npm run test-performance-llm -- --models=smollm2 --trials=5 --tokens=32 --headless --output=tsl-smollm2.json
npm run test-performance-webllm -- --models=SmolLM2-135M-Instruct-q0f32-MLC,SmolLM2-135M-Instruct-q0f16-MLC --trials=5 --tokens=32 --headless --output=webllm-smollm2.json
```

The WebLLM harness imports WebLLM 0.2.84 from ESM by default. Pass `--module=<url>` to test a local bundle or another pinned build.

### Phase 2: exact larger-model comparison

Both catalogs contain Qwen3.5-0.8B and Phi-1.5. These runs test the regime where weight bandwidth and memory capacity matter more. WebLLM provides Qwen3.5-0.8B `q0f16`, `q4f32`, and `q4f16` variants, while the current TSL loader expands BF16 weights to F32.

Suggested commands:

```sh
npm run test-performance-llm -- --models=qwen3.5-0.8b,phi-1.5 --trials=5 --tokens=32 --headless --output=tsl-larger.json
npm run test-performance-webllm -- --models=Qwen3.5-0.8B-q0f16-MLC,Qwen3.5-0.8B-q4f32_1-MLC,Qwen3.5-0.8B-q4f16_1-MLC,phi-1_5-q4f32_1-MLC,phi-1_5-q4f16_1-MLC --trials=5 --tokens=32 --headless --output=webllm-larger.json
```

### Phase 3: component attribution

Use focused experiments to explain the full-model gap:

1. Compare one-token F32 GEMV for representative matrix shapes.
2. Compare sequential one-token prefill with batched prefill at 32, 128, and 512 tokens.
3. Measure TSL prefill with and without intermediate vocabulary projections.
4. Measure full-logit readback versus GPU argmax.
5. Sweep decode position through 1, 32, 128, 512, and the context limit.
6. Measure cache reset time and bytes uploaded.
7. Capture GPU timestamps by kernel category: linear, attention, normalization, pointwise, logits, and sampling.

This sequence distinguishes compiler scheduling, precision, synchronization, and long-context attention instead of assigning the entire gap to quantization.

Item 3 has been implemented for the TSL runners. Re-run the existing TSL benchmark and compare `dispatchesPerForward`, TTFT, and `prefillTokensPerSecond` against the earlier Optimization 2 numbers in `llm_performance.md`. Decode throughput should stay close to the previous result because generated tokens still require logits.

## Conclusions

WebLLM's main lead comes from an offline ML compiler pipeline. It specializes the model, precision, memory layout, prefill program, decode program, attention implementation, and sampling path before the browser loads the model. The current TSL runner makes those choices at runtime from reusable operators and F32 arrays.

The TSL submission batching work closed a major runtime gap. The remaining high-value differences are GPU-side sampling, batched prefill, reduced-precision weights, and graph fusion. Quantization should dominate model capacity and larger-model decode. Batched prefill should dominate TTFT. GPU sampling offers the clearest small experiment because the existing measurements already show sensitivity to readback synchronization.

No source evidence supports a numeric WebLLM-over-TSL throughput claim yet. The proposed SmolLM2 `q0f32` benchmark can supply that number without mixing compiler gains with quantization.

## References

- Three.js measurements: `llm_performance.md`
- Three.js example: `examples/webgpu_llm.html`
- Three.js LLM implementation: `examples/jsm/gpgpu/llm/`
- WebLLM checkout: `../web-llm/`
- WebLLM repository: <https://github.com/mlc-ai/web-llm>
- WebLLM paper: <https://arxiv.org/html/2412.15803v2>
