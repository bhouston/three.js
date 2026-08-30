
// Real-GPU correctness regression tests for the GPGPU addons (PrefixSum/CountingSort). Unlike the
// mocked-renderer correctness tests in three.addons.unit.js, these build a real WebGPURenderer and
// run actual dispatches - see test/unit/UnitTestsAddonsGPU.html and the test-unit-addons-gpu*
// npm scripts. Separate from three.addons.perf.unit.js: these assert correctness, not timing.

//addons/gpgpu
import './addons/gpgpu/PrefixSum.gpu.tests.js';
import './addons/gpgpu/CountingSort.gpu.tests.js';
