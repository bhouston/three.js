
// Performance tests for the GPGPU addons (PrefixSum/CountingSort/BitonicSort). Unlike the regular
// addons unit tests, these need a real GPU and can take a while at large element counts - see
// `test/unit/UnitTestsAddonsPerf.html` and the `test-unit-addons-perf*` npm scripts.

//addons/gpgpu
import './addons/gpgpu/PrefixSum.perf.tests.js';
import './addons/gpgpu/CountingSort.perf.tests.js';
import './addons/gpgpu/BitonicSort.perf.tests.js';
