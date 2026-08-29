
// Performance tests for the GPGPU addons (CountingSort/BitonicSort). Unlike the regular addons
// unit tests, these need a real GPU and can take a while at large element counts - see
// `test/unit/UnitTestsAddonsPerf.html` and the `test-unit-addons-perf*` npm scripts.
//
// This branch has no standalone PrefixSum class (CountingSort inlines its prefix sum as a single-
// dispatch loop instead), so there's nothing to benchmark separately here.

//addons/gpgpu
import './addons/gpgpu/CountingSort.perf.tests.js';
import './addons/gpgpu/BitonicSort.perf.tests.js';
