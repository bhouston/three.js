// Performance tests for the GPGPU addons (currently FFT2D). Unlike the regular addons unit
// tests, these need a real GPU and can take a while at large sizes - see
// `test/unit/UnitTestsAddonsPerf.html` and the `test-unit-addons-perf*` npm scripts.

//addons/gpgpu
import './addons/gpgpu/FFT2D.sanity.tests.js';
import './addons/gpgpu/FFT2D.perf.tests.js';
