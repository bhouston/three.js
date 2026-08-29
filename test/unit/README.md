## Setup

- Execute `npm install` from the root folder

## Run

You can run the unit tests in two ways:

- Headless: Execute `npm run test-unit`, `npm run test-unit-addons` from the root folder.  
  In headless mode the tests will run in a headless browser.
- Headful: Execute `npm run test-unit-headful`, `npm run test-unit-addons-headful` from the root folder.  
  In headful mode, a browser window will open, and you can see the tests running.  
  While the headful mode is running you can also use any browser to navigate to http://localhost:8080/test/unit/UnitTests.html or http://localhost:8080/test/unit/UnitTestsAddons.html to run the tests in that browser.  
  Further changes to the library will not be reflected until the page is refreshed.

See [Installation](https://threejs.org/docs/#manual/introduction/Installation) for more information.

## Performance tests

The GPGPU addons (`PrefixSum`, `CountingSort`, `BitonicSort`) also have performance tests, separate
from the regular addons suite since they need a real GPU and run large (~1M element) benchmarks
repeatedly. Run them with `npm run test-unit-addons-perf` (headless) or
`npm run test-unit-addons-perf-headful` (opens a browser window). Each test logs timing stats
(min/max/mean/median over many runs) to the console; see `test/unit/addons/gpgpu/perf-utils.js`
and the `*.perf.tests.js` files to add more or change sizes/run counts.

## Notes

A small number of tests can only be run in a browser environment.

For browser tests, further changes to the library will not be reflected until the page is refreshed.

## Troubleshooting

When adding or updating tests, the most common cause of test failure is forgetting to change `QUnit.todo` to `QUnit.test` when the test is ready.

An error that indicates "no tests were found" means that an import statement could not be resolved. This is usually caused by a typo in the import path.

## Debugging

To debug a test, add `debugger;` to the test code. Then, run the test in a browser and open the developer tools. The test will stop at the `debugger` statement and you can inspect the code.

