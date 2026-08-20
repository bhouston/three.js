import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';
import { playwright } from '@vitest/browser-playwright';

const src = fileURLToPath( new URL( './src', import.meta.url ) );
const testUtils = fileURLToPath( new URL( './test/unit/utils', import.meta.url ) );

// Lane is chosen by filename: "*.browser.js" runs in the unit-browser
// project (real Chromium), everything else in unit-node (plain Node).
// See test/unit/VITEST_MIGRATION.md for the full rationale and playbook.
export default defineConfig( {

	resolve: {
		alias: {
			'@src': src,
			'@test-utils': testUtils,
		},
	},

	test: {

		projects: [
			{
				extends: true,
				test: {
					name: 'unit-node',
					environment: 'node',
					include: [ 'test/unit/src/**/*.js', 'test/unit/addons/**/*.js' ],
					exclude: [
						'test/unit/**/*.browser.js',
						// shared (non-test) helper modules imported by test files -
						// add any new ones here rather than under test/unit/src or
						// test/unit/addons directly, since those trees are otherwise
						// assumed to be all test files
						'test/unit/addons/utils/GaussianSplatTestUtils.js',
						'test/unit/utils/std-geometry-tests.js',
					],
					setupFiles: [ './test/unit/vitest-setup.js' ],
				},
			},
			{
				extends: true,
				test: {
					name: 'unit-browser',
					include: [ 'test/unit/src/**/*.browser.js', 'test/unit/addons/**/*.browser.js' ],
					setupFiles: [ './test/unit/vitest-setup.js' ],
					browser: {
						enabled: true,
						headless: true,
						provider: playwright(),
						instances: [
							{ browser: 'chromium' },
						],
					},
				},
			},
			{
				extends: true,
				test: {
					name: 'e2e',
					environment: 'node',
					include: [ 'test/e2e/e2e.test.js' ],
					// Examples own their timeouts internally (networkTimeout,
					// renderTimeout); a per-test vitest timeout would also have to
					// cover time spent queued behind other examples on the same
					// lane, so it's disabled here in favor of the job-level
					// timeout-minutes in CI.
					testTimeout: 0,
					hookTimeout: 5 * 60000,
					teardownTimeout: 60000,
					// vitest only runs 5 `test.concurrent` bodies at once by
					// default, silently overriding the e2e.test.js lane pool -
					// confirmed by timestamping actual start times, only 5 of 8
					// requested lanes were ever active concurrently until this
					// was raised. Passing --maxConcurrency on the CLI does NOT
					// reach a --project sub-config, so it has to be set here.
					// 64 comfortably covers any realistic E2E_WORKERS value;
					// real concurrency is still bounded by the lane pool itself.
					maxConcurrency: 64,
				},
			},
		],

	},

} );
