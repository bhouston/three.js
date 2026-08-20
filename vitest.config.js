import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';
import { playwright } from '@vitest/browser-playwright';

const src = fileURLToPath( new URL( './src', import.meta.url ) );
const testUtils = fileURLToPath( new URL( './test/unit/utils', import.meta.url ) );

// Lane is chosen by filename: "*.browser.js" runs in the unit-browser
// project (real Chromium), everything else in unit-node (plain Node).
// See test/unit/VITEST_SPIKE.md for the full rationale and playbook.
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
					include: [ 'test/unit/src/**/*.js' ],
					exclude: [ 'test/unit/src/**/*.tests.js', 'test/unit/src/**/*.browser.js' ],
					setupFiles: [ './test/unit/vitest-setup.js' ],
				},
			},
			{
				extends: true,
				test: {
					name: 'unit-browser',
					include: [ 'test/unit/src/**/*.browser.js' ],
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
		],

	},

} );
