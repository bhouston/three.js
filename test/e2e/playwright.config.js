import { defineConfig } from '@playwright/test';
import { fileURLToPath } from 'node:url';

export default defineConfig( {

	testDir: '.',
	testMatch: 'examples.spec.js',
	fullyParallel: true,
	workers: 1,
	retries: 2,
	timeout: 360000,
	outputDir: 'results',
	reporter: 'list',
	use: {
		browserName: 'chromium',
		channel: process.env.THREE_BROWSER_CHANNEL,
		headless: ! process.env.VISIBLE,
		viewport: { width: 800, height: 500 },
		baseURL: 'http://localhost:1234',
		launchOptions: {
			args: [ '--hide-scrollbars', '--enable-unsafe-webgpu', ...( process.platform === 'win32' ? [ '--use-angle=d3d11' ] : [] ) ]
		},
		trace: 'retain-on-failure'
	},
	webServer: {
		command: 'node utils/server.js -p 1234',
		url: 'http://localhost:1234',
		reuseExistingServer: ! process.env.CI,
		cwd: fileURLToPath( new URL( '../../', import.meta.url ) )
	}

} );
