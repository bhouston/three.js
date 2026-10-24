import { SSGI_QUALITY_PRESETS, resolveSSGISettings } from '../../../../examples/jsm/tsl/utils/SSGISettings.js';

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'TSL', () => {

		QUnit.module( 'SSGISettings', () => {

			QUnit.test( 'relative and authored local distances scale with the model', assert => {

				const relative = { radius: 0.25, thickness: 0.005 };
				for ( const metadata of [ {}, { localScale: 12 }, { localRadius: 3, localThickness: 0.04 } ] ) {

					const original = resolveSSGISettings( relative, 20, metadata, 1 );
					for ( const scale of [ 0.1, 2, 10 ] ) {

						const scaled = resolveSSGISettings( relative, 20 * scale, metadata, scale );
						assert.ok( Math.abs( scaled.radius / scale - original.radius ) < 1e-12, 'Relative search distance is invariant under uniform model scaling' );
						assert.ok( Math.abs( scaled.thickness / scale - original.thickness ) < 1e-12, 'Relative thickness is invariant under uniform model scaling' );

					}

				}

			} );

			QUnit.test( 'authored overrides do not change when unrelated bounds grow', assert => {

				const relative = { radius: 0.25, thickness: 0.005 };
				for ( const metadata of [ { localScale: 12 }, { localRadius: 3, localThickness: 0.04 }, { worldRadius: 3, worldThickness: 0.04 } ] ) {

					assert.deepEqual( resolveSSGISettings( relative, 20, metadata ), resolveSSGISettings( relative, 2000, metadata ), 'Explicit scales or distances isolate transport parameters from scene bounds' );

				}

				const fixed = resolveSSGISettings( relative, 20, { worldThickness: 0.04 }, 10 );
				assert.strictEqual( fixed.thickness, 0.04, 'World-unit thickness deliberately stays fixed under model scaling' );

			} );

			QUnit.test( 'work-budget presets do not author physical distance settings', assert => {

				for ( const preset of Object.values( SSGI_QUALITY_PRESETS ) ) {

					assert.strictEqual( preset.radius, undefined );
					assert.strictEqual( preset.thickness, undefined );

				}

			} );

		} );

	} );

} );
