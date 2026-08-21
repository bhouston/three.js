import { describe, expect, it } from 'vitest';
import { buildChannelActivations, getChannel } from '../../../../examples/jsm/neural-material/NeuralMaterialFormat.js';
import { classifyMaterialChannels, resolveMaterialChannelNodes } from '../../../../examples/jsm/neural-material/NeuralMaterialSource.js';

/**
 * Proves the actual feature this refactor is for: a caller can compose an
 * entirely custom channel array - new named outputs, with their own
 * activation/type/resolution/application logic, plus a channel that's
 * *always* constant and never trained at all - and every generic entry
 * point (`classifyMaterialChannels`, `resolveMaterialChannelNodes`,
 * `layoutChannels`, `buildChannelActivations`, and the apply-slice loop
 * `NeuralMaterialNodeMaterial`'s constructor runs) handles it correctly,
 * without editing anything in NeuralMaterialFormat.js/NeuralMaterialSource.js
 * itself. Mixes in one real built-in channel (`roughness`, via `getChannel`)
 * alongside the custom ones to prove built-in and caller-supplied channels
 * compose freely in the same array.
 */
describe( 'Addons > Neural > NeuralMaterial > custom channel composition', () => {

	function buildCustomChannels() {

		// A brand-new `float3`/'color'-shaped channel this file invents from
		// scratch - NeuralMaterialFormat.js has no idea "customTint" exists.
		const customTint = {
			key: 'customTint',
			size: 3,
			activation: 'sigmoid',
			type: 'color',
			nodeKeys: [ 'customTintNode' ],
			clampRange: [ 0, 1 ],
			defaultValue: [ 0.5, 0.5, 0.5 ],
			resolveNode: ( material ) => material.customTintNode || { constant: material.customTint || [ 0.5, 0.5, 0.5 ] },
			resolveConstant: ( material ) => material.customTint || [ 0.5, 0.5, 0.5 ],
			applyActive: ( target, sliceNode ) => {

				target.customTintApplied = sliceNode;

			},
			applyConstant: ( target, constantValue ) => {

				target.customTint = constantValue;

			}
		};

		// A channel that's *always* constant - never trained, no nodeKeys
		// presence check at all - for a value a caller wants surfaced through
		// `constantValues`/`applyConstant` on every material regardless of
		// whether anything on the material even resembles a node for it.
		const sensorGain = {
			key: 'sensorGain',
			size: 1,
			alwaysConstant: true,
			constantValue: 2.5,
			applyConstant: ( target, constantValue ) => {

				target.sensorGain = constantValue;

			}
		};

		return [ customTint, sensorGain, getChannel( 'roughness' ) ];

	}

	it( 'classifies a caller-supplied channel active iff its own nodeKeys are set on the material', () => {

		const channels = buildCustomChannels();
		const material = { customTintNode: {} };
		const { activeChannels, constantValues } = classifyMaterialChannels( material, channels );

		expect( activeChannels.map( ( c ) => c.key ) ).toEqual( [ 'customTint' ] );
		expect( constantValues.sensorGain ).toBe( 2.5 );
		expect( constantValues.roughness ).toBe( 1 ); // roughness's own documented default

	} );

	it( 'an alwaysConstant channel never checks nodeKeys and always surfaces via constantValues', () => {

		const channels = buildCustomChannels();
		// A property that would look like "presence" for a normal channel -
		// sensorGain has no nodeKeys at all, so this can't make it active.
		const material = { sensorGainNode: {} };
		const { activeChannels, constantValues } = classifyMaterialChannels( material, channels );

		expect( activeChannels.some( ( c ) => c.key === 'sensorGain' ) ).toBe( false );
		expect( constantValues.sensorGain ).toBe( 2.5 );

	} );

	it( 'resolveMaterialChannelNodes calls each channel\'s own resolveNode, including a custom one', () => {

		const channels = buildCustomChannels();
		const material = { customTint: [ 0.1, 0.2, 0.3 ] };
		const nodes = resolveMaterialChannelNodes( material, channels );

		expect( nodes.customTint ).toEqual( { constant: [ 0.1, 0.2, 0.3 ] } );
		// sensorGain declares no resolveNode - resolveMaterialChannelNodes
		// falls back to converting its literal constantValue into a node.
		expect( nodes.sensorGain ).toBeDefined();

	} );

	it( 'layoutChannels/buildChannelActivations work end-to-end for a fully custom active list', () => {

		const channels = buildCustomChannels();
		const material = { customTintNode: {}, roughnessNode: {} };
		const { activeChannels, totalChannels, packCount } = classifyMaterialChannels( material, channels );

		expect( activeChannels.map( ( c ) => c.key ) ).toEqual( [ 'customTint', 'roughness' ] );
		expect( totalChannels ).toBe( 4 ); // customTint(3) + roughness(1)
		expect( packCount ).toBe( 1 );

		const activations = buildChannelActivations( activeChannels );
		expect( activations ).toEqual( [ 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid' ] );

	} );

	it( 'applyActive/applyConstant round-trip onto a plain target object, simulating NeuralMaterialNodeMaterial\'s generic apply loop', () => {

		const channels = buildCustomChannels();
		const material = { customTintNode: {} };
		const { activeChannels, constantValues } = classifyMaterialChannels( material, channels );
		const activeKeys = new Set( activeChannels.map( ( c ) => c.key ) );

		// This mirrors exactly what NeuralMaterialNodeMaterial's constructor
		// does with a real slices map - here using a plain string stand-in for
		// "the trained slice node" since no network/renderer is involved.
		const target = {};
		for ( const channel of channels ) {

			if ( activeKeys.has( channel.key ) ) channel.applyActive( target, 'SLICE_' + channel.key );
			else channel.applyConstant( target, constantValues[ channel.key ] );

		}

		expect( target.customTintApplied ).toBe( 'SLICE_customTint' );
		expect( target.sensorGain ).toBe( 2.5 );
		// `material` never set `roughness`, so its resolved constant is
		// exactly `roughness`'s own `defaultValue` (1) - a total no-op, so
		// the built-in channel's applyConstant is skipped entirely (see
		// NeuralMaterialFormat.js's `constantEqualsDefault` pass) rather than
		// assigning a `roughnessNode` that would needlessly (and, for a
		// handful of other PBR properties, incorrectly - see that pass's doc
		// comment) differ from what an untouched material already does.
		expect( target.roughnessNode ).toBeUndefined();
		expect( target.roughness ).toBeUndefined();

	} );

	it( 'applyConstant is skipped entirely when the resolved value equals the channel\'s own default (a real regression: see NeuralMaterialFormat.js\'s constantEqualsDefault doc comment)', () => {

		const target = {};
		const roughnessChannel = getChannel( 'roughness' );

		roughnessChannel.applyConstant( target, 1 ); // roughness's own default
		expect( target.roughnessNode ).toBeUndefined();

		roughnessChannel.applyConstant( target, 0.4 ); // a genuine, non-default constant
		expect( target.roughnessNode ).toBeDefined();

	} );

} );
